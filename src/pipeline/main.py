"""Unified pipeline entry point — assembles providers, hooks, and runs.

This is the single place in the codebase that knows about specific LLM /
VLM providers.  Everything below this file consumes ABCs from
``src.llm.providers.base`` and ``src.pipeline.hooks``.

Run::

    python -m src.pipeline.main path/to/paper.pdf
    python -m src.pipeline.main path/to/paper.pdf --skip-tfid
    python -m src.pipeline.main path/to/paper.pdf --force-assembly
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import logging
import os
import subprocess
import sys
import time

# Ensure src is importable from project root
sys.path.insert(0, os.getcwd())

# MUST precede any torch / paddle / cv2 import: caps per-backend thread
# pools so several model libraries don't each spawn one thread per core
# (the load=30 oversubscription root cause).
import src.pipeline._threadcaps  # noqa: F401  (import for side effect)

from src.adjudication.global_assembly import GlobalAssembly
from src.adjudication.local_vars_builder import LocalVarsBuilder
from src.adjudication.pdf_parser import PDFParser
from src.adjudication.post_processor import PostProcessor
from src.extraction.common.content_recognizer import ContentRecognizer
from src.extraction.fusion import PointCountConsistency
from src.extraction.table.pipeline import TablePipeline
from src.extraction.table.scheme_seg_parser import SchemeSegParser
from src.llm.concurrency import configure_concurrency
from src.llm.config import load_label_reader_config, load_llm_config, load_vlm_config
from src.llm.fusion import ModalityRoutingPolicy
from src.llm.hooks import FigureInspectionHook, TableInspectionHook
from src.llm.inspectors import (
    ExactCellMatcher,
    FigureInspector,
    NearestPointMatcher,
    TableInspector,
)
from src.llm.providers.gemini import GeminiProvider
from src.parsing.active_area_detector import ActiveAreaDetector
from src.pipeline._memory import release_memory
from src.pipeline.figure_pipeline import FigurePipeline
from src.utils.config import load_config

logger = logging.getLogger(__name__)

CONFIG_PATH = "config.yaml"

# Exit code emitted when a PDF is skipped by the Step 0 pre-filter (review /
# non-flow paper).  batch_all.py maps this code to the ``skipped`` bucket so
# a deliberately-skipped paper isn't counted as a success (0) or a crash.
EXIT_SKIPPED = 3


# ───────────────────────────────────────────────────────────────── builders


def _build_provider_stack(no_vlm_inspection: bool):
    """Assemble the LLM/VLM provider and the two inspection hooks.

    Returns ``(provider, llm_cfg, vlm_cfg, figure_hooks, table_hooks, label_reader)``.
    ``label_reader`` is the VLM second reader for chart tick / cell labels
    (``vlm.label_reader`` in config.yaml; None when disabled or ``--no-vlm``).
    The hook lists are empty when ``no_vlm_inspection`` is true, so the
    pipelines run their classic extraction-only flow without ever
    contacting Gemini for inspection.  The LLM provider is always
    constructed (adjudication still needs it).
    """
    llm_cfg = load_llm_config(CONFIG_PATH)
    vlm_cfg = load_vlm_config(CONFIG_PATH)

    # Set the process-wide LLM concurrency cap.  The value is read from
    # config.yaml (``llm.max_concurrent``) so a single knob governs every
    # Gemini call site — adjudication, inspection, metadata, header judge.
    import yaml
    raw_cfg = yaml.safe_load(open(CONFIG_PATH))
    configure_concurrency(int(raw_cfg.get("llm", {}).get("max_concurrent", 5)))

    provider = GeminiProvider()   # GEMINI_API_KEY pulled from env
    logger.info(
        "main.providers gemini llm_model=%s vlm_model=%s",
        llm_cfg.model,
        vlm_cfg.model,
    )

    if no_vlm_inspection:
        logger.info("main.providers VLM inspection disabled by --no-vlm")
        return provider, llm_cfg, vlm_cfg, [], [], None

    label_reader = _build_label_reader(provider)

    fig_inspector = FigureInspector(
        vlm=provider,
        cfg=vlm_cfg,
        matcher=NearestPointMatcher(tol=0.05),
        policy=ModalityRoutingPolicy(numeric_tol=0.05),
    )
    tab_inspector = TableInspector(
        vlm=provider,
        cfg=vlm_cfg,
        matcher=ExactCellMatcher(),
        policy=ModalityRoutingPolicy(numeric_tol=0.05),
    )
    figure_hooks = [
        FigureInspectionHook(
            inspector=fig_inspector,
            consistency_checks=[PointCountConsistency()],
        )
    ]
    table_hooks = [
        TableInspectionHook(
            inspector=tab_inspector,
            consistency_checks=[],
        )
    ]
    return provider, llm_cfg, vlm_cfg, figure_hooks, table_hooks, label_reader


def _build_label_reader(default_provider):
    """VLM second reader for tick / heatmap-cell labels (design: VLM reads
    symbols on a numbered contact sheet, YOLO + robust_fit keep the geometry)."""
    try:
        lr_cfg = load_label_reader_config(CONFIG_PATH)
    except Exception as exc:
        logger.warning("main.label_reader config error: %s — disabled", exc)
        return None
    if not lr_cfg.enabled:
        logger.info("main.label_reader disabled")
        return None
    from src.extraction.figure.label_reader import VLMLabelReader
    vlm = default_provider
    if lr_cfg.provider != getattr(default_provider, "name", "gemini"):
        from src.llm.providers import get_vlm_provider
        try:
            vlm = get_vlm_provider(lr_cfg.provider)
        except Exception as exc:
            logger.warning("main.label_reader provider '%s' unavailable (%s) — disabled", lr_cfg.provider, exc)
            return None
    logger.info("main.label_reader provider=%s model=%s policy=%s", lr_cfg.provider, lr_cfg.model,
                lr_cfg.value_conflict_policy)
    reader = VLMLabelReader(vlm=vlm, cfg=lr_cfg, max_boxes=lr_cfg.max_boxes)
    reader.value_conflict_policy = lr_cfg.value_conflict_policy
    return reader


# ───────────────────────────────────────────────────────────────── steps


def run_step1_tfid(pdf_path: str) -> bool:
    print("\n=== Step 1: TF-ID Parsing ===")
    try:
        cfg = load_config()
        detector = ActiveAreaDetector()
        detections = detector.process_pdf(pdf_path)
        base_intermediate_dir = cfg.get("global", {}).get("intermediate_dir", "data/intermediate")
        basename = os.path.splitext(os.path.basename(pdf_path))[0]
        intermediate_dir = os.path.join(base_intermediate_dir, basename)
        saved_paths = detector.save_crops(pdf_path, detections, intermediate_dir)
        print(f"Saved {len(saved_paths)} crops (figures/tables) to {intermediate_dir}")
        # Resolve each crop to its real caption / label / footnote from the
        # PDF text layer (deterministic; feeds the text-window anchors and
        # both adjudication prompts).
        try:
            from src.parsing.caption_locator import write_contexts
            write_contexts(pdf_path, intermediate_dir)
        except Exception as exc:
            print(f"[CaptionLocator] failed (non-fatal): {exc}")
        # Release Florence-2 (~2GB) immediately — it is used only here.
        del detector
        release_memory()
        return True
    except Exception as exc:
        print(f"Step 1 Failed: {exc}")
        return False


def run_step44_global_vars(pdf_path: str, intermediate_dir: str, provider, llm_cfg, rebuild: bool = False) -> dict:
    """Step 4.4 — one LLM call per paper: paper-level reaction context and
    default conditions (each value with a verbatim quote and a scope), written
    to ``{intermediate_dir}/global_vars.json``.  See ``GlobalVarsBuilder``."""
    from src.adjudication.global_vars_builder import GlobalVarsBuilder
    from src.adjudication.post_processor import build_abbrev_map

    paper_text = PDFParser().extract_text(pdf_path)
    scheme_cond_text = ""
    cond_path = os.path.join(intermediate_dir, "scheme_conditions.txt")
    if os.path.exists(cond_path):
        with open(cond_path) as f:
            scheme_cond_text = f.read().strip()
    try:
        return GlobalVarsBuilder(llm=provider, llm_cfg=llm_cfg).build(
            pdf_path, intermediate_dir, paper_text,
            scheme_conditions=scheme_cond_text, abbrev_map=build_abbrev_map(paper_text), rebuild=rebuild,
        )
    except Exception as exc:
        print(f"   [GlobalVars] failed (non-fatal): {exc}")
        return {}


def run_step45_local_vars(pdf_path: str, intermediate_dir: str, provider, llm_cfg, rebuild: bool = False,
                          global_vars: dict = None) -> None:
    """Step 4.5 — one LocalVarsBuilder call per figure / table source.

    Each source gets its CaptionLocator context (real label, verbatim
    caption, footnote) so the LLM sees the figure's identity and the text
    window is anchored on the paper's own figure number.  ``rebuild=True``
    discards cached ``local_vars/*.json`` first (the cache has no staleness
    check of its own) — used by ``scripts/reassemble.py --rebuild-vars``.
    """
    from src.parsing.caption_locator import load_context
    from src.adjudication.source_discovery import apply_context_to_evidence

    paper_text = PDFParser().extract_text(pdf_path)
    local_vars_dir = os.path.join(intermediate_dir, "local_vars")
    if rebuild and os.path.isdir(local_vars_dir):
        for fn in glob.glob(os.path.join(local_vars_dir, "*_local_vars.json")):
            os.remove(fn)
        print(f"   [LocalVars] rebuild: cleared cached local_vars in {local_vars_dir}")
    os.makedirs(local_vars_dir, exist_ok=True)
    builder = LocalVarsBuilder(llm=provider, llm_cfg=llm_cfg)

    scheme_cond_text = ""
    cond_path = os.path.join(intermediate_dir, "scheme_conditions.txt")
    if os.path.exists(cond_path):
        with open(cond_path) as f:
            scheme_cond_text = f.read().strip()
        print(f"   [LocalVars] Loaded scheme_conditions.txt ({len(scheme_cond_text)} chars)")

    macro_cleaned_dir = os.path.join(intermediate_dir, "macro_cleaned")
    for jpath in glob.glob(os.path.join(macro_cleaned_dir, "*_evidence.json")):
        try:
            with open(jpath) as f:
                ev = json.load(f)
            src_id = ev.get("meta", {}).get(
                "figure_id",
                os.path.basename(jpath).replace("_evidence.json", ""),
            )
            ctx = load_context(intermediate_dir, src_id)
            builder.build(
                src_id, "figure", apply_context_to_evidence(ev, ctx, "figure"), paper_text, local_vars_dir,
                scheme_conditions=scheme_cond_text, context=ctx, global_vars=global_vars,
            )
        except Exception as exc:
            print(f"   [LocalVars] Figure error {jpath}: {exc}")

    tables_dir = os.path.join(intermediate_dir, "tables")
    if os.path.exists(tables_dir):
        for root, _, files in os.walk(tables_dir):
            for fname in files:
                if not fname.endswith("_evidence.json"):
                    continue
                ev_path = os.path.join(root, fname)
                try:
                    with open(ev_path) as f:
                        ev = json.load(f)
                    src_id = fname.replace("_evidence.json", "")
                    csv_path = ev.get("csv_path", "")
                    csv_head = ""
                    if csv_path and os.path.exists(csv_path):
                        with open(csv_path) as cf:
                            csv_head = "".join(cf.readlines()[:6])
                    ctx = load_context(intermediate_dir, src_id)
                    builder.build(
                        src_id, "table", apply_context_to_evidence(ev, ctx, "table"), paper_text, local_vars_dir,
                        csv_head=csv_head, scheme_conditions=scheme_cond_text, context=ctx, global_vars=global_vars,
                    )
                except Exception as exc:
                    print(f"   [LocalVars] Table error {ev_path}: {exc}")



# ───────────────────────────────────────────────────────────────── main


def process_one_pdf(
    pdf_path: str,
    args,
    provider,
    llm_cfg,
    vlm_cfg,
    figure_hooks,
    table_hooks,
    label_reader=None,
) -> str | None:
    """Run Steps 0-6 for ONE PDF.

    Heavy stage models (Florence-2, YOLO, TATR) are loaded and released
    within this call.  Process-level singletons (MolNexTR, and PaddleOCR
    via the ocr_backend cache) persist across calls so a ``--dir`` batch
    reuses them instead of paying MolNexTR's ``torch.load`` per PDF.

    Returns ``"skipped"`` when the Step 0 pre-filter rejects the paper
    (caller maps this to ``EXIT_SKIPPED``); ``None`` on a full run.
    Per-stage wall-clock is written to ``{intermediate_dir}/timing.json``.
    """
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    intermediate_dir = os.path.join("data/intermediate", basename)

    # Per-stage wall-clock (semantic keys, not step numbers — so renaming /
    # inserting a pipeline stage later doesn't leave cryptic "step4.5" keys).
    timings: dict[str, float] = {}
    _last = [time.perf_counter()]

    def _mark(name: str) -> None:
        now = time.perf_counter()
        timings[name] = round(now - _last[0], 1)
        _last[0] = now

    def _write_timing(status: str) -> None:
        out = {"status": status, **timings}
        if status == "ok":
            out["total"] = round(sum(timings.values()), 1)
        try:
            os.makedirs(intermediate_dir, exist_ok=True)
            with open(os.path.join(intermediate_dir, "timing.json"), "w") as f:
                json.dump(out, f, indent=2)
        except Exception as exc:
            print(f"[Timing] write failed: {exc}")
        line = "  ".join(f"{k}={v}s" for k, v in timings.items())
        print(f"[Timing] {status}: {line}"
              + (f"  total={out['total']}s" if "total" in out else ""))

    # ─── Step 0: Pre-filter (skip review articles + non-flow papers) ──
    # This is the EARLY-EXIT gate: it runs before any heavy local weight
    # (Florence-2 / YOLO / TATR / MolNexTR / PaddleOCR) is loaded, so a
    # review / non-flow paper costs only a 3-page text scan, not inference.
    if not getattr(args, "no_prefilter", False):
        from src.preprocessing.paper_filter import filter_paper
        pf = filter_paper(pdf_path)
        if not pf["is_relevant"]:
            print(f"[PreFilter] SKIP {basename}: {pf['reason']}")
            _mark("filter")
            _write_timing("skipped")
            return "skipped"
    _mark("filter")

    # ─── Step 1: TF-ID ──────────────────────────────────────────────
    figures_exist = len(glob.glob(os.path.join(intermediate_dir, "figures", "*.png"))) > 0
    if args.skip_tfid and figures_exist:
        print("\n=== Step 1: TF-ID Parsing (SKIPPED - intermediate figures found) ===")
    elif not run_step1_tfid(pdf_path):
        return

    _mark("tfid")

    # ─── Step 2-4: Figure pipeline (with module-6 hook) ────────────
    print("\n=== Step 2-4: Figure Extraction ===")
    try:
        from src.extraction.figure.metadata_vlm import FigureMetadataExtractor
        fig_metadata_extractor = FigureMetadataExtractor(vlm=provider, cfg=vlm_cfg)
        fig_pipeline = FigurePipeline(
            metadata_extractor=fig_metadata_extractor,
            post_extract_hooks=figure_hooks,
            label_reader=label_reader,
            value_conflict_policy=getattr(label_reader, "value_conflict_policy", "vlm"),
        )
        fig_pipeline.process_pdf_figures(pdf_path)
    except Exception as exc:
        print(f"[FigurePipeline] Error: {exc} — continuing to table extraction.")
    finally:
        try:
            del fig_pipeline
        except NameError:
            pass
        release_memory()

    _mark("figure")

    # ─── Step Table: Table pipeline (with module-12 hook) ──────────
    print("\n=== Step Table: Table Extraction ===")
    shared_content_rec = ContentRecognizer()
    from src.extraction.table.cell_vlm import TableCellExtractor
    from src.extraction.table.header_resolver import HeaderResolver
    tab_cell_extractor = TableCellExtractor(vlm=provider, cfg=vlm_cfg)
    tab_header_resolver = HeaderResolver(llm=provider, cfg=llm_cfg)
    tab_pipeline = TablePipeline(
        cell_extractor=tab_cell_extractor,
        header_resolver=tab_header_resolver,
        sequential_mode=True,
        content_recognizer=shared_content_rec,
        post_extract_hooks=table_hooks,
    )

    tables_dir = os.path.join(intermediate_dir, "tables")
    if os.path.exists(tables_dir):
        table_imgs = glob.glob(os.path.join(tables_dir, "*.png"))
        table_imgs = [f for f in table_imgs if "_body" not in f and "_crop" not in f]
        print(f"Processing {len(table_imgs)} tables...")
        for t_img in table_imgs:
            try:
                tab_pipeline.process_table(t_img, output_dir=tables_dir)
            except Exception as exc:
                print(f"Error processing table {t_img}: {exc}")
    else:
        print("No tables directory found.")

    # Release the table pipeline + its stage models.  The MolNexTR and
    # PaddleOCR singletons are NOT freed (they live in module-level
    # caches for cross-PDF reuse under --dir); this only drops the
    # TablePipeline container and its YOLO/TATR stage references.
    try:
        del tab_pipeline
        del shared_content_rec
    except NameError:
        pass
    release_memory()

    _mark("table")

    # ─── Step 3.5: Tab-Scheme-Seg (unchanged) ──────────────────────
    print("\n=== Step 3.5: Tab-Scheme-Seg (Scheme Parsing) ===")
    reactant_pool: dict = {}
    product_pool: dict = {}
    compound_pool: dict = {}
    scheme_conditions_texts: list[str] = []

    if os.path.exists(tables_dir):
        scheme_imgs = glob.glob(os.path.join(tables_dir, "**", "*_table_scheme_*.png"), recursive=True)
        if scheme_imgs:
            scheme_cfg = load_config().get("tables", {}).get("scheme_parsing", {})
            scheme_parser = SchemeSegParser(
                model_path=scheme_cfg.get("model_path", "models/tab-scheme-seg/best.pt"),
                conf_threshold=scheme_cfg.get("confidence_threshold", 0.3),
            )
            for sp in scheme_imgs:
                print(f"   Processing scheme: {os.path.basename(sp)}")
                result = scheme_parser.parse_scheme(sp)
                reactant_pool.update(result.get("reactant_pool", {}))
                product_pool.update(result.get("product_pool", {}))
                compound_pool.update(result.get("compound_pool", {}))
                ct = result.get("conditions_text", "")
                if ct:
                    scheme_conditions_texts.append(ct)
            del scheme_parser
            release_memory()
            print(
                f"   -> pools: reactant={len(reactant_pool)} product={len(product_pool)} "
                f"compound={len(compound_pool)}"
            )
        else:
            print("   No scheme images found.")

    if reactant_pool or product_pool or compound_pool:
        pool_path = os.path.join(intermediate_dir, "compound_pool.json")
        with open(pool_path, "w") as f:
            json.dump(
                {"reactant_pool": reactant_pool, "product_pool": product_pool, "compound_pool": compound_pool},
                f, indent=2,
            )
        total = len(reactant_pool) + len(product_pool) + len(compound_pool)
        print(f"   -> Compound pool ({total} total) -> {pool_path}")
    if scheme_conditions_texts:
        cond_path = os.path.join(intermediate_dir, "scheme_conditions.txt")
        with open(cond_path, "w") as f:
            f.write("\n".join(scheme_conditions_texts))
        print(f"   -> Scheme conditions saved -> {cond_path}")

    _mark("scheme")

    # ─── Step 4.4: Paper-level pool ────────────────────────────────
    print("\n=== Step 4.4: Build Paper-Level Global Vars ===")
    global_vars = run_step44_global_vars(pdf_path, intermediate_dir, provider, llm_cfg)
    _mark("global_vars")

    # ─── Step 4.5: Sub-Variable Libraries ──────────────────────────
    print("\n=== Step 4.5: Build Sub-Variable Libraries ===")
    run_step45_local_vars(pdf_path, intermediate_dir, provider, llm_cfg, global_vars=global_vars)

    _mark("local_vars")

    # ─── Step 5: Global Assembly ───────────────────────────────────
    print("\n=== Step 5: Global Assembly ===")
    assembler = GlobalAssembly(llm=provider, llm_cfg=llm_cfg)
    assembler.run(pdf_path, intermediate_dir, force=args.force_assembly)
    _mark("assembly")

    # ─── Step 6: Post-processing ───────────────────────────────────
    print("\n=== Step 6: Post-Processing (Normalisation) ===")
    post = PostProcessor()
    post.run(pdf_path, intermediate_dir, smiles_lookup=not args.no_smiles_lookup)
    _mark("post")

    _write_timing("ok")
    print("\n=== Pipeline Complete ===")
    return None


def _run_single(args) -> str | None:
    """Process ONE PDF under the single-instance lock.

    The lock lives here (not in ``main``) so the batch driver can spawn
    subprocesses that each lock themselves and serialise cleanly.
    Returns ``process_one_pdf``'s status (``"skipped"`` or ``None``).
    """
    from src.pipeline.single_instance import single_instance_lock

    with single_instance_lock("flowfigtabminer-pipeline"):
        provider, llm_cfg, vlm_cfg, figure_hooks, table_hooks, label_reader = _build_provider_stack(
            no_vlm_inspection=args.no_vlm
        )
        return process_one_pdf(
            args.pdf_path, args, provider, llm_cfg, vlm_cfg, figure_hooks, table_hooks,
            label_reader=label_reader,
        )


def _run_batch(args) -> None:
    """Process every PDF in ``args.dir`` in a SEPARATE subprocess, serially.

    Subprocess isolation is the durable fix for cross-PDF thread/memory
    accumulation: the in-process loop leaked C++ thread pools (paddle /
    torch / ultralytics) that ``del`` + gc cannot reclaim, pushing the
    load average from ~10 on the first PDF to 44 on the second.  Running
    each PDF as its own ``python -m src.pipeline.main <pdf>`` process lets
    the OS reclaim *everything* on exit — every PDF starts from a clean
    slate.

    The parent (this driver) does NOT take the lock; each child does, so
    the children serialise on the lock and two batches can't run heavy
    work concurrently either.
    """
    pdfs = sorted(glob.glob(os.path.join(args.dir, "*.pdf")))
    if not pdfs:
        print(f"Error: no *.pdf found in {args.dir}")
        return

    passthrough = []
    if args.skip_tfid:
        passthrough.append("--skip-tfid")
    if args.force_assembly:
        passthrough.append("--force-assembly")
    if args.no_smiles_lookup:
        passthrough.append("--no-smiles-lookup")
    if args.no_prefilter:
        passthrough.append("--no-prefilter")
    if args.no_vlm:
        passthrough.append("--no-vlm")

    for idx, pdf in enumerate(pdfs, 1):
        print(f"\n########## [{idx}/{len(pdfs)}] {os.path.basename(pdf)} (subprocess) ##########")
        cmd = [sys.executable, "-m", "src.pipeline.main", pdf] + passthrough
        result = subprocess.run(cmd)
        if result.returncode == EXIT_SKIPPED:
            print(f"[batch] {os.path.basename(pdf)} skipped by pre-filter (review/non-flow)")
        elif result.returncode != 0:
            print(f"[batch] {os.path.basename(pdf)} exited with code {result.returncode}")


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="FlowFigTabMiner Unified Pipeline")
    parser.add_argument("pdf_path", nargs="?", help="Path to a single input PDF")
    parser.add_argument("--dir", help="Process every *.pdf in this directory, "
                                      "each in its own subprocess (serial)")
    parser.add_argument("--skip-tfid", action="store_true",
                        help="Skip Step 1 if intermediate figures already exist")
    parser.add_argument("--force-assembly", action="store_true",
                        help="Force re-run Step 5 LLM even if _final.json already exists")
    parser.add_argument("--no-smiles-lookup", action="store_true",
                        help="Disable PubChem name→SMILES lookup in Step 6 (default: enabled)")
    parser.add_argument("--no-prefilter", action="store_true",
                        help="Disable Step 0 pre-filter (review/non-flow skip; default: enabled)")
    parser.add_argument("--no-vlm", action="store_true",
                        help="Skip VLM inspection hooks (paper modules 6 & 12). "
                             "Adjudication still uses Gemini.")
    args = parser.parse_args()

    if args.dir:
        # Batch driver: one subprocess per PDF, no lock here (children lock).
        _run_batch(args)
    elif args.pdf_path:
        if not os.path.exists(args.pdf_path):
            print(f"Error: PDF not found at {args.pdf_path}")
            return
        if _run_single(args) == "skipped":
            sys.exit(EXIT_SKIPPED)
    else:
        print("Error: provide a PDF path or --dir <directory>")


if __name__ == "__main__":
    main()
