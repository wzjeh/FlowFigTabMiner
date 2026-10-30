"""FlowFigTabMiner as Python functions: one function per pipeline step, and
``extract()`` that runs them in order.

    from src.pipeline.api import Paper, build_llm, extract

    extract("paper.pdf")                                  # whole pipeline, config.yaml provider
    extract("paper.pdf", provider="openai", steps=("prefilter", "detect"))

    paper = Paper("paper.pdf", work_root="extract", out_dir="results")
    llm = build_llm()
    figures(paper, llm); tables(paper, llm)               # single steps

Every step reads and writes only ``paper.work_dir`` (``work_root/<paper_id>``,
``paper_id`` = PDF file name without ``.pdf``) and ``paper.out_dir``.  The
step functions do the work and nothing else.  ``extract()`` adds the
bookkeeping: it records each step in ``work_dir/steps.json`` (status, seconds,
provider, model, code commit), skips steps already ``ok`` unless
``redo=True``, and refuses a step whose earlier steps have not run, so a
paper whose crops were never made cannot be assembled from nothing.  Running
a step drops the records of every later step: their outputs were built on
the old ones and are redone next time.

This is the single place in the codebase that builds LLM / VLM providers.
Everything below consumes ``src.llm.providers.base`` and ``src.pipeline.hooks``.
"""

from __future__ import annotations

import datetime
import glob
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional

# MUST precede any torch / cv2 import: caps per-backend thread pools so
# several model libraries don't each spawn one thread per core.
import src.pipeline._threadcaps  # noqa: F401  (import for side effect)

from src.adjudication.global_assembly import GlobalAssembly
from src.adjudication.local_vars_builder import LocalVarsBuilder
from src.adjudication.pdf_parser import PDFParser
from src.adjudication.post_processor import PostProcessor
from src.extraction.common.content_recognizer import ContentRecognizer
from src.extraction.common.molecule_processor import MoleculeProcessor
from src.extraction.fusion import PointCountConsistency
from src.extraction.table.pipeline import TablePipeline
from src.extraction.table.scheme_seg_parser import SchemeSegParser
from src.extraction.table.table_vlm import TableTranscriber
from src.llm.concurrency import configure_concurrency
from src.llm.config import load_label_reader_config, load_llm_config, load_table_reader_config, load_vlm_config
from src.llm.fusion import ModalityRoutingPolicy
from src.llm.hooks import FigureInspectionHook
from src.llm.inspectors import FigureInspector, NearestPointMatcher
from src.llm.providers import get_vlm_provider
from src.parsing.active_area_detector import ActiveAreaDetector
from src.pipeline._memory import release_memory
from src.pipeline.figure_pipeline import FigurePipeline
from src.utils.config import load_config

logger = logging.getLogger(__name__)

CONFIG_PATH = "config.yaml"

STEPS = ("prefilter", "detect", "figures", "tables", "schemes", "context", "assemble", "normalize")
# timing.json keys, kept from the single-function pipeline
_TIMING_KEYS = {"prefilter": "filter", "detect": "tfid", "figures": "figure", "tables": "table",
                "schemes": "scheme", "assemble": "assembly", "normalize": "post"}


@dataclass
class Paper:
    """One PDF and where its files go."""

    pdf: str
    work_root: str = "data/intermediate"
    out_dir: str = "data/final_output"

    @property
    def paper_id(self) -> str:
        return os.path.splitext(os.path.basename(self.pdf))[0]

    @property
    def work_dir(self) -> str:
        return os.path.join(self.work_root, self.paper_id)


@dataclass
class LLMStack:
    """Every LLM / VLM handle the steps need, built once per process."""

    provider: object
    llm_cfg: object
    vlm_cfg: object
    figure_hooks: list = field(default_factory=list)
    label_reader: object = None
    provider_name: str = ""
    model: str = ""
    _overrides: dict = field(default_factory=dict)
    _vendors: dict = field(default_factory=dict)

    def vendor(self, name: str):
        """One provider instance per vendor name within this stack.  A new
        stack reads the API key from the environment again (the web worker
        swaps keys between jobs)."""
        if name not in self._vendors:
            self._vendors[name] = get_vlm_provider(name)
        return self._vendors[name]

    def table_transcriber(self):
        """VLM table transcriber (``vlm.table_reader``) and its config."""
        cfg = load_table_reader_config(CONFIG_PATH, **self._overrides)
        logger.info("api.table_reader provider=%s model=%s min_text_agreement=%s",
                    cfg.provider, cfg.model, cfg.min_text_agreement)
        return TableTranscriber(vlm=self.vendor(cfg.provider), cfg=cfg), cfg


def build_llm(provider: Optional[str] = None, model: Optional[str] = None, no_vlm: bool = False) -> LLMStack:
    """Assemble the LLM/VLM providers from config.yaml.

    ``provider`` / ``model`` override ``llm.provider`` / ``llm.models`` for every
    call.  ``no_vlm`` drops the figure inspection hook and the label reader;
    the figure metadata call and the table transcriber are core extraction
    steps and always run, like adjudication itself.
    """
    overrides = {"provider": provider, "model": model}
    llm_cfg = load_llm_config(CONFIG_PATH, **overrides)
    vlm_cfg = load_vlm_config(CONFIG_PATH, **overrides)

    # One process-wide cap on in-flight LLM calls (``llm.max_concurrent``).
    configure_concurrency(int(load_config().get("llm", {}).get("max_concurrent", 5)))

    stack = LLMStack(provider=None, llm_cfg=llm_cfg, vlm_cfg=vlm_cfg,
                     provider_name=llm_cfg.provider, model=llm_cfg.model, _overrides=overrides)
    stack.provider = stack.vendor(llm_cfg.provider)
    logger.info("api.providers %s llm_model=%s vlm_model=%s", llm_cfg.provider, llm_cfg.model, vlm_cfg.model)
    if no_vlm:
        logger.info("api.providers VLM inspection disabled by --no-vlm")
        return stack

    stack.label_reader = _build_label_reader(overrides, stack)
    fig_inspector = FigureInspector(
        vlm=stack.vendor(vlm_cfg.provider),
        cfg=vlm_cfg,
        matcher=NearestPointMatcher(tol=0.05),
        policy=ModalityRoutingPolicy(numeric_tol=0.05),
    )
    stack.figure_hooks = [
        FigureInspectionHook(
            inspector=fig_inspector,
            consistency_checks=[PointCountConsistency()],
        )
    ]
    return stack


def _build_label_reader(overrides: dict, stack: LLMStack):
    """VLM second reader for tick / heatmap-cell labels (design: VLM reads
    symbols on a numbered contact sheet, YOLO + robust_fit keep the geometry)."""
    lr_cfg = load_label_reader_config(CONFIG_PATH, **overrides)
    if not lr_cfg.enabled:
        logger.info("api.label_reader disabled")
        return None
    from src.extraction.figure.label_reader import VLMLabelReader
    logger.info("api.label_reader provider=%s model=%s policy=%s", lr_cfg.provider, lr_cfg.model,
                lr_cfg.value_conflict_policy)
    reader = VLMLabelReader(vlm=stack.vendor(lr_cfg.provider), cfg=lr_cfg, max_boxes=lr_cfg.max_boxes)
    reader.value_conflict_policy = lr_cfg.value_conflict_policy
    return reader


# ───────────────────────────────────────────────────────────────── steps


def prefilter(paper: Paper) -> bool:
    """Step 0: False for a review article or a non-flow paper (a 3-page text
    scan, before any model loads)."""
    from src.preprocessing.paper_filter import filter_paper
    pf = filter_paper(paper.pdf)
    if not pf["is_relevant"]:
        print(f"[PreFilter] SKIP {paper.paper_id}: {pf['reason']}")
    return bool(pf["is_relevant"])


def detect(paper: Paper) -> int:
    """Step 1: TF-ID figure / table crops plus their captions from the PDF
    text layer.  Returns the number of crops."""
    print("\n=== Step 1: TF-ID Parsing ===")
    detector = ActiveAreaDetector()
    detections = detector.process_pdf(paper.pdf)
    saved_paths = detector.save_crops(paper.pdf, detections, paper.work_dir)
    print(f"Saved {len(saved_paths)} crops (figures/tables) to {paper.work_dir}")
    # Resolve each crop to its real caption / label / footnote from the
    # PDF text layer (deterministic; feeds the text-window anchors and
    # both adjudication prompts).
    try:
        from src.parsing.caption_locator import write_contexts
        write_contexts(paper.pdf, paper.work_dir)
    except Exception as exc:
        print(f"[CaptionLocator] failed (non-fatal): {exc}")
    # Release Florence-2 (~2GB) immediately — it is used only here.
    del detector
    release_memory()
    return len(saved_paths)


def figures(paper: Paper, llm: LLMStack) -> None:
    """Steps 2-4: figure crops → data points (YOLO + axis mapping + legend)."""
    print("\n=== Step 2-4: Figure Extraction ===")
    from src.extraction.figure.metadata_vlm import FigureMetadataExtractor
    fig_pipeline = FigurePipeline(
        metadata_extractor=FigureMetadataExtractor(vlm=llm.provider, cfg=llm.vlm_cfg),
        post_extract_hooks=llm.figure_hooks,
        label_reader=llm.label_reader,
        value_conflict_policy=getattr(llm.label_reader, "value_conflict_policy", "vlm"),
    )
    try:
        fig_pipeline.process_pdf_figures(paper.work_dir)
    finally:
        del fig_pipeline
        release_memory()


def tables(paper: Paper, llm: LLMStack) -> None:
    """Every ``tables/*.png`` crop through the table pipeline (VLM transcription
    + molecule YOLO + MolNexTR)."""
    print("\n=== Step Table: Table Extraction ===")
    shared_content_rec = ContentRecognizer()
    transcriber, table_cfg = llm.table_transcriber()
    tab_pipeline = TablePipeline(
        transcriber=transcriber,
        content_recognizer=shared_content_rec,
        min_text_agreement=table_cfg.min_text_agreement,
    )
    tables_dir = os.path.join(paper.work_dir, "tables")
    if os.path.exists(tables_dir):
        table_imgs = glob.glob(os.path.join(glob.escape(tables_dir), "*.png"))
        table_imgs = [f for f in table_imgs if "_body" not in f and "_crop" not in f]
        print(f"Processing {len(table_imgs)} tables...")
        for t_img in table_imgs:
            try:
                tab_pipeline.process_table(t_img, output_dir=tables_dir)
            except Exception as exc:
                print(f"Error processing table {t_img}: {exc}")
    else:
        print("No tables directory found.")
    # Release the table pipeline + its stage models.  The MolNexTR and OCR
    # singletons are NOT freed (module-level caches); this only drops the
    # TablePipeline container and its YOLO stage references.
    del tab_pipeline, shared_content_rec
    release_memory()


def figure_schemes(paper: Paper, llm: LLMStack) -> dict:
    """label -> SMILES harvested from the figure crops macro YOLO rejected.
    Forensics under ``{work_dir}/schemes/``; {} when there is nothing to read."""
    from src.extraction.scheme.figure_scheme_harvest import FigureSchemeHarvester, filtered_scheme_crops
    crops = filtered_scheme_crops(paper.work_dir)
    if not crops:
        return {}
    mol_cfg = load_config().get("tables", {}).get("molecule_detection", {})
    mp = MoleculeProcessor(model_path=mol_cfg.get("model_path"), conf_threshold=mol_cfg.get("confidence_threshold", 0.25))
    rec = ContentRecognizer()
    try:
        pool = FigureSchemeHarvester(mp, rec, llm.provider, llm.vlm_cfg).harvest(paper.work_dir)
    finally:
        del mp, rec
        release_memory()
    print(f"   -> figure schemes: {len(crops)} crops, {len(pool)} labelled structures")
    return pool


def schemes(paper: Paper, llm: LLMStack) -> None:
    """Step 3.5: reaction schemes (table-side Tab-Scheme-Seg and the figure
    crops filed as schemes) → ``compound_pool.json`` + ``scheme_conditions.txt``."""
    print("\n=== Step 3.5: Tab-Scheme-Seg (Scheme Parsing) ===")
    reactant_pool: dict = {}
    product_pool: dict = {}
    compound_pool: dict = {}
    scheme_conditions_texts: list[str] = []

    tables_dir = os.path.join(paper.work_dir, "tables")
    if os.path.exists(tables_dir):
        scheme_imgs = glob.glob(os.path.join(glob.escape(tables_dir), "**", "*_table_scheme_*.png"), recursive=True)
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

    # Schemes that TF-ID filed as figures (macro YOLO: "no target_image"):
    # the only place many papers draw the structure behind "3" / "c-9".
    fig_pool = figure_schemes(paper, llm)
    for k, v in fig_pool.items():
        if k not in reactant_pool and k not in product_pool:
            compound_pool.setdefault(k, v)

    if reactant_pool or product_pool or compound_pool:
        pool_path = os.path.join(paper.work_dir, "compound_pool.json")
        with open(pool_path, "w") as f:
            json.dump(
                {"reactant_pool": reactant_pool, "product_pool": product_pool, "compound_pool": compound_pool},
                f, indent=2,
            )
        total = len(reactant_pool) + len(product_pool) + len(compound_pool)
        print(f"   -> Compound pool ({total} total) -> {pool_path}")
    if scheme_conditions_texts:
        cond_path = os.path.join(paper.work_dir, "scheme_conditions.txt")
        with open(cond_path, "w") as f:
            f.write("\n".join(scheme_conditions_texts))
        print(f"   -> Scheme conditions saved -> {cond_path}")


def _scheme_conditions(work_dir: str) -> str:
    cond_path = os.path.join(work_dir, "scheme_conditions.txt")
    if not os.path.exists(cond_path):
        return ""
    with open(cond_path) as f:
        return f.read().strip()


def global_vars(paper: Paper, llm: LLMStack, rebuild: bool = False) -> dict:
    """Step 4.4 — one LLM call per paper: paper-level reaction context and
    default conditions (each value with a verbatim quote and a scope), written
    to ``{work_dir}/global_vars.json``.  See ``GlobalVarsBuilder``."""
    from src.adjudication.global_vars_builder import GlobalVarsBuilder
    from src.adjudication.post_processor import build_abbrev_map

    paper_text = PDFParser().extract_text(paper.pdf)
    try:
        return GlobalVarsBuilder(llm=llm.provider, llm_cfg=llm.llm_cfg).build(
            paper.pdf, paper.work_dir, paper_text,
            scheme_conditions=_scheme_conditions(paper.work_dir), abbrev_map=build_abbrev_map(paper_text),
            rebuild=rebuild,
        )
    except Exception as exc:
        print(f"   [GlobalVars] failed (non-fatal): {exc}")
        return {}


def local_vars(paper: Paper, llm: LLMStack, rebuild: bool = False, global_vars: dict = None) -> None:
    """Step 4.5 — one LocalVarsBuilder call per figure / table source.

    Each source gets its CaptionLocator context (real label, verbatim
    caption, footnote) so the LLM sees the figure's identity and the text
    window is anchored on the paper's own figure number.  ``rebuild=True``
    discards cached ``local_vars/*.json`` first (the cache has no staleness
    check of its own) — used by ``scripts/reassemble.py --rebuild-vars``.
    """
    from src.parsing.caption_locator import load_context
    from src.adjudication.source_discovery import apply_context_to_evidence, figure_value_problem, load_figure_evidence

    intermediate_dir = paper.work_dir
    paper_text = PDFParser().extract_text(paper.pdf)
    local_vars_dir = os.path.join(intermediate_dir, "local_vars")
    if rebuild and os.path.isdir(local_vars_dir):
        for fn in glob.glob(os.path.join(glob.escape(local_vars_dir), "*_local_vars.json")):
            os.remove(fn)
        print(f"   [LocalVars] rebuild: cleared cached local_vars in {local_vars_dir}")
    os.makedirs(local_vars_dir, exist_ok=True)
    builder = LocalVarsBuilder(llm=llm.provider, llm_cfg=llm.llm_cfg)

    scheme_cond_text = _scheme_conditions(intermediate_dir)
    if scheme_cond_text:
        print(f"   [LocalVars] Loaded scheme_conditions.txt ({len(scheme_cond_text)} chars)")

    macro_cleaned_dir = os.path.join(intermediate_dir, "macro_cleaned")
    for jpath in glob.glob(os.path.join(glob.escape(macro_cleaned_dir), "*_evidence.json")):
        try:
            ev = load_figure_evidence(jpath)
            src_id = ev.get("meta", {}).get(
                "figure_id",
                os.path.basename(jpath).replace("_evidence.json", ""),
            )
            if figure_value_problem(ev):
                continue                    # nothing to assemble (source_discovery records why)
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
                    if str(ev.get("parse_status", "ok")).startswith("failed"):
                        print(f"   [LocalVars] skip table {src_id}: {ev.get('parse_status')}")
                        continue
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


def context(paper: Paper, llm: LLMStack, rebuild: bool = False) -> dict:
    """Steps 4.4 + 4.5: paper-level and per-source variable libraries.
    Returns the seconds each half took (timing.json keeps both)."""
    t0 = time.perf_counter()
    print("\n=== Step 4.4: Build Paper-Level Global Vars ===")
    gv = global_vars(paper, llm, rebuild=rebuild)
    t1 = time.perf_counter()
    print("\n=== Step 4.5: Build Sub-Variable Libraries ===")
    local_vars(paper, llm, rebuild=rebuild, global_vars=gv)
    return {"global_vars": round(t1 - t0, 1), "local_vars": round(time.perf_counter() - t1, 1)}


def assemble(paper: Paper, llm: LLMStack, force: bool = False) -> Optional[str]:
    """Step 5: one LLM call per source → ``out_dir/<paper_id>_final.json``."""
    print("\n=== Step 5: Global Assembly ===")
    assembler = GlobalAssembly(llm=llm.provider, llm_cfg=llm.llm_cfg,
                               output_dir=paper.out_dir, intermediate_root=paper.work_root)
    return assembler.run(paper.pdf, paper.work_dir, force=force)


def normalize(paper: Paper, smiles_lookup: bool = True) -> Optional[str]:
    """Step 6: normalisation + validation → ``out_dir/<paper_id>_normalized.json``."""
    print("\n=== Step 6: Post-Processing (Normalisation) ===")
    return PostProcessor(output_dir=paper.out_dir).run(paper.pdf, paper.work_dir, smiles_lookup=smiles_lookup)


# ───────────────────────────────────────────────────────────────── extract


def _commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.abspath(__file__)), timeout=10).stdout.strip() or None
    except Exception:
        return None


def read_steps(paper: Paper) -> dict:
    """``work_dir/steps.json``: step name → {status, seconds, at, commit, provider, model, ...}."""
    path = os.path.join(paper.work_dir, "steps.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _write_steps(paper: Paper, record: dict) -> None:
    os.makedirs(paper.work_dir, exist_ok=True)
    tmp = os.path.join(paper.work_dir, "steps.json.tmp")
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2)
    os.replace(tmp, os.path.join(paper.work_dir, "steps.json"))


def _blocker(step: str, done: dict) -> Optional[str]:
    """Why ``step`` cannot run yet, or None.  Every earlier step must have run
    (a failed figure step still lets the tables be assembled, as before), the
    pre-filter must have passed and the crops must exist."""
    for prev in STEPS[: STEPS.index(step)]:
        rec = done.get(prev)
        if rec is None:
            return f"step '{prev}' has not run"
        if prev == "prefilter" and rec["status"] != "ok":
            return "the pre-filter skipped this paper"
        if prev == "detect" and rec["status"] != "ok":
            return "figure/table detection failed"
    return None


def extract(pdf: str, work_root: str = "data/intermediate", out_dir: str = "data/final_output",
            provider: Optional[str] = None, model: Optional[str] = None, steps=STEPS, redo: bool = False,
            use_prefilter: bool = True, smiles_lookup: bool = True, no_vlm: bool = False,
            llm: Optional[LLMStack] = None) -> dict:
    """Run ``steps`` (in pipeline order) for one PDF.

    Steps already ``ok`` in ``steps.json`` are skipped unless ``redo``; a
    ``redo`` also forces the assembly LLM call.  Returns a summary:
    ``{"status": "ok" | "skipped" | "failed" | "blocked", "steps": {...},
    "normalized": path or None}``.
    """
    paper = Paper(pdf, work_root=work_root, out_dir=out_dir)
    wanted = [s for s in STEPS if s in set(steps)]
    unknown = set(steps) - set(STEPS)
    if unknown:
        raise ValueError(f"unknown step(s) {sorted(unknown)}; steps are {STEPS}")
    done = read_steps(paper)
    commit = _commit()
    timings: dict = {}
    summary = {"status": "ok", "steps": {}, "normalized": None}

    for step in wanted:
        if not redo and done.get(step, {}).get("status") == "ok":
            print(f"[extract] {paper.paper_id}: '{step}' already done — skipped")
            summary["steps"][step] = "done"
            continue
        why = _blocker(step, done)
        if why:
            print(f"[extract] {paper.paper_id}: cannot run '{step}': {why}")
            summary["status"] = "blocked"
            summary["steps"][step] = "blocked"
            break
        needs_llm = step not in ("prefilter", "detect", "normalize")
        if needs_llm and llm is None:
            llm = build_llm(provider=provider, model=model, no_vlm=no_vlm)
        rec = {"at": datetime.datetime.now().isoformat(timespec="seconds"), "commit": commit}
        if needs_llm:
            rec.update(provider=llm.provider_name, model=llm.model)
        t0 = time.perf_counter()
        status, extra = "ok", {}
        try:
            if step == "prefilter":
                if use_prefilter and not prefilter(paper):
                    status = "skipped"
                extra = {} if use_prefilter else {"note": "pre-filter disabled"}
            elif step == "detect":
                extra = {"crops": detect(paper)}
            elif step == "figures":
                figures(paper, llm)
            elif step == "tables":
                tables(paper, llm)
            elif step == "schemes":
                schemes(paper, llm)
            elif step == "context":
                timings.update(context(paper, llm))
            elif step == "assemble":
                assemble(paper, llm, force=redo)
            elif step == "normalize":
                summary["normalized"] = normalize(paper, smiles_lookup=smiles_lookup)
        except Exception as exc:
            logger.exception("step %s failed for %s", step, paper.paper_id)
            print(f"[extract] {paper.paper_id}: step '{step}' failed: {exc}")
            status, extra = "failed", {"error": f"{type(exc).__name__}: {exc}"[:500]}
        seconds = round(time.perf_counter() - t0, 1)
        if step in _TIMING_KEYS:
            timings[_TIMING_KEYS[step]] = seconds
        done[step] = {"status": status, "seconds": seconds, **rec, **extra}
        for later in STEPS[STEPS.index(step) + 1:]:
            done.pop(later, None)               # built on the old output of this step
        _write_steps(paper, done)
        summary["steps"][step] = status
        if status == "skipped":
            summary["status"] = "skipped"
            break
        if status == "failed":
            summary["status"] = "failed"
            if step == "detect":
                break

    if timings:
        _write_timing(paper, summary["status"], timings)
    return summary


def _write_timing(paper: Paper, status: str, timings: dict) -> None:
    """Per-step wall-clock of this run → ``work_dir/timing.json``."""
    out = {"status": status, **timings}
    if status == "ok":
        out["total"] = round(sum(timings.values()), 1)
    try:
        os.makedirs(paper.work_dir, exist_ok=True)
        with open(os.path.join(paper.work_dir, "timing.json"), "w") as f:
            json.dump(out, f, indent=2)
    except Exception as exc:
        print(f"[Timing] write failed: {exc}")
    line = "  ".join(f"{k}={v}s" for k, v in timings.items())
    print(f"[Timing] {status}: {line}" + (f"  total={out['total']}s" if "total" in out else ""))
