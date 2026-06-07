import json
import logging
import os
from typing import Optional

from src.adjudication.pdf_parser import PDFParser
from src.adjudication.per_source_assembler import PerSourceAssembler
from src.adjudication.per_source_prompts import (
    CommonPreamble,
    FigurePromptBuilder,
    TablePromptBuilder,
)
from src.adjudication.source_discovery import discover
from src.llm.config import LLMConfig
from src.llm.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class GlobalAssembly:
    """Thin orchestrator — delegates the per-source LLM calls to
    ``PerSourceAssembler``.

    Old behaviour (one giant LLM call covering every figure + table at
    once) hit Gemini 2.5 Flash's 32k output-token cap on table-heavy
    papers; this class now fans the work out — one call per source — so
    each call has plenty of output headroom.  The downstream contract
    (``data/final_output/{basename}_final.json`` as a JSON array of
    reaction-record dicts) is unchanged.
    """

    def __init__(
        self,
        llm: LLMProvider,
        llm_cfg: LLMConfig,
        output_dir: str = "data/final_output",
        assembler: Optional[PerSourceAssembler] = None,
        intermediate_root: str = "data/intermediate",
    ):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.llm = llm
        self.llm_cfg = llm_cfg
        self.pdf_parser = PDFParser()
        self.intermediate_root = intermediate_root
        self.assembler = assembler or PerSourceAssembler(
            llm=llm, llm_cfg=llm_cfg,
            prompt_builders={
                "figure": FigurePromptBuilder(),
                "table":  TablePromptBuilder(),
            },
            max_workers=10,
            raw_dir=intermediate_root,
        )

    @staticmethod
    def _evidence_stale(out_file: str, intermediate_dir: str) -> bool:
        """True if any evidence JSON is newer than the cached _final.json.

        Without this, a re-extraction (newer evidence) would be masked by a
        stale cached final — the cause of Franklin's 0-records (03-25 final
        cache-hit over fresh 06-07 evidence).
        """
        import glob
        try:
            final_mtime = os.path.getmtime(out_file)
        except OSError:
            return True
        for pat in (
            os.path.join(intermediate_dir, "macro_cleaned", "*_evidence.json"),
            os.path.join(intermediate_dir, "tables", "**", "*_evidence.json"),
        ):
            for e in glob.glob(pat, recursive=True):
                if os.path.getmtime(e) > final_mtime:
                    return True
        return False

    def run(self, pdf_path, intermediate_dir=None, force=False):
        """
        Run global assembly for a PDF.
        If force=False and _final.json already exists, skip LLM call and return cached path.
        """
        basename = os.path.splitext(os.path.basename(pdf_path))[0]
        if not intermediate_dir:
            intermediate_dir = os.path.join("data/intermediate", basename)

        out_file = os.path.join(self.output_dir, f"{basename}_final.json")
        cache_ok = (not force) and os.path.exists(out_file)
        if cache_ok and self._evidence_stale(out_file, intermediate_dir):
            cache_ok = False  # evidence re-extracted since last assembly → rerun
            print("[GlobalAssembly] Evidence newer than cached final — re-running assembly.")
        if cache_ok:
            print(f"[GlobalAssembly] Cache hit — skipping LLM (use --force-assembly to rerun): {out_file}")
            # Still run Excel export from cached JSON
            try:
                with open(out_file) as f:
                    records = json.load(f)
                if isinstance(records, list):
                    self._save_excel(records, basename)
            except Exception as e:
                print(f"[GlobalAssembly] Excel from cache skipped: {e}")
            return out_file

        print(f"[GlobalAssembly] Assembling final dataset for {basename}...")

        # 1. Extract paper text (cached on disk by PDFParser).
        paper_text = self.pdf_parser.extract_text(pdf_path)

        # 2. Discover sources — figures + tables + their local_vars + CSVs +
        # dual-anchor 8 KB paper-text windows.
        packets = discover(self.intermediate_root, basename, paper_text)
        n_fig = sum(1 for p in packets if p.source_type == "figure")
        n_tab = sum(1 for p in packets if p.source_type == "table")
        print(f"   -> Discovered {len(packets)} source packets ({n_tab} tables, {n_fig} figures).")

        # 3. Load paper-level compound pools + scheme conditions (shared
        # preamble across every per-source call).
        pools = {}
        pool_path = os.path.join(intermediate_dir, "compound_pool.json")
        if os.path.exists(pool_path):
            try:
                pool_data = json.load(open(pool_path))
                # Backward-compat: old flat dict (no reactant_pool key) → treat as compound_pool.
                if not isinstance(pool_data.get("reactant_pool"), dict):
                    pools = {"compound_pool": pool_data}
                else:
                    pools = pool_data
                total = sum(len(pools.get(k, {})) for k in ("reactant_pool", "product_pool", "compound_pool"))
                print(f"   -> Loaded compound pool ({total} entries)")
            except Exception as exc:
                logger.warning("global_assembly compound_pool load failed: %s", exc)

        scheme_conditions = ""
        cond_path = os.path.join(intermediate_dir, "scheme_conditions.txt")
        if os.path.exists(cond_path):
            try:
                scheme_conditions = open(cond_path).read().strip()
                print(f"   -> Loaded scheme conditions text")
            except Exception as exc:
                logger.warning("global_assembly scheme_conditions load failed: %s", exc)

        # Abbreviation map extracted from FULL paper text (so every source —
        # even ones whose 8KB window misses the definition — can expand it).
        from src.adjudication.post_processor import build_abbrev_map
        abbrev_map = build_abbrev_map(paper_text)
        if abbrev_map:
            print(f"   -> Extracted {len(abbrev_map)} abbreviation definition(s)")

        preamble = CommonPreamble.build(pools, scheme_conditions, abbrev_map=abbrev_map)

        # 4. Fan-out per-source LLM calls.
        records = self.assembler.assemble(packets, preamble, basename)

        # 5. Persist final.json + Excel.
        with open(out_file, "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"   -> Saved final result to {out_file} ({len(records)} records)")

        if records:
            try:
                self._save_excel(records, basename)
            except Exception as exc:
                print(f"[GlobalAssembly] Excel export skipped: {exc}")

        return out_file

    def _save_excel(self, records, basename):
        import pandas as pd
        from src.adjudication.post_processor import PREFERRED_COLUMNS
        out_path = os.path.join(self.output_dir, f"{basename}_final.xlsx")
        flat = []
        for r in records:
            row = dict(r)
            conds = row.pop("conditions", {}) or {}
            other = row.pop("other_metrics", {}) or {}
            row.update(conds)
            row.update(other)
            flat.append(row)
        df = pd.DataFrame(flat)
        src_col = "source_table_or_figure"
        if src_col not in df.columns:
            df[src_col] = "unknown"

        # Apply fixed column order
        ordered = [c for c in PREFERRED_COLUMNS if c in df.columns]
        extra = sorted(c for c in df.columns if c not in PREFERRED_COLUMNS)
        df = df[ordered + extra]

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="All Records", index=False)
            for src, grp in df.groupby(src_col, sort=False):
                sheet = str(src)[:31]
                grp.to_excel(writer, sheet_name=sheet, index=False)
        print(f"   -> Saved Excel to {out_path}")
