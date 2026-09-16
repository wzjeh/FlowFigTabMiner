"""GlobalVarsBuilder — Step 4.4: ONE LLM call per paper that distils the
paper-level reaction context and default conditions into
``{intermediate_dir}/global_vars.json``.

Why
---
The paper describes a two-level parameter pool (global paper-wide defaults,
local per-figure/table overrides), but the code only ever built the local
level.  Conditions stated once in the General Procedure ("all reactions
were run in THF at -78 °C in a T-shaped micromixer …") therefore never
reached figure records, even when a table's footnote in the same paper had
already surfaced them.  This builder makes the global level a real artifact:

* input  = abstract + experimental / general-procedure section + every
  caption & footnote resolved by CaptionLocator + scheme-OCR conditions
  + abbreviation map;
* output = ``default_conditions`` where EVERY value carries a verbatim
  ``quote`` and a ``scope`` (``paper`` = author states it holds for all
  experiments; ``partial`` = holds for some) — only ``scope == "paper"``
  values may be inherited by records downstream (PostProcessor B3).

Cache: ``global_vars.json`` is reused unless it is older than the newest
``context/*.json`` / evidence file, or ``rebuild=True``.
"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Any, Dict, List, Optional

from src.adjudication.pdf_parser import _EXPERIMENTAL_ANCHORS
from src.llm.json_utils import sanitize_json_text
from src.llm.types import ChatMessage, Role

CONDITION_FIELDS = (
    "temperature_C", "residence_time_s", "flow_rate_mL_min", "solvent",
    "reactor_type", "pressure_bar", "catalyst", "additive",
)

_SYSTEM = (
    "You are an expert flow chemistry data analyst. You read ONE paper's experimental "
    "section, abstract, and all figure/table captions and footnotes, and produce a compact "
    "JSON summary of the paper-wide reaction context and the DEFAULT reaction conditions "
    "that hold across the paper's experiments.\n"
    "Rules:\n"
    "(1) Every condition value MUST be accompanied by a short verbatim quote from the text "
    "that states it. No quote → value must be null.\n"
    "(2) scope = \"paper\" ONLY when the text states the condition for all / typical "
    "experiments (general procedure, 'all reactions were …', 'unless otherwise noted'). "
    "If the value is one setting among several that were varied (e.g. a figure scans "
    "temperature), use scope = \"partial\".\n"
    "(3) Use the field units exactly: temperature in °C, residence time in seconds, flow rate "
    "in mL/min, pressure in bar. Expand solvent abbreviations (THF → tetrahydrofuran).\n"
    "(4) No SMILES. Output valid JSON only, no markdown fences."
)


def _abstract(paper_text: str, size: int = 2500) -> str:
    return (paper_text or "")[:size]


def _experimental_section(paper_text: str, size: int = 12000) -> str:
    low = re.sub(r"\s", " ", (paper_text or "").lower())
    for anchor in _EXPERIMENTAL_ANCHORS:
        idx = low.find(anchor)
        if idx != -1:
            start = max(0, idx - 300)
            return paper_text[start:start + size]
    return ""


def _collect_captions(intermediate_dir: str) -> List[Dict[str, str]]:
    out = []
    for p in sorted(glob.glob(os.path.join(intermediate_dir, "context", "*_context.json"))):
        try:
            c = json.load(open(p))
        except Exception:
            continue
        if not c.get("caption"):
            continue
        out.append({"label": c.get("label") or c["source_id"], "caption": c["caption"],
                    "footnote": c.get("footnote") or ""})
    # de-duplicate sub-figure crops that resolved to the same label
    seen, uniq = set(), []
    for c in out:
        if c["label"] in seen:
            continue
        seen.add(c["label"]); uniq.append(c)
    return uniq


def _newest_mtime(intermediate_dir: str) -> float:
    paths = glob.glob(os.path.join(intermediate_dir, "context", "*.json")) \
        + glob.glob(os.path.join(intermediate_dir, "macro_cleaned", "*_evidence.json")) \
        + glob.glob(os.path.join(intermediate_dir, "tables", "**", "*_evidence.json"), recursive=True)
    return max((os.path.getmtime(p) for p in paths), default=0.0)


class GlobalVarsBuilder:
    def __init__(self, llm, llm_cfg):
        self.llm = llm
        self.llm_cfg = llm_cfg

    # ------------------------------------------------------------------ #
    def build(self, pdf_path: str, intermediate_dir: str, paper_text: str,
              scheme_conditions: str = "", abbrev_map: Optional[Dict[str, str]] = None,
              rebuild: bool = False) -> Dict[str, Any]:
        out_path = os.path.join(intermediate_dir, "global_vars.json")
        if not rebuild and os.path.exists(out_path) and os.path.getmtime(out_path) >= _newest_mtime(intermediate_dir):
            print("[GlobalVarsBuilder] Cache hit: global_vars.json")
            try:
                return json.load(open(out_path))
            except Exception:
                pass

        captions = _collect_captions(intermediate_dir)
        user_prompt = self._build_prompt(paper_text, captions, scheme_conditions, abbrev_map or {})
        print(f"[GlobalVarsBuilder] Building paper-level pool ({len(captions)} captions, "
              f"{len(user_prompt)} chars prompt)...")
        try:
            resp = self.llm.chat(
                [ChatMessage(role=Role.SYSTEM, content=_SYSTEM),
                 ChatMessage(role=Role.USER, content=user_prompt)],
                self.llm_cfg,
            )
            result = json.loads(sanitize_json_text(resp.text or ""))
            if not isinstance(result, dict):
                raise ValueError("global_vars response is not an object")
        except Exception as exc:
            print(f"[GlobalVarsBuilder] failed ({exc}) — writing empty pool")
            result = {"reaction_context": {}, "default_conditions": {}, "source_index": {}}
        result = self._validate(result)
        result["_meta"] = {"n_captions": len(captions), "has_experimental": bool(_experimental_section(paper_text))}
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        n_paper = sum(1 for v in (result.get("default_conditions") or {}).values()
                      if isinstance(v, dict) and v.get("scope") == "paper" and v.get("value") is not None)
        print(f"[GlobalVarsBuilder] Saved -> {out_path} ({n_paper} paper-scope defaults)")
        return result

    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate(result: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce the no-quote-no-value rule deterministically."""
        dc = result.get("default_conditions") or {}
        clean: Dict[str, Any] = {}
        for field in CONDITION_FIELDS:
            v = dc.get(field)
            if not isinstance(v, dict):
                clean[field] = {"value": None, "quote": None, "scope": None}
                continue
            value, quote = v.get("value"), (v.get("quote") or "").strip()
            scope = v.get("scope") if v.get("scope") in ("paper", "partial") else None
            if value in (None, "", []) or not quote:
                value, scope = None, None
            clean[field] = {"value": value, "quote": quote or None, "scope": scope}
        result["default_conditions"] = clean
        result.setdefault("reaction_context", {})
        result.setdefault("source_index", {})
        return result

    @staticmethod
    def _build_prompt(paper_text: str, captions: List[Dict[str, str]], scheme_conditions: str,
                      abbrev_map: Dict[str, str]) -> str:
        cap_lines = "\n".join(
            f"- {c['label']}: {c['caption'][:400]}" + (f"\n    footnote: {c['footnote'][:400]}" if c['footnote'] else "")
            for c in captions
        ) or "(none)"
        abbr = "\n".join(f"  {a} = {full}" for a, full in sorted(abbrev_map.items())) or "(none)"
        exp = _experimental_section(paper_text) or "(no experimental section found)"
        labels = [c["label"] for c in captions]
        return f"""=== ABSTRACT / OPENING ===
{_abstract(paper_text)}

=== EXPERIMENTAL / GENERAL PROCEDURE (up to 12k chars) ===
{exp}

=== ALL FIGURE / TABLE CAPTIONS AND FOOTNOTES (PDF text layer) ===
{cap_lines}

=== SCHEME CONDITIONS (OCR of reaction-scheme images; may be noisy) ===
{scheme_conditions or '(none)'}

=== ABBREVIATIONS ===
{abbr}

=== OUTPUT SCHEMA ===
{{
  "reaction_context": {{
    "main_transformation": "<one sentence: what reaction the paper studies>",
    "substrate_class": "<e.g. aryl bromides bearing esters>",
    "main_product_name": "<the product family or the single product, if the paper has one>",
    "organometallic_reagent": "<e.g. n-BuLi (1.5 equiv) or null>",
    "electrophile": "<e.g. benzaldehyde or 'various' or null>",
    "reactor_description": "<e.g. two T-shaped micromixers + microtube reactors, 250 µm ID>"
  }},
  "default_conditions": {{
    "temperature_C":     {{"value": null, "quote": null, "scope": null}},
    "residence_time_s":  {{"value": null, "quote": null, "scope": null}},
    "flow_rate_mL_min":  {{"value": null, "quote": null, "scope": null}},
    "solvent":           {{"value": null, "quote": null, "scope": null}},
    "reactor_type":      {{"value": null, "quote": null, "scope": null}},
    "pressure_bar":      {{"value": null, "quote": null, "scope": null}},
    "catalyst":          {{"value": null, "quote": null, "scope": null}},
    "additive":          {{"value": null, "quote": null, "scope": null}}
  }},
  "source_index": {{
    "<label from {labels[:6]} ...>": {{"what_it_varies": "<e.g. temperature and residence time>",
                                     "conditions_in_caption": {{"temperature_C": null, "solvent": null, "residence_time_s": null}}}}
  }}
}}
Fill source_index for every label listed in the captions block. scope must be "paper" or "partial" (see rules)."""
