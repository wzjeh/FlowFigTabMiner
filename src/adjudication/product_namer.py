"""Name the product of a figure whose template only describes it.

"trapped product with benzaldehyde", "tridecafluorohexylstannane" (a family),
"benzyllithiums" (the intermediate): the assembler had the substrate, the
organolithium reagent and the electrophile in front of it but did not compose
the product.  One extra call per such figure asks for the specific compound;
the answer is used only when it names one compound, and the PubChem lookup in
the post-processor turns the name into a structure.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from src.adjudication.source_discovery import panel_line
from src.llm.json_utils import sanitize_json_text
from src.llm.types import ChatMessage, Role

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "product_namer.md"

_GENERIC_RE = re.compile(
    r"\b(products?|derivatives?|compounds?|epoxides?|esters?|ketones?|substituted|protonated|methylated|"
    r"stannylated|silylated|trapped|corresponding|alkyl|aryl|molecules?|intermediates?|adducts?)\b|lithiums?\b", re.I)


def is_generic_product(name: Optional[str], label: Optional[str], smiles: Optional[str], resolves=None) -> bool:
    """No structure, no label, and a name that describes rather than names.
    ``resolves(name) -> bool`` (PubChem) catches a family written like a name:
    "tridecafluorohexylstannane" for tributyl(tridecafluorohexyl)stannane."""
    if smiles or (label and str(label).strip()):
        return False
    name = (name or "").strip()
    if not name or _GENERIC_RE.search(name):
        return True
    return bool(resolves) and not resolves(name)


def _identity(tpl: Dict[str, Any]) -> Dict[str, Any]:
    rt = tpl.get("record_template") or {}
    out = {k: rt.get(k) for k in ("reactant1_name", "reactant1_smiles", "reactant2_name", "reactant2_smiles",
                                   "product_name", "product_label", "product_smiles")}
    smap = tpl.get("series_map") or {}
    series_ids = {s: {k: v for k, v in (m or {}).items() if isinstance(m, dict) and k.endswith(("_name", "_label", "_smiles"))}
                  for s, m in smap.items() if isinstance(m, dict)}
    out["series_identities"] = {s: v for s, v in series_ids.items() if v}
    return out


class ProductNamer:
    def __init__(self, llm, llm_cfg, resolves=None):
        self.llm = llm
        self.llm_cfg = llm_cfg
        self.resolves = resolves            # name -> bool; the post-processor's PubChem lookup (cached on disk)
        self.prompt = _PROMPT_PATH.read_text(encoding="utf-8")

    def refine(self, tpl: Dict[str, Any], packet, preamble, raw_dir: Optional[str] = None) -> Dict[str, Any]:
        """Return ``tpl`` with a specific product name when the template's is
        generic and the paper's text determines it; otherwise ``tpl`` unchanged."""
        rt = tpl.get("record_template") or {}
        if not is_generic_product(rt.get("product_name"), rt.get("product_label"), rt.get("product_smiles"), self.resolves):
            return tpl
        lv = packet.local_vars or {}
        fixed = lv.get("fixed_conditions") or {}
        ev = packet.evidence or {}
        meta = ev.get("meta") or {}
        ctx = packet.context or {}
        user = (
            f"=== SOURCE: {packet.human_label} ({packet.source_id}) ===\n"
            f"Caption: {ctx.get('caption') or meta.get('caption_pdf') or '(none)'}\n"
            + panel_line(ev)
            + f"Reaction context: {lv.get('reaction_context') or '(none)'}\n"
            f"Notes: {fixed.get('notes') or '(none)'}\n"
            f"Template identities: {json.dumps(_identity(tpl), ensure_ascii=False)}\n"
            f"(the current product_name is not a database name: it describes a family or an intermediate)\n\n"
            + preamble.render_pools_section() + "\n\n"
            f"=== PAPER TEXT ===\n{packet.text_window}\n"
        )
        try:
            resp = self.llm.chat([ChatMessage(role=Role.SYSTEM, content=self.prompt),
                                  ChatMessage(role=Role.USER, content=user)], self.llm_cfg)
            text = resp.text or ""
            if raw_dir:
                os.makedirs(raw_dir, exist_ok=True)
                with open(os.path.join(raw_dir, f"{packet.source_id}_product_raw.txt"), "w") as f:
                    f.write(text)
            ans = json.loads(sanitize_json_text(text), strict=False)
        except Exception as exc:
            logger.warning("product_namer %s failed: %s", packet.source_id, exc)
            return tpl
        if not isinstance(ans, dict):
            return tpl
        name = (ans.get("product_name") or "").strip() if isinstance(ans.get("product_name"), str) else ""
        label = (ans.get("product_label") or "").strip() if isinstance(ans.get("product_label"), str) else ""
        old = (rt.get("product_name") or "").strip()
        accepted = (bool(name) and len(name) <= 120 and name.lower() != old.lower() and not _GENERIC_RE.search(name)
                    and not re.search(r"\b(and|or)\b|[;/]", name))      # one compound, not a pair
        if not accepted and not label:
            logger.info("product_namer %s: no specific name (%s)", packet.source_id, ans.get("basis"))
            return tpl
        out = json.loads(json.dumps(tpl))
        rt2 = out.setdefault("record_template", {})
        if accepted:
            if old:
                rt2["product_name_generic"] = old
            rt2["product_name"] = name
        if label and not rt2.get("product_label") and re.fullmatch(r"(?:[ct]-)?\d{1,2}[a-z]{0,2}'?", label):
            rt2["product_label"] = label
        logger.info("product_namer %s: '%s' -> '%s' label=%s (%s)", packet.source_id, old, rt2.get("product_name"),
                    rt2.get("product_label"), ans.get("basis"))
        return out
