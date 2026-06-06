import os
import json

from src.llm.config import LLMConfig
from src.llm.providers.base import LLMProvider
from src.llm.types import ChatMessage, Role


class LocalVarsBuilder:
    """Build per-figure / per-table sub-variable library via the LLM.

    The ``LLMProvider`` is injected (no SDK import inside this module).
    """

    def __init__(self, llm: LLMProvider, llm_cfg: LLMConfig):
        self.llm = llm
        self.llm_cfg = llm_cfg

    def build(self, source_id, source_type, evidence_data, paper_text, output_dir, csv_head="", scheme_conditions=""):
        """
        Build a sub-variable library JSON for a single figure or table.

        Args:
            source_id:     e.g. "page_2_figure_1_t0" or "page_3_table_0"
            source_type:   "figure" or "table"
            evidence_data: parsed JSON dict from the evidence file
            paper_text:    full PDF text (cached by PDFParser)
            output_dir:    directory to write {source_id}_local_vars.json
            csv_head:      first 6 rows of the table CSV (table only)

        Returns:
            dict — parsed local vars JSON (from cache or freshly built)
        """
        out_path = os.path.join(output_dir, f"{source_id}_local_vars.json")

        # Cache check
        if os.path.exists(out_path):
            print(f"[LocalVarsBuilder] Cache hit: {source_id}")
            with open(out_path) as f:
                return json.load(f)

        print(f"[LocalVarsBuilder] Building local vars for {source_id} ({source_type})...")

        text_window = self._extract_text_window(source_id, source_type, paper_text)

        if source_type == "figure":
            system_prompt, user_prompt = self._build_figure_prompts(
                source_id, evidence_data, text_window
            )
        else:
            system_prompt, user_prompt = self._build_table_prompts(
                source_id, evidence_data, text_window, csv_head, scheme_conditions
            )

        llm_response = self.llm.chat(
            [
                ChatMessage(role=Role.SYSTEM, content=system_prompt),
                ChatMessage(role=Role.USER, content=user_prompt),
            ],
            self.llm_cfg,
        )
        result = self._clean_json(llm_response.text, source_id, source_type)

        os.makedirs(output_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"[LocalVarsBuilder] Saved -> {out_path}")
        return result

    # ------------------------------------------------------------------ #
    #  Prompt builders
    # ------------------------------------------------------------------ #

    _SOLVENT_ABBREV = (
        "Common solvent abbreviations: THF=tetrahydrofuran, Et2O=diethyl ether, "
        "DCM/CH2Cl2=dichloromethane, MeCN=acetonitrile, EtOAc=ethyl acetate, "
        "MeOH=methanol, toluene/PhMe, hexane/hex, MTBE=methyl tert-butyl ether, "
        "dioxane, DMF=dimethylformamide, DMSO. "
        "If you see these abbreviations in the context, treat them as valid solvent names."
    )

    def _build_figure_prompts(self, source_id, ev, text_window):
        system_prompt = (
            "You are an expert flow chemistry data analyst. "
            "Analyze a single extracted figure from a flow chemistry paper and produce "
            "a structured JSON sub-variable library documenting what it measures and how "
            "to interpret each axis and series.\n"
            "Rules:\n"
            "(1) No SMILES.\n"
            "(2) For fixed_conditions: search the **entire paper-text context** for "
            "conditions that apply across ALL data points of this figure. The context window "
            "may include TWO segments separated by `--- experimental section ---` or "
            "`--- local context ---` markers: the figure's local prose AND a separate excerpt "
            "from the paper's Experimental / Materials-and-Methods section. Conditions such as "
            "temperature, pressure, solvent, catalyst (loading), reactor_type, residence_time often "
            "appear ONLY in the experimental section — you MUST populate fixed_conditions from there "
            "too, not only from the figure's surrounding text.\n"
            "(3) Output valid JSON only, no markdown fences.\n"
            "(4) Each text field is suffixed with a source tag in square brackets, e.g. "
            "`Pressure (MPa) [src=vlm_metadata]`. Tag meanings: ``vlm_metadata`` = Gemini vision "
            "transcription (high confidence, clean characters); ``paddleocr`` = best-effort OCR "
            "(may have character drops, word-order issues); ``missing`` = the field was not "
            "found.  Calibrate your confidence accordingly when interpreting axis labels and "
            "legend names.\n"
            f"{self._SOLVENT_ABBREV}"
        )

        meta = ev.get("meta", {})
        raw_data = ev.get("raw_data", [])
        text_ev = ev.get("text_evidence", {})
        figure_type = meta.get("figure_type", "unknown")

        def _join_texts(items):
            """Extract text strings from [{text:..., source:...}, ...] or plain list.

            When a ``source`` key is present (per the per-field decisive
            source design), annotate the text with it so the LLM can
            calibrate its trust: ``vlm_metadata`` is Gemini-clean,
            ``paddleocr`` is best-effort, ``missing`` means absent.
            """
            if not items:
                return ""
            if isinstance(items[0], dict):
                parts = []
                for it in items:
                    txt = it.get("text", "")
                    if not txt:
                        continue
                    src = it.get("source")
                    parts.append(f"{txt} [src={src}]" if src else txt)
                return " | ".join(parts)
            return " | ".join(str(it) for it in items)

        x_title = _join_texts(text_ev.get("x_axis_title", []))
        yl_title = _join_texts(text_ev.get("y_axis_title", []))
        yr_title = _join_texts(text_ev.get("y_right_axis_title", []))
        legend_texts = _join_texts(text_ev.get("legend_text", []))
        chart_texts = _join_texts(text_ev.get("chart_text", []))

        # raw_data keys use Title-case: "X", "Y_Left", "Series"
        x_vals = [pt.get("X") for pt in raw_data if pt.get("X") is not None]
        yl_vals = [pt.get("Y_Left") for pt in raw_data if pt.get("Y_Left") is not None]
        unique_series = list({pt.get("Series", "") for pt in raw_data if pt.get("Series")})

        x_range = f"{min(x_vals):.3g} to {max(x_vals):.3g}" if x_vals else "N/A"
        yl_range = f"{min(yl_vals):.3g} to {max(yl_vals):.3g}" if yl_vals else "N/A"

        user_prompt = f"""=== FIGURE EVIDENCE ===
Source ID: {source_id}
Figure type: {figure_type}
X-axis label: {x_title}
Y-left-axis label: {yl_title}
Y-right-axis label: {yr_title}
Legend text: {legend_texts}
Chart text: {chart_texts}
Data point count: {len(raw_data)}
X range: {x_range}
Y_Left range: {yl_range}
Unique series: {unique_series}

=== RELEVANT PAPER TEXT CONTEXT (4000 chars) ===
{text_window}

=== OUTPUT SCHEMA ===
Output a single valid JSON object (no markdown):
{{
  "source_id": "{source_id}",
  "source_type": "figure",
  "figure_type": "<scatter|line|heatmap|bar|other>",
  "reaction_context": "<one-sentence description of what this figure shows>",
  "axis_semantics": {{
    "x_axis": {{"raw_label": "...", "semantic_meaning": "...", "maps_to_field": "..."}},
    "y_left_axis": {{"raw_label": "...", "semantic_meaning": "...", "maps_to_field": "..."}},
    "y_right_axis": null
  }},
  "series_semantics": {{
    "<series_name>": {{"role": "...", "metric": "...", "description": "..."}}
  }},
  "fixed_conditions": {{
    "temperature_C": null,
    "solvent": null,
    "catalyst": null,
    "reactor_type": null,
    "notes": "..."
  }},
  "data_interpretation_notes": "..."
}}
maps_to_field must be one of: conditions.temperature_C, conditions.residence_time_s,
conditions.flow_rate_mL_min, conditions.solvent, conditions.catalyst, conditions.pressure_bar,
conditions.reactor_type, yield_pct, conversion_pct, selectivity_pct, ee_pct, other_metrics.<name>"""

        return system_prompt, user_prompt

    def _build_table_prompts(self, source_id, ev, text_window, csv_head, scheme_conditions=""):
        system_prompt = (
            "You are an expert flow chemistry data analyst. "
            "Analyze a single extracted table from a flow chemistry paper and produce "
            "a structured JSON sub-variable library documenting what it measures and how "
            "to interpret each column.\n"
            "Rules:\n"
            "(1) No SMILES. "
            "(2) For fixed_conditions: search the table caption/note AND the **entire paper-text context** "
            "for conditions that apply uniformly to ALL rows of this table. The context window may include "
            "TWO segments separated by `--- experimental section ---` or `--- local context ---` markers: "
            "the table's local prose AND a separate excerpt from the paper's Experimental / "
            "Materials-and-Methods section. Conditions such as temperature, pressure, solvent, catalyst "
            "(loading), reactor_type, residence_time often appear ONLY in the experimental section — "
            "you MUST populate fixed_conditions from there too, not only from the caption or CSV. "
            "(3) Output valid JSON only, no markdown fences.\n"
            f"{self._SOLVENT_ABBREV}"
        )

        caption = ev.get("caption_text", "") or ""
        note = ev.get("table_note_text", "") or ""
        num_extracted = ev.get("num_extracted", 0)

        scheme_cond_block = ""
        if scheme_conditions:
            scheme_cond_block = f"""
=== SCHEME CONDITIONS (from reaction scheme image, may contain solvent/catalyst info) ===
{scheme_conditions}
"""

        user_prompt = f"""=== TABLE EVIDENCE ===
Source ID: {source_id}
Caption: {caption}
Table note: {note}
Extracted cell count: {num_extracted}
CSV preview (first 6 rows):
{csv_head}

=== RELEVANT PAPER TEXT CONTEXT (4000 chars) ===
{text_window}
{scheme_cond_block}
=== OUTPUT SCHEMA ===
Output a single valid JSON object (no markdown):
{{
  "source_id": "{source_id}",
  "source_type": "table",
  "figure_type": null,
  "reaction_context": "<one-sentence description of what this table shows>",
  "axis_semantics": null,
  "series_semantics": null,
  "column_semantics": {{
    "<column_header>": "<semantic meaning and which output field it maps to>"
  }},
  "fixed_conditions": {{
    "temperature_C": null,
    "residence_time_s": null,
    "solvent": null,
    "catalyst": null,
    "reactor_type": null,
    "notes": "..."
  }},
  "data_interpretation_notes": "..."
}}
For residence_time_s: convert minutes×60 if needed.
IMPORTANT: fixed_conditions should capture ANY condition that is constant across ALL rows of this table,
even if stated only in the paper text or caption (not as a CSV column). Common examples:
- "all reactions were performed at -78°C" → temperature_C: -78
- "using THF as solvent" → solvent: "tetrahydrofuran"
- "T-shaped micromixer connected to a capillary reactor" → reactor_type: "T-shaped micromixer + capillary reactor"
If a condition varies row-by-row (i.e. it IS a CSV column), leave it null in fixed_conditions."""

        return system_prompt, user_prompt

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _extract_text_window(self, source_id, source_type, paper_text):
        """Dual-anchor text window — see ``pdf_parser.extract_text_window``.

        Primary anchor: figure/table number derived from source_id.
        Experimental anchor: General Procedure / Materials and Methods
        heading.  Total ≈ 8 KB; paper-wide baselines (catalyst loading,
        temperature, solvent) reach the prompt even when they live in
        the experimental section far from the figure citation.
        """
        from src.adjudication.pdf_parser import extract_text_window
        return extract_text_window(
            paper_text,
            source_id,
            source_type,
            primary_size=6000,
            experimental_size=2000,
        )

    def _clean_json(self, content, source_id, source_type):
        """Strip markdown fences and parse JSON. Returns a fallback stub on failure."""
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content
            if content.rstrip().endswith("```"):
                content = content.rstrip().rsplit("\n", 1)[0]
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            print(f"[LocalVarsBuilder] JSON parse failed for {source_id}. Raw:\n{content[:500]}")
            return {
                "source_id": source_id,
                "source_type": source_type,
                "figure_type": None,
                "reaction_context": "",
                "axis_semantics": None,
                "series_semantics": None,
                "column_semantics": None,
                "fixed_conditions": {},
                "data_interpretation_notes": ""
            }
