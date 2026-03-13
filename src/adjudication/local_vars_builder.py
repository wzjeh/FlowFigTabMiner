import os
import json
from src.adjudication.llm_engine import LLMEngine


class LocalVarsBuilder:
    def __init__(self):
        self.llm = LLMEngine()

    def build(self, source_id, source_type, evidence_data, paper_text, output_dir, csv_head=""):
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
                source_id, evidence_data, text_window, csv_head
            )

        raw = self.llm.chat(system_prompt, user_prompt)
        result = self._clean_json(raw, source_id, source_type)

        os.makedirs(output_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"[LocalVarsBuilder] Saved -> {out_path}")
        return result

    # ------------------------------------------------------------------ #
    #  Prompt builders
    # ------------------------------------------------------------------ #

    def _build_figure_prompts(self, source_id, ev, text_window):
        system_prompt = (
            "You are an expert flow chemistry data analyst. "
            "Analyze a single extracted figure from a flow chemistry paper and produce "
            "a structured JSON sub-variable library documenting what it measures and how "
            "to interpret each axis and series.\n"
            "Rules:\n"
            "(1) No SMILES. (2) Only state fixed_conditions explicitly mentioned in the context. "
            "(3) Output valid JSON only, no markdown fences."
        )

        meta = ev.get("meta", {})
        raw_data = ev.get("raw_data", [])
        axes = ev.get("axes", {})
        series_list = ev.get("series", [])

        x_title = axes.get("x_axis_title", "")
        yl_title = axes.get("y_left_axis_title", "")
        yr_title = axes.get("y_right_axis_title", "")
        legend_texts = axes.get("legend_text", [])
        chart_texts = axes.get("chart_text", [])
        figure_type = meta.get("figure_type", "unknown")

        x_vals = [pt.get("x") for pt in raw_data if pt.get("x") is not None]
        yl_vals = [pt.get("y_left") for pt in raw_data if pt.get("y_left") is not None]
        unique_series = list({pt.get("series", "") for pt in raw_data if pt.get("series")})

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

    def _build_table_prompts(self, source_id, ev, text_window, csv_head):
        system_prompt = (
            "You are an expert flow chemistry data analyst. "
            "Analyze a single extracted table from a flow chemistry paper and produce "
            "a structured JSON sub-variable library documenting what it measures and how "
            "to interpret each column.\n"
            "Rules:\n"
            "(1) No SMILES. (2) Only state fixed_conditions explicitly mentioned in the context. "
            "(3) Output valid JSON only, no markdown fences."
        )

        caption = ev.get("caption_text", "") or ""
        note = ev.get("table_note_text", "") or ""
        num_extracted = ev.get("num_extracted", 0)

        user_prompt = f"""=== TABLE EVIDENCE ===
Source ID: {source_id}
Caption: {caption}
Table note: {note}
Extracted cell count: {num_extracted}
CSV preview (first 6 rows):
{csv_head}

=== RELEVANT PAPER TEXT CONTEXT (4000 chars) ===
{text_window}

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
For residence_time_s: convert minutes×60 if needed."""

        return system_prompt, user_prompt

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _extract_text_window(self, source_id, source_type, paper_text, window_size=4000):
        """
        Search paper_text for a ~4000-char window relevant to this source.
        Looks for figure/table number keywords derived from the source_id.
        Falls back to the first 4000 chars if nothing found.
        """
        if not paper_text:
            return ""

        # Build search keywords from source_id (e.g. "page_2_figure_1_t0" → "figure 1", "figure1")
        keywords = []
        parts = source_id.lower().split("_")
        if source_type == "figure":
            for i, p in enumerate(parts):
                if p == "figure" and i + 1 < len(parts):
                    num = parts[i + 1]
                    keywords += [f"figure {num}", f"fig. {num}", f"fig {num}", f"figure{num}"]
        elif source_type == "table":
            for i, p in enumerate(parts):
                if p == "table" and i + 1 < len(parts):
                    num = parts[i + 1]
                    keywords += [f"table {num}", f"table{num}"]

        text_lower = paper_text.lower()
        best_pos = -1
        for kw in keywords:
            idx = text_lower.find(kw)
            if idx != -1:
                if best_pos == -1 or idx < best_pos:
                    best_pos = idx

        if best_pos == -1:
            return paper_text[:window_size]

        start = max(0, best_pos - 500)
        end = min(len(paper_text), start + window_size)
        return paper_text[start:end]

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
