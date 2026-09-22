import os
import json
from src.adjudication.source_discovery import panel_line as _panel_line
import re
from src.adjudication.time_statements import find_residence_time_statements, fixed_candidates

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

    def build(self, source_id, source_type, evidence_data, paper_text, output_dir, csv_head="",
              scheme_conditions="", context=None, global_vars=None):
        """
        Build a sub-variable library JSON for a single figure or table.

        Args:
            source_id:     e.g. "page_2_figure_1_t0" or "page_3_table_0"
            source_type:   "figure" or "table"
            evidence_data: parsed JSON dict from the evidence file
            paper_text:    full PDF text (cached by PDFParser)
            output_dir:    directory to write {source_id}_local_vars.json
            csv_head:      first 6 rows of the table CSV (table only)
            context:       CaptionLocator context dict (label / caption /
                           footnote from the PDF text layer), or None

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

        context = context or {}
        text_window = self._extract_text_window(source_id, source_type, paper_text, context)
        global_block = self._render_global_vars(global_vars)

        # Residence-time statements the model may cite (caption / note / legend /
        # scheme text / paper window); nothing is ever computed from reactor
        # geometry (Zhao 2026-09-21).  Same rule for figures and tables: the
        # only difference is what "the data carries this quantity" means.
        if source_type == "figure":
            scan = self._figure_scan_text(evidence_data, context, text_window)
        else:
            scan = " ".join(str(x or "") for x in (evidence_data.get("caption_text"), evidence_data.get("table_note_text"),
                                                   scheme_conditions, text_window))
        time_cands = fixed_candidates(find_residence_time_statements(scan))
        for c in time_cands:
            c["scope"] = "near"
        if not time_cands and paper_text:
            # Nothing near the source: statements elsewhere in the paper are
            # offered for citation only (the model decides whether the
            # sentence describes this source; never filled deterministically).
            time_cands = fixed_candidates(find_residence_time_statements(paper_text))
            for c in time_cands:
                c["scope"] = "paper"
        if source_type == "figure":
            system_prompt, user_prompt = self._build_figure_prompts(
                source_id, evidence_data, text_window, context, time_cands
            )
        else:
            system_prompt, user_prompt = self._build_table_prompts(
                source_id, evidence_data, text_window, csv_head, scheme_conditions, context, time_cands
            )
        if global_block:
            user_prompt = user_prompt.replace("=== OUTPUT SCHEMA ===", global_block + "\n=== OUTPUT SCHEMA ===", 1)

        llm_response = self.llm.chat(
            [
                ChatMessage(role=Role.SYSTEM, content=system_prompt),
                ChatMessage(role=Role.USER, content=user_prompt),
            ],
            self.llm_cfg,
        )
        result = self._clean_json(llm_response.text, source_id, source_type)
        if source_type == "figure":
            result = self._enforce_chart_facts(result, evidence_data)
            result = self._enforce_time_candidates(result, time_cands, carried=self._figure_carried(result, evidence_data))
        else:
            result = self._enforce_time_candidates(result, time_cands, csv_head, evidence_data.get("header_row_count") or 1)

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

    def _build_figure_prompts(self, source_id, ev, text_window, context=None, time_cands=None):
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
            "too, not only from the figure's surrounding text. PRECEDENCE: the figure's own caption / "
            "footnote outranks the paper text and any paper-level default — a quantity the figure VARIES "
            "(its axes or legend series) must stay null in fixed_conditions.\n"
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
        context = context or {}
        label = context.get("label") or meta.get("label") or "(unresolved)"
        caption = context.get("caption") or meta.get("caption_pdf") or meta.get("caption") or ""
        footnote = context.get("footnote") or meta.get("footnote_pdf") or ""
        caption_src = context.get("caption_source") or meta.get("caption_source") or "missing"
        inner_text = (context.get("inner_text") or meta.get("inner_text") or "").strip()
        inner_block = (f"In-figure text from the PDF text layer (tick labels / legend / annotations, verbatim):\n{inner_text}\n"
                       if inner_text else "")

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
        dv_vals = [pt.get("Y_Right/Data_Value") for pt in raw_data if pt.get("Y_Right/Data_Value") is not None]
        dv_range = f"{min(dv_vals):.3g} to {max(dv_vals):.3g}" if dv_vals else "N/A"
        facts_block = self._render_chart_facts(meta.get("facts") or {}, len(dv_vals), len(raw_data))

        panel_line = _panel_line(ev)
        user_prompt = f"""=== FIGURE EVIDENCE ===
Source ID: {source_id}
Paper label: {label}
Caption [src={caption_src}]: {caption or '(none)'}
{panel_line}Footnote: {footnote or '(none)'}
{inner_block}Figure type: {figure_type}
X-axis label: {x_title}
Y-left-axis label: {yl_title}
Y-right-axis label: {yr_title}
Legend text: {legend_texts}
Chart text: {chart_texts}
Data point count: {len(raw_data)}
X range: {x_range}
Y_Left range: {yl_range}
Y_Right/Data_Value range: {dv_range} ({len(dv_vals)} of {len(raw_data)} points carry a value)
Unique series: {unique_series}
{facts_block}{self._render_time_block(time_cands, "figure")}
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
    "y_right_axis": null,
    "data_value": null
  }},
  "series_semantics": {{
    "<series_name>": {{"role": "...", "metric": "...", "description": "..."}}
  }},
  "fixed_conditions": {{
    "temperature_C": null,
    "residence_time_s": null,
    "residence_time_quote": null,
    "residence_time_2_s": null,
    "residence_time_2_quote": null,
    "solvent": null,
    "catalyst": null,
    "reactor_type": null,
    "notes": "..."
  }},
  "data_interpretation_notes": "..."
}}
maps_to_field must be one of: conditions.temperature_C, conditions.residence_time_s, conditions.residence_time_2_s, conditions.reaction_time_s,
conditions.flow_rate_mL_min, conditions.solvent, conditions.catalyst, conditions.pressure_bar,
conditions.reactor_type, yield_pct, conversion_pct, selectivity_pct, ee_pct, other_metrics.<name>
"data_value" describes the per-point "Y_Right/Data_Value" column when it is populated (same shape as
the axis entries; REQUIRED when CHART FACTS say the chart is a heatmap or point labels were read)."""

        return system_prompt, user_prompt

    @staticmethod
    def _render_global_vars(gv):
        """Paper-level pool (Step 4.4) as a prompt block.  Values with
        scope=paper are defaults for fixed_conditions unless THIS source's
        caption / footnote / CSV / text says otherwise."""
        if not isinstance(gv, dict):
            return ""
        rc = gv.get("reaction_context") or {}
        dc = gv.get("default_conditions") or {}
        lines = ["=== PAPER-LEVEL CONTEXT AND DEFAULT CONDITIONS (from the whole paper; Step 4.4) ===",
                 "Use scope=paper values as fixed_conditions defaults when this source does not state its own;",
                 "scope=partial values are hints only. Never override a value stated by this source."]
        for k in ("main_transformation", "substrate_class", "main_product_name", "organometallic_reagent",
                  "electrophile", "reactor_description"):
            if rc.get(k):
                lines.append(f"  {k}: {rc[k]}")
        for field, v in dc.items():
            if isinstance(v, dict) and v.get("value") is not None:
                q = (v.get("quote") or "")[:140]
                lines.append(f"  {field} = {v['value']!r}  [scope={v.get('scope')}]  quote: \"{q}\"")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _render_time_block(time_cands, kind):
        """Prompt block listing the verbatim residence-time statements the model may cite."""
        what = "THIS FIGURE" if kind == "figure" else "THIS TABLE"
        if not time_cands:
            return f"""
=== RESIDENCE-TIME STATEMENTS FOUND IN THE PAPER TEXT ===
(none) — fixed_conditions.residence_time_s stays null; never derive it from reactor volume, length or flow rate.
"""
        lines = "\n".join(f"- {c['value_s']:g} s{(' (step ' + str(c['step']) + ')') if c.get('step') else ''} — \"{c['quote'][:300]}\""
                          for c in time_cands)
        where = (f"NEAR {what}" if all(c.get("scope", "near") == "near" for c in time_cands)
                 else f"ELSEWHERE IN THE PAPER (none near {what.lower()} — cite one only if it clearly describes its runs)")
        varies = ("an axis or legend series" if kind == "figure" else "a column")
        return f"""
=== RESIDENCE-TIME STATEMENTS FOUND {where} (verbatim) ===
{lines}
fixed_conditions.residence_time_s may ONLY be one of these values (seconds), with residence_time_quote = that
sentence verbatim, and only when the sentence describes the conditions of {what}; a residence time that
{varies} of {what.lower()} varies stays null. Otherwise both are null.
Two-reactor systems: a "(step 2)" / tR2 statement goes to residence_time_2_s (+ residence_time_2_quote), the
first step (tR1, or an unnumbered tR) to residence_time_s. Never derive a residence time from reactor volume,
length or flow rate.
"""

    @staticmethod
    def _figure_scan_text(ev, context, text_window):
        """Text a figure may cite: its caption / footnote, the legend and chart
        text read from the image, and the paper window around it."""
        meta = (ev or {}).get("meta", {}) or {}
        context = context or {}
        text_ev = (ev or {}).get("text_evidence", {}) or {}
        parts = [context.get("caption") or meta.get("caption_pdf") or meta.get("caption"),
                 context.get("footnote") or meta.get("footnote_pdf")]
        for key in ("legend_text", "chart_text"):
            for it in text_ev.get(key) or []:
                parts.append(it.get("text") if isinstance(it, dict) else it)
        parts.append(text_window)
        return " ".join(str(x) for x in parts if x)

    @classmethod
    def _figure_carried(cls, result, ev):
        """Which residence-time fields the figure's data itself carries: an axis
        mapped to the field, or a legend series that states the time."""
        ax = result.get("axis_semantics") if isinstance(result, dict) else None
        axis_fields = {a.get("maps_to_field") for a in (ax or {}).values() if isinstance(a, dict)}
        series = {str(p.get("Series")) for p in ((ev or {}).get("raw_data") or []) if isinstance(p, dict) and p.get("Series")}
        sem = result.get("series_semantics") if isinstance(result, dict) else None
        series_text = " | ".join(sorted(series | set((sem or {}).keys() if isinstance(sem, dict) else ())))
        sem_json = json.dumps(sem) if isinstance(sem, dict) else ""
        return {
            "residence_time_s": "conditions.residence_time_s" in axis_fields
                                or bool(cls._TR_COLUMN_RE.search(series_text)) or "residence_time_s" in sem_json,
            "residence_time_2_s": "conditions.residence_time_2_s" in axis_fields
                                  or bool(cls._TR2_COLUMN_RE.search(series_text)) or "residence_time_2_s" in sem_json,
        }

    @staticmethod
    def _enforce_chart_facts(result, ev):
        """Deterministic guard: the pipeline's measured chart facts outrank
        the LLM's guess.  For a heatmap the axis roles are fixed by
        construction (X = residence time on the converted log axis, Y_Left =
        temperature, Y_Right/Data_Value = the yield read from the cell
        label), so a swapped / missing mapping is corrected here rather
        than propagated into every record."""
        meta = (ev or {}).get("meta", {}) or {}
        facts = meta.get("facts") or {}
        if not isinstance(result, dict) or facts.get("chart_type") != "heatmap":
            return result
        ax = result.get("axis_semantics")
        if not isinstance(ax, dict):
            ax = {}
        raw = (ev or {}).get("raw_data", []) or []
        yl = [r.get("Y_Left") for r in raw if isinstance(r, dict) and r.get("Y_Left") is not None]
        y_is_temp = bool(yl) and (min(yl) < 0 or max(yl) <= 400)
        fixes = []

        def _set(key, field, meaning):
            cur = ax.get(key) if isinstance(ax.get(key), dict) else {}
            if cur.get("maps_to_field") != field:
                fixes.append(f"{key}: {cur.get('maps_to_field')!r} -> {field!r}")
            ax[key] = {"raw_label": cur.get("raw_label", ""), "semantic_meaning": cur.get("semantic_meaning") or meaning,
                       "maps_to_field": field}

        x_cur = ax.get("x_axis") if isinstance(ax.get("x_axis"), dict) else {}
        if x_cur.get("maps_to_field") != "conditions.residence_time_2_s":
            _set("x_axis", "conditions.residence_time_s", "residence time (log axis converted to seconds)")
        if y_is_temp:
            _set("y_left_axis", "conditions.temperature_C", "reaction temperature (°C)")
        dv = ax.get("data_value") if isinstance(ax.get("data_value"), dict) else {}
        dv_field = dv.get("maps_to_field")
        if dv_field not in ("yield_pct", "conversion_pct", "selectivity_pct"):
            _set("data_value", "yield_pct", "yield (%) read from the heatmap cell label")
        result["axis_semantics"] = ax
        if result.get("figure_type") != "heatmap":
            fixes.append(f"figure_type: {result.get('figure_type')!r} -> 'heatmap'")
            result["figure_type"] = "heatmap"
        if fixes:
            note = "Axis roles enforced from measured chart facts (heatmap): " + "; ".join(fixes)
            result["data_interpretation_notes"] = (note + " | " + (result.get("data_interpretation_notes") or "")).strip(" |")
            print(f"[LocalVarsBuilder] heatmap guard applied: {fixes}")
        return result

    @staticmethod
    def _render_chart_facts(facts, n_dv, n_points):
        """Deterministic facts from the figure pipeline, stated as givens so
        the LLM does not re-guess chart type / axis scale from raw numbers."""
        if not facts:
            return ""
        lines = ["=== CHART FACTS (measured by the extraction pipeline — treat as given) ==="]
        ct = facts.get("chart_type")
        if ct == "heatmap":
            lines.append(
                "This chart is a HEATMAP (colour-binned yield map). Interpretation is FIXED:\n"
                "  - X = a condition axis, almost always residence time; values are already in physical\n"
                "    units (the log axis has been converted, so 0.01–100 means seconds, NOT exponents).\n"
                "  - Y_Left = the other condition axis, almost always temperature in °C.\n"
                "  - Y_Right/Data_Value = the measured outcome (yield %) read from the cell label.\n"
                "  Set figure_type=\"heatmap\", map x_axis and y_left_axis to condition fields, map\n"
                "  data_value to yield_pct (or conversion_pct if the caption says conversion)."
            )
        elif ct:
            lines.append(f"Chart type: {ct} (xy plot: X is the independent variable, Y_Left the plotted outcome/response).")
        if facts.get("x_scale"):
            lines.append(f"X axis scale: {facts['x_scale']}; Y_Left scale: {facts.get('y_left_scale')}")
        fit = facts.get("axis_fit") or {}
        if fit:
            lines.append(f"Axis calibration: X={'ok' if fit.get('x') else 'FAILED'}, "
                         f"Y_Left={'ok' if fit.get('y_left') else 'FAILED'}"
                         + (" — a FAILED axis means its values are unreliable; say so in data_interpretation_notes." if not (fit.get('x') and fit.get('y_left')) else ""))
        if n_dv:
            lines.append(f"Per-point value labels were read for {n_dv}/{n_points} points (Y_Right/Data_Value column).")
        smr = facts.get("series_matched_ratio")
        if smr is not None and smr == 0 and facts.get("n_series_legend", 0) > 0:
            lines.append("Legend-to-point colour matching FAILED: every point carries Series='Default'; "
                         "do NOT infer per-point conditions from the series name.")
        elif smr == 0:
            lines.append("No legend detected: series labels are not available for this chart.")
        return "\n".join(lines) + "\n"

    # a tR / tR1 / t1 column varies the first step; a tR2 / t2 column the second
    _TR2_COLUMN_RE = re.compile(r"\bt\s*_?\s*R?\s*2\b|\bR_?t\s*2\b|\bτ\s*_?2\b|residence\s+time.{0,12}(?:t\s*_?\s*R?\s*2|R2|second)\b", re.I)
    _TR_COLUMN_RE = re.compile(r"residence\s+time(?!.{0,12}(?:t\s*_?\s*R?\s*2|R2|second)\b)|\bres\.?\s*time|\bt\s*_?\s*R\s*1?\b|\bt\s*_?\s*1\b"
                               r"|\bR_?t\s*1?\b|retention time|\bτ\s*_?1?\b", re.I)
    _BATCH_RE = re.compile(r"batch|flask|vial|stirr", re.I)
    # (field, quote key, source key, admissible statement steps, column regex, auto-fill a single near candidate)
    # The second step is never auto-filled: whether a table belongs to a two-reactor
    # sequence is the model's reading of the table, not a property of the text nearby.
    _TIME_FIELDS = (("residence_time_s", "residence_time_quote", "residence_time_source", (None, 1), _TR_COLUMN_RE, True),
                    ("residence_time_2_s", "residence_time_2_quote", "residence_time_2_source", (2,), _TR2_COLUMN_RE, False))

    @classmethod
    def _enforce_time_candidates(cls, result, time_cands, csv_head="", header_rows=1, carried=None):
        """fixed_conditions.residence_time_s (first step) / residence_time_2_s
        (second step of a two-reactor system) are each either one of the quoted
        statements found in the paper text (with the sentence in the *_quote
        key) or null — never a computed or remembered number.  A table that
        varies that time (a column for it) keeps it null; a single near-table
        candidate for a flow table is filled deterministically.  The per-field
        verdict is stored in ``result["time_guard"]`` so the post-processor can
        hold every record of the source to the same rule.  ``carried`` (figures)
        replaces the column check: the axis / legend series carries the field."""
        fc = result.get("fixed_conditions") if isinstance(result, dict) else None
        if not isinstance(fc, dict):
            return result
        header = "\n".join((csv_head or "").split("\n")[:max(1, int(header_rows or 1))])
        is_batch = bool(cls._BATCH_RE.search(str(fc.get("reactor_type") or "")))
        guard = result.setdefault("time_guard", {})
        # the statements themselves, for the per-source prompt (rendered with local_vars)
        result["time_candidates"] = [{"value_s": c["value_s"], "step": c.get("step"), "scope": c.get("scope", "near"),
                                      "quote": c["quote"][:240]} for c in (time_cands or [])[:12]]
        for field, qkey, skey, steps, col_re, autofill in cls._TIME_FIELDS:
            cands = [c for c in (time_cands or []) if c.get("step") in steps]
            allowed = {round(float(c["value_s"]), 6): c for c in cands}
            val = fc.get(field)
            has_column = bool(carried.get(field)) if carried is not None else bool(col_re.search(header))
            # the same rule applies to every record of this table (post_processor.inherit_conditions):
            # a value comes from a column of the table or is one of the quoted statements
            guard[field] = {"column": has_column, "allowed": sorted(allowed)}
            if has_column:
                fc[field] = None; fc[qkey] = None
                continue
            match = None
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                match = next((c for a, c in allowed.items() if abs(float(val) - a) <= 1e-6 * max(1.0, abs(a))), None)
                if match is None:
                    print(f"[LocalVarsBuilder] {field}={val} is not a quoted statement — cleared")
                    fc[field] = None; fc[qkey] = None
            near = all(c.get("scope", "near") == "near" for c in cands)
            if autofill and fc.get(field) is None and len(allowed) == 1 and not is_batch and near:
                (v, c), = allowed.items()
                fc[field] = v; fc[qkey] = c["quote"]; fc[skey] = "paper_text_single_candidate"
            elif match is not None:
                fc[qkey] = fc.get(qkey) or match["quote"]; fc[skey] = "paper_text_quoted"
        return result

    def _build_table_prompts(self, source_id, ev, text_window, csv_head, scheme_conditions="", context=None, time_cands=None):
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

        context = context or {}
        # caption / note precedence already settled by apply_context_to_evidence
        caption = ev.get("caption_text", "") or ""
        note = ev.get("table_note_text", "") or ""
        label = context.get("label") or "(unresolved)"
        num_extracted = ev.get("num_extracted", 0)
        inner_text = (context.get("inner_text") or ev.get("inner_text") or "").strip()
        inner_block = (f"\n=== TABLE TEXT LAYER (verbatim from the PDF, row order; authoritative for numbers) ===\n{inner_text}\n"
                       if inner_text else "")

        time_block = self._render_time_block(time_cands, "table")
        scheme_cond_block = ""
        if scheme_conditions:
            scheme_cond_block = f"""
=== SCHEME CONDITIONS (from reaction scheme image, may contain solvent/catalyst info) ===
{scheme_conditions}
"""

        user_prompt = f"""=== TABLE EVIDENCE ===
Source ID: {source_id}
Paper label: {label}
Caption: {caption}
Table note: {note}
Extracted cell count: {num_extracted}
CSV preview (first 6 rows):
{csv_head}
{inner_block}
=== RELEVANT PAPER TEXT CONTEXT (4000 chars) ===
{text_window}
{scheme_cond_block}{time_block}
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
    "residence_time_quote": null,
    "residence_time_2_s": null,
    "residence_time_2_quote": null,
    "reaction_time_s": null,
    "solvent": null,
    "catalyst": null,
    "reactor_type": null,
    "notes": "..."
  }},
  "data_interpretation_notes": "..."
}}
For residence_time_s: convert minutes×60 if needed. Two-step flow tables (tR1 / tR2 columns, reactors R1 / R2):
column tR1 maps to conditions.residence_time_s, column tR2 to conditions.residence_time_2_s.
PRECEDENCE: the table's own caption / note / header outranks the paper text and the paper-level defaults:
if the caption or note states a condition, use it; if the table VARIES a quantity (a column for it, or
the caption says "effect of temperature"), leave that field null — never fill it from paper-level defaults.
A batch / macrobatch table has no flow_rate or residence_time defaults; its time column maps to
conditions.reaction_time_s (seconds), never to residence_time_s.
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

    def _extract_text_window(self, source_id, source_type, paper_text, context=None):
        """Dual-anchor text window — see ``pdf_parser.extract_text_window``.

        Primary anchor: the source's REAL paper label ("Figure 2") from the
        CaptionLocator context (every in-text mention), plus the caption's
        first words; falls back to the source_id-derived keyword when no
        context exists.  Experimental anchor: General Procedure / Materials
        and Methods heading.  Total ≈ 8 KB.
        """
        from src.adjudication.pdf_parser import extract_text_window
        context = context or {}
        caption = context.get("caption") or ""
        extra = (caption[:60],) if len(caption) >= 5 else None
        return extract_text_window(
            paper_text,
            source_id,
            source_type,
            primary_size=6000,
            experimental_size=2000,
            extra_anchor_keywords=extra,
            label=context.get("label"),
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
