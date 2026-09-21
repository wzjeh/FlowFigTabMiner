"""Per-source prompt builders for ``PerSourceAssembler``.

Each source (one figure or one table) gets its own LLM call.  The prompt
is built from a ``CommonPreamble`` (output schema, rules, compound
pools, domain knowledge — shared across all sources of one PDF) plus a
small source-specific block (axis labels + raw_data for figures, CSV +
caption for tables).

The Strategy pattern keeps figure / table prompt construction in
separate classes while sharing the heavy preamble.  Adding a new source
type (e.g. scheme) means writing one new builder, not editing
``per_source_assembler.py``.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from src.adjudication.source_discovery import SourcePacket


# ── Output schema and rules — verbatim from the old single-call GlobalAssembly
# prompt so PostProcessor's downstream parser keeps working.  Rule 3 and 6
# are rewritten for the per-source context.

_OUTPUT_SCHEMA = """For each reaction record, output one JSON object with these fields:
{
  "reactant1_smiles": "...",       // SMILES of reactant 1 if available, else null (do NOT guess)
  "reactant1_name": "...",         // name/label of reactant 1 if SMILES not available, else null
  "reactant2_smiles": "...",       // SMILES of reactant 2 if available, else null (do NOT guess)
  "reactant2_name": "...",         // name/label of reactant 2 if SMILES not available, else null
  "product_smiles": "...",         // SMILES if available in data, else null (do NOT guess)
  "product_name": "...",           // name/label if SMILES not available, else null
  "product_label": "...",          // e.g. "4a", "compound 3", null if absent
  "entry_number": null,            // table entry number, e.g. "1", "2a", null if absent
  "yield_pct": null,               // numeric yield %, null if absent
  "yield_type": null,              // how yield was measured: "isolated" | "GC" | "NMR" | "crude" | null
  "batch_yield_pct": null,         // yield of the analogous batch reaction for comparison
  "conversion_pct": null,          // numeric conversion %, null if absent
  "selectivity_pct": null,         // numeric selectivity %, null if absent
  "ee_pct": null,                  // enantiomeric excess %, null if absent
  "diastereomeric_ratio": null,    // diastereomeric ratio as string, null if absent
  "stoichiometry": null,           // molar ratio or equivalents, null if absent
  "reaction_class": "...",         // REQUIRED — one of: "nucleophilic addition" | "halogen-metal exchange"
                                   //   | "directed metalation" | "anionic cyclization" | "C-C coupling"
                                   //   | "C-N coupling" | "C-O coupling" | "polymerization" | "halogenation"
                                   //   | "oxidation" | "reduction" | "hydrogenation" | "esterification"
                                   //   | "amidation" | "alkylation" | "acylation" | "photocatalysis"
                                   //   | "hydrolysis" | "other".  PAPER-LEVEL label; same for every record.
  "paper_doi": null,               // DOI found in paper text, null if not found
  "conditions": {
    "temperature_C": null,
    "residence_time_s": null,      // FLOW only: residence time in seconds (minutes × 60); first reactor (tR1) in two-step systems
    "residence_time_2_s": null,    // FLOW, two-step systems only: second reactor residence time (tR2) in seconds
    "reaction_time_s": null,       // BATCH / flask only: reaction time in seconds; null for flow
    "flow_rate_mL_min": null,
    "flow_rate_stream1_mL_min": null,
    "flow_rate_stream2_mL_min": null,
    "solvent": null,               // expand abbreviations (THF→tetrahydrofuran, etc.)
    "catalyst": null,              // main body name only — no ligand, no loading
    "catalyst_metal": null,
    "catalyst_loading_pct": null,
    "ligand": null,
    "ligand_loading_pct": null,
    "additive": null,
    "pressure_bar": null,
    "reactor_type": null
  },
  "other_metrics": {},             // TON / TOF / productivity / K/S / etc.
  "source_table_or_figure": "...", // human-readable label e.g. "Table 1", "Figure 3"
  "data_correction_note": null,    // OCR / numeric correction note only
  "notes": null                    // narrative context not fitting any structured field
}"""

_BASE_RULES = """=== RULES ===
1. SUBSTRATE SCOPE TABLES: Each row is one reaction record. Extract every row.
2. OPTIMIZATION/SCREENING TABLES: Extract each condition set as a separate record.
3. PER-SOURCE COUNT CHECK: This call extracts records from ONE source only.
   For figures: emit exactly N records where N = the raw_data row count.
   For tables: emit exactly M records where M = (CSV data rows) minus the header rows count.
4. CONDITIONS: Fixed conditions stated in caption / paper text / experimental section
   apply to ALL records from this source.
5. SMILES: Use SMILES only if directly provided.  If only a name is given, set the *_name
   field and leave SMILES null — do NOT invent SMILES.
6. SOURCE RESTRICTION: Extract data ONLY from the source provided below; do NOT pull
   numbers from the Introduction, prior-work prose, or any other figure / table mentioned
   in the paper text.  Paper text is for CONTEXT (interpret labels, find shared conditions,
   pick the reaction_class) — never for new records.
7. OTHER METRICS: Put metrics that don't fit the schema (TON, TOF, productivity g/h,
   K/S, ee, dr, purity %) in the "other_metrics" dict.
8. NO HALLUCINATION: Every numeric value must come from the source's data or paper text.
   Use null for missing fields; do not invent.
9. OUTPUT: Return ONLY a valid JSON array.  No markdown fences, no explanation text.
10. SMILES LOOKUP:
    - If product_label matches a key in PRODUCT STRUCTURE POOL → set product_smiles.
    - If reactant label matches a key in REACTANT STRUCTURE POOL → set reactant1/2_smiles.
    - If only COMPOUND STRUCTURE POOL exists (no arrow detected), infer role from context.
    Do not modify SMILES strings.
11. SCHEME CONDITIONS: If Scheme Conditions are provided and a record lacks certain
    condition fields (temperature, solvent, catalyst), use Scheme Conditions as fallback.
12. LOCAL VARS: If the source has a "local_vars" block:
    - axis_semantics.x_axis.maps_to_field tells you which output field to fill from X
      (MANDATORY — never leave X unused).
    - y_left_axis / y_right_axis maps the Y values to output fields similarly.
    - fixed_conditions apply to ALL records from this source.
    - data_interpretation_notes explain ambiguous values.
    These OVERRIDE your own interpretation.
13. REACTION CLASS: REQUIRED on every record.  Read the paper text, identify the single
    main organolithium-enabled transformation of the paper, and assign that SAME class to
    every record from this source.  Use "other" if none fit.  Never leave reaction_class null.
14. NOTES: Only use notes for narrative context that doesn't fit any structured field.
    Do NOT repeat information already in other fields.
15. CATALYST SPLITTING: Write only the catalyst main body in "catalyst" (e.g. "Pd/C").
    Extract loading into catalyst_loading_pct.  Put ligand name in "ligand", ligand loading
    in "ligand_loading_pct".  Put other co-reagents in "additive".
16. PAPER TEXT WINDOW: The "RELEVANT PAPER TEXT" block may contain TWO segments separated by
    `--- experimental section ---` or `--- local context ---`.  The experimental segment
    holds paper-wide baselines (catalyst loading, default temperature, solvent, reactor type)
    that are often stated ONCE in Materials-and-Methods.  Use BOTH segments when filling
    conditions and reaction_class — do not ignore the experimental segment.
17. PRODUCT-COMPOSITION TABLES: If a table's column headers include product
    names/abbreviations (e.g. columns 'Temperature', 'Catalyst', then product
    columns like '3,4-DCAN', 'CAN', 'CNB', 'AZO' holding yield/selectivity/
    composition values), each product column is a distinct product. Keep ONE
    record per data row (per RULE 3 — do NOT unpivot): set product_name to the
    MAIN/target product (the primary product column header; expand abbreviations
    using the caption/footnote, e.g. "3,4-DCAN" → "3,4-dichloroaniline"), set
    yield_pct (or selectivity_pct/conversion_pct per what the table measures) to
    that product's cell value, and put the OTHER product columns into
    other_metrics (e.g. {"CAN_pct": ..., "AZO_pct": ...}). NEVER leave
    product_name null when product names appear in the column headers.
18. FIGURE FIXED PRODUCT: If a figure's series encodes a CONDITION (temperature,
    residence time, pressure) or is unnamed ("Default") rather than a chemical
    species, the measured product is usually a SINGLE fixed compound named in the
    caption / local_vars.reaction_context / series_semantics / paper text. Fill
    product_name with that fixed product — do NOT leave it null just because the
    series encodes a condition.
19. ABBREVIATIONS: If a product/reactant name is an all-caps abbreviation
    (e.g. "3,4-DCAN", "DCNB"), you MUST expand it to the full chemical name.
    Look it up in BOTH the ABBREVIATION DEFINITIONS block AND the paper text
    for a "full name (ABBR)" definition (e.g. "3,4-dichloroaniline (3,4-DCAN)"
    → use "3,4-dichloroaniline"). Only keep the bare abbreviation if no
    definition exists anywhere.
20. PAPER-LEVEL DEFAULTS: The PAPER-LEVEL block lists conditions the authors state for the
    whole paper (scope=paper, each with a verbatim quote). Precedence for every condition
    field: (a) this source's own data / caption / footnote / local_vars.fixed_conditions,
    then (b) scope=paper defaults, then (c) scope=partial values ONLY if the paper text
    for this source confirms them. When you fill a field from (b) or (c), list that field
    name in the record's "conditions_provenance" object as {"<field>": "paper_global"}.
    reaction_context (main product, reagent, electrophile) may fill product/reactant NAMES
    that this source leaves implicit (e.g. a figure that only shows yield vs conditions).
    PRECEDENCE OF THE SOURCE'S OWN TEXT: if the caption / footnote / header / text layer of
    THIS source mentions a quantity (e.g. "Effects of temperature and residence time",
    "tR = 0.055 s", "at -78 °C"), that field comes from there — or stays null if the source
    varies it — NEVER from a paper-level default. A source described as batch / macrobatch /
    flask never receives flow_rate or residence_time defaults: its time column ("t", "time",
    "reaction time") is conditions.reaction_time_s (seconds), residence_time_s stays null."""

_DOMAIN_KNOWLEDGE = """=== FLOW CHEMISTRY DOMAIN KNOWLEDGE ===
- For organolithium flow chemistry papers: if reactor_type is not explicitly stated in a source,
  it is typically a "T-shaped micromixer + capillary/coil reactor" setup. Apply this default
  ONLY if the paper context confirms flow microreactor use and local_vars.fixed_conditions.reactor_type is null.
- Solvent abbreviations: THF=tetrahydrofuran, Et2O=diethyl ether, DCM/CH2Cl2=dichloromethane,
  MeCN=acetonitrile, EtOAc=ethyl acetate, MeOH=methanol, toluene/PhMe, hexane, MTBE, dioxane,
  DMF=dimethylformamide, DMSO. Expand these abbreviations when filling the solvent field.
- If scheme_conditions contain solvent info, apply it to ALL records from tables in the same paper.
- Unit conversions: 1 MPa = 10 bar; convert pressure values accordingly into pressure_bar."""


_SYSTEM_PROMPT = (
    "You are an expert flow chemistry data extractor. "
    "Your sole task is to extract structured reaction data from one source "
    "(figure or table) of a flow chemistry paper. "
    "Be precise, grounded, and never hallucinate values not present in the provided text or data."
)


@dataclass(frozen=True)
class CommonPreamble:
    """Shared blocks reused across every per-source call within one PDF."""

    reactant_pool_json: str = ""
    product_pool_json: str = ""
    compound_pool_json: str = ""
    scheme_conditions: str = ""
    abbrev_lines: str = ""   # "ABBR = full name" lines, extracted full-text
    global_vars_json: str = ""   # paper-level pool (Step 4.4), rendered once per paper

    @classmethod
    def build(cls, pools: Dict[str, Any], scheme_conditions: str = "",
              abbrev_map: Dict[str, str] = None, global_vars: Dict[str, Any] = None) -> "CommonPreamble":
        """Build a preamble from a ``compound_pool.json`` dict, scheme text,
        and an abbreviation map (extracted from the FULL paper text so every
        source — even ones whose text window misses the definition — gets it)."""
        abbrev_lines = ""
        if abbrev_map:
            abbrev_lines = "\n".join(f"  {a} = {full}" for a, full in sorted(abbrev_map.items()))
        return cls(
            reactant_pool_json=json.dumps(pools.get("reactant_pool", {}), indent=2) if pools.get("reactant_pool") else "",
            product_pool_json=json.dumps(pools.get("product_pool", {}), indent=2) if pools.get("product_pool") else "",
            compound_pool_json=json.dumps(pools.get("compound_pool", {}), indent=2) if pools.get("compound_pool") else "",
            scheme_conditions=scheme_conditions or "",
            abbrev_lines=abbrev_lines,
            global_vars_json=cls._render_global_vars(global_vars),
        )

    @staticmethod
    def _render_global_vars(gv: Dict[str, Any] = None) -> str:
        if not isinstance(gv, dict):
            return ""
        rc = gv.get("reaction_context") or {}
        dc = {k: v for k, v in (gv.get("default_conditions") or {}).items()
              if isinstance(v, dict) and v.get("value") is not None}
        if not rc and not dc:
            return ""
        return json.dumps({"reaction_context": rc, "default_conditions": dc}, indent=2, ensure_ascii=False)

    def render_global_vars_section(self) -> str:
        if not self.global_vars_json:
            return ""
        return (
            "=== PAPER-LEVEL CONTEXT AND DEFAULT CONDITIONS (whole paper; see RULES §20) ===\n"
            + self.global_vars_json
        )

    def render_abbrev_section(self) -> str:
        if not self.abbrev_lines:
            return ""
        return (
            "=== ABBREVIATION DEFINITIONS (extracted from full paper text) ===\n"
            "If a product/reactant name equals one of these abbreviations, "
            "expand it to the full name:\n" + self.abbrev_lines
        )

    def render_pools_section(self) -> str:
        parts = []
        if self.reactant_pool_json:
            parts.append(
                "=== REACTANT STRUCTURE POOL (from scheme, left of reaction arrow) ===\n"
                "Use for reactant_smiles when label matches:\n" + self.reactant_pool_json
            )
        if self.product_pool_json:
            parts.append(
                "=== PRODUCT STRUCTURE POOL (from scheme, right of reaction arrow) ===\n"
                "Use for product_smiles when label matches:\n" + self.product_pool_json
            )
        if self.compound_pool_json:
            parts.append(
                "=== COMPOUND STRUCTURE POOL (role undetermined, no arrow detected) ===\n"
                + self.compound_pool_json
            )
        return "\n\n".join(parts)

    def render_scheme_conditions_section(self) -> str:
        if not self.scheme_conditions:
            return ""
        return (
            "=== SCHEME CONDITIONS (apply to all records from this paper unless source overrides) ===\n"
            + self.scheme_conditions
        )


class PerSourcePromptBuilder(ABC):
    """ABC: builds (system, user) prompts for one ``SourcePacket``."""

    @abstractmethod
    def build(self, packet: SourcePacket, preamble: CommonPreamble) -> Tuple[str, str]:
        ...

    # ── shared rendering helpers (override-safe) ────────────────────────

    def _render_local_vars(self, packet: SourcePacket) -> str:
        if not packet.local_vars:
            return "(no local_vars available for this source)"
        return json.dumps(packet.local_vars, indent=2, ensure_ascii=False)

    def _render_common_blocks(self, preamble: CommonPreamble, packet: SourcePacket) -> str:
        pools = preamble.render_pools_section()
        sch = preamble.render_scheme_conditions_section()
        abbr = preamble.render_abbrev_section()
        out = [_OUTPUT_SCHEMA, _BASE_RULES, _DOMAIN_KNOWLEDGE]
        gv = preamble.render_global_vars_section()
        if gv:
            out.append(gv)
        if pools:
            out.append(pools)
        if sch:
            out.append(sch)
        if abbr:
            out.append(abbr)
        out.append(
            "=== LOCAL VARIABLE LIBRARY FOR THIS SOURCE ===\n"
            "Use it to interpret axis/column meanings, apply fixed_conditions, and pick the\n"
            "reaction_class — see RULES §12.\n"
            + self._render_local_vars(packet)
        )
        out.append(
            "=== RELEVANT PAPER TEXT (~8 KB; may contain dual anchors — see RULES §16) ===\n"
            + (packet.text_window or "(no paper text available)")
        )
        return "\n\n".join(out)


class FigurePromptBuilder(PerSourcePromptBuilder):
    def build(self, packet: SourcePacket, preamble: CommonPreamble) -> Tuple[str, str]:
        ev = packet.evidence
        text_ev = ev.get("text_evidence", {})

        def _join(key: str) -> str:
            items = text_ev.get(key, []) or []
            parts = []
            for it in items:
                if isinstance(it, dict):
                    t = it.get("text", "")
                    src = it.get("source")
                    if t:
                        parts.append(f"{t} [src={src}]" if src else t)
                elif it:
                    parts.append(str(it))
            return " | ".join(parts) if parts else "(none)"

        raw_data = ev.get("raw_data", []) or []
        meta = ev.get("meta", {}) or {}
        figure_type = meta.get("figure_type", "unknown")
        title = meta.get("title", "")
        title_text = title.get("text") if isinstance(title, dict) else (title or "")
        ctx = packet.context or {}
        label = ctx.get("label") or meta.get("label") or "(unresolved)"
        caption = ctx.get("caption") or meta.get("caption_pdf") or meta.get("caption") or ""
        footnote = ctx.get("footnote") or meta.get("footnote_pdf") or ""
        caption_src = ctx.get("caption_source") or meta.get("caption_source") or "missing"
        inner_text = (ctx.get("inner_text") or meta.get("inner_text") or "").strip()
        inner_block = (f"In-figure text (PDF text layer, verbatim — tick labels / legend / annotations):\n{inner_text}\n"
                       if inner_text else "")

        facts = meta.get("facts") or {}
        facts_lines = ""
        if facts:
            facts_lines = "Chart facts (measured, treat as given): " + json.dumps(
                {k: facts.get(k) for k in ("chart_type", "x_scale", "y_left_scale", "axis_fit",
                                           "n_point_labels", "series_matched_ratio") if k in facts}) + "\n"
            if facts.get("chart_type") == "heatmap":
                facts_lines += (
                    "HEATMAP RULE: X and Y_Left are CONDITION axes (residence_time_s / temperature_C as the\n"
                    "local_vars axis_semantics say); the outcome of each point is Y_Right/Data_Value → yield_pct.\n"
                    "Never put a Y_Left value into yield_pct.  X values are already physical (seconds).\n"
                )

        figure_block = (
            f"=== THIS SOURCE: {packet.human_label} ({packet.source_id}) ===\n"
            f"Paper label: {label}\n"
            f"Caption [src={caption_src}]: {caption or '(none)'}\n"
            f"Footnote: {footnote or '(none)'}\n"
            f"Figure type: {figure_type}\n"
            + facts_lines + inner_block +
            f"Title: {title_text or '(none)'}\n"
            f"Axis labels (with [src=…] provenance tags):\n"
            f"  X-axis title: {_join('x_axis_title')}\n"
            f"  Y-left-axis title: {_join('y_axis_title')}\n"
            f"  Y-right-axis title: {_join('y_right_axis_title')}\n"
            f"  Legend text: {_join('legend_text')}\n"
            f"  Chart text: {_join('chart_text')}\n\n"
            f"=== RAW DATA ({len(raw_data)} data points — emit EXACTLY ONE record per point) ===\n"
            + json.dumps(raw_data, indent=2)
        )

        user_prompt = (
            "You are extracting reaction records from a SINGLE FIGURE of a flow chemistry paper.\n\n"
            + figure_block
            + "\n\n"
            + self._render_common_blocks(preamble, packet)
            + "\n\n=== TASK ===\n"
            f"Emit EXACTLY {len(raw_data)} reaction record(s) — one JSON object per raw_data point — "
            f"as a JSON array.  Tag every record with "
            f'"source_table_or_figure": "{packet.human_label}".\n'
            "Output the JSON array only.\n"
        )
        return _SYSTEM_PROMPT, user_prompt


class FigureTemplateBuilder(FigurePromptBuilder):
    """Design step D: ask for ONE figure template instead of N transcribed
    records.  The per-point numbers are then copied by
    ``figure_synthesis.synthesize_records`` — the LLM only decides semantics
    (which raw column is which field, what each legend series means, what
    the fixed product / reactants / conditions are)."""

    _TEMPLATE_SCHEMA = """=== FIGURE TEMPLATE OUTPUT (ONE JSON object, not an array) ===
{
  "record_template": { <every field of the record schema above, filled with everything that is SHARED
                        by all points of this figure: reactant1/2, product (name/label/SMILES — per RULE 18
                        the fixed product named in the caption / reaction_context), reaction_class, paper_doi,
                        yield_type, and EVERY condition that is constant for the figure (temperature_C,
                        solvent, residence_time_s, flow rates, catalyst, reactor_type … from the caption,
                        footnote, local_vars.fixed_conditions, paper text or PAPER-LEVEL defaults).
                        Leave null ONLY the fields that vary point-to-point, i.e. the fields named in
                        axis_map / series_map below.> },
  "axis_map": {
    "X": "<field path or null>",
    "Y_Left": "<field path or null>",
    "Y_Right/Data_Value": "<field path or null>"
  },
  "axis_transforms": { "X": {"scale": 1, "offset": 0} },   // only when a unit conversion is needed
                                                            // (e.g. minutes → seconds: scale 60); else omit
  "series_map": {
    "<legend series name exactly as in raw_data>": { "<field path>": <value>, ... }
  },
  "notes": "<one line on how the mapping was decided>"
}
Field paths: conditions.temperature_C, conditions.residence_time_s, conditions.residence_time_2_s, conditions.flow_rate_mL_min,
conditions.pressure_bar, conditions.solvent, conditions.catalyst, yield_pct, conversion_pct,
selectivity_pct, ee_pct, product_name, product_label, product_smiles, reactant1_name, reactant2_name,
other_metrics.<name>.
Rules for the template:
- axis_map MUST follow local_vars.axis_semantics (and the CHART FACTS / HEATMAP RULE when present).
  X values are already physical units; do not add transforms unless the axis label proves a unit mismatch.
- series_map: when a legend series encodes a condition ("-78 °C", "tR = 2 s"), map it to that condition
  field with a NUMERIC value; when it encodes a product/substrate ("3a", "Ar = Ph"), map it to
  product_label / product_name / reactant fields.  Every series in raw_data must appear.
- Do NOT copy any per-point numbers into record_template; the code copies them from raw_data.
- A condition that is fixed for the whole figure (e.g. "at -78 °C" in the caption while X is residence
  time) belongs in record_template.conditions — do not leave it null just because it is a number.
- product_name / reactant names must be filled whenever the caption, local_vars.reaction_context or the
  paper-level context names them (RULE 18); never leave the product null on a yield figure.
- Output the JSON object only."""

    def build(self, packet: SourcePacket, preamble: CommonPreamble) -> Tuple[str, str]:
        _, legacy_prompt = super().build(packet, preamble)
        # Strip the legacy "emit EXACTLY N records" task block and the raw
        # point dump; the code does the per-point work now.
        head = legacy_prompt.split("=== RAW DATA (", 1)[0]
        common = legacy_prompt.split("\n\n", 1)[1]
        common = common.split("=== THIS SOURCE:", 1)[1]
        common = "=== THIS SOURCE:" + common.split("=== RAW DATA (", 1)[0] + self._render_common_blocks(preamble, packet)
        raw_data = packet.evidence.get("raw_data", []) or []
        series = sorted({str(p.get("Series")) for p in raw_data if isinstance(p, dict) and p.get("Series") is not None})
        sample = json.dumps(raw_data[:8], indent=None)
        user_prompt = (
            "You are describing how to turn the data points of a SINGLE FIGURE of a flow chemistry paper "
            "into reaction records. Do NOT transcribe points; produce ONE template.\n\n"
            + common
            + f"\n\n=== RAW DATA SUMMARY ({len(raw_data)} points; first 8 shown verbatim) ===\n"
            f"Series present: {series}\n{sample}\n\n"
            + self._TEMPLATE_SCHEMA
        )
        return _SYSTEM_PROMPT, user_prompt


class TablePromptBuilder(PerSourcePromptBuilder):
    def build(self, packet: SourcePacket, preamble: CommonPreamble) -> Tuple[str, str]:
        ev = packet.evidence
        caption = ev.get("caption_text", "") or ""
        note = ev.get("table_note_text", "") or ""
        header_row_count = ev.get("header_row_count", 1)
        num_extracted = ev.get("num_extracted", 0)
        csv_lines = (packet.csv_content or "").splitlines()
        data_row_count = max(0, len(csv_lines) - int(header_row_count or 0))
        ctx = packet.context or {}
        inner_text = (ctx.get("inner_text") or ev.get("inner_text") or "").strip()
        inner_block = ""
        if inner_text:
            inner_block = (
                "\n\n=== TABLE TEXT LAYER (verbatim from the PDF, row order) ===\n"
                "Authoritative for entry numbers and numeric cells (yields, temperatures, times) when the CSV\n"
                "is incomplete or garbled; align its rows with the CSV rows by order / entry number.\n"
                + inner_text
            )

        table_block = (
            f"=== THIS SOURCE: {packet.human_label} ({packet.source_id}) ===\n"
            f"Caption: {caption or '(none)'}\n"
            f"Note: {note or '(none)'}\n"
            f"Header rows: {header_row_count}\n"
            f"Cells extracted: {num_extracted}\n\n"
            f"=== CSV CONTENT ({len(csv_lines)} total rows; {data_row_count} data rows after header) ===\n"
            + (packet.csv_content or "(empty)")
            + inner_block
        )

        user_prompt = (
            "You are extracting reaction records from a SINGLE TABLE of a flow chemistry paper.\n\n"
            + table_block
            + "\n\n"
            + self._render_common_blocks(preamble, packet)
            + "\n\n=== TASK ===\n"
            f"Emit EXACTLY {data_row_count} reaction record(s) — one JSON object per CSV data row "
            f"(after the {header_row_count} header row(s)) — as a JSON array.  Tag every record with "
            f'"source_table_or_figure": "{packet.human_label}".\n'
            "Output the JSON array only.\n"
        )
        return _SYSTEM_PROMPT, user_prompt
