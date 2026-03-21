"""
Step 6: PostProcessor — deterministic normalization of GlobalAssembly _final.json output.

Reads:  data/final_output/{basename}_final.json
Writes: data/final_output/{basename}_normalized.json
        data/final_output/{basename}_normalized.xlsx

Does NOT call LLM. All logic is rule-based.
"""

from __future__ import annotations

import os
import re
import json
import glob
import requests


# ---------------------------------------------------------------------------
# Solvent normalisation dictionary
# ---------------------------------------------------------------------------
SOLVENT_NORM = {
    # DCM / methylene chloride
    "dcm": "dichloromethane",
    "ch2cl2": "dichloromethane",
    "methylene chloride": "dichloromethane",
    "methylenechloride": "dichloromethane",
    # DCE
    "1,2-dichloroethane": "1,2-dichloroethane",
    "dce": "1,2-dichloroethane",
    "ethylene dichloride": "1,2-dichloroethane",
    # THF
    "thf": "tetrahydrofuran",
    # Diethyl ether
    "et2o": "diethyl ether",
    "ether": "diethyl ether",
    "diethylether": "diethyl ether",
    "etho": "diethyl ether",
    # Acetonitrile
    "mecn": "acetonitrile",
    "ch3cn": "acetonitrile",
    "ncme": "acetonitrile",
    # Ethyl acetate
    "etoacc": "ethyl acetate",
    "etoac": "ethyl acetate",
    "acoet": "ethyl acetate",
    # Methanol
    "meoh": "methanol",
    "ch3oh": "methanol",
    # Ethanol
    "etoh": "ethanol",
    "ch3ch2oh": "ethanol",
    # DMF
    "dmf": "dimethylformamide",
    # DMSO
    "dmso": "dimethyl sulfoxide",
    # Toluene
    "phme": "toluene",
    "tol": "toluene",
    # MTBE
    "mtbe": "methyl tert-butyl ether",
    "tbe": "methyl tert-butyl ether",
    # Dioxane
    "1,4-dioxane": "dioxane",
    # NMP
    "nmp": "N-methylpyrrolidone",
    # Hexane
    "n-hexane": "hexane",
    # Heptane
    "n-heptane": "heptane",
    # IPA / isopropanol
    "ipa": "isopropanol",
    "2-propanol": "isopropanol",
    "ipro": "isopropanol",
    # Chloroform
    "chcl3": "chloroform",
    # Water
    "h2o": "water",
    # Cyclohexane
    "c6h12": "cyclohexane",
    # Pentane
    "n-pentane": "pentane",
    # Acetone
    "me2co": "acetone",
}

# Reactor type controlled vocabulary
REACTOR_NORM = {
    "t-shaped micromixer + capillary/coil reactor": "T-mixer + capillary coil",
    "t-shape micromixer + capillary coil reactor": "T-mixer + capillary coil",
    "t-mixer + coil": "T-mixer + capillary coil",
    "microreactor coil": "capillary coil reactor",
    "coil reactor": "capillary coil reactor",
    "tubular reactor": "capillary coil reactor",
    "packed bed": "packed bed reactor",
    "packed-bed reactor": "packed bed reactor",
    "pfr": "plug-flow reactor",
    "cstr": "continuous stirred tank reactor",
    "flow microreactor": "flow microreactor",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _norm_solvent_token(token: str) -> str:
    """Normalise a single solvent token. Returns original if not in dict."""
    key = token.strip().lower()
    # Strip parenthetical aliases e.g. "THF (tetrahydrofuran)"
    key = re.sub(r"\s*\(.*?\)", "", key).strip()
    return SOLVENT_NORM.get(key, token.strip())


def normalise_solvent(raw) -> dict:
    """
    Returns dict with keys:
      solvent       — first/primary solvent (normalised)
      solvent_list  — list of all solvents (normalised), or None if single
    """
    if not raw:
        return {"solvent": None, "solvent_list": None}

    raw_str = str(raw).strip()
    # Split on / or + or standalone comma (not digit,digit as in IUPAC names like 1,2-dichloroethane)
    parts = re.split(r"\s*/\s*|\s+\+\s+|(?<!\d),(?!\d)", raw_str)
    normed = [_norm_solvent_token(p) for p in parts if p.strip()]
    if not normed:
        return {"solvent": None, "solvent_list": None}
    if len(normed) == 1:
        return {"solvent": normed[0], "solvent_list": None}
    return {"solvent": normed[0], "solvent_list": normed}


def normalise_reactor(raw) -> str | None:
    if not raw:
        return None
    key = str(raw).strip().lower()
    for pattern, replacement in REACTOR_NORM.items():
        if pattern in key:
            return replacement
    return raw


def extract_doi(text: str) -> str | None:
    """Extract first DOI from text."""
    if not text:
        return None
    m = re.search(r"\b(10\.\d{4,}/\S+?)[\s,;)\]\"']", text)
    if m:
        doi = m.group(1).rstrip(".")
        return doi
    return None


def extract_year(text: str) -> int | None:
    """Extract first 4-digit year (1990–2040) from text."""
    if not text:
        return None
    m = re.search(r"\b(199\d|20[0-3]\d)\b", text)
    return int(m.group(1)) if m else None


def infer_yield_type(record: dict) -> str | None:
    """Infer yield_type from notes and source fields."""
    haystack = " ".join(filter(None, [
        str(record.get("notes") or ""),
        str(record.get("source_table_or_figure") or ""),
    ])).lower()
    if "gc" in haystack:
        return "GC"
    if "nmr" in haystack:
        return "NMR"
    if "isolated" in haystack:
        return "isolated"
    if "crude" in haystack:
        return "crude"
    return None


def humanise_source(raw: str) -> str:
    """
    Convert file-path-style source name to human-readable label.
    e.g. "page_2_table_0_extracted.csv" -> "Table 1 (p.2)"
         "page_3_figure_0_t1"           -> "Figure 1 (p.3)"
    """
    if not raw:
        return raw
    raw = str(raw)
    # Already looks human-readable
    if re.match(r"(Table|Figure|Scheme)\s+\d", raw, re.IGNORECASE):
        return raw

    # page_N_table_M... or page_N_figure_M...
    m = re.match(r"page_(\d+)_(table|figure|fig)_(\d+)", raw, re.IGNORECASE)
    if m:
        page = int(m.group(1)) + 1  # 0-indexed → 1-indexed
        kind = m.group(2).capitalize()
        if kind == "Fig":
            kind = "Figure"
        idx = int(m.group(3)) + 1
        return f"{kind} {idx} (p.{page})"
    return raw


def lookup_smiles(name: str) -> str | None:
    """Query PubChem REST API for SMILES by compound name. Returns None on failure."""
    if not name or not name.strip():
        return None
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(name)}/property/IsomericSMILES/JSON"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return data["PropertyTable"]["Properties"][0]["IsomericSMILES"]
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Notes rescue parsers — extract structured info from free-text notes
# ---------------------------------------------------------------------------

def _extract_entry_number(notes: str) -> tuple:
    """Return (entry_number_str_or_None, remaining_notes)."""
    if not notes:
        return None, ""
    m = re.search(r'[Ee]ntry\s*(\d+[a-z]?)', notes)
    if m:
        val = m.group(1)
        remaining = (notes[:m.start()] + notes[m.end():]).strip()
        return val, remaining
    m = re.search(r'\b(\d+[a-z]?)\s*;', notes)
    if m:
        val = m.group(1)
        remaining = (notes[:m.start()] + notes[m.end():]).strip()
        return val, remaining
    return None, notes


def _extract_diastereomeric_ratio(notes: str) -> tuple:
    """Return (dr_str_or_None, remaining_notes)."""
    if not notes:
        return None, ""
    m = re.search(r'[Aa]nti:syn\s*=?\s*\d+:\d+', notes)
    if not m:
        m = re.search(r'\d+:\d+\s*anti:syn', notes, re.IGNORECASE)
    if m:
        val = m.group(0).strip()
        remaining = (notes[:m.start()] + notes[m.end():]).strip()
        return val, remaining
    return None, notes


def _extract_batch_yield(notes: str) -> tuple:
    """Return (batch_yield_float_or_None, remaining_notes)."""
    if not notes:
        return None, ""
    m = re.search(r'batch\s+yield\s*[:=]?\s*(\d+\.?\d*)\s*%', notes, re.IGNORECASE)
    if not m:
        m = re.search(r'batch.*?(\d+\.?\d*)\s*%', notes, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        remaining = (notes[:m.start()] + notes[m.end():]).strip()
        return val, remaining
    return None, notes


def _extract_stream_flow_rates(notes: str) -> tuple:
    """Return (stream1_or_None, stream2_or_None, remaining_notes)."""
    if not notes:
        return None, None, ""
    matches = list(re.finditer(r'Q\w*\s*=\s*(\d+\.?\d*)\s*mL/min', notes))
    s1 = float(matches[0].group(1)) if len(matches) >= 1 else None
    s2 = float(matches[1].group(1)) if len(matches) >= 2 else None
    remaining = notes
    for m in reversed(matches[:2]):
        remaining = remaining[:m.start()] + remaining[m.end():]
    return s1, s2, remaining.strip()


def _extract_stoichiometry(notes: str) -> tuple:
    """Return (stoichiometry_str_or_None, remaining_notes)."""
    if not notes:
        return None, ""
    m = re.search(r'\[M\]\s*/\s*\[\w+\]\s*=\s*[\d.]+', notes)
    if not m:
        m = re.search(r'[\d.]+\s*equiv', notes, re.IGNORECASE)
    if m:
        val = m.group(0).strip()
        remaining = (notes[:m.start()] + notes[m.end():]).strip()
        return val, remaining
    return None, notes


def build_reaction_smiles(record: dict) -> str | None:
    """
    Build a reaction SMILES string in ORD-compatible format:
        reactants>>reagents>>products
    where:
        reactants = reactant1_smiles[.reactant2_smiles]  (dot-joined, skip None)
        reagents  = empty  (catalyst/solvent SMILES are rarely available; kept for future)
        products  = product_smiles

    Returns None if both reactants and products are absent.
    """
    r1 = (record.get("reactant1_smiles") or "").strip()
    r2 = (record.get("reactant2_smiles") or "").strip()
    p  = (record.get("product_smiles")   or "").strip()

    if not r1 and not p:
        return None

    reactants = ".".join(s for s in [r1, r2] if s)
    return f"{reactants}>>{p}"


def _parse_catalyst_fields(record: dict) -> dict:
    """
    Split catalyst string into structured fields.
    Returns dict of updates to apply to record['conditions'].
    Only sets fields that are not already present.
    """
    conds = dict(record.get("conditions") or {})
    catalyst_raw = str(conds.get("catalyst") or "").strip()
    if not catalyst_raw:
        return {}

    updates = {}

    # Find "Name (X mol%)" patterns
    named_patterns = list(re.finditer(r'([^,(]+?)\s*\((\d+\.?\d*)\s*mol%\)', catalyst_raw))

    if named_patterns:
        first = named_patterns[0]
        if not conds.get("catalyst_loading_pct"):
            updates["catalyst_loading_pct"] = float(first.group(2))
        # Overwrite catalyst to just the main body name
        updates["catalyst"] = first.group(1).strip()

        if len(named_patterns) >= 2:
            second = named_patterns[1]
            if not conds.get("ligand"):
                updates["ligand"] = second.group(1).strip()
            if not conds.get("ligand_loading_pct"):
                updates["ligand_loading_pct"] = float(second.group(2))

        # Additive: text after last named pattern that has no loading
        last_end = named_patterns[-1].end()
        rest = catalyst_raw[last_end:].strip().strip(',').strip()
        if rest and not re.search(r'\d+\s*mol%', rest) and not conds.get("additive"):
            updates["additive"] = rest
    else:
        # Try bare "X mol%" in catalyst string
        m = re.search(r'(\d+\.?\d*)\s*mol%', catalyst_raw)
        if m and not conds.get("catalyst_loading_pct"):
            updates["catalyst_loading_pct"] = float(m.group(1))

    return updates


# Fixed Excel column order
PREFERRED_COLUMNS = [
    "entry_number",
    "reactant1_name", "reactant1_smiles",
    "reactant2_name", "reactant2_smiles",
    "product_name", "product_smiles", "product_label",
    "reaction_smiles",
    "reaction_class",
    "yield_pct", "yield_type", "batch_yield_pct",
    "conversion_pct", "selectivity_pct",
    "diastereomeric_ratio", "ee_pct",
    "temperature_C", "residence_time_s", "flow_rate_mL_min",
    "flow_rate_stream1_mL_min", "flow_rate_stream2_mL_min",
    "solvent", "solvent_list",
    "catalyst", "catalyst_metal", "catalyst_loading_pct",
    "ligand", "ligand_loading_pct", "additive",
    "pressure_bar", "reactor_type",
    "stoichiometry",
    "paper_doi", "paper_year",
    "source_table_or_figure",
    "data_correction_note",
    "notes",
]


# ---------------------------------------------------------------------------
# Main PostProcessor class
# ---------------------------------------------------------------------------

class PostProcessor:
    def __init__(self, output_dir="data/final_output"):
        self.output_dir = output_dir

    def run(self, pdf_path, intermediate_dir=None, smiles_lookup=False):
        """
        Normalise _final.json for one PDF and write _normalized.json + .xlsx.
        Returns path to _normalized.json or None if _final.json missing.
        """
        basename = os.path.splitext(os.path.basename(pdf_path))[0]
        if not intermediate_dir:
            intermediate_dir = os.path.join("data/intermediate", basename)

        final_path = os.path.join(self.output_dir, f"{basename}_final.json")
        if not os.path.exists(final_path):
            print(f"[PostProcessor] No _final.json found for {basename}, skipping.")
            return None

        with open(final_path) as f:
            try:
                records = json.load(f)
            except json.JSONDecodeError as e:
                print(f"[PostProcessor] Failed to parse {final_path}: {e}")
                return None

        if not isinstance(records, list):
            print(f"[PostProcessor] Expected list, got {type(records)}, skipping.")
            return None

        # Load paper text for DOI extraction (uses PDFParser if available)
        full_text = self._load_fulltext(pdf_path, intermediate_dir)
        paper_doi = extract_doi(full_text)
        paper_year = extract_year(full_text)

        print(f"[PostProcessor] {basename}: {len(records)} records, DOI={paper_doi}, year={paper_year}")

        normalised = []
        for rec in records:
            nr = dict(rec)

            # -- Solvent --
            conds = dict(nr.get("conditions") or {})
            sv = normalise_solvent(conds.get("solvent"))
            conds["solvent"] = sv["solvent"]
            if sv["solvent_list"]:
                conds["solvent_list"] = sv["solvent_list"]

            # -- Reactor type --
            conds["reactor_type"] = normalise_reactor(conds.get("reactor_type"))

            nr["conditions"] = conds

            # -- Source label --
            nr["source_table_or_figure"] = humanise_source(nr.get("source_table_or_figure", ""))

            # -- DOI + year (inject if not already set) --
            if not nr.get("paper_doi") and paper_doi:
                nr["paper_doi"] = paper_doi
            if not nr.get("paper_year") and paper_year:
                nr["paper_year"] = paper_year

            # -- yield_type (infer if null) --
            if not nr.get("yield_type"):
                nr["yield_type"] = infer_yield_type(nr)

            # -- SMILES lookup (optional, slow) --
            if smiles_lookup:
                if not nr.get("reactant1_smiles") and nr.get("reactant1_name"):
                    nr["reactant1_smiles"] = lookup_smiles(nr["reactant1_name"])
                if not nr.get("reactant2_smiles") and nr.get("reactant2_name"):
                    nr["reactant2_smiles"] = lookup_smiles(nr["reactant2_name"])
                if not nr.get("product_smiles") and nr.get("product_name"):
                    nr["product_smiles"] = lookup_smiles(nr["product_name"])

            # -- Notes rescue parsing --
            notes_raw = str(nr.get("notes") or "")
            remaining = notes_raw

            entry_num, remaining = _extract_entry_number(remaining)
            dr, remaining = _extract_diastereomeric_ratio(remaining)
            batch_yield, remaining = _extract_batch_yield(remaining)
            s1, s2, remaining = _extract_stream_flow_rates(remaining)
            stoich, remaining = _extract_stoichiometry(remaining)

            if entry_num and not nr.get("entry_number"):
                nr["entry_number"] = entry_num
            if dr and not nr.get("diastereomeric_ratio"):
                nr["diastereomeric_ratio"] = dr
            if batch_yield is not None and not nr.get("batch_yield_pct"):
                nr["batch_yield_pct"] = batch_yield
            if s1 is not None and not nr.get("flow_rate_stream1_mL_min"):
                nr["flow_rate_stream1_mL_min"] = s1
            if s2 is not None and not nr.get("flow_rate_stream2_mL_min"):
                nr["flow_rate_stream2_mL_min"] = s2
            if stoich and not nr.get("stoichiometry"):
                nr["stoichiometry"] = stoich

            # Clean notes: strip punctuation remnants, set null if empty
            remaining = remaining.strip().strip(';').strip(',').strip()
            nr["notes"] = remaining if remaining else None

            # -- Catalyst field splitting --
            cat_updates = _parse_catalyst_fields(nr)
            if cat_updates:
                conds = dict(nr.get("conditions") or {})
                conds.update(cat_updates)
                nr["conditions"] = conds

            # -- reaction_smiles (ORD-compatible: reactants>>reagents>>products) --
            if not nr.get("reaction_smiles"):
                nr["reaction_smiles"] = build_reaction_smiles(nr)

            normalised.append(nr)

        # Save _normalized.json
        norm_json = os.path.join(self.output_dir, f"{basename}_normalized.json")
        with open(norm_json, "w") as f:
            json.dump(normalised, f, indent=2, ensure_ascii=False)
        print(f"[PostProcessor] -> {norm_json}")

        # Save _normalized.xlsx
        try:
            self._save_excel(normalised, basename)
        except Exception as e:
            print(f"[PostProcessor] Excel export failed: {e}")

        return norm_json

    def _load_fulltext(self, pdf_path, intermediate_dir):
        """Try to load cached fulltext, otherwise extract via PDFParser."""
        # Check for cached _fulltext.txt
        cached = os.path.join(intermediate_dir, "_fulltext.txt")
        if os.path.exists(cached):
            with open(cached) as f:
                return f.read()
        # Fall back to PDFParser
        try:
            from src.adjudication.pdf_parser import PDFParser
            return PDFParser().extract_text(pdf_path)
        except Exception:
            return ""

    def _save_excel(self, records, basename):
        import pandas as pd
        out_path = os.path.join(self.output_dir, f"{basename}_normalized.xlsx")
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

        # Apply fixed column order: preferred first, then any extra columns alphabetically
        ordered = [c for c in PREFERRED_COLUMNS if c in df.columns]
        extra = sorted(c for c in df.columns if c not in PREFERRED_COLUMNS)
        df = df[ordered + extra]

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            # All Records sheet
            df.to_excel(writer, sheet_name="All Records", index=False)
            # Per-source sheets
            for src, grp in df.groupby(src_col, sort=False):
                sheet = str(src)[:31]
                grp.to_excel(writer, sheet_name=sheet, index=False)
        print(f"[PostProcessor] -> Excel: {out_path}")
