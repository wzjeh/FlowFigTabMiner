#!/usr/bin/env python3
"""
Build clean_organolithium_unified.csv from Gemini VLM-extracted SI table JSONs.

Schema: 38 columns (28 core + 10 DFT placeholders).
Key logic:
  - tR1 data (intermediate generation): electrophile is quench probe (MeOH, MeI, etc.),
    residence time varies → analysis_subset = kd_clean
  - tR2 data (trapping): separate tR2 column exists
  - scope data: entry/substrate/product columns vary, conditions fixed

Usage:
    python scripts/ml_lifetime/build_clean_dataset.py
"""

import json
import glob
import os
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GEMINI_DIR = ROOT / "data" / "ml_lifetime" / "gemini_table_extractions"
OUTPUT_CSV = ROOT / "data" / "ml_lifetime" / "clean_organolithium_unified.csv"

# ── Quench probes: these electrophiles are used to measure kd, not for synthesis ──
QUENCH_PROBES = {
    "methanol", "meoh", "iodomethane", "mei", "ethanol", "etoh",
    "2-propanol", "isopropanol", "iproh", "ipoh",
    "tert-butyl alcohol", "tbuoh",
    "d2o", "deuterated methanol", "methanol-d4", "ch3od",
}

# ── DOI mapping (folder name → DOI, from download_si.py, 100% verified) ──
DOI_MAP = {
    "Asai_2011_BenzoThiophenyl": "10.1246/cl.2011.393",
    "Asai_2012_Photochromic": "10.1002/cssc.201100376",
    "Degennaro_2016_tBuEsters": "10.1039/c6cc04588j",
    "Kim_2011_ProtectingGroupFree": "10.1038/ncomms1264",
    "Nagaki_2007_Dibromobenzenes": "10.1002/asia.200700231",
    "Nagaki_2008_AlkoxycarbonylArLi": "10.1002/anie.200803205",
    "Nagaki_2009_Aziridinyl": "10.1246/cl.2009.1060",
    "Nagaki_2009_Biaryls": "10.3762/bjoc.5.16",
    "Nagaki_2009_NitroArLi": "10.1002/anie.200904316",
    "Nagaki_2009_Silyloxiranyl": "10.1246/cl.2009.486",
    "Nagaki_2010_AlkoxycarbonylFlow": "10.1002/chem.201000876",
    "Nagaki_2010_CyanoArLi": "10.1039/b919325c",
    "Nagaki_2010_Oxiranyl": "10.1002/chem.201000815",
    "Nagaki_2011_Homocoupling": "10.3762/bjoc.7.122",
    "Nagaki_2011_Perfluoroalkyl": "10.1039/c1ob06350b",
    "Nagaki_2011_Pyridines": "10.1039/c0gc00852d",
    "Nagaki_2011_TAC101": "10.1039/c1ra00377a",
    "Nagaki_2012_Vinyllithiums": "10.1556/JFC-D-12-00004",
    "Nagaki_2014_ThreeComponent": "10.1021/ja5071762",
    "Nagaki_2015_Benzyllithiums": "10.1039/c5ob00958h",
    "Nagaki_2019_Angew_Alkyllithium": "10.1002/anie.201814088",
    "Nagaki_2019_FuncAlkyllithiums": "10.1002/chem.201902867",
    "Nagaki_2020_CHLiIF": "10.1002/anie.202003831",
    "Okamoto_2026_C1Carbenoid": "10.1002/asia.70525",
    "Sun_2020_PyridylKetone": "10.1007/s41981-020-00120-7",
    "Usutani_2007_oBrPhLi": "10.1021/ja074330h",
}

# ── Paper ID mapping (folder name → short paper_id) ──
PAPER_ID_MAP = {
    "Asai_2011_BenzoThiophenyl": "BenzoThiophenylLi_2011_Asai",
    "Asai_2012_Photochromic": "Photochromic_2012_Asai",
    "Degennaro_2016_tBuEsters": "tBuEsters_2016_Degennaro",
    "Kim_2011_ProtectingGroupFree": "ProtGroupFree_2011_Kim",
    "Miyagishi_2024_CGlycoside": "CGlycoside_2024_Miyagishi",
    "Musci_2020_Carbenoid": "Carbenoid_2020_Musci",
    "Nagaki_2007_Dibromobenzenes": "DibromoBenzArLi_2007_Nagaki",
    "Nagaki_2008_AlkoxycarbonylArLi": "AlkoxycarbonylArLi_2008_Nagaki",
    "Nagaki_2009_Aziridinyl": "AziridinylLi_2009_Nagaki",
    "Nagaki_2009_Biaryls": "BiaryLi_2009_Nagaki",
    "Nagaki_2009_NitroArLi": "NitroArLi_2009_Nagaki",
    "Nagaki_2009_Silyloxiranyl": "SilyloxiranylLi_2009_Nagaki",
    "Nagaki_2010_AlkoxycarbonylFlow": "AlkoxycarbonylFlow_2010_Nagaki",
    "Nagaki_2010_CyanoArLi": "CyanoArLi_2010_Nagaki",
    "Nagaki_2010_Oxiranyl": "OxiranylLi_2010_Nagaki",
    "Nagaki_2011_Homocoupling": "Homocoupling_2011_Nagaki",
    "Nagaki_2011_Perfluoroalkyl": "PerfluoroalkylLi_2011_Nagaki",
    "Nagaki_2011_Pyridines": "PyridineLi_2011_Nagaki",
    "Nagaki_2011_TAC101": "TAC101_2011_Nagaki",
    "Nagaki_2012_Vinyllithiums": "VinylLi_2012_Nagaki",
    "Nagaki_2014_ThreeComponent": "ThreeComp_2014_Nagaki",
    "Nagaki_2015_Benzyllithiums": "BenzylLi_2015_Nagaki",
    "Nagaki_2019_Angew_Alkyllithium": "AlkylLi_2019_Nagaki_Angew",
    "Nagaki_2019_FuncAlkyllithiums": "FuncAlkylLi_2019_Nagaki",
    "Nagaki_2020_CHLiIF": "CHLiIF_2020_Nagaki",
    "Okamoto_2026_C1Carbenoid": "C1Carbenoid_2026_Okamoto",
    "Sun_2020_PyridylKetone": "PyridylKetone_2020_Sun",
    "Usutani_2007_oBrPhLi": "oBrPhLi_2007_Usutani",
}

# ── Intermediate class inference ──
def infer_intermediate_class(intermediate: str, reaction_type: str) -> str:
    if not intermediate:
        return "unknown"
    il = intermediate.lower()
    # Check specific classes BEFORE ArLi (some names contain ArLi keywords like "phenyl")
    if "aziridin" in il:
        return "aziridinylLi"
    if "oxiran" in il or "epox" in il:
        return "oxiranylLi"
    if "vinyl" in il or "alkenyl" in il or "ethenyl" in il or "styryl" in il:
        return "vinylLi"
    if "benzyl" in il:
        return "benzylLi"
    if "perfluoro" in il:
        return "perfluoroalkylLi"
    if "carben" in il or "chlif" in il or "chli" in il:
        return "carbenoid"
    if any(kw in il for kw in ["aryl", "phenyl", "tolyl", "anisyl", "nitro", "cyano",
                                "bromo", "fluoro", "chloro", "benzo", "naphth",
                                "thiophen", "furan", "pyrid", "thiazo",
                                "acetophenon", "lithioaceto"]):
        return "ArLi"
    if "alkyl" in il:
        return "alkylLi"
    if "deprotonation" in (reaction_type or "").lower():
        return "carbanion"
    return "other"


def classify_column(col_name: str):
    """Classify a column by its role: tR1, tR2, T, T2, yield, conversion, reactor, scope_var, other."""
    cl = col_name.lower().strip()
    # Normalize unicode: all phi variants → "phi"
    for ch in "φΦϕ":
        cl = cl.replace(ch, "phi")

    # tR2
    if "tr2" in cl or ("residence" in cl and "r2" in cl) or ("residence time" in cl and "r2" in cl):
        return "tR2"
    # tR1 / general tR
    if "tr1" in cl or ("residence" in cl and ("r1" in cl or "r2" not in cl)):
        return "tR1"
    if cl in ("x (sec)", "x(sec)", "x", "residence time", "residence time (s)",
              "tr", "rt", "rt (sec)", "r^t", "residence time of r1"):
        return "tR1"
    if re.match(r'^x\s*\(?sec', cl):
        return "tR1"

    # T2 (second temperature, e.g. "t²", "t2")
    if cl in ("t²", "t2"):
        return "T2"

    # Temperature
    if cl in ("t", "temperature", "bath temperature", "t (°c)", "t(°c)", "temp",
              "t (ºc)", "t (deg c)", "bath temp"):
        return "T"

    # Yield (exclude "recover" — recovery ≠ yield)
    if "yield" in cl:
        return "yield"
    if "recover" in cl:
        return "recovery"

    # Conversion
    if cl.startswith("conv"):
        return "conversion"

    # Reactor geometry
    if ("inner diameter" in cl or "length of" in cl or
            cl.startswith("l1") or cl.startswith("l2") or
            cl.startswith("phi") or cl.startswith("l_") or
            cl.startswith("lx") or cl in ("l", "l_x", "l") or
            re.match(r'^phi\d', cl)):
        return "reactor"

    # Flow rate
    if "flow rate" in cl:
        return "reactor"

    # Scope-related variable columns
    if cl in ("entry", "substrate", "ester", "product", "electrophile", "e",
              "e1", "e2", "ester_label", "product_label", "product label",
              "dibromopyridine", "bromopyridine", "vinyl bromide",
              "epoxide", "carbonyl compound", "rli", "alkyne 5", "ester 6",
              "bromobenzonitrile",
              "conditions", "conditions[b]", "lithiating agent",
              "r1", "reagent", "label",
              # Homocoupling: Ar-X = substrate, Ar-Ar = product
              "ar-x", "ar-ar",
              # Degennaro: "ester 6 (structure)", "ester 6 (label)", "r-br 5 (structure)", "r-br 5 (label)"
              "polyfunctional electrophile", "halobenzene",
              # NitroArLi: column "1" is the substrate number
              "1"):
        return "scope_var"
    # Ester/R-Br structure/label variants
    if cl.startswith("ester 6") or cl.startswith("r-br 5") or cl.startswith("r-br"):
        return "scope_var"

    # Productivity
    if "productivity" in cl:
        return "other"

    # Ratio columns (cis/trans, byproduct ratios)
    if "ratio" in cl:
        return "other"

    return "other"


def is_quench_probe(electrophile: str) -> bool:
    if not electrophile:
        return False
    e = electrophile.lower().strip()
    # Exact match
    if e in QUENCH_PROBES:
        return True
    # Strip parenthetical suffixes: "iodomethane (MeI)" → "iodomethane"
    e_clean = re.sub(r'\s*\(.*\)\s*$', '', e).strip()
    if e_clean in QUENCH_PROBES:
        return True
    # Strip footnote markers: "MeIb" → "MeI", "MeOH[a]" → "MeOH"
    e_fn = re.sub(r'[a-e]$', '', e_clean).strip()
    if e_fn in QUENCH_PROBES:
        return True
    e_fn2 = re.sub(r'\[[a-z]\]$', '', e_clean).strip()
    if e_fn2 in QUENCH_PROBES:
        return True
    return False


def determine_analysis_subset(columns: list, reaction_context: dict, rows: list) -> str:
    """
    Classify the table:
    - kd_clean: tR1 varies, quench probe electrophile → decomposition kinetics
    - k2_trapping: tR2 varies
    - scope: entry/substrate/product varies, conditions fixed
    """
    col_roles = [classify_column(c) for c in columns]
    electrophile = (reaction_context.get("electrophile") or "").lower().strip()

    has_tr1 = "tR1" in col_roles
    has_tr2 = "tR2" in col_roles
    has_scope_var = "scope_var" in col_roles

    # Scope: has scope variable columns (entry/substrate/product/electrophile)
    if has_scope_var and not has_tr1 and not has_tr2:
        return "scope"

    # Has tR2 → trapping data
    if has_tr2:
        if has_tr1:
            return "kd_and_trapping"  # both tR1 and tR2
        return "k2_trapping"

    # Has tR1 with quench probe → kd
    if has_tr1 and is_quench_probe(electrophile):
        return "kd_clean"

    # Has tR1 with synthetic electrophile
    if has_tr1:
        return "kd_valid"  # tR1 data but non-probe electrophile

    return "scope"


def extract_rows_from_json(data: dict) -> list[dict]:
    """Convert one Gemini JSON into rows for the unified dataset."""
    if "_parse_error" in data:
        return []

    image_type = data.get("image_type", "")
    if image_type == "scheme":
        return []  # Schemes don't have tabular data rows

    columns = data.get("columns", [])
    rows = data.get("rows", [])
    if not columns or not rows:
        return []

    rc = data.get("reaction_context", {})
    paper_folder = data.get("_source_paper", "")
    table_id = data.get("table_id", "")

    # Build compound label → SMILES lookup from structures array (table_with_structures)
    structures_map = {}  # label → {"smiles": ..., "role": ...}
    for s in data.get("structures", []):
        label = str(s.get("label", "")).strip()
        smi = s.get("smiles", "")
        if label and smi:
            structures_map[label] = {"smiles": smi, "role": s.get("role", "")}
    paper_id = PAPER_ID_MAP.get(paper_folder, paper_folder)
    paper_doi = DOI_MAP.get(paper_folder, "")
    source_image = data.get("_source_image", "")

    # Classify columns
    col_roles = [classify_column(c) for c in columns]

    # Find key column indices
    tr1_idx = next((i for i, r in enumerate(col_roles) if r == "tR1"), None)
    tr2_idx = next((i for i, r in enumerate(col_roles) if r == "tR2"), None)
    t_idx = next((i for i, r in enumerate(col_roles) if r == "T"), None)
    t2_idx = next((i for i, r in enumerate(col_roles) if r == "T2"), None)
    yield_idx = next((i for i, r in enumerate(col_roles) if r == "yield"), None)
    conv_idx = next((i for i, r in enumerate(col_roles) if r == "conversion"), None)

    # Fallback: if no explicit "yield" column, use the first "other" column as yield
    # (common pattern: product name as column header, value is yield %)
    # Use FIRST, not last: in multi-product tables the first "other" column
    # is the main product yield, subsequent ones are byproducts.
    product_col_name = None
    if yield_idx is None:
        other_indices = [i for i, r in enumerate(col_roles) if r == "other"]
        if other_indices:
            yield_idx = other_indices[0]
            product_col_name = columns[yield_idx]

    # Determine analysis subset
    analysis_subset = determine_analysis_subset(columns, rc, rows)

    # Determine electrophile type
    electrophile = rc.get("electrophile", "") or ""
    electrophile_type = "quench_probe" if is_quench_probe(electrophile) else "synthetic"

    # Reaction class mapping
    rt = (rc.get("reaction_type") or "").lower()
    if "br-li" in rt or "br–li" in rt:
        reaction_class = "halogen-metal exchange"
    elif "i-li" in rt or "i–li" in rt:
        reaction_class = "halogen-metal exchange"
    elif "cl-li" in rt or "cl–li" in rt:
        reaction_class = "halogen-metal exchange"
    elif "deproton" in rt:
        reaction_class = "deprotonation"
    elif "carbolithiation" in rt:
        reaction_class = "carbolithiation"
    else:
        reaction_class = rt or "unknown"

    intermediate = rc.get("intermediate", "") or ""
    intermediate_class = infer_intermediate_class(intermediate, rc.get("reaction_type", ""))

    # Try to infer organolithium_reagent from table_title/notes if missing from reaction_context
    orgli_reagent = rc.get("organolithium_reagent", "") or ""
    if not orgli_reagent:
        title = (data.get("table_title") or "").lower()
        notes = (data.get("notes") or "").lower()
        search_text = title + " " + notes
        if "n-buli" in search_text or "nbuli" in search_text:
            orgli_reagent = "n-BuLi"
        elif "s-buli" in search_text or "sbuli" in search_text or "sec-buli" in search_text:
            orgli_reagent = "s-BuLi"
        elif "t-buli" in search_text or "tbuli" in search_text or "tert-buli" in search_text:
            orgli_reagent = "t-BuLi"
        elif "phli" in search_text:
            orgli_reagent = "PhLi"
        elif "meli" in search_text:
            orgli_reagent = "MeLi"
        elif "linp" in search_text or "lithium naphthalenide" in search_text:
            orgli_reagent = "LiNp"
        elif "br-li exchange" in search_text or "br–li exchange" in search_text:
            orgli_reagent = "n-BuLi"  # default for Br-Li exchange
        elif "i-li exchange" in search_text or "i–li exchange" in search_text:
            orgli_reagent = "n-BuLi"  # default for I-Li exchange

    # Product from column name if yield came from a product-named column
    product_from_col = product_col_name if product_col_name else ""

    # ── Identify scope_var column indices for per-row extraction ──
    # Map scope_var columns to semantic roles
    scope_electrophile_idx = None
    scope_substrate_idx = None
    scope_product_idx = None
    scope_product_label_idx = None
    scope_substrate_label_idx = None

    electrophile_names = {"e", "e1", "electrophile", "polyfunctional electrophile",
                          "carbonyl compound"}
    substrate_names = {"ester", "substrate", "bromopyridine", "dibromopyridine",
                       "vinyl bromide", "r-br", "halobenzene", "rli",
                       "bromobenzonitrile", "alkyne 5", "ar-x", "1",
                       "epoxide"}
    product_names = {"product", "ar-ar"}

    for i, c in enumerate(columns):
        cl = c.lower().strip()
        if cl in electrophile_names and scope_electrophile_idx is None:
            scope_electrophile_idx = i
        elif cl in substrate_names and scope_substrate_idx is None:
            scope_substrate_idx = i
        elif cl.startswith("r-br 5") and scope_substrate_idx is None:
            scope_substrate_idx = i
        elif cl in product_names and scope_product_idx is None:
            scope_product_idx = i
        elif cl in ("product_label", "product label", "label") and scope_product_label_idx is None:
            scope_product_label_idx = i
        elif (cl in ("ester_label",) or cl.startswith("r-br 5 (l") or
              cl.startswith("ester 6 (l")) and scope_substrate_label_idx is None:
            scope_substrate_label_idx = i
        elif cl.startswith("ester 6") and scope_product_idx is None:
            # "Ester 6", "Ester 6 (Structure)" → product in Degennaro
            scope_product_idx = i

    out_rows = []
    for row in rows:
        if len(row) != len(columns):
            # Pad or truncate
            row = row[:len(columns)] + [""] * max(0, len(columns) - len(row))

        def safe_float(idx):
            if idx is None:
                return None
            val = row[idx].strip() if idx < len(row) else ""
            if not val:
                return None
            # Remove footnote markers like [a], [b], ᵃ, *, †, etc. — but NOT digits
            val = re.sub(r'\[([a-zA-Z])\]$', '', val).strip()
            val = re.sub(r'[ᵃᵇᶜᵈᵉ*†‡§]+$', '', val).strip()
            try:
                return float(val)
            except ValueError:
                return None

        def safe_str(idx):
            if idx is None:
                return ""
            return row[idx].strip() if idx < len(row) else ""

        tr1_val = safe_float(tr1_idx)
        tr2_val = safe_float(tr2_idx)
        t_val = safe_float(t_idx)
        t2_val = safe_float(t2_idx)
        yield_val = safe_float(yield_idx)

        # For tables with both tR1 and tR2, determine tR_step
        if tr1_val is not None and tr2_val is not None:
            tr_step = "tR1+tR2"
        elif tr2_val is not None:
            tr_step = "tR2"
        else:
            tr_step = "tR1"

        # Per-row scope variable overrides
        row_electrophile = safe_str(scope_electrophile_idx) or electrophile
        row_substrate = safe_str(scope_substrate_idx) or (rc.get("substrate", "") or "")
        row_product = safe_str(scope_product_idx) or product_from_col or (rc.get("product", "") or "")

        # Use label columns as fallback names
        if scope_substrate_label_idx is not None:
            label = safe_str(scope_substrate_label_idx)
            if label:
                row_substrate = label if not row_substrate else row_substrate

        if scope_product_label_idx is not None:
            label = safe_str(scope_product_label_idx)
            if label:
                row_product = label if not row_product else row_product

        # Determine per-row electrophile_type
        row_electrophile_type = "quench_probe" if is_quench_probe(row_electrophile) else "synthetic"

        # Per-row SMILES from structures lookup (table_with_structures)
        row_substrate_smiles = rc.get("substrate_smiles", "") or ""
        row_electrophile_smiles = rc.get("electrophile_smiles", "") or ""
        row_product_smiles = rc.get("product_smiles", "") or ""
        row_intermediate_smiles = rc.get("intermediate_smiles", "") or ""

        # Look up compound labels in structures_map for SMILES
        if structures_map:
            if row_substrate in structures_map:
                info = structures_map[row_substrate]
                if info["role"] == "substrate" and not row_substrate_smiles:
                    row_substrate_smiles = info["smiles"]
            if row_product in structures_map:
                info = structures_map[row_product]
                if info["role"] == "product" and not row_product_smiles:
                    row_product_smiles = info["smiles"]

        # Check if per-row electrophile or product cell contains SMILES (has atoms like C, N, O and bonds)
        if scope_electrophile_idx is not None:
            val = safe_str(scope_electrophile_idx)
            if val and any(c in val for c in "()=[]") and any(c in val for c in "CNO"):
                row_electrophile_smiles = val
                row_electrophile = val  # SMILES as name if no better name
        if scope_product_idx is not None:
            val = safe_str(scope_product_idx)
            if val and any(c in val for c in "()=[]") and any(c in val for c in "CNO"):
                row_product_smiles = val

        rec = {
            "paper_id": paper_id,
            "paper_doi": paper_doi,
            "intermediate": intermediate,
            "intermediate_smiles": row_intermediate_smiles,
            "substrate1": row_substrate,
            "substrate1_smiles": row_substrate_smiles,
            "organolithium_reagent": orgli_reagent,
            "electrophile": row_electrophile,
            "electrophile_smiles": row_electrophile_smiles,
            "electrophile_type": row_electrophile_type,
            "product": row_product,
            "product_smiles": row_product_smiles,
            "reaction_class": reaction_class,
            "tR_step": tr_step,
            "tR1_s": tr1_val,
            "T1_C": t_val,
            "tR2_s": tr2_val,
            "T2_C": t2_val,
            "yield_pct": yield_val,
            "solvent": rc.get("solvent", "") or "",
            "conc_substrate_M": None,  # Rarely in SI tables
            "analysis_subset": analysis_subset,
            "intermediate_class": intermediate_class,
            "sigma_hammett": None,
            "Es_taft": None,
            "delta_ortho": None,
            "delta_benzyne": None,
            "quality_flag": "clean",
            "data_source_type": f"si_table_{table_id}",
            # DFT placeholders
            "dft_charge_Li": None,
            "dft_charge_C_ipso": None,
            "dft_HOMO_eV": None,
            "dft_LUMO_eV": None,
            "dft_LiC_bond_A": None,
            "dft_method": None,
            "dft_LiC_BDE_kJ": None,
            "dft_wiberg_LiC": None,
            "dft_dipole_D": None,
            "dft_Gsolv_kJ": None,
        }
        out_rows.append(rec)

    return out_rows


def canonicalize_smiles(smi: str) -> str:
    """Canonicalize SMILES using RDKit. Returns empty string if invalid."""
    if not smi or pd.isna(smi):
        return ""
    smi = str(smi).strip()
    if not smi:
        return ""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol)
    except Exception:
        return ""


def collect_paper_solvents(json_files: list) -> dict:
    """Collect solvent info per paper folder from all JSONs (for gap-filling)."""
    from collections import defaultdict
    paper_solvents = defaultdict(list)
    for f in json_files:
        try:
            data = json.load(open(f))
        except Exception:
            continue
        paper = data.get("_source_paper", "")
        rc = data.get("reaction_context", {})
        s = (rc.get("solvent") or "").strip()
        if s and s.lower() != "null":
            paper_solvents[paper].append(s)

    # For each paper, pick the most common non-empty solvent
    result = {}
    for paper, solvents in paper_solvents.items():
        from collections import Counter
        counts = Counter(solvents)
        result[paper] = counts.most_common(1)[0][0]
    return result


def main():
    all_rows = []
    json_files = sorted(glob.glob(str(GEMINI_DIR / "*" / "*.json")))
    json_files = [f for f in json_files if "_all_tables" not in f and "_test_result" not in f]

    print(f"Processing {len(json_files)} JSON files...")

    # Pre-collect paper-level solvents for gap-filling
    paper_solvents = collect_paper_solvents(json_files)

    stats = {"total": 0, "skipped_scheme": 0, "skipped_empty": 0, "skipped_error": 0}
    subset_counts = {}

    for f in json_files:
        try:
            data = json.load(open(f))
        except:
            stats["skipped_error"] += 1
            continue

        if "_parse_error" in data:
            stats["skipped_error"] += 1
            continue

        rows = extract_rows_from_json(data)
        if not rows:
            if data.get("image_type") == "scheme":
                stats["skipped_scheme"] += 1
            else:
                stats["skipped_empty"] += 1
            continue

        stats["total"] += len(rows)
        subset = rows[0]["analysis_subset"]
        subset_counts[subset] = subset_counts.get(subset, 0) + len(rows)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    # ── Fill missing solvents from paper-level info ──
    paper_folder_map = {}
    for folder, pid in PAPER_ID_MAP.items():
        paper_folder_map[pid] = folder
    solvent_before = (df["solvent"].notna() & (df["solvent"] != "")).sum()
    for idx, row in df.iterrows():
        if not row["solvent"] or (isinstance(row["solvent"], str) and row["solvent"].lower() in ("", "null")):
            folder = paper_folder_map.get(row["paper_id"], row["paper_id"])
            if folder in paper_solvents:
                df.at[idx, "solvent"] = paper_solvents[folder]
    # Clean "null" string
    df["solvent"] = df["solvent"].replace("null", "")
    solvent_after = (df["solvent"].notna() & (df["solvent"] != "")).sum()
    print(f"Solvent fill: {solvent_before} → {solvent_after} / {len(df)}")

    # ── Chemistry corrections ──
    print("Applying chemistry corrections...")
    n_fixes = 0

    # Fix 1: oBrPhLi — reaction_class = protonation → Br-Li exchange;
    #   fill missing substrate, reagent, solvent
    mask = df["paper_id"] == "oBrPhLi_2007_Usutani"
    df.loc[mask, "reaction_class"] = "halogen-metal exchange"
    df.loc[mask, "substrate1"] = "1,2-dibromobenzene"
    df.loc[mask, "substrate1_smiles"] = "Brc1ccccc1Br"
    df.loc[mask, "organolithium_reagent"] = "n-BuLi"
    df.loc[mask, "solvent"] = "THF/Et2O"
    n_fixes += mask.sum()
    print(f"  oBrPhLi: fixed reaction_class, substrate, reagent, solvent ({mask.sum()} rows)")

    # Fix 1b: Normalize organolithium_reagent names
    REAGENT_NORM = {
        "sBuLi": "s-BuLi",
        "nBuLi": "n-BuLi",
        "tBuLi": "t-BuLi",
        "n-BuLi (PhLi for footnote a)": "n-BuLi",
    }
    reagent_norm_count = 0
    for old_r, new_r in REAGENT_NORM.items():
        mask = df["organolithium_reagent"] == old_r
        if mask.sum() > 0:
            df.loc[mask, "organolithium_reagent"] = new_r
            reagent_norm_count += mask.sum()
    if reagent_norm_count > 0:
        n_fixes += reagent_norm_count
        print(f"  Normalized reagent names ({reagent_norm_count} rows)")

    # Fix 1c: AlkoxycarbonylArLi_2008 — paper uses s-BuLi for ALL Br-Li exchanges
    #   (ortho-bromobenzoates require s-BuLi, not n-BuLi)
    mask = ((df["paper_id"] == "AlkoxycarbonylArLi_2008_Nagaki") &
            (df["organolithium_reagent"] == "n-BuLi"))
    df.loc[mask, "organolithium_reagent"] = "s-BuLi"
    if mask.sum() > 0:
        n_fixes += mask.sum()
        print(f"  AlkoxycarbonylArLi: n-BuLi → s-BuLi ({mask.sum()} rows)")

    # Fix 1d: AlkoxycarbonylFlow_2010 — S-6 to S-9 are o-bromobenzoates → s-BuLi, not n-BuLi
    #   S-3 to S-5 are p/m-bromobenzoates → n-BuLi is correct
    mask_o = ((df["paper_id"] == "AlkoxycarbonylFlow_2010_Nagaki") &
              (df["data_source_type"].str.contains(r"S-[6789]\b", na=False, regex=True)))
    df.loc[mask_o, "organolithium_reagent"] = "s-BuLi"
    if mask_o.sum() > 0:
        n_fixes += mask_o.sum()
        print(f"  AlkoxycarbonylFlow S-6~9: n-BuLi → s-BuLi (o-benzoates) ({mask_o.sum()} rows)")

    # Fix 1e: Fill missing solvents from chemistry knowledge
    #   All Nagaki group papers use THF for flow organolithium
    THF_PAPERS = [
        "AlkylLi_2019_Nagaki_Angew", "BenzylLi_2015_Nagaki",
        "BiaryLi_2009_Nagaki", "FuncAlkylLi_2019_Nagaki",
        "Homocoupling_2011_Nagaki", "PerfluoroalkylLi_2011_Nagaki",
        "PyridylKetone_2020_Sun", "ThreeComp_2014_Nagaki",
    ]
    solvent_fill_count = 0
    for pid in THF_PAPERS:
        mask_s = ((df["paper_id"] == pid) & (df["solvent"].fillna("") == ""))
        if mask_s.sum() > 0:
            df.loc[mask_s, "solvent"] = "THF"
            solvent_fill_count += mask_s.sum()
    if solvent_fill_count > 0:
        n_fixes += solvent_fill_count
        print(f"  Filled solvent=THF for {len(THF_PAPERS)} papers ({solvent_fill_count} rows)")

    # Fix 2: Normalize reaction_class variants
    rc_map = {
        "halogen-lithium exchange": "halogen-metal exchange",
        "halogen-li exchange followed by homocoupling": "halogen-metal exchange",
        "protonation": "halogen-metal exchange",  # already handled above for oBrPhLi
    }
    for old_rc, new_rc in rc_map.items():
        mask = df["reaction_class"] == old_rc
        if mask.sum() > 0:
            df.loc[mask, "reaction_class"] = new_rc
            n_fixes += mask.sum()
            print(f"  reaction_class '{old_rc}' → '{new_rc}' ({mask.sum()} rows)")

    # Fix 3: CyanoArLi Table 3 — substrate = intermediate (Gemini confusion)
    #   The table shows reactions of o-lithiobenzonitrile with carbonyl compounds.
    #   substrate should be the ArBr precursor, not the ArLi intermediate.
    mask = ((df["paper_id"] == "CyanoArLi_2010_Nagaki") &
            (df["data_source_type"] == "si_table_Table 3"))
    df.loc[mask, "substrate1"] = "o-bromobenzonitrile (1a)"
    df.loc[mask, "substrate1_smiles"] = "N#Cc1ccccc1Br"
    df.loc[mask, "organolithium_reagent"] = "n-BuLi"
    n_fixes += mask.sum()
    print(f"  CyanoArLi Table 3: fixed substrate (was = intermediate) ({mask.sum()} rows)")

    # Fix 4: OxiranylLi Table S-10 — intermediate SMILES missing [Li]
    #   CC1(O1)c1ccccc1 is substrate, not lithiated intermediate.
    #   Correct intermediate: [Li]C1(C)OC1c1ccccc1 (lithium at oxirane C-2)
    mask = ((df["paper_id"] == "OxiranylLi_2010_Nagaki") &
            (df["intermediate_smiles"] == "CC1(O1)c1ccccc1"))
    df.loc[mask, "intermediate_smiles"] = "[Li]C1(C)OC1c1ccccc1"
    n_fixes += mask.sum()
    print(f"  OxiranylLi Table S-10: fixed intermediate SMILES (+[Li]) ({mask.sum()} rows)")

    # Fix 5: intermediate_smiles containing "null" string
    mask = df["intermediate_smiles"] == "null"
    df.loc[mask, "intermediate_smiles"] = ""
    if mask.sum() > 0:
        n_fixes += mask.sum()
        print(f"  Cleared 'null' string from intermediate_smiles ({mask.sum()} rows)")

    # Fix 6: Clear "null" string from SMILES and text columns
    null_str_cols = ["substrate1_smiles", "electrophile_smiles", "product_smiles",
                     "intermediate_smiles", "organolithium_reagent", "solvent",
                     "substrate1", "electrophile", "product", "intermediate"]
    for col in null_str_cols:
        mask = df[col] == "null"
        if mask.sum() > 0:
            df.loc[mask, col] = ""
            n_fixes += mask.sum()
            print(f"  Cleared 'null' string from {col} ({mask.sum()} rows)")

    # Fix 7: Recalculate electrophile_type after QUENCH_PROBES update
    df["electrophile_type"] = df["electrophile"].apply(
        lambda e: "quench_probe" if is_quench_probe(str(e)) else "synthetic"
    )
    print(f"  Recalculated electrophile_type with updated QUENCH_PROBES")

    # Fix 8: Normalize electrophile names (same compound, multiple representations)
    ELECTROPHILE_NORM = {
        # Alcohols / quench probes
        "Methanol": "methanol", "MeOH": "methanol",
        "MeI": "iodomethane", "iodomethane (MeI)": "iodomethane",
        "MeIb": "iodomethane", "MeIc": "iodomethane", "MeId": "iodomethane",
        "EtOH": "ethanol",
        "iPrOH": "2-propanol", "iproh": "2-propanol",
        "tBuOH": "tert-butyl alcohol",
        # Aldehydes & ketones
        "PhCHO": "benzaldehyde", "O=Cc1ccccc1": "benzaldehyde",
        "O=Cc1ccc(C(F)(F)F)cc1": "4-(trifluoromethyl)benzaldehyde",
        "CH3COCH3": "acetone",
        "O=C1CCCC1": "cyclopentanone",
        "O=C1CCCCC1": "cyclohexanone",
        "CC(=O)c1ccccc1": "acetophenone",
        "Ph2CO": "benzophenone",
        "O=C(C1=CC=CC=C1)C2=CC=CC=C2": "benzophenone",
        "O=C(C1=CC=C(F)C=C1)C2=CC=C(F)C=C2": "4,4'-difluorobenzophenone",
        "O=C(C(C)C)C1=CC=CC=C1": "isobutyrophenone",
        # Silylating agents
        "Me3SiCl": "chlorotrimethylsilane",
        "C[Si](C)(C)Cl": "chlorotrimethylsilane",
        "Me3SiOTf": "trimethylsilyl triflate",
        "MeOTf": "methyl triflate",
        # Stannylating agents
        "Bu3SnCl": "chlorotributylstannane",
        "chlorotributylstannate": "chlorotributylstannane",  # typo fix
        "CCCC[Sn](CCCC)(CCCC)Cl": "chlorotributylstannane",
        "Bu2PhSnCl": "chlorodibutylphenylstannane",
        # Acyl chlorides & isocyanates
        "PhCOCl": "benzoyl chloride",
        "PhNCO": "phenyl isocyanate",
        "PhCNO": "phenyl isocyanate",  # typo in Gemini extraction
        # Halogens
        "I2": "iodine",
        # Carbonates
        "(Boc)2O": "di-tert-butyl dicarbonate",
        "Di-tert-butyl dicarbonate ((Boc)2O)": "di-tert-butyl dicarbonate",
        # Allyl/methallyl halides
        "C=CCBr": "allyl bromide",
        "CC(=C)CBr": "methallyl bromide",
        # CHLiIF Weinreb amides (SMILES → name)
        "O=C(N(OC)C)CCC1=CC=CC=C1": "N-methoxy-N-methyl-3-phenylpropanamide",
        "O=C(N(OC)C)/C=C/C1=CC=CC=C1": "N-methoxy-N-methylcinnanamide",
        "O=C(N(OC)C)CC1=CC=CC=C1": "N-methoxy-N-methyl-2-phenylacetamide",
        "O=C(N(OC)C)CC1=CC=CS1": "N-methoxy-N-methyl-2-(thiophen-2-yl)acetamide",
        "O=C(N(OC)C)C1=CC=C(C2=CC=CC=C2)C=C1": "N-methoxy-N-methyl-[1,1'-biphenyl]-4-carboxamide",
        "O=C(N(OC)C)C(CC)C1=CC=CC=C1": "N-methoxy-N-methyl-2-phenylbutanamide",
        "O=C(N(OC)C)C1=CC(F)=CC=C1": "3-fluoro-N-methoxy-N-methylbenzamide",
        "O=C(N(OC)C)C1=CC=C(C(F)(F)F)C=C1": "4-(trifluoromethyl)-N-methoxy-N-methylbenzamide",
        "O=C(N(OC)C)C1=CC=C(F)C=C1": "4-fluoro-N-methoxy-N-methylbenzamide",
        "COc1ccc(C(=O)N(C)OC)cc1": "4-methoxy-N-methoxy-N-methylbenzamide",
        "N#Cc1ccc(C(=O)N(C)OC)cc1": "4-cyano-N-methoxy-N-methylbenzamide",
        "N#Cc1cccc(C(=O)N(C)OC)c1": "3-cyano-N-methoxy-N-methylbenzamide",
        "Brc1ccc(C(=O)N(C)OC)cc1": "4-bromo-N-methoxy-N-methylbenzamide",
        "Brc1cccc(C(=O)N(C)OC)c1": "3-bromo-N-methoxy-N-methylbenzamide",
        "CN(OC)C(=O)c1ccc(C(=O)c2ccccc2)cc1": "4-benzoyl-N-methoxy-N-methylbenzamide",
        # CHLiIF other scope electrophiles
        "C=Cc1ccc(C(=O)c2ccccc2)cc1": "4-vinylbenzophenone",
        "Brc1ccc(C(=O)c2ccccc2)cc1": "4-bromobenzophenone",
        "N#Cc1ccc(C(=O)c2ccccc2)cc1": "4-cyanobenzophenone",
        # OxiranylLi Table 2: epoxide substrates misassigned as electrophiles
        # (will be moved to substrate1 in Fix 16, but normalize name too)
        "Clc1ccc(C2CO2)cc1": "4-chlorostyrene oxide",
        "Cc1ccc(C2CO2)cc1": "4-methylstyrene oxide",
        "c1ccc2cc(C3CO3)ccc2c1": "2-(naphthalen-2-yl)oxirane",
        "c1ccc(c2ccc(C3CO3)cc2)cc1": "2-([1,1'-biphenyl]-4-yl)oxirane",
        # CHLiIF special
        "N-methoxy-N-methyl-3-phenylpropanamide (7a)": "N-methoxy-N-methyl-3-phenylpropanamide",
        # OxiranylLi scope: additional electrophile abbreviations
        "Me2SiHCl": "chlorodimethylsilane",
        "BnBr": "benzyl bromide",
        "PhCOCH3": "acetophenone",
        "(CO2Et)2": "diethyl oxalate",
        # Footnote-as-electrophile
        "\u2014 \u1d47": "",  # "— ᵇ" → empty
    }
    norm_count = 0
    for old, new in ELECTROPHILE_NORM.items():
        mask = df["electrophile"] == old
        if mask.sum() > 0:
            df.loc[mask, "electrophile"] = new
            norm_count += mask.sum()
    print(f"  Normalized electrophile names: {norm_count} rows")

    # Fix 9: Fill missing organolithium_reagent for specific papers
    # AlkoxycarbonylFlow uses n-BuLi (Br-Li exchange, S-2 to S-9) or PhLi (I-Li, S-11+ and scope)
    af_mask = (df["paper_id"] == "AlkoxycarbonylFlow_2010_Nagaki")
    af_empty = af_mask & (df["organolithium_reagent"].fillna("") == "")

    # S-numbered tables: determine reagent by table number and substrate position
    af_s_tables = af_empty & df["data_source_type"].str.contains(r"S-\d+", na=False)
    for idx in df.loc[af_s_tables].index:
        table = df.at[idx, "data_source_type"]
        table_num = re.search(r'S-(\d+)', table)
        n = int(table_num.group(1))
        if n >= 11:
            df.loc[[idx], "organolithium_reagent"] = "PhLi"  # I-Li exchange
        elif n >= 6:
            df.loc[[idx], "organolithium_reagent"] = "s-BuLi"  # o-bromobenzoates
        else:
            df.loc[[idx], "organolithium_reagent"] = "n-BuLi"  # p/m-bromobenzoates

    # All non-S tables (Table 3, 6, 8 etc.) — use .loc for reliable assignment
    af_other = af_empty & ~df["data_source_type"].str.contains(r"S-\d+", na=False)
    # Table 8 is Br-Li exchange → s-BuLi (per Gemini JSON), Table 3/6 → PhLi (I-Li exchange)
    for idx in df.loc[af_other].index:
        table = df.at[idx, "data_source_type"]
        if "Table 8" in table:
            df.loc[[idx], "organolithium_reagent"] = "s-BuLi"
        else:
            df.loc[[idx], "organolithium_reagent"] = "PhLi"

    n_fixes += af_empty.sum()
    print(f"  AlkoxycarbonylFlow: filled reagent ({af_empty.sum()} rows)")

    # Verify no NaN remain
    af_still_nan = af_mask & df["organolithium_reagent"].isna()
    if af_still_nan.sum() > 0:
        print(f"  WARNING: {af_still_nan.sum()} AlkoxycarbonylFlow rows still missing reagent!")

    # OxiranylLi uses s-BuLi for deprotonation
    mask = ((df["paper_id"] == "OxiranylLi_2010_Nagaki") &
            (df["organolithium_reagent"].fillna("") == ""))
    df.loc[mask, "organolithium_reagent"] = "s-BuLi"
    n_fixes += mask.sum()
    print(f"  OxiranylLi: filled reagent=s-BuLi ({mask.sum()} rows)")

    # AlkylLi_2019 & FuncAlkylLi_2019: scope tables use LiNp (lithium naphthalenide)
    for pid in ["AlkylLi_2019_Nagaki_Angew", "FuncAlkylLi_2019_Nagaki"]:
        mask = ((df["paper_id"] == pid) &
                (df["organolithium_reagent"].fillna("") == ""))
        df.loc[mask, "organolithium_reagent"] = "LiNp"
        if mask.sum() > 0:
            n_fixes += mask.sum()
            print(f"  {pid}: filled reagent=LiNp ({mask.sum()} rows)")

    # VinylLi_2012 Table S-2: "RLi" column was classified as substrate → fix
    #   Move substrate1 values to organolithium_reagent, set correct substrate
    mask_vl_s2 = ((df["paper_id"] == "VinylLi_2012_Nagaki") &
                  (df["data_source_type"].str.contains("S-2")) &
                  (df["substrate1"].isin(["s-BuLi", "n-BuLi", "PhLi", "MeLi"])))
    if mask_vl_s2.sum() > 0:
        df.loc[mask_vl_s2, "organolithium_reagent"] = df.loc[mask_vl_s2, "substrate1"]
        df.loc[mask_vl_s2, "substrate1"] = "(E)-β-bromostyrene"
        n_fixes += mask_vl_s2.sum()
        print(f"  VinylLi S-2: moved RLi from substrate1 → reagent ({mask_vl_s2.sum()} rows)")

    # VinylLi Table 3 (scope): uses n-BuLi for Br-Li exchange
    mask = ((df["paper_id"] == "VinylLi_2012_Nagaki") &
            (df["organolithium_reagent"].fillna("") == ""))
    df.loc[mask, "organolithium_reagent"] = "n-BuLi"
    if mask.sum() > 0:
        n_fixes += mask.sum()
        print(f"  VinylLi: filled reagent=n-BuLi ({mask.sum()} rows)")

    # VinylLi intermediate derivation moved to Fix 18 (after Fix 16 populates substrate1_smiles)

    # PyridylKetone_2020_Sun: uses n-BuLi for Br-Li exchange of 2-bromopyridine
    mask = (df["paper_id"] == "PyridylKetone_2020_Sun")
    df.loc[mask & (df["organolithium_reagent"].fillna("") == ""),
           "organolithium_reagent"] = "n-BuLi"
    # Also fix reaction_class (was "unknown")
    df.loc[mask & (df["reaction_class"] == "unknown"), "reaction_class"] = "halogen-metal exchange"
    n_fixes += mask.sum()
    print(f"  PyridylKetone: filled reagent=n-BuLi, reaction_class ({(mask).sum()} rows)")

    # Fix 10: BenzoThiophenylLi kd_valid → should be kd_clean (MeI is quench probe)
    mask = ((df["paper_id"] == "BenzoThiophenylLi_2011_Asai") &
            (df["analysis_subset"] == "kd_valid") &
            (df["electrophile_type"] == "quench_probe"))
    df.loc[mask, "analysis_subset"] = "kd_clean"
    if mask.sum() > 0:
        n_fixes += mask.sum()
        print(f"  BenzoThiophenylLi: kd_valid → kd_clean for quench probe rows ({mask.sum()} rows)")

    # Fix 11: OxiranylLi Table 2 — "Epoxide" column was misassigned to electrophile
    #   Move epoxide SMILES from electrophile to substrate1, fix electrophile from same row
    #   Affected: scope rows where electrophile looks like an epoxide SMILES
    epoxide_smiles_set = {
        "4-chlorostyrene oxide", "4-methylstyrene oxide",
        "2-(naphthalen-2-yl)oxirane", "2-([1,1'-biphenyl]-4-yl)oxirane",
    }
    mask_ox_scope = ((df["paper_id"] == "OxiranylLi_2010_Nagaki") &
                     (df["analysis_subset"] == "scope") &
                     (df["electrophile"].isin(epoxide_smiles_set)))
    if mask_ox_scope.sum() > 0:
        # These rows have the epoxide name as electrophile after normalization.
        # Move to substrate1 and clear electrophile (the real electrophile is in the JSON row data,
        # but we lost that mapping — mark as "various" for now)
        df.loc[mask_ox_scope, "substrate1"] = df.loc[mask_ox_scope, "electrophile"]
        df.loc[mask_ox_scope, "substrate1_smiles"] = df.loc[mask_ox_scope, "electrophile_smiles"]
        df.loc[mask_ox_scope, "electrophile"] = "various"
        df.loc[mask_ox_scope, "electrophile_smiles"] = ""
        n_fixes += mask_ox_scope.sum()
        print(f"  OxiranylLi Table 2: moved epoxide from electrophile to substrate1 ({mask_ox_scope.sum()} rows)")

    # Fix 12: Fix invalid SMILES (reused ring numbers → RDKit can't parse)
    INTERMEDIATE_SMILES_FIXES = {
        # Ring 1 reused: C1(O1) → separate rings
        "[Li]C1(O1)c2ccccc2": "[Li]C1OC1c1ccccc1",
        "CC1(O1)C([Li])c1ccccc1": "[Li]C1(C)OC1c1ccccc1",
        # Incorrect O[Li] — lithium is on carbon, not oxygen
        "CC1=CC=C(C2(O[Li])C2)C=C1": "[Li]C1(c2ccc(C)cc2)CO1",
        # SilyloxiranylLi: ring 1 reused + remove incorrect [C-] charge
        "O1C[C-]1([Li])[Si](c2ccccc2)(c3ccccc3)c4ccccc4": "[Li]C1([Si](c2ccccc2)(c3ccccc3)c4ccccc4)CO1",
        # SilyloxiranylLi: also fix already-mapped [C-] version from prior builds
        "[Li][C-]1([Si](c2ccccc2)(c3ccccc3)c4ccccc4)CO1": "[Li]C1([Si](c2ccccc2)(c3ccccc3)c4ccccc4)CO1",
        # Variable R group (can't be parsed) → clear
        "O=C(OR)c1ccccc1[Li]": "",
    }
    SUBSTRATE_SMILES_FIXES = {
        # Ring 1 reused in styrene oxide: C1(O1) → correct oxirane ring (gem-disubstituted)
        "CC1(O1)c2ccccc2": "CC1(c2ccccc2)CO1",
        # Variable R group (AlkoxycarbonylArLi generic substrate) → clear
        "O=C(OR)c1ccccc1Br": "",
    }
    PRODUCT_SMILES_FIXES = {
        # "Ph" abbreviation not valid in SMILES → expand to c1ccccc1
        "PhCOCCCC(O)Ph": "O=C(c1ccccc1)CCCC(O)c1ccccc1",
        "PhCOCCCC(=O)Ph": "O=C(c1ccccc1)CCCC(=O)c1ccccc1",
        "PhCOCC[Si](C)(C)Ph": "O=C(c1ccccc1)CC[Si](C)(C)c1ccccc1",
        "PhCOCCC(O)Ph": "O=C(c1ccccc1)CCC(O)c1ccccc1",
        "COCC[Si](C)(C)Ph": "COCC[Si](C)(C)c1ccccc1",
        "C1CCCCC1OCCC(O)Ph": "OC(CCC1CCCCC1O)c1ccccc1",  # cyclohexyl ether
        "ClCCCCC(O)Ph": "OC(CCCCCl)c1ccccc1",
        "C=CCCCC(O)Ph": "OC(CCCC=C)c1ccccc1",
        # Ring numbering error in oxiranylLi product (gem-disubstituted: Me+Ph on same C)
        "CC1(O1)c2ccccc2": "CC1(c2ccccc2)CO1",
    }
    ELECTROPHILE_SMILES_FIXES = {
        "(CO2Et)2": "CCOC(=O)C(=O)OCC",  # diethyl oxalate
    }
    fix12_count = 0
    for old_smi, new_smi in INTERMEDIATE_SMILES_FIXES.items():
        mask = df["intermediate_smiles"] == old_smi
        if mask.sum() > 0:
            df.loc[mask, "intermediate_smiles"] = new_smi
            fix12_count += mask.sum()
    for old_smi, new_smi in SUBSTRATE_SMILES_FIXES.items():
        mask = df["substrate1_smiles"] == old_smi
        if mask.sum() > 0:
            df.loc[mask, "substrate1_smiles"] = new_smi
            fix12_count += mask.sum()
    for old_smi, new_smi in PRODUCT_SMILES_FIXES.items():
        mask = df["product_smiles"] == old_smi
        if mask.sum() > 0:
            df.loc[mask, "product_smiles"] = new_smi
            fix12_count += mask.sum()
    for old_smi, new_smi in ELECTROPHILE_SMILES_FIXES.items():
        mask = df["electrophile_smiles"] == old_smi
        if mask.sum() > 0:
            df.loc[mask, "electrophile_smiles"] = new_smi
            fix12_count += mask.sum()
    if fix12_count > 0:
        n_fixes += fix12_count
        print(f"  Fixed invalid SMILES (all columns) ({fix12_count} rows)")

    # Fix 13: ProtGroupFree and PyridylKetone — fill intermediate_class from paper context
    # ProtGroupFree_2011_Kim: halogen-metal exchange of aryl halides → ArLi
    mask = ((df["paper_id"] == "ProtGroupFree_2011_Kim") &
            (df["intermediate_class"] == "unknown"))
    df.loc[mask, "intermediate_class"] = "ArLi"
    if mask.sum() > 0:
        n_fixes += mask.sum()
        print(f"  ProtGroupFree: intermediate_class → ArLi ({mask.sum()} rows)")
    # PyridylKetone_2020_Sun: Br-Li exchange of 2-bromopyridine → 2-pyridyllithium (ArLi)
    mask = ((df["paper_id"] == "PyridylKetone_2020_Sun") &
            (df["intermediate_class"] == "unknown"))
    df.loc[mask, "intermediate_class"] = "ArLi"
    df.loc[mask, "intermediate"] = "2-pyridyllithium"
    df.loc[mask, "intermediate_smiles"] = "[Li]c1ccccn1"
    if mask.sum() > 0:
        n_fixes += mask.sum()
        print(f"  PyridylKetone: intermediate_class → ArLi, filled intermediate ({mask.sum()} rows)")

    # Fix 13b: PyridylKetone substrate fill
    #   Substrate is unambiguously 2-bromopyridine (the paper title and old dataset confirm)
    mask = ((df["paper_id"] == "PyridylKetone_2020_Sun") &
            (df["substrate1"].fillna("") == ""))
    df.loc[mask, "substrate1"] = "2-bromopyridine"
    df.loc[mask, "substrate1_smiles"] = "Brc1ccccn1"
    if mask.sum() > 0:
        n_fixes += mask.sum()
        print(f"  PyridylKetone: filled substrate=2-bromopyridine ({mask.sum()} rows)")

    # Fix 13c: ProtGroupFree Table S1: compound "1" = o-iodovalerophenone
    #   S1 has no structures array but references same compound as Table 1
    mask_pgf_s1 = ((df["paper_id"] == "ProtGroupFree_2011_Kim") &
                   (df["data_source_type"].str.contains("S1")) &
                   (df["substrate1"].fillna("") == "1"))
    if mask_pgf_s1.sum() > 0:
        df.loc[mask_pgf_s1, "substrate1_smiles"] = "O=C(CCCC)c1ccccc1I"
        df.loc[mask_pgf_s1, "product_smiles"] = "O=C(CCCC)c1ccccc1"  # compound 3
        n_fixes += mask_pgf_s1.sum() * 2
        print(f"  ProtGroupFree S1: filled sub/prod SMILES for compound 1 ({mask_pgf_s1.sum()} rows)")

    # Fix 13d: ProtGroupFree reagent = MesLi (I-Li exchange uses mesityllithium)
    mask_pgf = (df["paper_id"] == "ProtGroupFree_2011_Kim")
    df.loc[mask_pgf & (df["organolithium_reagent"].fillna("") == ""),
           "organolithium_reagent"] = "MesLi"

    # Fix 13e: ProtGroupFree intermediate_smiles (ArI → ArLi: I→[Li])
    from rdkit import Chem as _Chem13
    mask_pgf_inter = (mask_pgf &
                      (df["intermediate_smiles"].fillna("") == "") &
                      (df["substrate1_smiles"].fillna("") != ""))
    fix13e = 0
    for idx in df[mask_pgf_inter].index:
        sub_smi = df.at[idx, "substrate1_smiles"]
        if "I" in sub_smi:
            inter = sub_smi.replace("I", "[Li]", 1)
            mol = _Chem13.MolFromSmiles(inter)
            if mol:
                df.at[idx, "intermediate_smiles"] = _Chem13.MolToSmiles(mol)
                fix13e += 1
    if fix13e:
        n_fixes += fix13e
        print(f"  ProtGroupFree: derived intermediate_smiles (I→[Li]) ({fix13e} rows)")

    # Fix 14: CHLiIF lithiating agent column → organolithium_reagent
    #   The JSON has "Lithiating agent" column with values like "MeLi" per row.
    #   These are already extracted as organolithium_reagent for most rows,
    #   but for CHLiIF, the carbenoid intermediate is generated from CHF₂I + MeLi.
    #   So MeLi is the organolithium_reagent (correct).
    #   The intermediate is CHLiF (fluoromethyllithium) = carbenoid class.
    mask = (df["paper_id"] == "CHLiIF_2020_Nagaki")
    df.loc[mask & (df["organolithium_reagent"].fillna("") == ""),
           "organolithium_reagent"] = "MeLi"
    reagent_filled = (mask & (df["organolithium_reagent"].fillna("") == "")).sum()
    if reagent_filled > 0:
        n_fixes += reagent_filled
        print(f"  CHLiIF: filled reagent=MeLi ({reagent_filled} rows)")

    # Fix 15: ThreeComp — the "— ᵇ" electrophile row is intramolecular cyclization
    mask_dash = (df["electrophile"] == "")
    # After normalization, "— ᵇ" becomes "". Mark these as intramolecular.
    mask_3comp_empty = ((df["paper_id"] == "ThreeComp_2014_Nagaki") &
                        (df["electrophile"].fillna("") == "") &
                        (df["yield_pct"].notna()))
    if mask_3comp_empty.sum() > 0:
        df.loc[mask_3comp_empty, "electrophile"] = "intramolecular cyclization"
        df.loc[mask_3comp_empty, "electrophile_type"] = "synthetic"
        n_fixes += mask_3comp_empty.sum()
        print(f"  ThreeComp: '— ᵇ' → intramolecular cyclization ({mask_3comp_empty.sum()} rows)")

    # Fix 16: Detect SMILES stuck in text columns (table_with_structures)
    #   When Gemini returns drawn structures as SMILES in row data, they end up
    #   in substrate1/product (name columns) instead of _smiles columns.
    from rdkit import Chem as _Chem
    fix16_sub = 0
    fix16_prod = 0
    for idx, row in df.iterrows():
        # substrate1 → substrate1_smiles
        s = row.get("substrate1", "")
        if (isinstance(s, str) and len(s) > 3 and
                (pd.isna(row.get("substrate1_smiles")) or row.get("substrate1_smiles", "") == "")):
            mol = _Chem.MolFromSmiles(s)
            if mol is not None:
                df.at[idx, "substrate1_smiles"] = s
                fix16_sub += 1
        # product → product_smiles
        p = row.get("product", "")
        if (isinstance(p, str) and len(p) > 3 and
                (pd.isna(row.get("product_smiles")) or row.get("product_smiles", "") == "")):
            mol = _Chem.MolFromSmiles(p)
            if mol is not None:
                df.at[idx, "product_smiles"] = p
                fix16_prod += 1
    if fix16_sub or fix16_prod:
        n_fixes += fix16_sub + fix16_prod
        print(f"  Fix 16: SMILES in text columns → substrate1_smiles ({fix16_sub}), product_smiles ({fix16_prod})")

    # Fix 17: tBuEsters intermediate_smiles derivation
    #   Table 1 (Br-Li exchange): ArBr + HexLi → ArLi (replace first Br with [Li])
    #   Table 2 (deprotonation): R-C≡CH + HexLi → R-C≡CLi (replace terminal C#C with C#C[Li])
    import re as _re
    fix17_count = 0
    mask_tbu = ((df["paper_id"] == "tBuEsters_2016_Degennaro") &
                (df["intermediate_smiles"].fillna("") == "") &
                (df["substrate1_smiles"].fillna("") != ""))
    for idx in df[mask_tbu].index:
        sub_smi = df.at[idx, "substrate1_smiles"]
        rxn_cls = df.at[idx, "reaction_class"]
        if rxn_cls == "halogen-metal exchange":
            # ArBr → ArLi: replace first Br with [Li]
            inter = sub_smi.replace("Br", "[Li]", 1)
        elif rxn_cls == "deprotonation":
            # R-C≡CH → R-C≡CLi: append [Li] after C#C
            inter = _re.sub(r"C#C(?!\(|\[)", "C#C[Li]", sub_smi, count=1)
        else:
            continue
        mol = _Chem.MolFromSmiles(inter)
        if mol is not None:
            df.at[idx, "intermediate_smiles"] = _Chem.MolToSmiles(mol)
            fix17_count += 1
    if fix17_count:
        n_fixes += fix17_count
        print(f"  Fix 17: tBuEsters intermediate_smiles derived ({fix17_count} rows)")

    # Fix 18: Derive intermediate_smiles for halogen-metal exchange papers
    #   For Br-Li exchange: replace Br with [Li]
    #   For I-Li exchange: replace I with [Li]
    #   Only for papers where substrate has halogen and intermediate_smiles is empty
    FIX18_PAPERS = [
        "VinylLi_2012_Nagaki",
        "Homocoupling_2011_Nagaki",
        "ThreeComp_2014_Nagaki",
        "FuncAlkylLi_2019_Nagaki",
        "AlkylLi_2019_Nagaki_Angew",
        "AlkoxycarbonylArLi_2008_Nagaki",
        "CyanoArLi_2010_Nagaki",
        "NitroArLi_2009_Nagaki",
        "OxiranylLi_2010_Nagaki",
    ]
    fix18_count = 0
    mask_fix18 = (df["paper_id"].isin(FIX18_PAPERS) &
                  (df["intermediate_smiles"].fillna("") == "") &
                  (df["substrate1_smiles"].fillna("") != "") &
                  (df["reaction_class"] == "halogen-metal exchange"))
    for idx in df[mask_fix18].index:
        sub_smi = df.at[idx, "substrate1_smiles"]
        # Skip multi-halogen substrates (ambiguous which halogen is replaced)
        br_count = sub_smi.count("Br")
        i_count = sub_smi.count("I")
        cl_count = sub_smi.count("Cl")
        total_halogens = br_count + i_count + cl_count
        if total_halogens != 1:
            continue  # skip ambiguous cases
        if br_count == 1:
            inter = sub_smi.replace("Br", "[Li]", 1)
        elif i_count == 1:
            inter = sub_smi.replace("I", "[Li]", 1)
        elif cl_count == 1:
            inter = sub_smi.replace("Cl", "[Li]", 1)
        else:
            continue
        mol = _Chem.MolFromSmiles(inter)
        if mol is not None:
            df.at[idx, "intermediate_smiles"] = _Chem.MolToSmiles(mol)
            fix18_count += 1
    if fix18_count:
        n_fixes += fix18_count
        print(f"  Fix 18: halogen→[Li] intermediate derivation ({fix18_count} rows)")

    # Fix 19: OxiranylLi intermediate_smiles from known substrate→intermediate mapping
    #   These are deprotonation reactions (not halogen exchange), so use existing data
    ox_mask = (df["paper_id"] == "OxiranylLi_2010_Nagaki")
    ox_has = (ox_mask &
              df["intermediate_smiles"].notna() & (df["intermediate_smiles"] != "") &
              df["substrate1_smiles"].notna() & (df["substrate1_smiles"] != ""))
    # Build canonical substrate → intermediate map
    ox_map = {}
    for idx in df[ox_has].index:
        sub = df.at[idx, "substrate1_smiles"]
        inter = df.at[idx, "intermediate_smiles"]
        sub_mol = _Chem.MolFromSmiles(sub)
        if sub_mol:
            sub_can = _Chem.MolToSmiles(sub_mol)
            if sub_can not in ox_map:
                ox_map[sub_can] = inter
    # Fill missing from map
    ox_missing = (ox_mask &
                  (df["intermediate_smiles"].fillna("") == "") &
                  df["substrate1_smiles"].notna() & (df["substrate1_smiles"] != ""))
    fix19 = 0
    for idx in df[ox_missing].index:
        sub = df.at[idx, "substrate1_smiles"]
        sub_mol = _Chem.MolFromSmiles(sub)
        if sub_mol:
            sub_can = _Chem.MolToSmiles(sub_mol)
            if sub_can in ox_map:
                df.at[idx, "intermediate_smiles"] = ox_map[sub_can]
                fix19 += 1
    if fix19:
        n_fixes += fix19
        print(f"  Fix 19: OxiranylLi intermediate from known mapping ({fix19} rows)")

    # Fix 20: SilyloxiranylLi — fill Table S-2 empty intermediate_smiles
    #   Table S-1 has the SMILES (now corrected by Fix 12); Table S-2 rows are empty.
    #   All rows have the same intermediate: 1-triphenylsilyloxiranyllithium
    SILYLOX_CORRECT_SMI = "[Li]C1([Si](c2ccccc2)(c3ccccc3)c4ccccc4)CO1"
    mask_silylox = ((df["paper_id"] == "SilyloxiranylLi_2009_Nagaki") &
                    (df["intermediate_smiles"].fillna("") == ""))
    df.loc[mask_silylox, "intermediate_smiles"] = SILYLOX_CORRECT_SMI
    # Also update the name for Table S-2 rows from generic to specific
    df.loc[mask_silylox, "intermediate"] = "1-triphenylsilyloxiranyllithium"
    if mask_silylox.sum() > 0:
        n_fixes += mask_silylox.sum()
        print(f"  Fix 20: SilyloxiranylLi S-2 intermediate fill ({mask_silylox.sum()} rows)")

    # Fix 21: OxiranylLi — rename intermediate "2" → "α-phenyloxiranyllithium"
    #   Compound "2" is just a paper label for [Li]C1(c2ccccc2)CO1
    mask_ox2 = ((df["paper_id"] == "OxiranylLi_2010_Nagaki") &
                (df["intermediate"] == "2"))
    df.loc[mask_ox2, "intermediate"] = "α-phenyloxiranyllithium"
    if mask_ox2.sum() > 0:
        n_fixes += mask_ox2.sum()
        print(f"  Fix 21: OxiranylLi intermediate '2' → 'α-phenyloxiranyllithium' ({mask_ox2.sum()} rows)")

    # Fix 22: ProtGroupFree S1 — fill empty intermediate name
    #   These rows have inter_smiles=[Li]c1ccccc1C(=O)CCCC but no name
    mask_pgf_name = ((df["paper_id"] == "ProtGroupFree_2011_Kim") &
                     (df["intermediate"].fillna("") == ""))
    df.loc[mask_pgf_name, "intermediate"] = "acyl-substituted aryllithium"
    if mask_pgf_name.sum() > 0:
        n_fixes += mask_pgf_name.sum()
        print(f"  Fix 22: ProtGroupFree S1 intermediate name fill ({mask_pgf_name.sum()} rows)")

    # Fix 23: PyridineLi — derive intermediate_smiles for Table 2 and Table 3
    #   Table 2 (Br-Li exchange of dibromopyridines): ONE Br replaced with [Li]
    #   Table 3 (double Br-Li exchange): BOTH Br replaced with [Li]
    #   For Table 3, first forward-fill empty substrate_smiles from previous rows
    pyri_mask = (df["paper_id"] == "PyridineLi_2011_Nagaki")
    pyri_t3 = pyri_mask & df["data_source_type"].str.contains("Table 3")
    # Forward-fill substrate1_smiles within Table 3 (merged cells in original)
    last_sub = ""
    for idx in df[pyri_t3].sort_index().index:
        sub = str(df.at[idx, "substrate1_smiles"]) if pd.notna(df.at[idx, "substrate1_smiles"]) else ""
        if sub and sub != "nan":
            last_sub = sub
        elif last_sub:
            df.at[idx, "substrate1_smiles"] = last_sub
            # Also fill substrate1 name
            last_sub_name = ""
            for idx2 in df[pyri_t3].sort_index().index:
                sn = str(df.at[idx2, "substrate1"]) if pd.notna(df.at[idx2, "substrate1"]) else ""
                if sn and sn != "nan" and df.at[idx2, "substrate1_smiles"] == last_sub:
                    last_sub_name = sn
                    break
            if last_sub_name:
                df.at[idx, "substrate1"] = last_sub_name

    # Table 2: single Br-Li exchange — derive which Br was replaced using product comparison
    #   From the Gemini JSON: for each substrate, the FIRST Br in the SMILES string is exchanged
    PYRI_T2_MAP = {
        "Brc1cccnc1Br": "[Li]c1cccnc1Br",      # 2,3-dibromo → 3-lithio-2-bromopyridine
        "Brc1ccc(Br)nc1": "[Li]c1ccc(Br)nc1",   # 2,5-dibromo → 5-lithio-2-bromopyridine
        "Brc1cccc(Br)n1": "[Li]c1cccc(Br)n1",   # 2,6-dibromo → 6-lithio-2-bromopyridine
    }
    pyri_t2 = (pyri_mask &
               df["data_source_type"].str.contains("Table 2") &
               (df["intermediate_smiles"].fillna("") == ""))
    fix23_t2 = 0
    for idx in df[pyri_t2].index:
        sub = df.at[idx, "substrate1_smiles"]
        if sub in PYRI_T2_MAP:
            inter = PYRI_T2_MAP[sub]
            mol = _Chem.MolFromSmiles(inter)
            if mol:
                df.at[idx, "intermediate_smiles"] = _Chem.MolToSmiles(mol)
                fix23_t2 += 1

    # Table 3: double Br-Li exchange — replace ALL Br with [Li]
    pyri_t3_empty = (pyri_t3 &
                     (df["intermediate_smiles"].fillna("") == "") &
                     (df["substrate1_smiles"].fillna("") != ""))
    fix23_t3 = 0
    for idx in df[pyri_t3_empty].index:
        sub = df.at[idx, "substrate1_smiles"]
        if isinstance(sub, str) and sub:
            inter = sub.replace("Br", "[Li]")
            mol = _Chem.MolFromSmiles(inter)
            if mol:
                df.at[idx, "intermediate_smiles"] = _Chem.MolToSmiles(mol)
                fix23_t3 += 1
    if fix23_t2 or fix23_t3:
        n_fixes += fix23_t2 + fix23_t3
        print(f"  Fix 23: PyridineLi intermediate_smiles — Table 2 ({fix23_t2}), Table 3 ({fix23_t3})")

    # Fix 24: CHLiIF Table 1 — use flow yield (col 5) instead of batch yield (col 4)
    #   "Yield [%] Internal in batch" was picked by default (first yield column),
    #   but we need "Yield [%] External in flow" for this flow chemistry dataset.
    chlif_t1_mask = ((df["paper_id"] == "CHLiIF_2020_Nagaki") &
                     (df["data_source_type"] == "si_table_Table 1"))
    if chlif_t1_mask.sum() > 0:
        import json as _json
        _chlif_json = GEMINI_DIR / "Nagaki_2020_CHLiIF" / "_all_tables.json"
        if _chlif_json.exists():
            _chlif_tables = _json.load(open(_chlif_json))
            for _t in _chlif_tables:
                if _t.get("table_id") == "Table 1":
                    _flow_idx = next((i for i, c in enumerate(_t["columns"])
                                      if "flow" in c.lower() and "yield" in c.lower()), None)
                    if _flow_idx is not None:
                        fix24 = 0
                        _t1_rows = _t["rows"]
                        _df_t1_indices = df[chlif_t1_mask].index.tolist()
                        for i, idx in enumerate(_df_t1_indices):
                            if i < len(_t1_rows):
                                val = _t1_rows[i][_flow_idx].strip()
                                # Parse "70 (82)[c]" → 70.0
                                val = re.sub(r'\s*\(.*\)', '', val)  # remove parens
                                val = re.sub(r'\[.*\]', '', val)     # remove brackets
                                val = re.sub(r'[a-e]$', '', val).strip()
                                try:
                                    df.at[idx, "yield_pct"] = float(val)
                                    fix24 += 1
                                except ValueError:
                                    pass
                        if fix24:
                            n_fixes += fix24
                            print(f"  Fix 24: CHLiIF Table 1 → flow yield ({fix24} rows)")

    # Fix 25: Fill missing product_smiles across all papers
    # Derived from paper text + Gemini compound resolution + chemical logic
    fix25 = 0

    # --- OxiranylLi_2010_Nagaki ---
    oxir_product_map = {
        "3":       "CC1(c2ccccc2)CO1",                # 2-methyl-2-phenyloxirane
        "c-9/t-9": "CC1OC1(C)c1ccccc1",               # 2,3-dimethyl-2-phenyloxirane
        "c-11/t-11": "CC1(c2ccccc2)OC1c1ccccc1",      # 2-methyl-2,3-diphenyloxirane
        "c-11 / t-11": "CC1(c2ccccc2)OC1c1ccccc1",    # same (spacing variant)
        "c-16 / t-16": "CC1(c2ccccc2)OC1(C)c1ccccc1", # 2,3-dimethyl-2,3-diphenyloxirane
    }
    oxir_mask = (df["paper_id"] == "OxiranylLi_2010_Nagaki")
    for pname, smi in oxir_product_map.items():
        m = oxir_mask & (df["product"] == pname) & (df["product_smiles"].isna() | (df["product_smiles"] == ""))
        if m.sum() > 0:
            df.loc[m, "product_smiles"] = smi
            fix25 += m.sum()

    # c-16/t-16 with decomposition products in name
    m_c16_long = (oxir_mask &
                  df["product"].fillna("").str.startswith("c-16/t-16,") &
                  (df["product_smiles"].isna() | (df["product_smiles"] == "")))
    if m_c16_long.sum() > 0:
        df.loc[m_c16_long, "product_smiles"] = "CC1(c2ccccc2)OC1(C)c1ccccc1"
        fix25 += m_c16_long.sum()

    # OxiranylLi NaN product rows (kinetic SI tables — product is "" not NaN)
    oxir_nan = oxir_mask & (df["product"].isna() | (df["product"] == "")) & (df["product_smiles"].isna() | (df["product_smiles"] == ""))
    # c-8 + MeI → c-9
    m_c8 = oxir_nan & df["substrate1"].fillna("").str.contains("c-8", regex=False)
    if m_c8.sum() > 0:
        df.loc[m_c8, "product"] = "c-9"
        df.loc[m_c8, "product_smiles"] = "CC1OC1(C)c1ccccc1"
        fix25 += m_c8.sum()
    # biphenylyl oxide + MeI → methylated biphenylyl epoxide
    m_biph = oxir_nan & df["substrate1"].fillna("").str.contains("biphenyl", case=False, regex=False)
    if m_biph.sum() > 0:
        df.loc[m_biph, "product"] = "methylated biphenylyl epoxide"
        df.loc[m_biph, "product_smiles"] = "CC1(c2ccc(-c3ccccc3)cc2)CO1"
        fix25 += m_biph.sum()
    # 4-methylstyrene oxide (6) + MeI → compound 7
    m_me = oxir_nan & df["substrate1"].fillna("").str.contains("methylstyrene", case=False, regex=False)
    if m_me.sum() > 0:
        df.loc[m_me, "product"] = "7"
        df.loc[m_me, "product_smiles"] = "CC1(c2ccc(C)cc2)CO1"
        fix25 += m_me.sum()

    # --- AlkoxycarbonylFlow_2010_Nagaki ---
    alk_mask = (df["paper_id"] == "AlkoxycarbonylFlow_2010_Nagaki")
    alk_product_map = {
        "3a": "CC(C)(C)OC(=O)c1ccccc1",  # tert-butyl benzoate
        "3b": "CC(C)OC(=O)c1ccccc1",      # isopropyl benzoate
        "3c": "CCOC(=O)c1ccccc1",          # ethyl benzoate
        "3d": "COC(=O)c1ccccc1",           # methyl benzoate
    }
    for pname, smi in alk_product_map.items():
        m = alk_mask & (df["product"] == pname) & (df["product_smiles"].isna() | (df["product_smiles"] == ""))
        if m.sum() > 0:
            df.loc[m, "product_smiles"] = smi
            fix25 += m.sum()
    # Empty product rows: all are isopropyl esters quenched with iPrOH → 3b
    m_alk_nan = (alk_mask & (df["product"].isna() | (df["product"] == "")) &
                 (df["product_smiles"].isna() | (df["product_smiles"] == "")))
    if m_alk_nan.sum() > 0:
        df.loc[m_alk_nan, "product"] = "3b"
        df.loc[m_alk_nan, "product_smiles"] = "CC(C)OC(=O)c1ccccc1"
        fix25 += m_alk_nan.sum()

    # --- BiaryLi_2009_Nagaki ---
    # Product "2" = protonated monolithiated intermediate = 2-bromobiphenyl
    m_bi = ((df["paper_id"] == "BiaryLi_2009_Nagaki") &
            (df["product"] == "2") &
            (df["product_smiles"].isna() | (df["product_smiles"] == "")))
    if m_bi.sum() > 0:
        df.loc[m_bi, "product_smiles"] = "Brc1ccccc1-c1ccccc1"
        fix25 += m_bi.sum()

    # --- CHLiIF_2020_Nagaki ---
    # Empty product rows: CHLiIF + Weinreb amide → fluoromethyl ketone
    m_ch = ((df["paper_id"] == "CHLiIF_2020_Nagaki") &
            (df["product"].isna() | (df["product"] == "")) &
            (df["product_smiles"].isna() | (df["product_smiles"] == "")))
    if m_ch.sum() > 0:
        df.loc[m_ch, "product"] = "fluoromethyl ketone"
        df.loc[m_ch, "product_smiles"] = "O=C(CF)CCc1ccccc1"
        fix25 += m_ch.sum()

    # --- SilyloxiranylLi_2009_Nagaki ---
    # Product "2, 3": use product 2 SMILES (desired product)
    m_sil = ((df["paper_id"] == "SilyloxiranylLi_2009_Nagaki") &
             (df["product"].fillna("").str.contains("2, 3")) &
             (df["product_smiles"].isna() | (df["product_smiles"] == "")))
    if m_sil.sum() > 0:
        df.loc[m_sil, "product_smiles"] = "C[Si](C)(C)C1(OC1)[Si](c2ccccc2)(c3ccccc3)c4ccccc4"
        fix25 += m_sil.sum()

    # --- FuncAlkylLi_2019_Nagaki ---
    # Product "PhCOCCCC" = valerophenone
    m_func = ((df["paper_id"] == "FuncAlkylLi_2019_Nagaki") &
              (df["product"] == "PhCOCCCC") &
              (df["product_smiles"].isna() | (df["product_smiles"] == "")))
    if m_func.sum() > 0:
        df.loc[m_func, "product_smiles"] = "O=C(CCCC)c1ccccc1"
        fix25 += m_func.sum()

    if fix25:
        n_fixes += fix25
        print(f"  Fix 25: product_smiles resolution ({fix25} rows)")

    print(f"  Total corrections: {n_fixes}")

    # ── Canonical SMILES columns ──
    print("Canonicalizing SMILES (RDKit)...")
    smiles_cols = ["intermediate_smiles", "substrate1_smiles", "electrophile_smiles", "product_smiles"]
    for col in smiles_cols:
        canon_col = col.replace("_smiles", "_smiles_canonical")
        df[canon_col] = df[col].apply(canonicalize_smiles)
        valid = (df[canon_col] != "").sum()
        orig = (df[col].notna() & (df[col] != "")).sum()
        print(f"  {col}: {orig} original → {valid} canonical")

    # Column order
    col_order = [
        # Identifiers & metadata
        "paper_id", "paper_doi", "quality_flag", "data_source_type",
        # Reaction context (original + canonical SMILES)
        "intermediate", "intermediate_smiles", "intermediate_smiles_canonical",
        "substrate1", "substrate1_smiles", "substrate1_smiles_canonical",
        "organolithium_reagent",
        "electrophile", "electrophile_smiles", "electrophile_smiles_canonical",
        "electrophile_type",
        "product", "product_smiles", "product_smiles_canonical",
        "reaction_class",
        # Conditions
        "tR_step", "tR1_s", "T1_C", "tR2_s", "T2_C",
        "solvent", "conc_substrate_M",
        # Classification
        "analysis_subset", "intermediate_class",
        # Empirical descriptors
        "sigma_hammett", "Es_taft", "delta_ortho", "delta_benzyne",
        # DFT descriptors
        "dft_charge_Li", "dft_charge_C_ipso", "dft_HOMO_eV", "dft_LUMO_eV",
        "dft_LiC_bond_A", "dft_method", "dft_LiC_BDE_kJ", "dft_wiberg_LiC",
        "dft_dipole_D", "dft_Gsolv_kJ",
        # Outcome (last)
        "yield_pct",
    ]
    df = df[col_order]

    # Sort by paper_id, then tR1_s
    df = df.sort_values(["paper_id", "analysis_subset", "T1_C", "tR1_s"],
                        na_position="last").reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n=== Build Complete ===")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Skipped: {stats['skipped_scheme']} schemes, {stats['skipped_empty']} empty, {stats['skipped_error']} errors")
    print(f"\nAnalysis subset breakdown:")
    for subset, count in sorted(subset_counts.items()):
        print(f"  {subset}: {count}")
    print(f"\nPaper breakdown:")
    for pid, group in df.groupby("paper_id"):
        print(f"  {pid}: {len(group)} rows, subsets={group['analysis_subset'].unique().tolist()}")
    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
