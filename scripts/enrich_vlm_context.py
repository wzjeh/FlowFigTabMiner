"""
Enrich the VLM tR sub-dataset with context information.

Four enrichment levels:
  Level 1: Paper-level v10 backfill (for vlm_only table rows)
  Level 2: table_context SMILES → per-row reactant/product
  Level 3: VLM reaction_description parsing → chemical names
  Level 4: Cross-figure context propagation within same paper

Input:
  data/final_output/organolithium_tr_subdataset_vlm.csv

Output:
  data/final_output/organolithium_tr_subdataset_vlm_enriched.csv

Usage:
  cd /Users/zhaowenyuan/Projects/FlowFigTabMiner
  python scripts/enrich_vlm_context.py
"""

import os
import csv
import json
import re
from collections import defaultdict

import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_CSV = os.path.join(PROJECT_ROOT, "data/final_output/organolithium_tr_subdataset_vlm.csv")
V10_HEATMAP = os.path.join(PROJECT_ROOT, "data/final_output/dataset_comparison/organolithium_tr_heatmap_points_v10.csv")
VLM_HEATMAP_DIR = os.path.join(PROJECT_ROOT, "data/vlm_input_tr/vlm_output/heatmaps")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data/final_output/organolithium_tr_subdataset_vlm_enriched.csv")

# ── Stats tracking ──
STATS = defaultdict(int)


def load_v10_paper_context():
    """Load v10 context grouped by paper_dir (paper-level, not figure-level)."""
    paper_ctx = {}
    if not os.path.exists(V10_HEATMAP):
        return paper_ctx

    with open(V10_HEATMAP) as f:
        for row in csv.DictReader(f):
            pdir = row.get("paper_dir", "")
            if "zz_" in pdir:
                continue
            if pdir not in paper_ctx:
                paper_ctx[pdir] = {
                    "reaction_class": row.get("reaction_class", ""),
                    "organolithium_role_v5": row.get("organolithium_role_v5", ""),
                    "reagent_family_v2": row.get("reagent_family_v2", ""),
                    "solvent": row.get("solvent", ""),
                    "reactor_type": row.get("reactor_type", ""),
                    "paper_doi": row.get("paper_doi", ""),
                    "paper_year": row.get("paper_year", ""),
                }
    return paper_ctx


def load_v10_figure_context():
    """Load v10 context grouped by review_id (figure-level, has reactant/product)."""
    fig_ctx = {}
    if not os.path.exists(V10_HEATMAP):
        return fig_ctx

    with open(V10_HEATMAP) as f:
        for row in csv.DictReader(f):
            pdir = row.get("paper_dir", "")
            if "zz_" in pdir:
                continue
            rid = row.get("review_id_v10", "")
            if rid and rid not in fig_ctx:
                fig_ctx[rid] = {
                    "reactant1_name": row.get("reactant1_name", ""),
                    "reactant2_name": row.get("reactant2_name", ""),
                    "product_name": row.get("product_name", ""),
                }
    return fig_ctx


# ── Manual context for papers not in v10 ──

MANUAL_PAPER_CONTEXT = {
    # Degennaro 2016: directed ortho-lithiation of heterocycles
    "Degennaro": {
        "reaction_class": "C-H functionalization",
        "organolithium_role": "deprotonating lithium base",
        "reagent_family": "n-BuLi",
        "solvent": "THF",
        "reactor_type": "tube reactor",
        "reactant1_name": "Boc-protected pyrrolidine",
        "reactant2_name": "n-BuLi",
        "product_name": "α-lithiated Boc-pyrrolidine + Boc2O trapped product (6a)",
        "paper_doi": "10.1002/chem.201504482",
        "paper_year": "2016",
    },
    # ── New papers (2024-04-03 batch) ──
    "Nagaki et al. 2019 - Alkyllithium compounds bearing": {
        "reaction_class": "halogen-metal exchange",
        "organolithium_role": "C-C bond-forming organolithium",
        "reagent_family": "s-BuLi",
        "solvent": "THF",
        "reactor_type": "flow microreactor",
        "paper_year": "2019",
    },
    "Musci et al. 2020": {
        "reaction_class": "directed metalation",
        "organolithium_role": "deprotonating lithium base",
        "reagent_family": "LDA",
        "solvent": "THF",
        "reactor_type": "flow microreactor",
        "paper_doi": "10.1021/acs.orglett.0c01085",
        "paper_year": "2020",
    },
    "Nagaki et al. 2015 - Benzyllithiums": {
        "reaction_class": "reductive lithiation",
        "organolithium_role": "C-C bond-forming organolithium",
        "reagent_family": "LiNp",
        "solvent": "THF",
        "reactor_type": "flow microreactor",
        "paper_doi": "10.1039/c5ob00958h",
        "paper_year": "2015",
    },
    "Nagaki et al. 2016 - Integration of borylation": {
        "reaction_class": "C-C coupling",
        "organolithium_role": "C-C bond-forming organolithium",
        "reagent_family": "n-BuLi",
        "solvent": "THF",
        "reactor_type": "monolithic flow reactor",
        "paper_doi": "10.1039/c5cy02098k",
        "paper_year": "2016",
    },
    "Okamoto et al. 2026": {
        "reaction_class": "reductive lithiation",
        "organolithium_role": "C-C bond-forming organolithium",
        "reagent_family": "n-BuLi",
        "solvent": "THF",
        "reactor_type": "flow microreactor",
        "paper_year": "2026",
    },
    "Miyagishi et al. 2024": {
        "reaction_class": "halogen-metal exchange",
        "organolithium_role": "C-C bond-forming organolithium",
        "reagent_family": "n-BuLi",
        "solvent": "THF",
        "reactor_type": "flow microreactor",
        "paper_year": "2024",
    },
    "Nagaki et al. 2019 - Generation and reaction of functional alkyl": {
        "reaction_class": "halogen-metal exchange",
        "organolithium_role": "C-C bond-forming organolithium",
        "reagent_family": "s-BuLi",
        "solvent": "THF",
        "reactor_type": "flow microreactor",
        "paper_year": "2019",
    },
    # Sun 2020: rapid construction of organolithium intermediates
    "Sun2020": {
        "reaction_class": "C-C coupling",
        "organolithium_role": "C-C bond-forming organolithium",
        "reagent_family": "n-BuLi",
        "solvent": "THF",
        "reactor_type": "tube reactor",
        "reactant1_name": "aryl halide",
        "reactant2_name": "n-BuLi",
        "product_name": "aryl-alkyl coupled product",
        "paper_doi": "10.1039/D0RE00078G",
        "paper_year": "2020",
    },
}


def _is_empty(val):
    """Check if a field value is effectively empty."""
    return not val or val in ("", "missing_or_ambiguous", "Unclassified")


def _set_if_empty(row, field, value, source_tag):
    """Set field only if currently empty. Returns True if set."""
    if _is_empty(row.get(field)) and value and not _is_empty(value):
        row[field] = value
        STATS[f"{source_tag}:{field}"] += 1
        return True
    return False


# ═══════════════════════════════════════════════════════════════
#  Level 1: Paper-level v10 backfill for vlm_only rows
# ═══════════════════════════════════════════════════════════════

def enrich_level1_paper_backfill(rows, v10_paper_ctx, v10_fig_ctx):
    """For vlm_only rows, transfer paper-level context from v10 or manual."""
    for r in rows:
        if r["context_source"] != "vlm_only":
            continue

        paper = r["paper"]

        # Try v10 paper-level context first (paper "80")
        v10p = v10_paper_ctx.get(paper, {})
        if v10p:
            _set_if_empty(r, "reaction_class", v10p.get("reaction_class"), "L1_v10")
            _set_if_empty(r, "organolithium_role", v10p.get("organolithium_role_v5"), "L1_v10")
            _set_if_empty(r, "reagent_family", v10p.get("reagent_family_v2"), "L1_v10")
            _set_if_empty(r, "solvent", v10p.get("solvent"), "L1_v10")
            _set_if_empty(r, "reactor_type", v10p.get("reactor_type"), "L1_v10")
            _set_if_empty(r, "paper_doi", v10p.get("paper_doi"), "L1_v10")
            _set_if_empty(r, "paper_year", v10p.get("paper_year"), "L1_v10")
            r["context_source"] = "v10_paper_backfill"
            STATS["L1_v10_matched"] += 1
            continue

        # Try manual context (Degennaro, Sun2020)
        for key, manual in MANUAL_PAPER_CONTEXT.items():
            if key.lower() in paper.lower():
                _set_if_empty(r, "reaction_class", manual.get("reaction_class"), "L1_manual")
                _set_if_empty(r, "organolithium_role", manual.get("organolithium_role"), "L1_manual")
                _set_if_empty(r, "reagent_family", manual.get("reagent_family"), "L1_manual")
                _set_if_empty(r, "solvent", manual.get("solvent"), "L1_manual")
                _set_if_empty(r, "reactor_type", manual.get("reactor_type"), "L1_manual")
                _set_if_empty(r, "reactant1_name", manual.get("reactant1_name"), "L1_manual")
                _set_if_empty(r, "reactant2_name", manual.get("reactant2_name"), "L1_manual")
                _set_if_empty(r, "product_name", manual.get("product_name"), "L1_manual")
                _set_if_empty(r, "paper_doi", manual.get("paper_doi"), "L1_manual")
                _set_if_empty(r, "paper_year", manual.get("paper_year"), "L1_manual")
                r["context_source"] = "manual_paper_context"
                STATS["L1_manual_matched"] += 1
                break


# ═══════════════════════════════════════════════════════════════
#  Level 2: table_context SMILES → per-row reactant/product
# ═══════════════════════════════════════════════════════════════

# Mapping of table_context JSON keys → output fields
TABLE_CONTEXT_FIELD_MAP = {
    # Ar1-X, Ar2-X, Product (paper 80)
    "Ar1-X": "reactant1_name",
    "Ar2-X": "reactant2_name",
    "Product": "product_name",
    # Common alternative keys
    "Reactant": "reactant1_name",
    "Electrophile": "reactant2_name",
    "Substrate": "reactant1_name",
}


def enrich_level2_table_context(rows):
    """Extract SMILES/names from table_context JSON into reactant/product fields."""
    for r in rows:
        tc = r.get("table_context", "")
        if not tc:
            continue
        try:
            ctx = json.loads(tc)
        except (json.JSONDecodeError, TypeError):
            continue

        for tc_key, out_field in TABLE_CONTEXT_FIELD_MAP.items():
            if tc_key in ctx and ctx[tc_key]:
                _set_if_empty(r, out_field, ctx[tc_key], "L2_table_ctx")


# ═══════════════════════════════════════════════════════════════
#  Level 3: VLM reaction_description parsing
# ═══════════════════════════════════════════════════════════════

# Pattern: "Lithiation of {substrate} with {lithium_reagent} followed by trapping with {electrophile} to give {product}"
RE_LITHIATION = re.compile(
    r"[Ll]ithiation\s+of\s+(.+?)\s+with\s+(\S+)\s+followed\s+by\s+trapping\s+with\s+(\S+)"
    r"(?:\s+to\s+give\s+(.+?))?$",
    re.IGNORECASE,
)

# Pattern: "{species} generated from {compound_id}"
RE_GENERATED_FROM = re.compile(
    r"(\S+(?:\s+\S+)?lithio\S*)\s+generated\s+from\s+(\S+)",
    re.IGNORECASE,
)

# Pattern: "R = {substituent}; effect of ..."
RE_R_SUBSTITUENT = re.compile(
    r"R\s*=\s*([\w-]+)",
    re.IGNORECASE,
)


def enrich_level3_description_parsing(rows):
    """Parse VLM reaction_description to extract chemical names."""
    for r in rows:
        desc = r.get("reaction_description", "")
        elec = r.get("electrophile_or_substrate", "")

        if not desc:
            continue

        # Pattern 1: "Lithiation of X with Y followed by trapping with Z to give W"
        m = RE_LITHIATION.search(desc)
        if m:
            substrate = m.group(1).strip()
            lithium_reagent = m.group(2).strip()
            electrophile = m.group(3).strip()
            product = (m.group(4) or "").strip()
            _set_if_empty(r, "reactant1_name", substrate, "L3_desc_lithiation")
            _set_if_empty(r, "reactant2_name", lithium_reagent, "L3_desc_lithiation")
            if product:
                _set_if_empty(r, "product_name", product, "L3_desc_lithiation")
            _set_if_empty(r, "electrophile_or_substrate", electrophile, "L3_desc_lithiation")
            STATS["L3_lithiation_parsed"] += 1
            continue

        # Pattern 2: "{lithio-species} generated from {id}"
        m = RE_GENERATED_FROM.search(desc)
        if m:
            species = m.group(1).strip()
            compound_id = m.group(2).strip()
            # The lithio-species IS the reactive intermediate (formed from reactant1 + reactant2)
            # reactant1 is the parent arene, e.g. "p-bromobenzonitrile" for "p-lithiobenzonitrile"
            # We can set a descriptive name
            parent_name = species.replace("lithio", "halo")  # approximate
            _set_if_empty(r, "reactant1_name", f"{species} (from {compound_id})", "L3_desc_generated")
            STATS["L3_generated_parsed"] += 1
            continue

        # Use electrophile_or_substrate field if it has real chemical info
        if elec and elec not in ("None", "N/A", ""):
            # Clean up "X substituent (R = X)" → just use as context
            if "substituent" not in elec.lower():
                _set_if_empty(r, "reactant2_name", elec, "L3_elec_field")
            STATS["L3_elec_used"] += 1


# ═══════════════════════════════════════════════════════════════
#  Level 4: Cross-figure context propagation
# ═══════════════════════════════════════════════════════════════

def enrich_level4_v10_figure_backfill(rows, v10_fig_ctx):
    """For v10_paper_backfill rows still missing reactant/product,
    try v10 figure-level context if consistent across review_ids."""
    # Build paper → v10 figure context map
    # from load_v10_figure_context(), we need paper mapping
    # We match via existing enriched rows in the same paper that have v10_index
    by_paper = defaultdict(list)
    for r in rows:
        by_paper[r["paper"]].append(r)

    for paper, paper_rows in by_paper.items():
        # Collect v10 figure context from rows that have v10_index
        fig_products = set()
        fig_r1 = set()
        fig_r2 = set()
        for r in paper_rows:
            vid = r.get("v10_index", "")
            if vid and vid in v10_fig_ctx:
                fc = v10_fig_ctx[vid]
                if not _is_empty(fc.get("product_name")):
                    fig_products.add(fc["product_name"])
                if not _is_empty(fc.get("reactant1_name")):
                    fig_r1.add(fc["reactant1_name"])
                if not _is_empty(fc.get("reactant2_name")):
                    fig_r2.add(fc["reactant2_name"])

        # Only propagate if v10 figures have single consistent value
        for r in paper_rows:
            if r.get("context_source") not in ("v10_paper_backfill", "manual_paper_context"):
                continue
            if len(fig_products) == 1:
                _set_if_empty(r, "product_name", fig_products.copy().pop(), "L4_v10_fig")
            if len(fig_r1) == 1:
                _set_if_empty(r, "reactant1_name", fig_r1.copy().pop(), "L4_v10_fig")
            if len(fig_r2) == 1:
                _set_if_empty(r, "reactant2_name", fig_r2.copy().pop(), "L4_v10_fig")


def enrich_level4_cross_figure(rows):
    """Propagate context from other figures in the same paper."""
    # Group by paper
    by_paper = defaultdict(list)
    for r in rows:
        by_paper[r["paper"]].append(r)

    for paper, paper_rows in by_paper.items():
        # Collect all non-empty values per field
        field_pool = {}
        propagatable = [
            "reaction_class", "organolithium_role", "reagent_family",
            "solvent", "reactor_type", "paper_doi", "paper_year",
        ]
        for field in propagatable:
            values = set()
            for r in paper_rows:
                v = r.get(field, "")
                if not _is_empty(v):
                    values.add(v)
            if len(values) == 1:
                # Unique value across paper → safe to propagate
                field_pool[field] = values.pop()

        # Apply to rows with gaps
        for r in paper_rows:
            for field, value in field_pool.items():
                _set_if_empty(r, field, value, "L4_cross_fig")

    # Also propagate reactant/product within same paper if single consistent value
    for field_name, tag in [
        ("reactant1_name", "L4_cross_fig_r1"),
        ("reactant2_name", "L4_cross_fig_r2"),
        ("product_name", "L4_cross_fig_prod"),
    ]:
        for paper, paper_rows in by_paper.items():
            vals = set()
            for r in paper_rows:
                v = r.get(field_name, "")
                if not _is_empty(v):
                    vals.add(v)
            if len(vals) == 1:
                val = vals.pop()
                for r in paper_rows:
                    _set_if_empty(r, field_name, val, tag)


# ═══════════════════════════════════════════════════════════════
#  Level 5: Known paper-level product/reactant from literature
#  (for papers where v10 never had product but chemistry is known)
# ═══════════════════════════════════════════════════════════════

# Map: partial paper name → fields to fill
# Only fill if field is still empty; only for papers with clear single reaction
KNOWN_PAPER_CHEMISTRY = {
    "Nagaki et al. 2010 - A flow microreactor system enables": {
        # Lithiation of aryl bromide ester → aryllithium (then trapped with electrophile)
        # Heatmap shows optimization of lithiation step; product = aryllithium intermediate
        "product_name": "tert-butyl 4-(lithio)benzoate (aryllithium intermediate)",
    },
    "Nagaki et al. 2010 - Generation and reactions of oxiranyl": {
        # sBuLi + epoxide → oxiranyllithium intermediate
        "reactant1_name": "epibromohydrin",
        "product_name": "oxiranyllithium intermediate",
    },
    "Nagaki et al. 2010 - Generation and reaction of cyano": {
        # n-BuLi + halobenzonitrile → lithiobenzonitrile
        "product_name": "lithiobenzonitrile (cyano-aryllithium)",
    },
    "Nagaki et al. 2012 - Cross-coupling of aryllithium": {
        # Aryllithium + aryl halide → biaryl (Murahashi coupling)
        "reactant1_name": "aryllithium",
        "product_name": "biaryl cross-coupling product",
    },
    "Sun et al. 2020": {
        "reactant1_name": "aryl halide",
        "reactant2_name": "n-BuLi",
        "product_name": "aryl-alkyl coupling product",
    },
    "Asai et al. 2012": {
        # Halogen-metal exchange of aryl bromide → aryllithium → diarylethene coupling
        "reactant2_name": "n-BuLi",
    },
    "three-component-coupling": {
        "reactant1_name": "dihalobenzene (benzyne precursor)",
    },
    # ── New papers (2024-04-03 batch) ──
    "Nagaki et al. 2019 - Alkyllithium compounds bearing electrophilic": {
        "reactant2_name": "s-BuLi",
    },
    "Musci et al. 2020": {
        "reactant1_name": "chloroiodomethane (ICH2Cl)",
        "reactant2_name": "LDA",
        "product_name": "chloroiodomethane-d1 (CHD(I)(Cl))",
    },
    "Nagaki et al. 2015 - Benzyllithiums": {
        "reactant1_name": "p-propanoylbenzyl chloride",
        "reactant2_name": "lithium naphthalenide (LiNp)",
    },
    "Nagaki et al. 2016 - Integration of borylation": {
        "reactant1_name": "bromobenzene",
        "reactant2_name": "n-BuLi",
    },
    "Okamoto et al. 2026": {
        "reactant2_name": "n-BuLi",
    },
    "Miyagishi et al. 2024": {
        "reactant2_name": "n-BuLi",
    },
    "Nagaki et al. 2019 - Generation and reaction of functional alkyl": {
        "reactant2_name": "s-BuLi",
    },
}


def enrich_level5_known_chemistry(rows):
    """Fill product/reactant from known paper chemistry literature."""
    for r in rows:
        paper = r.get("paper", "")
        for paper_prefix, fields in KNOWN_PAPER_CHEMISTRY.items():
            if paper.startswith(paper_prefix):
                for field, value in fields.items():
                    _set_if_empty(r, field, value, "L5_known")
                break


# ═══════════════════════════════════════════════════════════════
#  Level 6: tR step classification (tR1 / tR2 / tR_generic)
# ═══════════════════════════════════════════════════════════════

# Regex patterns: ordered list, first match wins
_TR1_PATTERNS = [
    re.compile(r"residence\s+time\s+in\s+R1", re.IGNORECASE),
    re.compile(r"[tT]\^?R1\s*[\(/]"),   # t^R1 (s), t^R1/s, tR1 (s)
    re.compile(r"[tT]\^?R1$"),           # bare t^R1
    re.compile(r"tR1\s*\(s\)"),          # tR1 (s)
]
_TR2_PATTERNS = [
    re.compile(r"[tT]\^?R2\s*[\(/]"),   # t^R2/s, t^R2 (s)
    re.compile(r"[tT]\^?R2$"),           # bare t^R2
    re.compile(r"tR2\s*\(s\)"),          # tR2 (s)
]

# For generic tR (no R1/R2 suffix), classify by paper
PAPER_TR_STEP_RULES = {
    "Nagaki et al. 2010 - A flow microreactor system enables": "tR1",
    "Nagaki et al. 2010 - Generation and reactions of oxiranyl": "tR1",
    "Nagaki et al. 2008 - Aryllithium compounds bearing alkoxycarbonyl": "tR1",
    # Paper 80: heatmaps + Residence Time tables = Suzuki coupling (third step)
    # → tR_generic; tR1 tables already classified by regex
    # New papers — generic tR is tR1 (lithiation step optimization)
    "Nagaki et al. 2015 - Benzyllithiums": "tR1",
    "Musci et al. 2020": "tR1",
    "Okamoto et al. 2026": "tR1",
}


def _classify_tr_step_from_axis(x_title):
    """Return 'tR1', 'tR2', or None (generic)."""
    for pat in _TR1_PATTERNS:
        if pat.search(x_title):
            return "tR1"
    for pat in _TR2_PATTERNS:
        if pat.search(x_title):
            return "tR2"
    return None


def enrich_level6_tr_step(rows):
    """Classify tR step and populate tR1_s/tR2_s/T1_C/T2_C."""
    for r in rows:
        x_title = r.get("x_axis_title", "")
        y_title = r.get("y_axis_title", "")

        # Step 1: try axis title regex
        step = _classify_tr_step_from_axis(x_title)

        # Step 2: fall back to paper-level rules for generic tR
        if step is None:
            paper = r.get("paper", "")
            for prefix, rule_step in PAPER_TR_STEP_RULES.items():
                if paper.startswith(prefix):
                    step = rule_step
                    STATS["L6_paper_rule"] += 1
                    break

        # Step 3: default to tR_generic
        if step is None:
            step = "tR_generic"
            STATS["L6_generic"] += 1
        else:
            STATS[f"L6_{step}"] += 1

        r["tR_step"] = step

        # Populate tR1_s / tR2_s
        tR_s = r.get("tR_s", "")
        if step == "tR1" or step == "tR_generic":
            r["tR1_s"] = tR_s
            r["tR2_s"] = ""
        elif step == "tR2":
            r["tR1_s"] = ""
            r["tR2_s"] = tR_s

        # Populate T1_C / T2_C
        T_C = r.get("T_C", "")
        has_T1_label = "T^1" in y_title or "T1" in y_title
        has_T2_label = "T^2" in y_title or "T2" in y_title

        if has_T1_label:
            r["T1_C"] = T_C
            r["T2_C"] = ""
        elif has_T2_label:
            r["T1_C"] = ""
            r["T2_C"] = T_C
        elif step == "tR2":
            r["T1_C"] = ""
            r["T2_C"] = T_C
        else:
            # tR1 or tR_generic: temperature is for step 1
            r["T1_C"] = T_C
            r["T2_C"] = ""


def _load_vlm_molecule_in_figure():
    """Load molecule_in_figure from VLM heatmap JSONs, keyed by image filename."""
    lookup = {}
    for jf in glob.glob(os.path.join(VLM_HEATMAP_DIR, "*.json")):
        base = os.path.basename(jf).replace(".json", ".png")
        img_key = f"heatmaps/{base}"
        try:
            with open(jf) as f:
                data = json.load(f)
            mol = data.get("context", {}).get("molecule_in_figure") or ""
            if mol:
                lookup[img_key] = mol
        except (json.JSONDecodeError, KeyError):
            pass
    return lookup


# ═══════════════════════════════════════════════════════════════
#  Level 7: Five-position chemical model
# ═══════════════════════════════════════════════════════════════

# Each entry: paper_prefix → mapping rules
# Values are either:
#   - string starting with "@" → copy from that field (e.g. "@reactant1_name")
#   - plain string → hardcoded literal value
#   - None → leave empty
FIVE_POSITION_MAP = {
    "Nagaki et al. 2010 - A flow microreactor system enables": {
        "substrate1": "@reactant1_name",
        "organolithium_reagent": "s-BuLi",
        "intermediate": "tert-butyl 4-(lithio)benzoate (aryllithium)",
        "substrate2_electrophile": "@electrophile_or_substrate",
        "product_final": "@product_name",
    },
    "Nagaki et al. 2010 - Generation and reactions of oxiranyl": {
        "substrate1": "@reactant1_name",
        "organolithium_reagent": "s-BuLi",
        "intermediate": "oxiranyllithium",
        "substrate2_electrophile": "@electrophile_or_substrate",
        "product_final": "@product_name",
    },
    "Nagaki et al. 2008 - Aryllithium compounds bearing alkoxycarbonyl": {
        "substrate1": "@reactant1_name",
        "organolithium_reagent": "s-BuLi",
        "intermediate": "tert-butyl o-(lithio)benzoate (aryllithium)",
        "substrate2_electrophile": "@electrophile_or_substrate",
        "product_final": "@product_name",
    },
    "Nagaki et al. 2010 - Generation and reaction of cyano": {
        # reactant1_name IS the lithio-species (intermediate)
        "substrate1": None,  # halobenzonitrile (not separately tracked)
        "organolithium_reagent": "n-BuLi",
        "intermediate": "@reactant1_name",
        "substrate2_electrophile": "@electrophile_or_substrate",
        "product_final": "@product_name",
    },
    "Nagaki et al. 2010 - Cross-coupling in a flow": {
        "substrate1": "@reactant1_name",
        "organolithium_reagent": "n-BuLi",
        "intermediate": "p-methoxyphenyllithium",
        "substrate2_electrophile": "@reactant2_name",
        "product_final": "@product_name",
    },
    "Nagaki et al. 2012 - Cross-coupling of aryllithium": {
        "substrate1": None,
        "organolithium_reagent": "n-BuLi",
        "intermediate": "@reactant1_name",
        "substrate2_electrophile": "@reactant2_name",
        "product_final": "@product_name",
    },
    "Nagaki et al. 2011 - Homocoupling": {
        "substrate1": "@reactant1_name",
        "organolithium_reagent": "n-BuLi",
        "intermediate": "aryllithium",
        "substrate2_electrophile": "FeCl3 (oxidative coupling catalyst)",
        "product_final": "@product_name",
    },
    "Homocoupling of aryl halides in flow": {
        "substrate1": "@reactant1_name",
        "organolithium_reagent": "n-BuLi",
        "intermediate": "aryllithium",
        "substrate2_electrophile": "FeCl3 (oxidative coupling catalyst)",
        "product_final": "@product_name",
    },
    "Asai et al. 2012": {
        "substrate1": "@reactant1_name",
        "organolithium_reagent": "n-BuLi",
        "intermediate": "heteroaryllithium",
        "substrate2_electrophile": "octafluorocyclopentene",
        "product_final": "@product_name",
    },
    "three-component-coupling": {
        "substrate1": "@reactant1_name",
        "organolithium_reagent": "PhLi",
        "intermediate": "@molecule_in_figure",  # VLM has it for sub-figs b,c
        "substrate2_electrophile": "@electrophile_or_substrate",
        "product_final": "@product_name",
    },
    "Degennaro": {
        "substrate1": "@reactant1_name",
        "organolithium_reagent": "n-BuLi",
        "intermediate": "alpha-lithiated Boc-pyrrolidine",
        "substrate2_electrophile": "Boc2O (di-tert-butyl dicarbonate)",
        "product_final": "@product_name",
    },
    "Sun et al. 2020": {
        "substrate1": "@reactant1_name",
        "organolithium_reagent": "n-BuLi",
        "intermediate": "2-pyridyllithium",
        "substrate2_electrophile": None,
        "product_final": "@product_name",
    },
    "example1": {
        "substrate1": "@reactant1_name",
        "organolithium_reagent": None,
        "intermediate": "@molecule_in_figure",
        "substrate2_electrophile": None,
        "product_final": "@product_name",
    },
}

# ── New papers (2024-04-03 batch) ──
FIVE_POSITION_MAP["Nagaki et al. 2019 - Alkyllithium compounds bearing electrophilic"] = {
    # Angew Chem 2019: alkylLi with EWG (bromoalkyl nitrile, bromoalkyl ester, etc.)
    "substrate1": "@reactant1_name",
    "organolithium_reagent": "s-BuLi",
    "intermediate": "alkyllithium bearing EWG",
    "substrate2_electrophile": "@electrophile_or_substrate",
    "product_final": "@product_name",
}
FIVE_POSITION_MAP["Musci et al. 2020"] = {
    # chloroiodomethyllithium carbenoid via LDA
    "substrate1": "chloroiodomethane (ICH2Cl)",
    "organolithium_reagent": "LDA",
    "intermediate": "chloroiodomethyllithium (CHLi(I)(Cl))",
    "substrate2_electrophile": "@electrophile_or_substrate",
    "product_final": "@product_name",
}
FIVE_POSITION_MAP["Nagaki et al. 2015 - Benzyllithiums"] = {
    # benzylLi bearing aldehyde carbonyl
    "substrate1": "p-propanoylbenzyl chloride",
    "organolithium_reagent": "lithium naphthalenide (LiNp)",
    "intermediate": "benzyllithium bearing aldehyde",
    "substrate2_electrophile": "@electrophile_or_substrate",
    "product_final": "@product_name",
}
FIVE_POSITION_MAP["Nagaki et al. 2016 - Integration of borylation"] = {
    # ArLi → borylation → Suzuki
    "substrate1": "bromobenzene",
    "organolithium_reagent": "n-BuLi",
    "intermediate": "phenyllithium (then borylated)",
    "substrate2_electrophile": "@electrophile_or_substrate",
    "product_final": "@product_name",
}
FIVE_POSITION_MAP["Okamoto et al. 2026"] = {
    # C1 carbenoid from organosulfur
    "substrate1": "@reactant1_name",
    "organolithium_reagent": "n-BuLi",
    "intermediate": "anionic C1 carbenoid (thio-substituted)",
    "substrate2_electrophile": "@electrophile_or_substrate",
    "product_final": "@product_name",
}
FIVE_POSITION_MAP["Miyagishi et al. 2024"] = {
    # C-glycoside from unstable organolithium
    "substrate1": "@reactant1_name",
    "organolithium_reagent": "n-BuLi",
    "intermediate": "aryllithium (for C-glycosylation)",
    "substrate2_electrophile": "@electrophile_or_substrate",
    "product_final": "@product_name",
}
FIVE_POSITION_MAP["Nagaki et al. 2019 - Generation and reaction of functional alkyl"] = {
    # functional alkylLi via Br/Li exchange
    "substrate1": "@reactant1_name",
    "organolithium_reagent": "s-BuLi",
    "intermediate": "functional alkyllithium",
    "substrate2_electrophile": "@electrophile_or_substrate",
    "product_final": "@product_name",
}

# Paper 80: uses same substrate1/substrate2 SMILES from table_context
FIVE_POSITION_MAP["80"] = {
    "substrate1": "@reactant1_name",
    "organolithium_reagent": "n-BuLi",
    "intermediate": "aryllithium (then borylated to arylboronate)",
    "substrate2_electrophile": "@reactant2_name",
    "product_final": "@product_name",
}

# Intermediate parsing from molecule_in_figure
_RE_INTERMEDIATE = re.compile(
    r"(?:aryllithium|organolithium)\s+intermediate[:\s]*(.+?)(?:;|$)",
    re.IGNORECASE,
)


def _resolve_field(row, spec):
    """Resolve a field spec: '@field_name' → row[field_name], else literal."""
    if spec is None:
        return ""
    if spec.startswith("@"):
        field = spec[1:]
        val = row.get(field, "")
        # For molecule_in_figure, try to extract intermediate description
        if field == "molecule_in_figure" and val:
            m = _RE_INTERMEDIATE.search(val)
            if m:
                return m.group(1).strip()
            # If it starts with "organolithium intermediate:", take the rest
            if "intermediate:" in val.lower():
                idx = val.lower().index("intermediate:")
                return val[idx + len("intermediate:"):].strip().split(";")[0].strip()
            return val
        # Skip "substituent" values for electrophile
        if field == "electrophile_or_substrate":
            if not val or "substituent" in val.lower():
                return ""
        return val if val and not _is_empty(val) else ""
    return spec


# Fallback intermediate names when molecule_in_figure is empty
FALLBACK_INTERMEDIATES = {
    "three-component-coupling": "aryllithium (via carbolithiation of benzyne)",
    "example1": "organolithium intermediate",
    "Nagaki et al. 2010 - Generation and reaction of cyano":
        "lithiobenzonitrile (cyano-aryllithium)",
    "Musci et al. 2020": "chloroiodomethyllithium (CHLi(I)(Cl))",
    "Nagaki et al. 2015 - Benzyllithiums": "benzyllithium bearing aldehyde carbonyl",
    "Nagaki et al. 2016 - Integration of borylation": "phenyllithium (then borylated to arylboronate)",
    "Okamoto et al. 2026": "anionic C1 carbenoid (thio-substituted)",
    "Miyagishi et al. 2024": "aryllithium (for C-glycosylation)",
    "Nagaki et al. 2019 - Alkyllithium compounds bearing":
        "alkyllithium bearing electrophilic functional group",
    "Nagaki et al. 2019 - Generation and reaction of functional alkyl":
        "functional alkyllithium",
}


def enrich_level7_five_position(rows, vlm_mol_lookup=None):
    """Map current reactant1/2/product into the 5-position chemical model."""
    vlm_mol_lookup = vlm_mol_lookup or {}

    for r in rows:
        paper = r.get("paper", "")

        # Inject molecule_in_figure from VLM JSONs if not already present
        if not r.get("molecule_in_figure"):
            img = r.get("image_file", "")
            r["molecule_in_figure"] = vlm_mol_lookup.get(img, "")

        # Find matching rule
        rule = None
        for prefix, mapping in FIVE_POSITION_MAP.items():
            if paper.startswith(prefix) or prefix.lower() in paper.lower():
                rule = mapping
                break

        if not rule:
            STATS["L7_no_rule"] += 1
            continue

        for pos in ["substrate1", "organolithium_reagent", "intermediate",
                     "substrate2_electrophile", "product_final"]:
            spec = rule.get(pos)
            val = _resolve_field(r, spec)
            if val:
                _set_if_empty(r, pos, val, "L7_5pos")

        # Fallback intermediate from hardcoded knowledge
        if _is_empty(r.get("intermediate")):
            for prefix, fallback in FALLBACK_INTERMEDIATES.items():
                if paper.startswith(prefix) or prefix.lower() in paper.lower():
                    _set_if_empty(r, "intermediate", fallback, "L7_fallback_int")
                    break

        STATS["L7_mapped"] += 1


# ═══════════════════════════════════════════════════════════════
#  Level 8: Reaction conditions from local_vars + substrate2 backfill
# ═══════════════════════════════════════════════════════════════

# Paper-level conditions from local_vars notes
# Fields: substrate2_electrophile, conc_substrate_M, conc_orgLi_M,
#         equiv_orgLi, conc_electrophile_M, flow_rate_total_mL_min
PAPER_CONDITIONS = {
    "Nagaki et al. 2010 - A flow microreactor system enables": {
        "substrate2_electrophile": "ROH (alcohol quench)",
        "electrophile_type": "quench_probe",
        "conc_substrate_M": "0.10",
        "conc_orgLi_M": "0.42",
        "equiv_orgLi": "1.1",
    },
    "Nagaki et al. 2010 - Generation and reactions of oxiranyl": {
        # Heatmaps = deprotonation step; tables have specific electrophiles (MeI, etc.)
        # Don't blanket-assign electrophile here; table rows get it from table_context
        "electrophile_type": "quench_probe",  # heatmaps use quench; tables override below
        "conc_substrate_M": "0.10",
        "conc_orgLi_M": "0.75",
        "conc_electrophile_M": "0.45",
    },
    "Nagaki et al. 2008 - Aryllithium compounds bearing alkoxycarbonyl": {
        "substrate2_electrophile": "ROH (alcohol quench)",
        "electrophile_type": "quench_probe",
        "conc_substrate_M": "0.10",
        "conc_orgLi_M": "0.42",
        "equiv_orgLi": "1.1",
    },
    "Nagaki et al. 2010 - Generation and reaction of cyano": {
        "substrate2_electrophile": "MeOH (methanol quench)",
        "electrophile_type": "quench_probe",
        "conc_substrate_M": "0.10",
        "equiv_orgLi": "1.1",
    },
    "Nagaki et al. 2010 - Cross-coupling in a flow": {
        "electrophile_type": "synthetic",
        "conc_substrate_M": "0.10",
        "conc_orgLi_M": "0.42",
        "equiv_orgLi": "1.1",
    },
    "three-component-coupling": {
        # PhCHO only for sub-figs b,c (from VLM electrophile_or_substrate);
        # t0/t3/t4 have no confirmed electrophile — do NOT blanket-assign
        "electrophile_type": "synthetic",
    },
    "Sun et al. 2020": {
        "substrate2_electrophile": "ethyl benzoate (PhCO₂Et)",
        "electrophile_type": "synthetic",
        "conc_substrate_M": "0.20",
        "conc_orgLi_M": "0.20",
        "conc_electrophile_M": "0.30",
    },
    "Degennaro": {
        # Already has substrate2_electrophile = Boc2O from Level 7
        "electrophile_type": "synthetic",
        "equiv_orgLi": "3.0",
        "equiv_electrophile": "3.0",
        "conc_electrophile_M": "0.16",
    },
    "Asai et al. 2012": {
        # Already has substrate2_electrophile = octafluorocyclopentene
        "electrophile_type": "synthetic",
        "conc_substrate_M": "0.10",
        "conc_orgLi_M": "0.42",
        "equiv_orgLi": "1.05",
        "conc_electrophile_M": "0.15",
    },
    "Nagaki et al. 2012 - Cross-coupling of aryllithium": {
        "electrophile_type": "synthetic",
    },
    "Nagaki et al. 2011 - Homocoupling": {
        "electrophile_type": "synthetic",
        "conc_substrate_M": "0.10",
        "conc_orgLi_M": "0.40",
    },
    "Homocoupling of aryl halides in flow": {
        # Same paper, different naming in v10
        "electrophile_type": "synthetic",
        "conc_substrate_M": "0.10",
        "conc_orgLi_M": "0.40",
    },
    "80": {
        "electrophile_type": "synthetic",
        "conc_orgLi_M": "0.50",
    },
    # ── New papers (2024-04-03 batch) ──
    "Nagaki et al. 2019 - Alkyllithium compounds bearing electrophilic": {
        "electrophile_type": "synthetic",
    },
    "Musci et al. 2020": {
        "electrophile_type": "quench_probe",
        "substrate2_electrophile": "CD3OD (deuterium quench)",
    },
    "Nagaki et al. 2015 - Benzyllithiums": {
        "electrophile_type": "quench_probe",
        "substrate2_electrophile": "MeOH (methanol quench)",
    },
    "Nagaki et al. 2016 - Integration of borylation": {
        "electrophile_type": "synthetic",
        "substrate2_electrophile": "p-bromobenzonitrile",
    },
    "Okamoto et al. 2026": {
        "electrophile_type": "synthetic",
    },
    "Miyagishi et al. 2024": {
        "electrophile_type": "synthetic",
    },
    "Nagaki et al. 2019 - Generation and reaction of functional alkyl": {
        "electrophile_type": "synthetic",
    },
}


def enrich_level8_conditions(rows):
    """Fill reaction conditions and backfill substrate2 from local_vars knowledge."""
    for r in rows:
        paper = r.get("paper", "")
        for prefix, conds in PAPER_CONDITIONS.items():
            if paper.startswith(prefix) or prefix.lower() in paper.lower():
                for field, value in conds.items():
                    _set_if_empty(r, field, value, "L8_conditions")
                break


# ═══════════════════════════════════════════════════════════════
#  Level 9: SMILES assignment for all chemical species
# ═══════════════════════════════════════════════════════════════

# Text name → canonical SMILES mapping
# Sources: PubChem CID lookups + paper schemes
NAME_TO_SMILES = {
    # ── Substrates (ArX, alkyl halides) ──
    "tert-butyl p-bromobenzoate":       "CC(C)(C)OC(=O)c1ccc(Br)cc1",
    "tert-butyl o-bromobenzoate":       "CC(C)(C)OC(=O)c1ccccc1Br",
    "epibromohydrin":                   "BrCC1CO1",
    "p-bromoanisole":                   "COc1ccc(Br)cc1",
    "bromobenzene":                     "Brc1ccccc1",
    "2-bromoiodobenzene":               "Brc1ccccc1I",
    "1,2-diiodobenzene":                "Ic1ccccc1I",
    "dihalobenzene (benzyne precursor)": "Ic1ccccc1I",  # most common in three-component
    "fluoroiodomethane (1)":            "FCI",
    "aryl halide":                      "Brc1ccccn1",  # 2-bromopyridine (Sun2020)
    "Boc-protected pyrrolidine":        "CC(C)(C)OC(=O)N1CCCC1",
    # ── Organolithium reagents ──
    "n-BuLi":  "CCCC[Li]",
    "s-BuLi":  "CCC([Li])C",
    "PhLi":    "[Li]c1ccccc1",
    # ── Intermediates (ArLi) ──
    "tert-butyl 4-(lithio)benzoate (aryllithium)":       "CC(C)(C)OC(=O)c1ccc([Li])cc1",
    "tert-butyl o-(lithio)benzoate (aryllithium)":       "CC(C)(C)OC(=O)c1ccccc1[Li]",
    "oxiranyllithium":                                    "[Li]C1CO1",
    "p-methoxyphenyllithium":                             "COc1ccc([Li])cc1",
    "phenyllithium":                                      "[Li]c1ccccc1",
    "aryllithium":                                        "[Li]c1ccccc1",  # generic → PhLi
    "heteroaryllithium":                                  "",  # per-row SMILES from substrate
    "p-lithiobenzonitrile (from 1c)":                     "N#Cc1ccc([Li])cc1",
    "o-lithiobenzonitrile (from 1a)":                     "N#Cc1ccccc1[Li]",
    "m-lithiobenzonitrile (from 1b)":                     "N#Cc1cccc([Li])c1",
    "aryllithium (via carbolithiation of benzyne)":       "",  # complex; sub-figure dependent
    "aryllithium (then borylated to arylboronate)":       "[Li]c1ccccc1",  # first intermediate
    "2-pyridyllithium":                                   "[Li]c1ccccn1",
    "alpha-lithiated Boc-pyrrolidine":                    "CC(C)(C)OC(=O)N1CCC([Li])C1",
    "Li-CH2-F (fluoromethyllithium), lifetime 13 ms at -60 °C": "[Li]CF",
    "CHLi(I)(F) (iodofluoromethyllithium), lifetime 82 ms at -40 °C": "[Li]C(F)I",
    "(Br, Li substituents on benzene ring)":  "Brc1ccccc1[Li]",
    "(I, Li substituents on benzene ring)":   "Ic1ccccc1[Li]",
    # ── Products ──
    "tert-butyl 4-(lithio)benzoate (aryllithium intermediate)": "CC(C)(C)OC(=O)c1ccc([Li])cc1",
    "oxiranyllithium intermediate":          "[Li]C1CO1",
    "tert-butyl benzoate":                   "CC(C)(C)OC(=O)c1ccccc1",
    "lithiobenzonitrile (cyano-aryllithium)": "N#Cc1ccc([Li])cc1",  # representative (para)
    "p-methoxybiphenyl":                     "COc1ccc(-c2ccccc2)cc1",
    "4,4'-dimethoxybiphenyl":                "COc1ccc(-c2ccc(OC)cc2)cc1",
    "biphenyl-4-carbonitrile":               "N#Cc1ccc(-c2ccccc2)cc1",
    "biaryl cross-coupling product":         "",  # per-row from reactant SMILES
    "alcohol product":                       "",  # per sub-figure
    "aryl-alkyl coupling product":           "",  # generic
    "α-lithiated Boc-pyrrolidine + Boc2O trapped product (6a)":
        "CC(C)(C)OC(=O)C1CCCN1C(=O)OC(C)(C)C",  # 2,2-di-Boc-pyrrolidine
    # ── New intermediates (2024-04-03 batch) ──
    "chloroiodomethyllithium (CHLi(I)(Cl))":     "[Li]C(Cl)I",
    "benzyllithium bearing aldehyde carbonyl":    "",  # per-substrate
    "benzyllithium bearing aldehyde":             "",
    "phenyllithium (then borylated)":             "[Li]c1ccccc1",
    "phenyllithium (then borylated to arylboronate)": "[Li]c1ccccc1",
    "anionic C1 carbenoid (thio-substituted)":    "",  # complex structure
    "aryllithium (for C-glycosylation)":          "",  # per-substrate
    "alkyllithium bearing EWG":                   "",  # per-substrate
    "alkyllithium bearing electrophilic functional group": "",
    "functional alkyllithium":                    "",  # per-substrate
    # ── New substrates ──
    "chloroiodomethane (ICH2Cl)":                 "ClCI",
    "p-propanoylbenzyl chloride":                 "CCC(=O)c1ccc(CCl)cc1",
    # ── New reagents ──
    "LDA":                                       "CC(C)[Li]N(C(C)C)C(C)C",  # approximate
    "lithium naphthalenide (LiNp)":              "[Li]",  # radical anion, no simple SMILES
    # ── Electrophiles ──
    "ROH (alcohol quench)":               "CO",  # MeOH as representative
    "MeOH (methanol quench)":             "CO",
    "octafluorocyclopentene":             "FC1=C(F)C(F)(F)C(F)(F)C1(F)F",
    "PhCHO (benzaldehyde)":               "O=Cc1ccccc1",
    "FeCl3 (oxidative coupling catalyst)": "[Fe](Cl)(Cl)Cl",
    "ethyl benzoate (PhCO₂Et)":           "CCOC(=O)c1ccccc1",
    "Boc2O (di-tert-butyl dicarbonate)":  "CC(C)(C)OC(=O)OC(=O)OC(C)(C)C",
    "p-iodobenzonitrile":                 "N#Cc1ccc(I)cc1",
    "various (MeI for standard quench)":  "CI",  # iodomethane
}

# Heteroaryllithium: derive SMILES from substrate SMILES (Br → Li)
def _substrate_to_aryllithium_smiles(substrate_smiles):
    """Convert ArBr SMILES → ArLi SMILES (replace first Br with [Li])."""
    if not substrate_smiles:
        return ""
    # Simple substitution: first Br → [Li]
    if "Br" in substrate_smiles:
        return substrate_smiles.replace("Br", "[Li]", 1)
    if "I" in substrate_smiles:
        return substrate_smiles.replace("I", "[Li]", 1)
    return ""


def enrich_level9_smiles(rows):
    """Add SMILES columns for substrate1, intermediate, product_final, etc."""
    smiles_fields = [
        ("substrate1", "substrate1_smiles"),
        ("organolithium_reagent", "organolithium_smiles"),
        ("intermediate", "intermediate_smiles"),
        ("substrate2_electrophile", "substrate2_smiles"),
        ("product_final", "product_smiles"),
    ]

    for r in rows:
        for src_field, smi_field in smiles_fields:
            val = r.get(src_field, "")
            if not val:
                r[smi_field] = ""
                continue

            # If it already looks like SMILES, use directly
            if any(ch in val for ch in ["c1", "C(", "Cc1", "(=O)", "[Li]", "#"]):
                r[smi_field] = val
                STATS[f"L9_{smi_field}_direct"] += 1
                continue

            # Lookup in NAME_TO_SMILES
            smi = NAME_TO_SMILES.get(val, "")
            if smi:
                r[smi_field] = smi
                STATS[f"L9_{smi_field}_lookup"] += 1
                continue

            # Special: heteroaryllithium → derive from substrate
            if src_field == "intermediate" and val == "heteroaryllithium":
                sub_smi = r.get("substrate1", "")
                if sub_smi and any(ch in sub_smi for ch in ["c1", "Br"]):
                    r[smi_field] = _substrate_to_aryllithium_smiles(sub_smi)
                    STATS[f"L9_{smi_field}_derived"] += 1
                    continue

            # Special: generic aryllithium for paper 80 → derive from substrate
            if src_field == "intermediate" and "aryllithium" in val.lower():
                sub_smi = r.get("substrate1", "")
                if sub_smi and any(ch in sub_smi for ch in ["c1", "Br"]):
                    r[smi_field] = _substrate_to_aryllithium_smiles(sub_smi)
                    STATS[f"L9_{smi_field}_derived"] += 1
                    continue

            r[smi_field] = ""
            STATS[f"L9_{smi_field}_missing"] += 1


# ═══════════════════════════════════════════════════════════════
#  Level 10: Column cleanup — drop redundant columns for ML
# ═══════════════════════════════════════════════════════════════

# Columns to remove: pipeline metadata, parsed-away sources, near-empty
COLUMNS_TO_DROP = {
    "point_id",           # sequential, regenerate if needed
    "image_file",         # pipeline artifact
    "v10_index",          # internal index
    "x_axis_title",       # encoded in tR_step
    "x_axis_scale",       # pipeline metadata
    "y_axis_title",       # encoded in T1_C/T2_C
    "y_axis_scale",       # pipeline metadata
    "sub_figure_label",   # pipeline metadata
    "reaction_description",  # parsed into other fields
    "compound_ids",       # 7.5% fill, VLM artifact
    "electrophile_or_substrate",  # parsed into substrate2_electrophile
    "caption",            # 0% fill
    "table_context",      # 3.4%, raw JSON already parsed
    "molecule_in_figure", # parsed into intermediate
    "coordinate_source",  # 100% "vlm" constant
    "context_source",     # pipeline provenance
    "source_type",        # heatmap/table, no ML value
    "organolithium_role", # redundant with organolithium_reagent
    "reagent_family",     # redundant with organolithium_reagent
    "tR_s",               # split into tR1_s/tR2_s
    "T_C",                # split into T1_C/T2_C
    "equiv_electrophile", # 0.3% fill (4 rows)
    # reactant1/2_name and product_name are the text originals,
    # now replaced by substrate1/intermediate/product_final + SMILES
    "reactant1_name",
    "reactant2_name",
    "product_name",
}


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    # Load input
    with open(INPUT_CSV) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    print(f"[INPUT] {len(rows)} rows loaded from {INPUT_CSV}")

    # Load v10 context
    v10_paper = load_v10_paper_context()
    v10_fig = load_v10_figure_context()
    print(f"[v10] {len(v10_paper)} papers, {len(v10_fig)} figures")

    # Run enrichment levels
    print("\n── Level 1: Paper-level v10 backfill ──")
    enrich_level1_paper_backfill(rows, v10_paper, v10_fig)

    print("── Level 2: table_context SMILES extraction ──")
    enrich_level2_table_context(rows)

    print("── Level 3: VLM reaction_description parsing ──")
    enrich_level3_description_parsing(rows)

    print("── Level 4a: v10 figure-level backfill for table rows ──")
    enrich_level4_v10_figure_backfill(rows, v10_fig)

    print("── Level 4b: Cross-figure context propagation ──")
    enrich_level4_cross_figure(rows)

    print("── Level 5: Known paper chemistry from literature ──")
    enrich_level5_known_chemistry(rows)

    print("── Level 6: tR step classification ──")
    enrich_level6_tr_step(rows)

    print("── Level 7: Five-position chemical model ──")
    vlm_mol = _load_vlm_molecule_in_figure()
    print(f"  [VLM] Loaded molecule_in_figure for {len(vlm_mol)} heatmaps")
    enrich_level7_five_position(rows, vlm_mol)

    print("── Level 8: Reaction conditions + substrate2 backfill ──")
    enrich_level8_conditions(rows)

    print("── Level 9: SMILES assignment ──")
    enrich_level9_smiles(rows)

    print("── Level 10: Column cleanup ──")

    # Extend fieldnames with new columns (before cleanup)
    new_cols = [
        "tR_step", "tR1_s", "tR2_s", "T1_C", "T2_C",
        "substrate1", "organolithium_reagent", "intermediate",
        "substrate2_electrophile", "electrophile_type", "product_final",
        "molecule_in_figure",
        "conc_substrate_M", "conc_orgLi_M", "equiv_orgLi",
        "conc_electrophile_M", "equiv_electrophile",
        # SMILES columns
        "substrate1_smiles", "organolithium_smiles", "intermediate_smiles",
        "substrate2_smiles", "product_smiles",
    ]
    for col in new_cols:
        if col not in fieldnames:
            fieldnames.append(col)

    # Ensure all rows have the new fields
    for r in rows:
        for col in new_cols:
            r.setdefault(col, "")

    # Drop redundant columns
    fieldnames = [c for c in fieldnames if c not in COLUMNS_TO_DROP]
    for r in rows:
        for col in COLUMNS_TO_DROP:
            r.pop(col, None)
    print(f"  Dropped {len(COLUMNS_TO_DROP)} columns → {len(fieldnames)} remain")

    # Write output
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Report
    print(f"\n{'=' * 65}")
    print(f"ENRICHED DATASET: {OUTPUT_CSV}")
    print(f"{'=' * 65}")
    print(f"Total rows: {len(rows)}")

    # Enrichment stats
    print(f"\n── Enrichment stats ──")
    for key in sorted(STATS.keys()):
        print(f"  {key:40s}: {STATS[key]}")

    # Final coverage
    print(f"\n── Final context coverage ──")
    fields = [
        "reaction_class", "solvent", "reactor_type",
        "paper_doi", "paper_year",
    ]
    for f in fields:
        filled = sum(1 for r in rows if not _is_empty(r.get(f)))
        pct = filled / len(rows) * 100
        print(f"  {f:25s}: {filled:>5}/{len(rows)} ({pct:5.1f}%)")

    # tR step distribution
    print(f"\n── tR step classification ──")
    step_counts = defaultdict(int)
    for r in rows:
        step_counts[r.get("tR_step", "")] += 1
    for step, cnt in sorted(step_counts.items()):
        print(f"  {step:15s}: {cnt:>5}")

    # Five-position model coverage
    print(f"\n── Five-position model coverage ──")
    for field in ["substrate1", "organolithium_reagent", "intermediate",
                   "substrate2_electrophile", "electrophile_type", "product_final"]:
        filled = sum(1 for r in rows if not _is_empty(r.get(field)))
        pct = filled / len(rows) * 100
        print(f"  {field:25s}: {filled:>5}/{len(rows)} ({pct:5.1f}%)")

    # Electrophile type breakdown
    print(f"\n── Electrophile type breakdown ──")
    etype_counts = defaultdict(int)
    for r in rows:
        et = r.get("electrophile_type", "") or "(empty)"
        etype_counts[et] += 1
    for et, cnt in sorted(etype_counts.items()):
        print(f"  {et:20s}: {cnt:>5}")

    # Reaction conditions coverage
    print(f"\n── Reaction conditions coverage ──")
    for field in ["conc_substrate_M", "conc_orgLi_M", "equiv_orgLi",
                   "conc_electrophile_M"]:
        filled = sum(1 for r in rows if not _is_empty(r.get(field)))
        pct = filled / len(rows) * 100
        print(f"  {field:25s}: {filled:>5}/{len(rows)} ({pct:5.1f}%)")

    # SMILES coverage
    print(f"\n── SMILES coverage ──")
    for field in ["substrate1_smiles", "organolithium_smiles", "intermediate_smiles",
                   "substrate2_smiles", "product_smiles"]:
        filled = sum(1 for r in rows if r.get(field, "").strip())
        pct = filled / len(rows) * 100
        print(f"  {field:25s}: {filled:>5}/{len(rows)} ({pct:5.1f}%)")

    # Sample per paper (3 rows each)
    print(f"\n── Sample rows per paper (tR_step + 5-position) ──")
    seen_papers = set()
    for r in rows:
        p = r["paper"]
        if p in seen_papers:
            continue
        seen_papers.add(p)
        paper_rows = [x for x in rows if x["paper"] == p]
        sample = paper_rows[:2]
        for s in sample:
            print(f"  {p[:40]:40s} | step={s.get('tR_step',''):10s} "
                  f"| sub1={str(s.get('substrate1',''))[:25]:25s} "
                  f"| OrgLi={str(s.get('organolithium_reagent',''))[:10]:10s} "
                  f"| int={str(s.get('intermediate',''))[:30]:30s} "
                  f"| sub2={str(s.get('substrate2_electrophile',''))[:25]:25s} "
                  f"| prod={str(s.get('product_final',''))[:30]}")

    # Remaining gaps (chemical fields + SMILES)
    print(f"\n── Remaining gaps ──")
    for field in ["substrate1", "organolithium_reagent", "intermediate",
                   "substrate1_smiles", "organolithium_smiles",
                   "intermediate_smiles", "substrate2_smiles", "product_smiles"]:
        missing = [r for r in rows if not r.get(field, "").strip()]
        if missing:
            papers = defaultdict(int)
            for r in missing:
                papers[r["paper"][:50]] += 1
            print(f"  {field} ({len(missing)} missing):")
            for p, cnt in sorted(papers.items(), key=lambda x: -x[1]):
                print(f"    {p}: {cnt} pts")

    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
