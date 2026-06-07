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
import time
import unicodedata
from collections import Counter
import requests

from src.adjudication.entity_pool import (
    build_global_entity_pool,
    resolve_record_smiles,
)


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


# Common OCR confusions in chemical names from figure legends.
# Conservative: only di/de prefix on halogens (a clear "di"→"de" misread),
# NOT on legitimate "de" words (dehydro, demethylation, dehalogenation).
_CHEM_NAME_FIXES = [
    (re.compile(r'\bdechloro', re.IGNORECASE), 'dichloro'),
    (re.compile(r'\bdebromo', re.IGNORECASE), 'dibromo'),
    (re.compile(r'\bdefluoro', re.IGNORECASE), 'difluoro'),
    (re.compile(r'\bdeiodo', re.IGNORECASE), 'diiodo'),
]


def normalize_chem_name(name: str | None) -> str | None:
    """Clean common OCR errors in a chemical name.

    Written back to the record so it fixes ``product_name`` AND feeds a
    clean name to PubChem:
    - locant separator misread as decimal: ``3.4-`` → ``3,4-``
    - di/de halogen-prefix confusion: ``dechloro`` → ``dichloro``
    - truncated common suffixes: ``...benzen`` → ``...benzene``,
      ``...anilin`` → ``...aniline``
    """
    if not name or not name.strip():
        return name
    s = name.strip()
    s = re.sub(r'(\d)\.(\d)', r'\1,\2', s)            # 3.4- → 3,4-
    for pat, repl in _CHEM_NAME_FIXES:
        s = pat.sub(repl, s)
    s = re.sub(r'benzen\b', 'benzene', s)             # truncated suffix
    s = re.sub(r'anilin\b', 'aniline', s)
    return s


# Abbreviation resolution: chemistry papers define abbreviations inline as
# "full name (ABBR)", e.g. "3,4-dichloroaniline (3,4-DCAN)".  Extract these
# so a product_name left as the bare abbreviation ("3,4-DCAN") can be
# restored to the full name (then resolved to SMILES).
_ABBREV_RE = re.compile(
    r'([\w,\-]*[A-Za-z][\w,\-]*)\s*'                  # full name: single no-space token, has a letter
    r'\(([0-9,\-]*[A-Z]{2,}[A-Z0-9,\-]*)\)'          # (ABBR): >=2 consecutive caps
)


def build_abbrev_map(text) -> dict:
    """Map ``abbr.lower() -> full name`` from inline "name (ABBR)" defs.

    ABBR requires >=2 consecutive capitals, so units / refs like "(2.0 MPa)"
    or "(Fig. 1)" are NOT captured.  The full name must contain a lowercase
    letter (so it's a real name, not another acronym).
    """
    out: dict = {}
    if not text:
        return out
    for name, abbr in _ABBREV_RE.findall(text):
        name = name.strip(" -,")
        abbr = abbr.strip()
        # name must look like a chemical name: has a lowercase letter, decent
        # length, and is not a generic English word (reactant/product/...).
        if (len(name) >= 4 and any(c.islower() for c in name)
                and name.lower() not in _NON_CHEM_WORDS):
            out.setdefault(abbr.lower(), name)
    return out


# Generic words that can sit before "(ABBR)" but are NOT chemical names.
_NON_CHEM_WORDS = {
    "reactant", "reactants", "product", "products", "compound", "compounds",
    "intermediate", "intermediates", "mixture", "solution", "substrate",
    "substrates", "reagent", "reagents", "catalyst", "material", "materials",
    "sample", "target", "reaction", "molecule", "species", "derivative",
    "analogue", "analog", "byproduct", "by-product", "starting",
}


# name→SMILES disk cache: avoids repeat HTTP across the batch and keeps us
# under PubChem's 5 req/s ceiling.  Loaded lazily, persisted by flush.
_SMILES_CACHE: dict | None = None
_SMILES_CACHE_PATH = "data/cache/smiles_lookup.json"


def _smiles_cache() -> dict:
    global _SMILES_CACHE
    if _SMILES_CACHE is None:
        try:
            with open(_SMILES_CACHE_PATH) as f:
                _SMILES_CACHE = json.load(f)
        except Exception:
            _SMILES_CACHE = {}
    return _SMILES_CACHE


def flush_smiles_cache() -> None:
    """Persist the in-memory name→SMILES cache to disk."""
    if _SMILES_CACHE is None:
        return
    try:
        os.makedirs(os.path.dirname(_SMILES_CACHE_PATH), exist_ok=True)
        with open(_SMILES_CACHE_PATH, "w") as f:
            json.dump(_SMILES_CACHE, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        print(f"[PostProcessor] smiles cache save failed: {exc}")


def lookup_smiles(name: str) -> str | None:
    """name→SMILES via PubChem REST — cached + rate-limited.

    PubChem's hard limit is 5 req/s, so we sleep 0.25s before each
    *network* call (cache hits skip it) and retry 503 with exponential
    backoff.  Results (including None) are cached so the batch never
    re-queries the same name.
    """
    if not name or not name.strip():
        return None
    cache = _smiles_cache()
    key = name.strip().lower()
    if key in cache:
        return cache[key]

    # PubChem (2025) renamed the property field to "SMILES"; older docs use
    # "IsomericSMILES". Request the new field, read either to be safe.
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{requests.utils.quote(name)}/property/SMILES/JSON"
    )
    result = None
    for attempt in range(3):
        time.sleep(0.25)  # ≤4 req/s, under PubChem's 5/s ceiling
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                props = resp.json()["PropertyTable"]["Properties"][0]
                result = props.get("SMILES") or props.get("IsomericSMILES")
                break
            if resp.status_code == 503:
                time.sleep(2 ** attempt)  # service busy — back off and retry
                continue
            break  # 404 / 400 — name not found, don't retry
        except Exception:
            time.sleep(2 ** attempt)
    cache[key] = result
    return result


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


# ---------------------------------------------------------------------------
# Reaction class normalisation (19-type controlled vocabulary)
# ---------------------------------------------------------------------------

_REACTION_CLASS_SYNONYMS: dict[str, set[str]] = {
    # ── General organic ──────────────────────────────────────────────────────
    "hydrogenation":         {"hydrogenation", "h2 addition", "catalytic hydrogenation"},
    "nitration":             {"nitration"},
    "oxidation":             {"oxidation", "oxidative"},
    "reduction":             {"reduction", "reductive", "birch reduction"},
    "photoreduction":        {"photoreduction", "photo-reduction"},
    "esterification":        {"esterification"},
    "amidation":             {"amidation", "amide coupling"},
    "halogenation":          {"halogenation", "chlorination", "bromination",
                              "fluorination", "iodination"},
    "alkylation":            {"alkylation", "c-alkylation", "n-alkylation",
                              "o-alkylation"},
    "acylation":             {"acylation", "friedel-crafts acylation"},
    "isomerization":         {"isomerization", "isomerisation"},
    "dehydration":           {"dehydration"},
    "dehydrogenation":       {"dehydrogenation"},
    "polymerization":        {"polymerization", "polymerisation",
                              "raft polymerization", "anionic polymerization",
                              "living polymerization", "ring-opening polymerization",
                              "rop"},
    "hydrolysis":            {"hydrolysis"},
    "photocatalysis":        {"photocatalysis", "photo-catalysis",
                              "photoredox", "photo-oxidation"},
    # ── Coupling reactions ────────────────────────────────────────────────────
    "C-C coupling":          {"c-c coupling", "cross-coupling", "suzuki",
                              "suzuki-miyaura", "heck", "negishi", "sonogashira",
                              "kumada", "buchwald", "carbolithiation",
                              "grignard addition", "reformatsky"},
    "C-N coupling":          {"c-n coupling", "buchwald-hartwig",
                              "ullmann coupling"},
    "C-O coupling":          {"c-o coupling", "etherification",
                              "williamson ether"},
    # ── Organolithium-specific ────────────────────────────────────────────────
    "nucleophilic addition": {"nucleophilic addition", "addition to carbonyl",
                              "organolithium addition", "rli addition",
                              "addition to aldehyde", "addition to ketone",
                              "addition to imine", "addition to ester",
                              "carbanion addition", "addition reaction",
                              "1,2-addition", "1,4-addition", "conjugate addition"},
    "halogen-metal exchange": {"halogen-metal exchange", "halogen-lithium exchange",
                               "lithium-halogen exchange", "lhx", "hmx",
                               "transmetalation from halide",
                               "aryllithium generation"},
    "directed metalation":   {"directed metalation", "directed ortho metalation",
                              "dom", "lateral metalation", "lateral lithiation",
                              "directed lithiation", "deprotonative metalation",
                              "c-h deprotonation", "deprotonation",
                              "benzylic deprotonation", "allylic deprotonation",
                              "alpha-deprotonation"},
    "anionic cyclization":   {"anionic cyclization", "anionic ring closure",
                              "carbanion cyclization", "intramolecular addition",
                              "intramolecular carbolithiation",
                              "anionic cascade", "anionic rearrangement",
                              "brook rearrangement", "retro-brook rearrangement"},
    # ── Catch-all ─────────────────────────────────────────────────────────────
    "other":                 {"other"},
}

# Tokens that indicate a mistaken classification (LLM describing a metric, not a reaction)
_REACTION_CLASS_REJECT = {
    "conversion", "selectivity", "yield", "production",
    "regime", "synthesis", "process", "purge",
}

# Canonical set for fast membership check (used for unknown-class logging)
_CANONICAL_CLASSES: set[str] = set(_REACTION_CLASS_SYNONYMS.keys())

# Path for accumulating unknown reaction classes found during processing
_UNKNOWN_CLASS_LOG = "evaluation/unknown_reaction_classes.txt"


def _normalize_reaction_class(value: str | None) -> str | None:
    """
    Map free-text reaction_class to a controlled vocabulary entry.
    Returns None if the value looks like a metric rather than a reaction type.
    """
    if not value or not isinstance(value, str):
        return value
    v = value.strip().lower()
    if not v:
        return value
    # Reject metric-like values
    if any(tok in v for tok in _REACTION_CLASS_REJECT):
        return value  # return as-is; caller decides whether to keep or null
    # Pass 1: exact match (v must be in the synonym set)
    for canonical, synonyms in _REACTION_CLASS_SYNONYMS.items():
        if v in synonyms:
            return canonical
    # Pass 2: substring match (any synonym is contained in v)
    for canonical, synonyms in _REACTION_CLASS_SYNONYMS.items():
        if any(s in v for s in synonyms):
            return canonical
    return value  # unrecognised but not a metric — keep original


_PAPER_LEVEL_REACTION_RULES: list[tuple[str, list[str]]] = [
    ("polymerization", [r"\bpolymeri[sz]ation\b", r"\bpolymer(s|ic)?\b", r"\btelechelic\b"]),
    ("amidation", [r"\bamidation\b", r"\bamide\b"]),
    ("C-N coupling", [r"\bamination\b", r"\bc-n coupling\b"]),
    ("C-O coupling", [r"\bc-o coupling\b", r"\betherification\b"]),
    ("photocatalysis", [r"\bphotocatal", r"\bphotoredox\b"]),
    ("oxidation", [r"\boxidation\b", r"\boxidative\b"]),
    ("reduction", [r"\breduction\b", r"\breductive\b", r"\bdibal\b"]),
    ("hydrogenation", [r"\bhydrogenation\b"]),
    ("halogenation", [r"\bhalogenation\b", r"\biodination\b", r"\bbromination\b", r"\bchlorination\b", r"\bfluorination\b"]),
    ("acylation", [r"\bacylation\b"]),
    ("alkylation", [r"\balkylation\b", r"\bnucleophilic substitution\b"]),
    ("hydrolysis", [r"\bhydrolysis\b"]),
    ("anionic cyclization", [r"\bcycli[sz]ation\b", r"\bring closure\b", r"\banionic cyclization\b"]),
    (
        "halogen-metal exchange",
        [
            r"\bhalogen[\s-]metal exchange\b",
            r"\bhalogen[\s-]lithium exchange\b",
            r"\blithium[\s-]halogen exchange\b",
            r"\biodine[\s-]lithium exchange\b",
            r"\bbromine[\s-]lithium exchange\b",
            r"\bbr\s*[-/]\s*li exchange\b",
            r"\bi\s*[-/]\s*li exchange\b",
            r"\bhalogen dance\b",
        ],
    ),
    (
        "C-C coupling",
        [
            r"\bc-c coupling\b",
            r"\bcross-coupling\b",
            r"\bcoupling\b",
            r"\bhomocoupling\b",
            r"\bmurahashi\b",
            r"\bsuzuki\b",
            r"\bcyanation\b",
            r"\barylation\b",
            r"\bcarbolithiation\b",
            r"\bglycosylation\b",
        ],
    ),
    (
        "directed metalation",
        [
            r"\bdirected metalation\b",
            r"\bdirected lithiation\b",
            r"\bdeprotolithiation\b",
            r"\bdeprotonation\b",
            r"\bmetalation\b",
            r"\bmetalation-substitution\b",
            r"\bortho[- ]?lithiation\b",
            r"\blateral metalation\b",
            r"\blaterally lithiated\b",
            r"\blithiation\b",
        ],
    ),
    (
        "nucleophilic addition",
        [
            r"\bnucleophilic addition\b",
            r"\baddition to\b",
            r"\bpropargylation\b",
            r"\bketone(s)?\b",
            r"\baldehyde(s)?\b",
            r"\bimine(s)?\b",
            r"\bacid chloride(s)?\b",
            r"\bketo ester(s)?\b",
            r"\bketoamide(s)?\b",
        ],
    ),
]


def _norm_text_key(text: str | None) -> str:
    text = str(text or "").strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = " ".join(text.split())
    return text.lower()


def _collect_intermediate_reaction_context(intermediate_dir: str | None) -> str:
    if not intermediate_dir or not os.path.isdir(intermediate_dir):
        return ""
    snippets: list[str] = []
    for path in glob.glob(os.path.join(intermediate_dir, "**", "*local_vars.json"), recursive=True)[:20]:
        try:
            with open(path, encoding="utf-8", errors="ignore") as fh:
                obj = json.load(fh)
        except Exception:
            continue
        for key in ("reaction_context", "data_interpretation_notes"):
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                snippets.append(val.strip())
        fixed = obj.get("fixed_conditions") or {}
        if isinstance(fixed, dict):
            note = fixed.get("notes")
            if isinstance(note, str) and note.strip():
                snippets.append(note.strip())
    scheme_path = os.path.join(intermediate_dir, "scheme_conditions.txt")
    if os.path.exists(scheme_path):
        try:
            with open(scheme_path, encoding="utf-8", errors="ignore") as fh:
                txt = fh.read().strip()
        except Exception:
            txt = ""
        if txt:
            snippets.append(txt)
    return " | ".join(snippets[:12])


def _infer_paper_level_class_from_text(text: str | None) -> tuple[str | None, str | None]:
    haystack = _norm_text_key(text)
    if not haystack:
        return None, None
    for label, patterns in _PAPER_LEVEL_REACTION_RULES:
        for pattern in patterns:
            if re.search(pattern, haystack):
                return label, pattern
    return None, None


def _normalize_record_level_class_for_vote(value: str | None) -> str | None:
    normalized = _normalize_reaction_class(value)
    low = _norm_text_key(normalized)
    if low in {"", "nan", "unspecified"}:
        return None
    if low == "carbolithiation":
        return "C-C coupling"
    if low == "nucleophilic substitution":
        return "alkylation"
    return normalized


def _apply_paper_level_reaction_class(records: list[dict], basename: str, intermediate_dir: str | None) -> list[dict]:
    title_class, title_pattern = _infer_paper_level_class_from_text(basename)
    context = _collect_intermediate_reaction_context(intermediate_dir)
    context_class, context_pattern = _infer_paper_level_class_from_text(context)

    votes = Counter()
    for rec in records:
        vote = _normalize_record_level_class_for_vote(rec.get("reaction_class"))
        if vote:
            votes[vote] += 1

    total_votes = sum(votes.values())
    top_class = None
    top_share = 0.0
    if votes:
        top_class, top_n = votes.most_common(1)[0]
        top_share = top_n / total_votes if total_votes else 0.0

    final_class = None
    assignment = None
    evidence = None
    needs_review = False
    if title_class:
        final_class = title_class
        assignment = "title_keyword"
        evidence = title_pattern
        if top_class and top_class != final_class and top_share >= 0.35:
            assignment = "title_keyword_over_majority"
            needs_review = True
    elif context_class:
        final_class = context_class
        assignment = "intermediate_context_keyword"
        evidence = context_pattern
        if top_class and top_class != final_class and top_share >= 0.35:
            assignment = "intermediate_context_over_majority"
            needs_review = True
    elif top_class:
        final_class = top_class
        if top_share >= 0.75:
            assignment = "majority_vote_high_conf"
        elif top_share >= 0.5:
            assignment = "majority_vote_low_conf"
            needs_review = True
        else:
            assignment = "plurality_vote"
            needs_review = True
        evidence = f"top_share={top_share:.3f}"
    else:
        final_class = "other"
        assignment = "fallback_other"
        needs_review = True

    for rec in records:
        rec["reaction_class_record_level"] = rec.get("reaction_class")
        rec["reaction_class_paper_level"] = final_class
        rec["reaction_class_assignment"] = assignment
        rec["reaction_class_evidence"] = evidence
        rec["reaction_class_needs_review"] = needs_review
        rec["reaction_class"] = final_class
    return records


# ---------------------------------------------------------------------------
# Unit parsing utilities (for future ORD export and condition normalisation)
# ---------------------------------------------------------------------------

def _strip_to_float(s: str | None) -> float | None:
    """
    Extract the first numeric value from a string like "80 °C", "2.5 bar",
    "30 s", "-78°C".  Returns None if no number found.
    """
    if s is None:
        return None
    m = re.search(r"-?\d+\.?\d*", str(s))
    return float(m.group(0)) if m else None


def _parse_value_and_unit(s: str | None) -> tuple[float | None, str | None]:
    """
    Split a condition string like "80 °C" into (80.0, "°C").
    Returns (None, None) if no number found.
    """
    if s is None:
        return None, None
    m = re.match(r"\s*(-?\d+\.?\d*)\s*(.*)", str(s).strip())
    if not m:
        return None, None
    value = float(m.group(1))
    unit = m.group(2).strip() or None
    return value, unit


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
    "reaction_class_record_level",
    "reaction_class_paper_level",
    "reaction_class",
    "reaction_class_assignment",
    "reaction_class_evidence",
    "reaction_class_needs_review",
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
        abbrev_map = build_abbrev_map(full_text)  # "name (ABBR)" → full name

        # Paper-level compound identity pool: merge label/name→SMILES across the
        # whole paper (scheme pool + every table CSV + records that already
        # resolved) so a structure recognised once backfills references to the
        # same compound elsewhere.  Built from the RAW records (pre-normalised)
        # plus on-disk evidence; used below to backfill before the PubChem step.
        scheme_pools = {}
        pool_path = os.path.join(intermediate_dir, "compound_pool.json")
        if os.path.exists(pool_path):
            try:
                pd = json.load(open(pool_path))
                scheme_pools = pd if isinstance(pd.get("reactant_pool"), dict) else {"compound_pool": pd}
            except Exception:
                scheme_pools = {}
        entity_pool = build_global_entity_pool(intermediate_dir, scheme_pools, records)
        if entity_pool.size:
            print(f"[PostProcessor] entity pool: {len(entity_pool.label_to_smiles)} labels, "
                  f"{len(entity_pool.name_to_smiles)} names")

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

            # -- Reaction class --
            if nr.get("reaction_class"):
                nr["reaction_class"] = _normalize_reaction_class(nr["reaction_class"])

            # Log unknown reaction classes for future taxonomy expansion
            rc = nr.get("reaction_class")
            if rc and rc not in _CANONICAL_CLASSES:
                self._log_unknown_class(rc, basename)

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

            # -- Chemical-name normalization + abbreviation resolution
            #    (always — fixes OCR errors, restores bare abbreviations like
            #    "3,4-DCAN" → "3,4-dichloroaniline", feeds clean names to PubChem) --
            for _nk in ("reactant1_name", "reactant2_name", "product_name"):
                if nr.get(_nk):
                    nr[_nk] = normalize_chem_name(nr[_nk])
                    if nr[_nk] and nr[_nk].lower() in abbrev_map:
                        nr[_nk] = abbrev_map[nr[_nk].lower()]

            # -- Entity-pool SMILES backfill (deterministic, no network) --
            #    Fills SMILES the per-source LLM left null by matching this
            #    record's label/name against the paper-level pool.  Runs before
            #    PubChem so the slow network lookup only handles what's left.
            resolve_record_smiles(nr, entity_pool)

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

        normalised = _apply_paper_level_reaction_class(normalised, basename, intermediate_dir)

        # Save _normalized.json
        norm_json = os.path.join(self.output_dir, f"{basename}_normalized.json")
        with open(norm_json, "w") as f:
            json.dump(normalised, f, indent=2, ensure_ascii=False)
        print(f"[PostProcessor] -> {norm_json}")
        flush_smiles_cache()  # persist any name→SMILES queries from this PDF

        # Save _normalized.xlsx
        try:
            self._save_excel(normalised, basename)
        except Exception as e:
            print(f"[PostProcessor] Excel export failed: {e}")

        return norm_json

    def _log_unknown_class(self, reaction_class: str, basename: str) -> None:
        """
        Append unrecognised reaction_class values to evaluation/unknown_reaction_classes.txt
        for periodic human review and taxonomy expansion.
        Each line: "reaction_class_value  |  source_pdf"
        """
        try:
            os.makedirs(os.path.dirname(_UNKNOWN_CLASS_LOG), exist_ok=True)
            # Read existing lines to avoid duplicates for this value+basename pair
            existing: set[str] = set()
            if os.path.exists(_UNKNOWN_CLASS_LOG):
                with open(_UNKNOWN_CLASS_LOG, encoding="utf-8") as f:
                    existing = {ln.strip() for ln in f if ln.strip()}
            entry = f"{reaction_class}  |  {basename}"
            if entry not in existing:
                with open(_UNKNOWN_CLASS_LOG, "a", encoding="utf-8") as f:
                    f.write(entry + "\n")
                print(f"[PostProcessor] Unknown reaction_class logged: {reaction_class!r}")
        except Exception:
            pass  # logging failure must not break the pipeline

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
