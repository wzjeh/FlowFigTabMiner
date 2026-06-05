"""Field-role catalogue.

Determines which side (pipeline vs VLM) is *empirically* better for a
given field type, based on the benchmark in paper SI Table S5:

- Figure numeric coordinates: pipeline F1 = 0.892 vs Gemini 0.575 → pipeline wins.
- Table cell text:             pipeline F1 = 0.827 vs Gemini 0.98 → VLM wins.
- Table SMILES:                pipeline F1 = 0.795 vs Gemini 0.944 → VLM wins.

The role drives ``ModalityRoutingPolicy`` — change a role, the policy
follows automatically, no policy code to touch.
"""

from __future__ import annotations

import enum
from typing import Any


class FieldRole(str, enum.Enum):
    """The semantic role of a field, which decides routing."""

    NUMERIC_PIPELINE = "numeric_pipeline"      # x, y, yield, T, t_R, …
    TEXTUAL_VLM = "textual_vlm"                # column header, footnote, free text
    STRUCTURAL_VLM = "structural_vlm"          # SMILES, compound name with structure
    IDENTIFIER_AGREE = "identifier_agree"      # compound labels (1a, 2b) — only trust when agree
    UNKNOWN = "unknown"                        # let policy decide; default = conflict


#: Field name → role mapping, used by ``classify_field``.
#:
#: Add an entry here, not in the policy code, to extend the catalogue.
#: Lower-case keys for case-insensitive matching against record column
#: names and dict keys.
FIELD_ROLES: dict[str, FieldRole] = {
    # ── figure numeric coordinates ─────────────────────────────────
    "x": FieldRole.NUMERIC_PIPELINE,
    "y": FieldRole.NUMERIC_PIPELINE,
    "value": FieldRole.NUMERIC_PIPELINE,           # heatmap Z value
    "y_left": FieldRole.NUMERIC_PIPELINE,
    "y_right": FieldRole.NUMERIC_PIPELINE,
    # ── figure textual annotations ─────────────────────────────────
    "series": FieldRole.TEXTUAL_VLM,
    "axis_label": FieldRole.TEXTUAL_VLM,
    "x_label": FieldRole.TEXTUAL_VLM,
    "y_label": FieldRole.TEXTUAL_VLM,
    "x_unit": FieldRole.TEXTUAL_VLM,
    "y_unit": FieldRole.TEXTUAL_VLM,
    # ── table numeric columns (common organolithium-paper headers) ─
    "t_r": FieldRole.NUMERIC_PIPELINE,
    "tr": FieldRole.NUMERIC_PIPELINE,
    "tr_s": FieldRole.NUMERIC_PIPELINE,
    "t (°c)": FieldRole.NUMERIC_PIPELINE,
    "t_c": FieldRole.NUMERIC_PIPELINE,
    "temperature": FieldRole.NUMERIC_PIPELINE,
    "yield": FieldRole.NUMERIC_PIPELINE,
    "yield (%)": FieldRole.NUMERIC_PIPELINE,
    "conversion": FieldRole.NUMERIC_PIPELINE,
    "selectivity": FieldRole.NUMERIC_PIPELINE,
    "ee": FieldRole.NUMERIC_PIPELINE,
    # ── table textual columns ─────────────────────────────────────
    "header": FieldRole.TEXTUAL_VLM,
    "column": FieldRole.TEXTUAL_VLM,
    "footnote": FieldRole.TEXTUAL_VLM,
    "compound_name": FieldRole.TEXTUAL_VLM,
    "solvent": FieldRole.TEXTUAL_VLM,
    "electrophile": FieldRole.TEXTUAL_VLM,
    # ── chemistry-structural ──────────────────────────────────────
    "smiles": FieldRole.STRUCTURAL_VLM,
    "substrate_smiles": FieldRole.STRUCTURAL_VLM,
    "product_smiles": FieldRole.STRUCTURAL_VLM,
    # ── ambiguous identifiers ─────────────────────────────────────
    "entry": FieldRole.IDENTIFIER_AGREE,
    "row": FieldRole.IDENTIFIER_AGREE,
    "col": FieldRole.IDENTIFIER_AGREE,
}


def _looks_numeric(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    try:
        float(str(value).strip().rstrip("%").rstrip())
        return True
    except (TypeError, ValueError):
        return False


def classify_field(key: str, value: Any | None = None) -> FieldRole:
    """Return the role for a ``(key, value)`` pair.

    The lookup is case-insensitive and ignores trailing units (``"T (°C)"``
    matches both ``t (°c)`` and ``t``).  If the catalogue has no entry, a
    last-ditch heuristic on ``value`` distinguishes numeric from textual.
    """
    if key is None:
        return FieldRole.UNKNOWN
    norm = key.lower().strip()
    if norm in FIELD_ROLES:
        return FIELD_ROLES[norm]
    # strip parenthesised unit, e.g. "yield (%)" → "yield"
    base = norm.split("(")[0].strip()
    if base in FIELD_ROLES:
        return FIELD_ROLES[base]
    # heuristic fallback
    if value is not None and _looks_numeric(value):
        return FieldRole.NUMERIC_PIPELINE
    return FieldRole.UNKNOWN
