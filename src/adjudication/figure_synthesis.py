"""Deterministic figure-record synthesis (design step D).

Before: the per-source LLM was asked to *transcribe* every raw_data point into
a full reaction record ("emit EXACTLY N records").  For a 500-point heatmap
that is 500 hand-copied numbers per call — rounding, dropped points, swapped
axes, and ~30k output tokens.  On the 30-paper evaluation only 81.6 % of
heatmap records matched their raw point.

Now: the LLM answers ONE semantic question per figure — a ``figure_template``
(record template + axis map + series map) — and this module turns raw_data
into records with plain code.  Numbers are copied, never re-typed.

Template shape (produced by ``FigureTemplateBuilder``)::

    {
      "record_template": { <every _OUTPUT_SCHEMA field, per-point numerics null> },
      "axis_map": {"X": "conditions.residence_time_s",
                   "Y_Left": "conditions.temperature_C",
                   "Y_Right/Data_Value": "yield_pct"},          # field path or null
      "axis_transforms": {"X": {"scale": 1, "offset": 0}, ...},  # optional unit fixes
      "series_map": {"-78 °C": {"conditions.temperature_C": -78},
                     "3a": {"product_label": "3a"}},              # per-series fields
      "notes": "..."
    }
"""

from __future__ import annotations

import copy
import math
import statistics
from typing import Any, Dict, List, Optional

ALLOWED_PREFIXES = ("conditions.", "other_metrics.")
ALLOWED_TOP = {
    "yield_pct", "conversion_pct", "selectivity_pct", "ee_pct", "batch_yield_pct",
    "product_name", "product_label", "product_smiles", "reactant1_name", "reactant2_name",
    "reactant1_smiles", "reactant2_smiles", "entry_number", "yield_type", "stoichiometry",
    "diastereomeric_ratio",
}
RAW_COLS = ("X", "Y_Left", "Y_Right/Data_Value")
OUTCOME_FIELDS = {"yield_pct", "conversion_pct", "selectivity_pct", "ee_pct", "batch_yield_pct"}


def _valid_path(path: Optional[str]) -> bool:
    if not path or not isinstance(path, str):
        return False
    if path in ALLOWED_TOP:
        return True
    return any(path.startswith(p) and len(path) > len(p) for p in ALLOWED_PREFIXES)


def _set_path(rec: Dict[str, Any], path: str, value: Any) -> None:
    if "." in path:
        head, tail = path.split(".", 1)
        sub = rec.get(head)
        if not isinstance(sub, dict):
            sub = {}
            rec[head] = sub
        sub[tail] = value
    else:
        rec[path] = value


def _transform(value: float, tf: Optional[Dict[str, Any]]) -> float:
    if not isinstance(tf, dict):
        return value
    try:
        return value * float(tf.get("scale", 1) or 1) + float(tf.get("offset", 0) or 0)
    except Exception:
        return value


def snap_levels(values: List[Optional[float]], gap: float = 2.0) -> List[Optional[float]]:
    """Snap noisy axis readings to their discrete levels (heatmap rows).

    Coordinate mapping returns e.g. -77.6, -77.7, -0.02 for the -78 / 0 °C
    rows.  Values are 1-D clustered (sorted, split where the gap exceeds
    ``gap``) and every member takes the rounded cluster median.  Charts with
    continuous axes are unaffected in practice because their clusters are
    singletons.
    """
    idx = [i for i, v in enumerate(values) if isinstance(v, (int, float))]
    if not idx:
        return values
    order = sorted(idx, key=lambda i: values[i])
    out = list(values)
    cluster: List[int] = []

    def flush():
        if cluster:
            med = statistics.median(values[i] for i in cluster)
            snapped = float(round(med))
            for i in cluster:
                out[i] = snapped

    prev = None
    for i in order:
        v = values[i]
        if prev is not None and v - prev > gap:
            flush(); cluster = []
        cluster.append(i); prev = v
    flush()
    return out


def snap_levels_log(values: List[Optional[float]], gap: float = 0.12) -> List[Optional[float]]:
    """Cluster positive readings in log10 space (split where the gap exceeds
    ``gap`` decades) and give every member the cluster's geometric median.
    Removes per-point jitter along a heatmap's residence-time columns without
    moving columns onto tick positions (they often sit between ticks)."""
    idx = [i for i, v in enumerate(values) if isinstance(v, (int, float)) and not math.isnan(v) and v > 0]
    if not idx:
        return values
    logs = {i: math.log10(values[i]) for i in idx}
    order = sorted(idx, key=lambda i: logs[i])
    out = list(values)
    cluster: List[int] = []

    def flush():
        if cluster:
            med = statistics.median(logs[i] for i in cluster)
            for i in cluster:
                out[i] = float(10 ** med)

    prev = None
    for i in order:
        if prev is not None and logs[i] - prev > gap:
            flush(); cluster = []
        cluster.append(i); prev = logs[i]
    flush()
    return out


def snap_to_ticks(values: List[Optional[float]], ticks: List[float], tol: float = 3.0) -> List[Optional[float]]:
    """Snap each reading to the nearest OCR'd axis tick when within ``tol``;
    readings farther away are left as they are (or handled by ``snap_levels``)."""
    if not ticks:
        return values
    out = []
    for v in values:
        if isinstance(v, (int, float)) and not math.isnan(v):
            t = min(ticks, key=lambda x: abs(x - v))
            out.append(float(t) if abs(t - v) <= tol else v)
        else:
            out.append(v)
    return out


def _num(v: Any) -> Optional[float]:
    """Numeric or None; NaN (pandas → JSON) counts as missing."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return float(v)


def validate_template(tpl: Any) -> Optional[str]:
    """Return an error string if the template is unusable, else None."""
    if not isinstance(tpl, dict):
        return "template is not an object"
    if not isinstance(tpl.get("record_template"), dict):
        return "record_template missing"
    am = tpl.get("axis_map")
    if not isinstance(am, dict):
        return "axis_map missing"
    mapped = [c for c in RAW_COLS if _valid_path(am.get(c))]
    if not mapped:
        return "axis_map maps no raw column to a known field"
    return None


def synthesize_records(
    tpl: Dict[str, Any],
    raw_data: List[Dict[str, Any]],
    human_label: str,
    facts: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Build one record per raw point from the template.  Pure function."""
    facts = facts or {}
    template = tpl["record_template"]
    axis_map = {c: (p if _valid_path(p) else None) for c, p in (tpl.get("axis_map") or {}).items()}
    transforms = tpl.get("axis_transforms") or {}
    series_map = tpl.get("series_map") or {}
    if not isinstance(series_map, dict):
        series_map = {}

    # Heatmap rows are discrete levels → snap the Y_Left readings: to the
    # OCR'd tick values when the pipeline recorded them, else by clustering.
    y_left = [_num(p.get("Y_Left")) if isinstance(p, dict) else None for p in raw_data]
    if facts.get("chart_type") == "heatmap" and (axis_map.get("Y_Left") or "").startswith("conditions."):
        ticks = [t for t in (facts.get("y_left_ticks") or []) if isinstance(t, (int, float))]
        # Heatmap rows often sit BETWEEN axis ticks (-28 °C on a 0/-20/-40 axis):
        # snap to a tick when one is close, otherwise to the clustered row level.
        y_left = snap_levels(snap_to_ticks(y_left, ticks) if ticks else y_left)
    # Heatmap columns (residence time on a log axis): same-column jitter → column median.
    x_vals = [_num(p.get("X")) if isinstance(p, dict) else None for p in raw_data]
    if facts.get("chart_type") == "heatmap" and (axis_map.get("X") or "").startswith("conditions."):
        x_vals = snap_levels_log(x_vals)

    records: List[Dict[str, Any]] = []
    for i, pt in enumerate(raw_data):
        if not isinstance(pt, dict):
            continue
        rec = copy.deepcopy(template)
        rec.setdefault("conditions", {})
        rec.setdefault("other_metrics", {})
        if not isinstance(rec["conditions"], dict):
            rec["conditions"] = {}
        if not isinstance(rec["other_metrics"], dict):
            rec["other_metrics"] = {}

        data_fields = []                       # paths set from the point / its series (never guarded)
        series = pt.get("Series")
        smap = series_map.get(series) if series is not None else None
        if isinstance(smap, dict):
            for path, val in smap.items():
                # A legend series may set conditions / identities, never an
                # outcome constant: "<20%" → yield_pct=10 would fabricate a
                # measurement for every point whose cell label was not read.
                if _valid_path(path) and path not in OUTCOME_FIELDS:
                    if path.endswith(("_name", "_label", "_smiles")) and isinstance(val, (int, float)) and not isinstance(val, bool):
                        val = str(val)          # identities are strings
                    _set_path(rec, path, val)
                    data_fields.append(path)
        elif series and series != "Default":
            rec["other_metrics"]["series"] = series

        for col in RAW_COLS:
            path = axis_map.get(col)
            val = y_left[i] if col == "Y_Left" else (x_vals[i] if col == "X" else _num(pt.get(col)))
            if not path or val is None:
                continue
            _set_path(rec, path, _transform(float(val), transforms.get(col)))
            data_fields.append(path)

        rec["source_table_or_figure"] = human_label
        rec["__synthesized"] = True
        rec["__data_fields"] = sorted(set(data_fields))
        records.append(rec)
    return records
