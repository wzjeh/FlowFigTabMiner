"""Figure / table inspection (paper modules 6 and 12).

Public surface
--------------
- ``FigureInspector``  — wraps a ``VLMProvider`` and a ``PointMatcher``.
- ``TableInspector``   — wraps a ``VLMProvider`` and a ``CellMatcher``.
- ``PointMatcher`` / ``CellMatcher`` ABCs + default implementations.

Typical usage in the pipeline assembly point (``src/pipeline/main.py``)::

    vlm = get_vlm_provider("gemini")
    fig_inspector = FigureInspector(
        vlm=vlm,
        matcher=NearestPointMatcher(tol=0.05),
        cfg=load_vlm_config("config.yaml"),
    )
    tab_inspector = TableInspector(
        vlm=vlm,
        matcher=ExactCellMatcher(),
        cfg=load_vlm_config("config.yaml"),
    )
"""

from __future__ import annotations

from src.llm.inspectors.figure import FigureInspector
from src.llm.inspectors.matchers import (
    CellMatcher,
    ExactCellMatcher,
    NearestPointMatcher,
    PointMatcher,
)
from src.llm.inspectors.table import TableInspector

__all__ = [
    "FigureInspector",
    "TableInspector",
    "PointMatcher",
    "CellMatcher",
    "NearestPointMatcher",
    "ExactCellMatcher",
]
