"""Figure inspection (paper module 6).

Public surface
--------------
- ``FigureInspector``  — wraps a ``VLMProvider`` and a ``PointMatcher``.
- ``PointMatcher`` ABC + ``NearestPointMatcher`` default implementation.

Typical usage in the pipeline assembly point (``src/pipeline/main.py``)::

    vlm = get_vlm_provider("gemini")
    fig_inspector = FigureInspector(
        vlm=vlm,
        matcher=NearestPointMatcher(tol=0.05),
        cfg=load_vlm_config("config.yaml"),
    )
"""

from __future__ import annotations

from src.llm.inspectors.figure import FigureInspector
from src.llm.inspectors.matchers import NearestPointMatcher, PointMatcher

__all__ = ["FigureInspector", "PointMatcher", "NearestPointMatcher"]
