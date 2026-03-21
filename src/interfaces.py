"""
Core interface contracts for FlowFigTabMiner components.

Using typing.Protocol (structural subtyping) so existing classes satisfy
these interfaces without any inheritance changes.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable, Any


# ── Extractor ─────────────────────────────────────────────────────────────────

@runtime_checkable
class Extractor(Protocol):
    """Processes a single image/PDF region and returns structured data."""

    def process(self, image_path: str, output_dir: str, **kwargs: Any) -> dict:
        """
        Args:
            image_path: Absolute path to the input image crop.
            output_dir:  Directory where outputs (CSV, JSON, debug images) should be written.
            **kwargs:    Extractor-specific options.

        Returns:
            A dict with at least:
              - "status": str  — "success" | "filtered" | "error"
              - "output_path": str | None  — primary output file (CSV or JSON)
        """
        ...


# ── Recognizer ────────────────────────────────────────────────────────────────

@runtime_checkable
class Recognizer(Protocol):
    """Detects objects / regions within an image and returns detection results."""

    def detect(self, image_path: str, **kwargs: Any) -> list[dict]:
        """
        Args:
            image_path: Absolute path to the input image.
            **kwargs:   Model-specific options (confidence threshold, device, …).

        Returns:
            List of detection dicts. Each dict must contain at least:
              - "label": str   — class name
              - "bbox": list   — [x1, y1, x2, y2] in pixel coordinates
              - "score": float — confidence score in [0, 1]
        """
        ...


# ── Assembler ─────────────────────────────────────────────────────────────────

@runtime_checkable
class Assembler(Protocol):
    """Aggregates per-source evidence into a unified output record."""

    def assemble(self, evidence: list[dict], **kwargs: Any) -> dict:
        """
        Args:
            evidence: List of per-figure/table evidence dicts produced by Extractors.
            **kwargs: Assembler-specific options.

        Returns:
            A single aggregated record dict ready for downstream post-processing.
        """
        ...
