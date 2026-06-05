"""Header row resolver — 1-shot LLM judge per table.

A single LLM call decides whether row 0 of the table is a header (and
how many header rows there are).  The judge sees three independent
readings of row 0 (PaddleOCR, Gemini cell extractor, and TATR's own
header_regions flag if available) plus row 1 from Gemini, applies the
strong-header-word heuristic in the prompt, and emits
``{header_row_count, confidence, reason}``.

The output is wrapped into a typed ``FieldValue`` by the assembler so
downstream code can audit the provenance.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from src.evidence.types import FieldSource, FieldValue
from src.llm.config import LLMConfig
from src.llm.providers.base import LLMProvider
from src.llm.types import ChatMessage, Role

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "header_judge.md"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text()


class HeaderJudgmentResponse(BaseModel):
    """Schema enforced on the LLM response."""

    header_row_count: Literal[0, 1, 2]
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str


class HeaderJudgment(BaseModel):
    """Wrapper carrying the parsed judgment plus provenance for the
    assembler.  Materializes into a FieldValue + a raw confidence."""

    field_value: FieldValue
    confidence: float


class HeaderResolver:
    """1-shot LLM judge to decide ``header_row_count`` ∈ {0, 1, 2}.

    Uses ``LLMProvider.chat`` rather than ``.inspect`` — no image needed
    once Gemini has already transcribed the rows; we just need a quick
    text judgement.

    Failure modes are explicit: if the LLM call errors or the judgment
    confidence falls below ``min_confidence``, the resolver returns
    ``header_row_count=1`` (the most common case) tagged with
    ``source=MISSING`` so downstream consumers know it's a default.
    """

    def __init__(
        self,
        llm: LLMProvider,
        cfg: LLMConfig,
        min_confidence: float = 0.5,
    ):
        self.llm = llm
        self.cfg = cfg
        self.min_confidence = min_confidence

    def resolve(
        self,
        paddle_row0: list[str] | None,
        vlm_row0: list[str] | None,
        vlm_row1: list[str] | None,
    ) -> HeaderJudgment:
        user_prompt = (
            "## Inputs\n\n"
            f"PaddleOCR reading of row 0:  {paddle_row0 if paddle_row0 else '(unavailable)'}\n"
            f"Gemini reading of row 0:     {vlm_row0 if vlm_row0 else '(unavailable)'}\n"
            f"Gemini reading of row 1:     {vlm_row1 if vlm_row1 else '(unavailable)'}\n\n"
            "## Task\n\n"
            "Decide ``header_row_count`` per the schema."
        )

        try:
            response = self.llm.chat(
                [
                    ChatMessage(role=Role.SYSTEM, content=_SYSTEM_PROMPT),
                    ChatMessage(role=Role.USER, content=user_prompt),
                ],
                self.cfg,
            )
        except Exception as exc:
            logger.warning("header_resolver: chat failed (%s); defaulting to 1", exc)
            return self._fallback(notes=f"llm call failed: {exc}")

        # Cheap manual JSON parse; LLM provider chat() doesn't use
        # structured outputs, but the prompt asks for strict JSON.  Fall
        # back gracefully.
        import json
        from src.llm.json_utils import sanitize_json_text

        try:
            parsed = json.loads(sanitize_json_text(response.text))
            judgement = HeaderJudgmentResponse.model_validate(parsed)
        except Exception as exc:
            logger.warning("header_resolver: parse failed (%s); defaulting to 1", exc)
            return self._fallback(notes=f"unparseable response: {response.text[:200]}")

        if judgement.confidence < self.min_confidence:
            logger.info(
                "header_resolver low confidence (%.2f < %.2f) — defaulting to 1 (reason: %s)",
                judgement.confidence, self.min_confidence, judgement.reason,
            )
            return self._fallback(
                notes=f"low confidence ({judgement.confidence:.2f}): {judgement.reason}",
            )

        return HeaderJudgment(
            field_value=FieldValue(
                value=judgement.header_row_count,
                source=FieldSource.LLM_JUDGE,
                model_id=response.model,
                latency_ms=response.latency_ms,
                notes=judgement.reason,
            ),
            confidence=judgement.confidence,
        )

    @staticmethod
    def _fallback(notes: str) -> HeaderJudgment:
        return HeaderJudgment(
            field_value=FieldValue(
                value=1,
                source=FieldSource.MISSING,
                notes=notes,
            ),
            confidence=0.0,
        )
