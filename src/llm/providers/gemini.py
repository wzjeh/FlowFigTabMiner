"""Gemini provider (google-genai SDK).

This is the only file in the project that is allowed to ``import google.genai``.
All other code accesses Gemini through ``LLMProvider`` / ``VLMProvider``.

Environment
-----------
- ``GEMINI_API_KEY`` (required) — Google AI Studio API key.
- Model id is supplied by the typed ``LLMConfig`` / ``VLMConfig`` passed
  per-call.  The configured model name flows straight to the SDK.

Logging
-------
Every call emits a structured log record (``logger.info`` with extra
fields).  Token counts are populated from the SDK response when present.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time

from google import genai
from google.genai import types as genai_types

from src.llm.cache import ResponseCache
from src.llm.config import LLMConfig, VLMConfig
from src.llm.errors import (
    LLMProviderError,
    ParseError,
    RateLimitError,
)
from src.llm.providers._registry import register_llm_provider, register_vlm_provider
from src.llm.providers.base import LLMProvider, VLMProvider
from src.llm.types import ChatMessage, LLMResponse, Role, VLMImage

logger = logging.getLogger(__name__)

_ENV_VAR = "GEMINI_API_KEY"


def _strip_json_fence(text: str) -> str:
    """Remove the leading/trailing ```json fences Gemini sometimes emits."""
    out = text.strip()
    if out.startswith("```"):
        out = re.sub(r"^```(?:json)?\s*", "", out)
        out = re.sub(r"\s*```$", "", out)
    return out


def _extract_balanced_json(text: str) -> str | None:
    """Best-effort recovery: find the first balanced JSON object/array."""
    for opener, closer in (("{", "}"), ("[", "]")):
        i = text.find(opener)
        if i < 0:
            continue
        depth = 0
        for j in range(i, len(text)):
            ch = text[j]
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[i : j + 1]
    return None


@register_llm_provider("gemini")
@register_vlm_provider("gemini")
class GeminiProvider(LLMProvider, VLMProvider):
    """Concrete Gemini provider.

    A single instance can be reused across calls; the SDK ``Client``
    manages its own HTTP pool.
    """

    def __init__(self, api_key: str | None = None, cache: ResponseCache | None = None):
        key = api_key or os.environ.get(_ENV_VAR)
        if not key:
            raise LLMProviderError(
                f"{_ENV_VAR} not set; refusing to construct GeminiProvider"
            )
        self._client = genai.Client(api_key=key)
        self._cache = cache or ResponseCache()

    @property
    def name(self) -> str:
        return "gemini"

    # ------------------------------------------------------------------ LLM

    def chat(self, messages: list[ChatMessage], cfg: LLMConfig) -> LLMResponse:
        """Plain text chat (no images).

        Concatenates the system + user messages into a Gemini conversation
        ``Content`` list.  Caching key is the (model, full message stream)
        tuple; reordering a system prompt invalidates the cache, as it
        should.
        """
        system_parts = [m.content for m in messages if m.role == Role.SYSTEM]
        non_system = [m for m in messages if m.role != Role.SYSTEM]

        cache_key = self._cache.key(
            "gemini.chat",
            cfg.model,
            cfg.temperature,
            [m.model_dump() for m in messages],
        )
        if (hit := self._cache.get(cache_key)) is not None:
            logger.info("gemini.chat cache hit (%s)", cfg.model)
            return LLMResponse(**hit, cache_hit=True)

        contents: list[genai_types.Content] = []
        for m in non_system:
            role = "user" if m.role == Role.USER else "model"
            contents.append(
                genai_types.Content(role=role, parts=[genai_types.Part.from_text(text=m.content)])
            )

        config = genai_types.GenerateContentConfig(
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_output_tokens,
            system_instruction="\n\n".join(system_parts) if system_parts else None,
        )

        t0 = time.perf_counter()
        try:
            response = self._client.models.generate_content(
                model=cfg.model, contents=contents, config=config
            )
        except Exception as exc:
            self._wrap_and_raise(exc)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        text = (response.text or "").strip()
        if not text:
            raise LLMProviderError(f"gemini returned empty completion (model={cfg.model})")

        usage = getattr(response, "usage_metadata", None)
        tokens_in = getattr(usage, "prompt_token_count", None) if usage else None
        tokens_out = getattr(usage, "candidates_token_count", None) if usage else None

        out = LLMResponse(
            text=text,
            model=cfg.model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=elapsed_ms,
            cache_hit=False,
        )
        # Re-roundtripping LLMResponse → dict drops the cache_hit override,
        # which is what we want: a hit on a subsequent run is still a hit.
        self._cache.set(cache_key, out.model_dump(exclude={"cache_hit"}))
        logger.info(
            "gemini.chat model=%s latency=%.0fms tokens_in=%s tokens_out=%s",
            cfg.model,
            elapsed_ms,
            tokens_in,
            tokens_out,
        )
        return out

    # ------------------------------------------------------------------ VLM

    def inspect(
        self,
        image: VLMImage,
        system_prompt: str,
        user_prompt: str,
        cfg: VLMConfig,
    ) -> tuple[LLMResponse, dict]:
        """Send ``image`` + prompts, return ``(metadata, parsed_json)``.

        The cache key includes a hash of the image bytes so re-runs over
        the same crop avoid paying twice.
        """
        img_bytes = image.path.read_bytes()
        cache_key = self._cache.key(
            "gemini.inspect",
            cfg.model,
            cfg.temperature,
            system_prompt,
            user_prompt,
            img_bytes,
        )
        if (hit := self._cache.get(cache_key)) is not None:
            logger.info("gemini.inspect cache hit (%s)", cfg.model)
            meta = LLMResponse(**hit["meta"], cache_hit=True)
            return meta, hit["parsed"]

        config = genai_types.GenerateContentConfig(
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_output_tokens,
            system_instruction=system_prompt or None,
        )

        t0 = time.perf_counter()
        try:
            response = self._client.models.generate_content(
                model=cfg.model,
                contents=[
                    genai_types.Content(
                        role="user",
                        parts=[
                            genai_types.Part.from_bytes(
                                data=img_bytes, mime_type=image.mime_type
                            ),
                            genai_types.Part.from_text(text=user_prompt),
                        ],
                    )
                ],
                config=config,
            )
        except Exception as exc:
            self._wrap_and_raise(exc)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        raw = (response.text or "").strip()
        if not raw:
            raise LLMProviderError(f"gemini returned empty inspection (model={cfg.model})")

        parsed = self._parse_json(raw)

        usage = getattr(response, "usage_metadata", None)
        tokens_in = getattr(usage, "prompt_token_count", None) if usage else None
        tokens_out = getattr(usage, "candidates_token_count", None) if usage else None

        meta = LLMResponse(
            text=raw,
            model=cfg.model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=elapsed_ms,
            cache_hit=False,
        )
        self._cache.set(
            cache_key,
            {"meta": meta.model_dump(exclude={"cache_hit"}), "parsed": parsed},
        )
        logger.info(
            "gemini.inspect model=%s image=%s latency=%.0fms tokens_in=%s tokens_out=%s",
            cfg.model,
            image.path.name,
            elapsed_ms,
            tokens_in,
            tokens_out,
        )
        return meta, parsed

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _parse_json(raw: str) -> dict:
        text = _strip_json_fence(raw)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            recovered = _extract_balanced_json(text)
            if recovered is None:
                raise ParseError("gemini response is not JSON", raw_text=raw)
            try:
                return json.loads(recovered)
            except json.JSONDecodeError as exc:
                raise ParseError(f"could not decode recovered JSON: {exc}", raw_text=raw)

    @staticmethod
    def _wrap_and_raise(exc: Exception) -> None:
        """Convert SDK exceptions to typed errors."""
        msg = str(exc).lower()
        if "quota" in msg or "rate" in msg or "429" in msg:
            raise RateLimitError(str(exc)) from exc
        raise LLMProviderError(str(exc)) from exc
