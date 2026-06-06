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
import random
import re
import time
from typing import Any, Callable, Optional, Type

from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel

from src.llm.cache import ResponseCache
from src.llm.concurrency import acquire_llm_slot
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


def _backoff_call(
    call: Callable[[], Any],
    *,
    max_retries: int,
    label: str,
) -> tuple[Any, int]:
    """Run ``call`` with exponential backoff on RateLimitError.

    Delays grow as 1 s, 2 s, 4 s, ... up to ``max_retries`` attempts.
    Other LLMProviderError types propagate immediately (no retry).

    Returns ``(result, retry_count)`` so callers can record how often a
    call was rate-limited.  ``retry_count`` is 0 on first-try success.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            return call(), attempt
        except RateLimitError as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            sleep_s = (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning(
                "%s rate-limited (attempt %d/%d), sleeping %.1fs before retry",
                label,
                attempt + 1,
                max_retries + 1,
                sleep_s,
            )
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


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
            cfg.max_output_tokens,
            cfg.thinking_budget,
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
            thinking_config=genai_types.ThinkingConfig(thinking_budget=cfg.thinking_budget),
        )

        def _call() -> Any:
            with acquire_llm_slot():
                try:
                    return self._client.models.generate_content(
                        model=cfg.model, contents=contents, config=config
                    )
                except Exception as exc:
                    self._wrap_and_raise(exc)

        t0 = time.perf_counter()
        response, retry_count = _backoff_call(_call, max_retries=cfg.max_retries, label="gemini.chat")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        text = (response.text or "").strip()
        if not text:
            raise LLMProviderError(f"gemini returned empty completion (model={cfg.model})")

        usage = getattr(response, "usage_metadata", None)
        tokens_in = getattr(usage, "prompt_token_count", None) if usage else None
        tokens_out = getattr(usage, "candidates_token_count", None) if usage else None
        tokens_thinking = getattr(usage, "thoughts_token_count", None) if usage else None
        finish_reason = None
        try:
            finish_reason = str(response.candidates[0].finish_reason)
        except Exception:
            pass

        out = LLMResponse(
            text=text,
            model=cfg.model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=elapsed_ms,
            cache_hit=False,
            finish_reason=finish_reason,
            retry_count=retry_count,
        )
        # Re-roundtripping LLMResponse → dict drops the cache_hit override,
        # which is what we want: a hit on a subsequent run is still a hit.
        self._cache.set(cache_key, out.model_dump(exclude={"cache_hit"}))
        logger.info(
            "gemini.chat model=%s latency=%.0fms tokens_in=%s tokens_out=%s thinking=%s finish=%s retry=%d",
            cfg.model,
            elapsed_ms,
            tokens_in,
            tokens_out,
            tokens_thinking,
            finish_reason,
            retry_count,
        )
        return out

    # ------------------------------------------------------------------ VLM

    def inspect(
        self,
        image: VLMImage,
        system_prompt: str,
        user_prompt: str,
        cfg: VLMConfig,
        response_schema: Optional[Type[BaseModel]] = None,
    ) -> tuple[LLMResponse, dict]:
        """Send ``image`` + prompts, return ``(metadata, parsed_json)``.

        Parameters
        ----------
        response_schema:
            Optional Pydantic model class.  When supplied, switches Gemini
            into structured-output mode (``response_mime_type="application/json"``
            + ``response_schema``) so the SDK guarantees the returned text
            is JSON conforming to the schema.  Callers can then validate
            with ``Model.model_validate(parsed)`` instead of writing
            tolerant regex.

        The cache key includes a hash of the image bytes plus the schema
        name so different schemas don't share entries.
        """
        img_bytes = image.path.read_bytes()
        schema_tag = response_schema.__name__ if response_schema else "free"
        cache_key = self._cache.key(
            "gemini.inspect",
            cfg.model,
            cfg.temperature,
            cfg.max_output_tokens,
            cfg.thinking_budget,
            system_prompt,
            user_prompt,
            img_bytes,
            schema_tag,
        )
        if (hit := self._cache.get(cache_key)) is not None:
            logger.info("gemini.inspect cache hit (%s, schema=%s)", cfg.model, schema_tag)
            meta = LLMResponse(**hit["meta"], cache_hit=True)
            return meta, hit["parsed"]

        config_kwargs: dict[str, Any] = dict(
            temperature=cfg.temperature,
            max_output_tokens=cfg.max_output_tokens,
            system_instruction=system_prompt or None,
            thinking_config=genai_types.ThinkingConfig(thinking_budget=cfg.thinking_budget),
        )
        if response_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema

        config = genai_types.GenerateContentConfig(**config_kwargs)

        contents = [
            genai_types.Content(
                role="user",
                parts=[
                    genai_types.Part.from_bytes(
                        data=img_bytes, mime_type=image.mime_type
                    ),
                    genai_types.Part.from_text(text=user_prompt),
                ],
            )
        ]

        def _call() -> Any:
            with acquire_llm_slot():
                try:
                    return self._client.models.generate_content(
                        model=cfg.model, contents=contents, config=config
                    )
                except Exception as exc:
                    self._wrap_and_raise(exc)

        t0 = time.perf_counter()
        response, retry_count = _backoff_call(_call, max_retries=cfg.max_retries, label="gemini.inspect")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        raw = (response.text or "").strip()
        if not raw:
            raise LLMProviderError(f"gemini returned empty inspection (model={cfg.model})")

        # Structured mode → SDK already guarantees JSON; just json.loads.
        # Free mode → fall back to the tolerant parser for fences / comments.
        if response_schema is not None:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ParseError(
                    f"gemini structured output not parseable as JSON (schema={schema_tag}): {exc}",
                    raw_text=raw,
                )
        else:
            parsed = self._parse_json(raw)

        usage = getattr(response, "usage_metadata", None)
        tokens_in = getattr(usage, "prompt_token_count", None) if usage else None
        tokens_out = getattr(usage, "candidates_token_count", None) if usage else None
        tokens_thinking = getattr(usage, "thoughts_token_count", None) if usage else None
        finish_reason = None
        try:
            finish_reason = str(response.candidates[0].finish_reason)
        except Exception:
            pass

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
            "gemini.inspect model=%s image=%s schema=%s latency=%.0fms tokens_in=%s tokens_out=%s thinking=%s finish=%s retry=%d",
            cfg.model,
            image.path.name,
            schema_tag,
            elapsed_ms,
            tokens_in,
            tokens_out,
            tokens_thinking,
            finish_reason,
            retry_count,
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
