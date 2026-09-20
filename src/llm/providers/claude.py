"""Claude provider (Anthropic SDK).

The only file that imports ``anthropic``; everything else goes through
``LLMProvider`` / ``VLMProvider``.  Mirrors ``gemini.py``: disk response cache
keyed on (model, prompts, image bytes, schema name), global concurrency
slot, exponential backoff on rate limits.

Environment
-----------
- ``ANTHROPIC_API_KEY`` (required unless a client is injected).
- Model id from the typed config (``claude-sonnet-5`` by default in
  config.yaml; ``claude-sonnet-4-6`` also works).

Notes
-----
- Current-generation Claude models reject sampling parameters, so
  ``cfg.temperature`` is NOT forwarded; determinism comes from structured
  outputs + low effort.
- Structured mode uses ``client.messages.parse(output_format=<pydantic>)``;
  free mode uses ``messages.create`` and a balanced-JSON extraction.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from typing import Any, Optional, Type

from pydantic import BaseModel

from src.llm.cache import ResponseCache
from src.llm.concurrency import acquire_llm_slot
from src.llm.config import LLMConfig, VLMConfig
from src.llm.errors import LLMProviderError, ParseError, RateLimitError
from src.llm.providers._registry import register_llm_provider, register_vlm_provider
from src.llm.providers.base import LLMProvider, VLMProvider
from src.llm.providers.gemini import _backoff_call, _extract_balanced_json, _strip_json_fence
from src.llm.types import ChatMessage, LLMResponse, Role, VLMImage

logger = logging.getLogger(__name__)

_ENV_VAR = "ANTHROPIC_API_KEY"
DEFAULT_EFFORT = "low"      # reading / extraction calls: cheap and deterministic


def _load_sdk():
    try:
        import anthropic  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover
        raise LLMProviderError("anthropic SDK not installed (pip install anthropic)") from exc
    return anthropic


@register_llm_provider("claude")
@register_vlm_provider("claude")
class ClaudeProvider(LLMProvider, VLMProvider):
    def __init__(self, api_key: str | None = None, cache: ResponseCache | None = None, client: Any = None):
        self._sdk = None
        if client is None:
            key = api_key or os.environ.get(_ENV_VAR)
            if not key:
                raise LLMProviderError(f"{_ENV_VAR} not set; refusing to construct ClaudeProvider")
            self._sdk = _load_sdk()
            client = self._sdk.Anthropic(api_key=key)
        self._client = client
        self._cache = cache or ResponseCache()

    @property
    def name(self) -> str:
        return "claude"

    # ------------------------------------------------------------------ errors
    def _wrap_and_raise(self, exc: Exception) -> None:
        name = type(exc).__name__
        status = getattr(exc, "status_code", None)
        if name == "RateLimitError" or status == 429 or "rate" in str(exc).lower():
            raise RateLimitError(str(exc)) from exc
        raise LLMProviderError(f"claude call failed: {exc}") from exc

    @staticmethod
    def _usage(resp: Any) -> tuple[Optional[int], Optional[int]]:
        u = getattr(resp, "usage", None)
        return (getattr(u, "input_tokens", None), getattr(u, "output_tokens", None)) if u else (None, None)

    @staticmethod
    def _text(resp: Any) -> str:
        parts = []
        for block in getattr(resp, "content", []) or []:
            if getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", "") or "")
        return "\n".join(parts).strip()

    # ------------------------------------------------------------------ LLM
    def chat(self, messages: list[ChatMessage], cfg: LLMConfig) -> LLMResponse:
        system_parts = [m.content for m in messages if m.role == Role.SYSTEM]
        turns = [{"role": "user" if m.role == Role.USER else "assistant", "content": m.content}
                 for m in messages if m.role != Role.SYSTEM]
        cache_key = self._cache.key("claude.chat", cfg.model, cfg.max_output_tokens, [m.model_dump() for m in messages])
        if (hit := self._cache.get(cache_key)) is not None:
            logger.info("claude.chat cache hit (%s)", cfg.model)
            return LLMResponse(**hit, cache_hit=True)
        kwargs: dict[str, Any] = dict(model=cfg.model, max_tokens=cfg.max_output_tokens, messages=turns,
                                      output_config={"effort": DEFAULT_EFFORT})
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)

        def _call() -> Any:
            with acquire_llm_slot():
                try:
                    return self._client.messages.create(**kwargs)
                except Exception as exc:
                    self._wrap_and_raise(exc)

        t0 = time.perf_counter()
        resp, retry_count = _backoff_call(_call, max_retries=cfg.max_retries, label="claude.chat")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        text = self._text(resp)
        if not text:
            raise LLMProviderError(f"claude returned empty completion (model={cfg.model}, stop={getattr(resp, 'stop_reason', None)})")
        tin, tout = self._usage(resp)
        out = LLMResponse(text=text, model=cfg.model, tokens_in=tin, tokens_out=tout, latency_ms=elapsed_ms,
                          cache_hit=False, finish_reason=getattr(resp, "stop_reason", None), retry_count=retry_count)
        self._cache.set(cache_key, out.model_dump(exclude={"cache_hit"}))
        logger.info("claude.chat model=%s latency=%.0fms tokens_in=%s tokens_out=%s finish=%s retry=%d",
                    cfg.model, elapsed_ms, tin, tout, out.finish_reason, retry_count)
        return out

    # ------------------------------------------------------------------ VLM
    def inspect(self, image: VLMImage, system_prompt: str, user_prompt: str, cfg: VLMConfig,
                response_schema: Optional[Type[BaseModel]] = None) -> tuple[LLMResponse, dict]:
        img_bytes = image.path.read_bytes()
        schema_tag = response_schema.__name__ if response_schema else "free"
        cache_key = self._cache.key("claude.inspect", cfg.model, cfg.max_output_tokens, system_prompt, user_prompt,
                                    img_bytes, schema_tag)
        if (hit := self._cache.get(cache_key)) is not None:
            logger.info("claude.inspect cache hit (%s, schema=%s)", cfg.model, schema_tag)
            return LLMResponse(**hit["meta"], cache_hit=True), hit["parsed"]
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": image.mime_type,
                                         "data": base64.standard_b64encode(img_bytes).decode("utf-8")}},
            {"type": "text", "text": user_prompt},
        ]
        kwargs: dict[str, Any] = dict(model=cfg.model, max_tokens=cfg.max_output_tokens,
                                      messages=[{"role": "user", "content": content}],
                                      output_config={"effort": DEFAULT_EFFORT})
        if system_prompt:
            kwargs["system"] = system_prompt

        def _call() -> Any:
            with acquire_llm_slot():
                try:
                    if response_schema is not None:
                        return self._client.messages.parse(output_format=response_schema, **kwargs)
                    return self._client.messages.create(**kwargs)
                except Exception as exc:
                    self._wrap_and_raise(exc)

        t0 = time.perf_counter()
        resp, retry_count = _backoff_call(_call, max_retries=cfg.max_retries, label="claude.inspect")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if getattr(resp, "stop_reason", None) == "refusal":
            raise LLMProviderError(f"claude refused the request (model={cfg.model})")
        if response_schema is not None:
            parsed_obj = getattr(resp, "parsed_output", None)
            if parsed_obj is None:
                raise ParseError("claude structured output missing parsed_output", raw_text=self._text(resp))
            parsed = parsed_obj.model_dump() if hasattr(parsed_obj, "model_dump") else dict(parsed_obj)
        else:
            raw = self._text(resp)
            cand = _strip_json_fence(raw)
            body = _extract_balanced_json(cand) or cand
            import json as _json
            try:
                parsed = _json.loads(body)
            except Exception as exc:
                raise ParseError(f"claude free-mode JSON parse failed: {exc}", raw_text=raw) from exc
        tin, tout = self._usage(resp)
        meta = LLMResponse(text="", model=cfg.model, tokens_in=tin, tokens_out=tout, latency_ms=elapsed_ms,
                           cache_hit=False, finish_reason=getattr(resp, "stop_reason", None), retry_count=retry_count)
        self._cache.set(cache_key, {"meta": meta.model_dump(exclude={"cache_hit"}), "parsed": parsed})
        logger.info("claude.inspect model=%s image=%s schema=%s latency=%.0fms tokens_in=%s tokens_out=%s finish=%s retry=%d",
                    cfg.model, image.path.name, schema_tag, elapsed_ms, tin, tout, meta.finish_reason, retry_count)
        return meta, parsed
