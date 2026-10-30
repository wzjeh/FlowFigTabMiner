"""OpenAI provider (openai SDK, Chat Completions).

The only file that imports ``openai``; everything else goes through
``LLMProvider`` / ``VLMProvider``.  Mirrors ``claude.py``: disk response cache
keyed on (model, prompts, image bytes, schema name), global concurrency slot,
exponential backoff on rate limits and server errors.

Environment
-----------
- ``OPENAI_API_KEY`` (required unless a client is injected).
- Model id from the typed config (``llm.models.openai`` in config.yaml).

Notes
-----
- Reasoning models (gpt-5 family) reject sampling parameters, so
  ``cfg.temperature`` is NOT forwarded; ``reasoning_effort`` is low, as for
  Claude.  gpt-4.x models take no ``reasoning_effort``.
- Structured mode uses ``client.chat.completions.parse(response_format=<pydantic>)``;
  free mode uses ``create`` and a balanced-JSON extraction.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from typing import Any, Optional, Type

from pydantic import BaseModel

from src.llm.cache import ResponseCache
from src.llm.concurrency import acquire_llm_slot
from src.llm.config import LLMConfig, VLMConfig
from src.llm.errors import LLMProviderError, ParseError, RateLimitError, TransientError
from src.llm.providers._registry import register_llm_provider, register_vlm_provider
from src.llm.providers.base import LLMProvider, VLMProvider
from src.llm.providers._common import _backoff_call, _extract_balanced_json, _strip_json_fence
from src.llm.types import ChatMessage, LLMResponse, Role, VLMImage

logger = logging.getLogger(__name__)

_ENV_VAR = "OPENAI_API_KEY"
DEFAULT_EFFORT = "low"      # reading / extraction calls: cheap and steady


def _load_sdk():
    try:
        import openai  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover
        raise LLMProviderError("openai SDK not installed (pip install openai)") from exc
    return openai


@register_llm_provider("openai")
@register_vlm_provider("openai")
class OpenAIProvider(LLMProvider, VLMProvider):
    def __init__(self, api_key: str | None = None, cache: ResponseCache | None = None, client: Any = None):
        if client is None:
            key = api_key or os.environ.get(_ENV_VAR)
            if not key:
                raise LLMProviderError(f"{_ENV_VAR} not set; refusing to construct OpenAIProvider")
            client = _load_sdk().OpenAI(api_key=key)
        self._client = client
        self._cache = cache or ResponseCache()

    @property
    def name(self) -> str:
        return "openai"

    # ------------------------------------------------------------------ errors
    @staticmethod
    def _wrap_and_raise(exc: Exception) -> None:
        name = type(exc).__name__
        status = getattr(exc, "status_code", None)
        if name == "RateLimitError" or status == 429:
            raise RateLimitError(str(exc)) from exc
        if (status is not None and status >= 500) or name in ("APITimeoutError", "APIConnectionError", "InternalServerError"):
            raise TransientError(str(exc)) from exc      # one source must not be lost to a server hiccup
        raise LLMProviderError(f"openai call failed: {exc}") from exc

    @staticmethod
    def _request(model: str, max_tokens: int, messages: list) -> dict:
        kwargs: dict[str, Any] = dict(model=model, max_completion_tokens=max_tokens, messages=messages)
        if not model.startswith("gpt-4"):
            kwargs["reasoning_effort"] = DEFAULT_EFFORT
        return kwargs

    @staticmethod
    def _usage(resp: Any) -> tuple[Optional[int], Optional[int]]:
        u = getattr(resp, "usage", None)
        return (getattr(u, "prompt_tokens", None), getattr(u, "completion_tokens", None)) if u else (None, None)

    # ------------------------------------------------------------------ LLM
    def chat(self, messages: list[ChatMessage], cfg: LLMConfig) -> LLMResponse:
        cache_key = self._cache.key("openai.chat", cfg.model, cfg.max_output_tokens, [m.model_dump() for m in messages])
        if (hit := self._cache.get(cache_key)) is not None:
            logger.info("openai.chat cache hit (%s)", cfg.model)
            return LLMResponse(**hit, cache_hit=True)
        turns = [{"role": {Role.SYSTEM: "system", Role.USER: "user", Role.ASSISTANT: "assistant"}[m.role],
                  "content": m.content} for m in messages]
        kwargs = self._request(cfg.model, cfg.max_output_tokens, turns)

        def _call() -> Any:
            with acquire_llm_slot():
                try:
                    return self._client.with_options(timeout=cfg.timeout_s).chat.completions.create(**kwargs)
                except Exception as exc:
                    self._wrap_and_raise(exc)

        t0 = time.perf_counter()
        resp, retry_count = _backoff_call(_call, max_retries=cfg.max_retries, label="openai.chat")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        if not text:
            raise LLMProviderError(f"openai returned empty completion (model={cfg.model}, finish={choice.finish_reason})")
        tin, tout = self._usage(resp)
        out = LLMResponse(text=text, model=cfg.model, tokens_in=tin, tokens_out=tout, latency_ms=elapsed_ms,
                          cache_hit=False, finish_reason=choice.finish_reason, retry_count=retry_count)
        self._cache.set(cache_key, out.model_dump(exclude={"cache_hit"}))
        logger.info("openai.chat model=%s latency=%.0fms tokens_in=%s tokens_out=%s finish=%s retry=%d",
                    cfg.model, elapsed_ms, tin, tout, out.finish_reason, retry_count)
        return out

    # ------------------------------------------------------------------ VLM
    def inspect(self, image: VLMImage, system_prompt: str, user_prompt: str, cfg: VLMConfig,
                response_schema: Optional[Type[BaseModel]] = None) -> tuple[LLMResponse, dict]:
        img_bytes = image.path.read_bytes()
        schema_tag = response_schema.__name__ if response_schema else "free"
        cache_key = self._cache.key("openai.inspect", cfg.model, cfg.max_output_tokens, system_prompt, user_prompt,
                                    img_bytes, schema_tag)
        if (hit := self._cache.get(cache_key)) is not None:
            logger.info("openai.inspect cache hit (%s, schema=%s)", cfg.model, schema_tag)
            return LLMResponse(**hit["meta"], cache_hit=True), hit["parsed"]
        data_url = f"data:{image.mime_type};base64," + base64.standard_b64encode(img_bytes).decode("utf-8")
        turns = [{"role": "system", "content": system_prompt}] if system_prompt else []
        turns.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url}},
                                                  {"type": "text", "text": user_prompt}]})
        kwargs = self._request(cfg.model, cfg.max_output_tokens, turns)

        def _call() -> Any:
            with acquire_llm_slot():
                try:
                    client = self._client.with_options(timeout=cfg.timeout_s)
                    if response_schema is not None:
                        return client.chat.completions.parse(response_format=response_schema, **kwargs)
                    return client.chat.completions.create(**kwargs)
                except Exception as exc:
                    self._wrap_and_raise(exc)

        t0 = time.perf_counter()
        resp, retry_count = _backoff_call(_call, max_retries=cfg.max_retries, label="openai.inspect")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        choice = resp.choices[0]
        if getattr(choice.message, "refusal", None):
            raise LLMProviderError(f"openai refused the request (model={cfg.model})")
        raw = (choice.message.content or "").strip()
        if response_schema is not None:
            parsed_obj = getattr(choice.message, "parsed", None)
            if parsed_obj is None:
                raise ParseError(f"openai structured output missing (finish={choice.finish_reason})", raw_text=raw)
            parsed = parsed_obj.model_dump()
        else:
            body = _extract_balanced_json(_strip_json_fence(raw)) or raw
            try:
                parsed = json.loads(body)
            except Exception as exc:
                raise ParseError(f"openai free-mode JSON parse failed: {exc}", raw_text=raw) from exc
        tin, tout = self._usage(resp)
        meta = LLMResponse(text="", model=cfg.model, tokens_in=tin, tokens_out=tout, latency_ms=elapsed_ms,
                           cache_hit=False, finish_reason=choice.finish_reason, retry_count=retry_count)
        self._cache.set(cache_key, {"meta": meta.model_dump(exclude={"cache_hit"}), "parsed": parsed})
        logger.info("openai.inspect model=%s image=%s schema=%s latency=%.0fms tokens_in=%s tokens_out=%s finish=%s retry=%d",
                    cfg.model, image.path.name, schema_tag, elapsed_ms, tin, tout, meta.finish_reason, retry_count)
        return meta, parsed
