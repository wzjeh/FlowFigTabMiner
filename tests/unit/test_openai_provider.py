"""OpenAIProvider with a mocked OpenAI client (no network needed)."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from src.llm.config import LLMConfig, VLMConfig
from src.llm.errors import LLMProviderError, ParseError, RateLimitError, TransientError
from src.llm.providers.openai import OpenAIProvider
from src.llm.types import ChatMessage, Role, VLMImage


class _Schema(BaseModel):
    readings: list[str] = []


class _NoCache:
    def key(self, *parts): return "k"
    def get(self, key): return None
    def set(self, key, value): pass


def _client():
    c = MagicMock()
    c.with_options.return_value = c
    return c


def _resp(content="", parsed=None, finish="stop", refusal=None):
    msg = SimpleNamespace(content=content, parsed=parsed, refusal=refusal)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)],
                           usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7))


def _vcfg(model="gpt-5-nano"):
    return VLMConfig(provider="openai", model=model, max_retries=0, max_output_tokens=512, timeout_s=60.0)


def test_inspect_sends_data_url_and_schema(tmp_path):
    img = tmp_path / "s.png"; img.write_bytes(b"\x89PNG fake")
    client = _client(); client.chat.completions.parse.return_value = _resp(parsed=_Schema(readings=["10"]))
    meta, parsed = OpenAIProvider(client=client, cache=_NoCache()).inspect(
        VLMImage(path=img), "sys", "read", _vcfg(), response_schema=_Schema)
    kw = client.chat.completions.parse.call_args.kwargs
    assert kw["response_format"] is _Schema and kw["model"] == "gpt-5-nano" and "temperature" not in kw
    assert kw["reasoning_effort"] == "low" and kw["max_completion_tokens"] == 512
    assert kw["messages"][0] == {"role": "system", "content": "sys"}
    part = kw["messages"][1]["content"][0]
    assert part["type"] == "image_url" and part["image_url"]["url"].startswith("data:image/png;base64,")
    assert parsed == {"readings": ["10"]} and meta.tokens_in == 11 and meta.tokens_out == 7
    assert client.with_options.call_args.kwargs == {"timeout": 60.0}


def test_free_mode_json_and_gpt4_has_no_effort(tmp_path):
    img = tmp_path / "s.png"; img.write_bytes(b"x")
    client = _client(); client.chat.completions.create.return_value = _resp(content='```json\n{"a": 1}\n```')
    _, parsed = OpenAIProvider(client=client, cache=_NoCache()).inspect(VLMImage(path=img), "", "u", _vcfg("gpt-4.1-nano"))
    kw = client.chat.completions.create.call_args.kwargs
    assert parsed == {"a": 1} and "reasoning_effort" not in kw and kw["messages"][0]["role"] == "user"


def test_missing_parsed_and_refusal(tmp_path):
    img = tmp_path / "s.png"; img.write_bytes(b"x")
    client = _client(); client.chat.completions.parse.return_value = _resp(parsed=None, finish="length")
    with pytest.raises(ParseError):
        OpenAIProvider(client=client, cache=_NoCache()).inspect(VLMImage(path=img), "", "u", _vcfg(), response_schema=_Schema)
    client.chat.completions.parse.return_value = _resp(refusal="no")
    with pytest.raises(LLMProviderError):
        OpenAIProvider(client=client, cache=_NoCache()).inspect(VLMImage(path=img), "", "u", _vcfg(), response_schema=_Schema)


def test_chat_roundtrip():
    client = _client(); client.chat.completions.create.return_value = _resp(content="hello")
    out = OpenAIProvider(client=client, cache=_NoCache()).chat(
        [ChatMessage(role=Role.SYSTEM, content="be brief"), ChatMessage(role=Role.USER, content="hi")],
        LLMConfig(provider="openai", model="gpt-5-nano", max_retries=0))
    kw = client.chat.completions.create.call_args.kwargs
    assert out.text == "hello" and kw["messages"] == [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]


def test_error_mapping():
    class RateLimitErr(Exception):
        status_code = 429
    class InternalServerError(Exception):
        status_code = 503
    with pytest.raises(RateLimitError):
        OpenAIProvider._wrap_and_raise(RateLimitErr("slow down"))
    for exc in (InternalServerError("x"), type("APITimeoutError", (Exception,), {})("t")):
        with pytest.raises(TransientError):
            OpenAIProvider._wrap_and_raise(exc)
    with pytest.raises(LLMProviderError):
        OpenAIProvider._wrap_and_raise(ValueError("400 bad"))


def test_missing_key_refuses(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMProviderError):
        OpenAIProvider()
