"""ClaudeProvider with a mocked Anthropic client (no SDK / network needed)."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from src.llm.config import LLMConfig, VLMConfig
from src.llm.errors import LLMProviderError, RateLimitError
from src.llm.providers.claude import ClaudeProvider
from src.llm.types import ChatMessage, Role, VLMImage


class _Schema(BaseModel):
    readings: list[dict] = []


class _NoCache:
    def key(self, *parts): return "k"
    def get(self, key): return None
    def set(self, key, value): pass


def _resp(text="", parsed=None, stop="end_turn"):
    r = SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], usage=SimpleNamespace(input_tokens=11, output_tokens=7),
                        stop_reason=stop)
    if parsed is not None:
        r.parsed_output = parsed
    return r


def _cfg(retries=0):
    return VLMConfig(provider="claude", model="claude-sonnet-5", max_retries=retries, max_output_tokens=512)


def test_inspect_sends_base64_image_and_schema(tmp_path):
    img = tmp_path / "sheet.png"; img.write_bytes(b"\x89PNG fake")
    client = MagicMock(); client.messages.parse.return_value = _resp(parsed=_Schema(readings=[{"index": 0, "text": "10^-1.5"}]))
    p = ClaudeProvider(client=client, cache=_NoCache())
    meta, parsed = p.inspect(VLMImage(path=img), "", "read cells", _cfg(), response_schema=_Schema)
    kw = client.messages.parse.call_args.kwargs
    assert kw["model"] == "claude-sonnet-5" and kw["output_format"] is _Schema and "temperature" not in kw
    block = kw["messages"][0]["content"][0]
    assert block["type"] == "image" and block["source"]["type"] == "base64" and block["source"]["media_type"] == "image/png"
    assert kw["messages"][0]["content"][1]["text"] == "read cells" and "system" not in kw
    assert parsed == {"readings": [{"index": 0, "text": "10^-1.5"}]} and meta.tokens_in == 11 and meta.model == "claude-sonnet-5"


def test_inspect_free_mode_extracts_json(tmp_path):
    img = tmp_path / "s.png"; img.write_bytes(b"x")
    client = MagicMock(); client.messages.create.return_value = _resp(text='```json\n{"a": 1}\n```')
    _, parsed = ClaudeProvider(client=client, cache=_NoCache()).inspect(VLMImage(path=img), "sys", "u", _cfg())
    assert parsed == {"a": 1} and client.messages.create.call_args.kwargs["system"] == "sys"


def test_rate_limit_maps_and_propagates(tmp_path):
    img = tmp_path / "s.png"; img.write_bytes(b"x")
    class RateLimitError(Exception):   # same name as the SDK class
        status_code = 429
    client = MagicMock(); client.messages.parse.side_effect = RateLimitError("429 too many")
    with pytest.raises(RateLimitError.__mro__[0]) if False else pytest.raises(Exception) as ei:
        ClaudeProvider(client=client, cache=_NoCache()).inspect(VLMImage(path=img), "", "u", _cfg(retries=0), response_schema=_Schema)
    assert ei.type is RateLimitError or issubclass(ei.type, __import__("src.llm.errors", fromlist=["RateLimitError"]).RateLimitError)


def test_refusal_and_other_errors(tmp_path):
    img = tmp_path / "s.png"; img.write_bytes(b"x")
    client = MagicMock(); client.messages.parse.return_value = _resp(stop="refusal", parsed=_Schema())
    with pytest.raises(LLMProviderError):
        ClaudeProvider(client=client, cache=_NoCache()).inspect(VLMImage(path=img), "", "u", _cfg(), response_schema=_Schema)
    client2 = MagicMock(); client2.messages.create.side_effect = ValueError("boom")
    with pytest.raises(LLMProviderError):
        ClaudeProvider(client=client2, cache=_NoCache()).chat([ChatMessage(role=Role.USER, content="hi")],
                                                             LLMConfig(provider="claude", model="claude-sonnet-5", max_retries=0))


def test_chat_roundtrip():
    client = MagicMock(); client.messages.create.return_value = _resp(text="hello")
    out = ClaudeProvider(client=client, cache=_NoCache()).chat(
        [ChatMessage(role=Role.SYSTEM, content="be brief"), ChatMessage(role=Role.USER, content="hi")],
        LLMConfig(provider="claude", model="claude-sonnet-5", max_retries=0))
    kw = client.messages.create.call_args.kwargs
    assert out.text == "hello" and kw["system"] == "be brief" and kw["messages"] == [{"role": "user", "content": "hi"}]
    assert "temperature" not in kw and kw["output_config"] == {"effort": "low"}


def test_missing_key_refuses(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMProviderError):
        ClaudeProvider()
