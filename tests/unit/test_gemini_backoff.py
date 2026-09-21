"""Transient provider failures (500 / 503 / timeout) are retried like rate limits; other errors are not."""
import pytest

from src.llm.errors import LLMProviderError, RateLimitError, TransientError
from src.llm.providers import gemini
from src.llm.providers.gemini import GeminiProvider, _backoff_call


def test_wrap_maps_server_side_failures_to_transient():
    for msg, typ in (("500 INTERNAL. {'error': {'code': 500, 'message': 'An internal error has occurred'}}", TransientError),
                     ("503 UNAVAILABLE. The model is overloaded.", TransientError),
                     ("Request timed out", TransientError),
                     ("429 RESOURCE_EXHAUSTED quota", RateLimitError),
                     ("400 INVALID_ARGUMENT", LLMProviderError)):
        with pytest.raises(typ) as ei:
            GeminiProvider._wrap_and_raise(RuntimeError(msg))
        assert type(ei.value) is typ


def test_backoff_retries_transient_then_succeeds(monkeypatch):
    monkeypatch.setattr(gemini.time, "sleep", lambda s: None)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise TransientError("500 INTERNAL")
        return "ok"

    assert _backoff_call(flaky, max_retries=3, label="t") == ("ok", 2)
    with pytest.raises(LLMProviderError):
        _backoff_call(lambda: (_ for _ in ()).throw(LLMProviderError("400 bad request")), max_retries=3, label="t")
    calls.clear()
    with pytest.raises(TransientError):
        _backoff_call(lambda: (calls.append(1), (_ for _ in ()).throw(TransientError("503")))[1], max_retries=2, label="t")
    assert len(calls) == 3
