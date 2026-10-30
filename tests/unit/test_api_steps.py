"""extract(): step bookkeeping in steps.json (skip done, refuse out of order, record failures)."""
import json

import pytest

import src.pipeline.api as api


@pytest.fixture
def fake(monkeypatch, tmp_path):
    """Replace every step with a recorder; no models, no LLM."""
    calls = []
    monkeypatch.setattr(api, "build_llm", lambda **kw: api.LLMStack(provider=None, llm_cfg=None, vlm_cfg=None,
                                                                     provider_name="claude", model="m"))
    monkeypatch.setattr(api, "prefilter", lambda paper: calls.append("prefilter") or True)
    monkeypatch.setattr(api, "detect", lambda paper: calls.append("detect") or 3)
    for name in ("figures", "tables", "schemes"):
        monkeypatch.setattr(api, name, lambda paper, llm, _n=name: calls.append(_n))
    monkeypatch.setattr(api, "context", lambda paper, llm: calls.append("context") or {"global_vars": 0.1, "local_vars": 0.2})
    monkeypatch.setattr(api, "assemble", lambda paper, llm, force=False: calls.append(("assemble", force)))
    monkeypatch.setattr(api, "normalize", lambda paper, smiles_lookup=True: calls.append("normalize") or "out.json")
    pdf = tmp_path / "10.1000_x.pdf"; pdf.write_bytes(b"%PDF")
    kw = dict(work_root=str(tmp_path / "w"), out_dir=str(tmp_path / "o"))
    return calls, str(pdf), kw


def _steps(kw):
    return json.load(open(f"{kw['work_root']}/10.1000_x/steps.json"))


def test_full_run_then_skip_then_redo(fake):
    calls, pdf, kw = fake
    s = api.extract(pdf, **kw)
    assert s["status"] == "ok" and s["normalized"] == "out.json" and len(calls) == 8
    rec = _steps(kw)
    assert list(rec) == list(api.STEPS) and rec["detect"]["crops"] == 3 and rec["figures"]["model"] == "m"
    timing = json.load(open(f"{kw['work_root']}/10.1000_x/timing.json"))
    assert list(timing)[:7] == ["status", "filter", "tfid", "figure", "table", "scheme", "global_vars"]
    calls.clear()
    assert api.extract(pdf, **kw)["steps"]["tables"] == "done" and calls == []
    api.extract(pdf, steps=("assemble",), redo=True, **kw)
    assert calls == [("assemble", True)]


def test_out_of_order_step_is_refused(fake):
    calls, pdf, kw = fake
    s = api.extract(pdf, steps=("prefilter", "detect", "assemble"), **kw)
    assert s["status"] == "blocked" and s["steps"]["assemble"] == "blocked" and "assemble" not in str(calls)


def test_prefilter_skip_stops_and_blocks(fake, monkeypatch):
    calls, pdf, kw = fake
    monkeypatch.setattr(api, "prefilter", lambda paper: False)
    assert api.extract(pdf, **kw)["status"] == "skipped" and calls == []
    assert api.extract(pdf, steps=("detect",), **kw)["status"] == "blocked"


def test_failed_step_is_recorded_and_retried(fake, monkeypatch):
    calls, pdf, kw = fake
    def boom(paper, llm):
        raise RuntimeError("figure stage crashed")
    monkeypatch.setattr(api, "figures", boom)
    s = api.extract(pdf, **kw)
    assert s["status"] == "failed" and s["steps"]["figures"] == "failed" and s["steps"]["normalize"] == "ok"
    assert "crashed" in _steps(kw)["figures"]["error"]
    monkeypatch.setattr(api, "figures", lambda paper, llm: calls.append("figures-again"))
    calls.clear()
    assert api.extract(pdf, **kw)["status"] == "ok"
    assert calls == ["figures-again", "tables", "schemes", "context", ("assemble", False), "normalize"]


def test_rerunning_a_step_invalidates_later_ones(fake):
    calls, pdf, kw = fake
    api.extract(pdf, **kw)
    api.extract(pdf, steps=("prefilter", "detect"), redo=True, **kw)
    assert list(_steps(kw)) == ["prefilter", "detect"]
    assert api.extract(pdf, steps=("assemble",), **kw)["status"] == "blocked"


def test_detect_failure_stops(fake, monkeypatch):
    calls, pdf, kw = fake
    monkeypatch.setattr(api, "detect", lambda paper: (_ for _ in ()).throw(RuntimeError("no pages")))
    s = api.extract(pdf, **kw)
    assert s["status"] == "failed" and "figures" not in s["steps"]


def test_unknown_step(fake):
    _, pdf, kw = fake
    with pytest.raises(ValueError):
        api.extract(pdf, steps=("detect", "ocr"), **kw)
