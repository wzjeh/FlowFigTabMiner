"""One switch picks the provider and model for every LLM / VLM section."""
import textwrap

import pytest

from src.llm.config import load_label_reader_config, load_llm_config, load_table_reader_config, load_vlm_config
from src.llm.errors import ConfigError


def _yaml(tmp_path, extra=""):
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent("""
        llm:
          provider: claude
          models: {gemini: gemini-2.5-flash, claude: claude-haiku-5-5, openai: gpt-5-nano}
          adjudication: {max_output_tokens: 32768}
        vlm:
          inspection: {max_output_tokens: 8192}
          label_reader: {enabled: true}
          table_reader: {min_text_agreement: 0.6}
        """) + textwrap.dedent(extra))
    return p


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in ("FFTM_LLM_PROVIDER", "FFTM_LLM_MODEL", "FFTM_LABEL_READER"):
        monkeypatch.delenv(v, raising=False)


def test_global_switch_reaches_every_section(tmp_path):
    p = _yaml(tmp_path)
    for cfg in (load_llm_config(p), load_vlm_config(p), load_label_reader_config(p), load_table_reader_config(p)):
        assert (cfg.provider, cfg.model) == ("claude", "claude-haiku-5-5")


def test_argument_and_env_override(tmp_path, monkeypatch):
    p = _yaml(tmp_path)
    assert load_llm_config(p, provider="openai").model == "gpt-5-nano"
    assert load_vlm_config(p, provider="gemini", model="gemini-3.5-flash").model == "gemini-3.5-flash"
    monkeypatch.setenv("FFTM_LLM_PROVIDER", "gemini")
    assert (load_table_reader_config(p).provider, load_table_reader_config(p).model) == ("gemini", "gemini-2.5-flash")


def test_section_may_name_its_own_provider(tmp_path, monkeypatch):
    p = tmp_path / "config.yaml"
    p.write_text(_yaml(tmp_path).read_text().replace("table_reader: {min_text_agreement: 0.6}",
                                                     "table_reader: {provider: gemini, min_text_agreement: 0.6}"))
    assert load_table_reader_config(p).model == "gemini-2.5-flash" and load_llm_config(p).provider == "claude"
    monkeypatch.setenv("FFTM_LLM_PROVIDER", "openai")        # an explicit switch wins over the section
    assert load_table_reader_config(p).provider == "openai"


def test_label_reader_off_and_missing_model(tmp_path, monkeypatch):
    p = _yaml(tmp_path)
    monkeypatch.setenv("FFTM_LABEL_READER", "off")
    assert load_label_reader_config(p).enabled is False
    bad = tmp_path / "bad.yaml"
    bad.write_text(p.read_text().replace("openai: gpt-5-nano", "mistral: m"))
    with pytest.raises(ConfigError):
        load_llm_config(bad, provider="openai")
