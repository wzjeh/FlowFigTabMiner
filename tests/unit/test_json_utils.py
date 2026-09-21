"""sanitize_json_text: string-aware structure and comment handling."""
import json

from src.llm.json_utils import sanitize_json_text


def test_url_inside_string_is_not_a_comment():
    raw = '```json\n[{"paper_doi": "https://doi.org/10.1002/anie.201713031", "yield_pct": 87}]\n```'
    assert json.loads(sanitize_json_text(raw)) == [{"paper_doi": "https://doi.org/10.1002/anie.201713031", "yield_pct": 87}]


def test_brackets_inside_strings_do_not_break_array_extraction():
    raw = 'Here you go:\n[{"source_table_or_figure": "Table 4 (p.7) — electrophiles.[a", "note": "Yield [%]"}, {"x": 1}]'
    out = json.loads(sanitize_json_text(raw))
    assert isinstance(out, list) and len(out) == 2 and out[0]["note"] == "Yield [%]"


def test_real_comments_and_trailing_commas_still_removed():
    raw = '{"a": 1, // comment\n "b": [1, 2,], /* block */ "c": "x"}'
    assert json.loads(sanitize_json_text(raw)) == {"a": 1, "b": [1, 2], "c": "x"}


def test_truncated_array_recovers_complete_records():
    raw = '[{"a": 1}, {"a": 2}, {"a": 3, "b": "cut off'
    assert json.loads(sanitize_json_text(raw)) == [{"a": 1}, {"a": 2}]
