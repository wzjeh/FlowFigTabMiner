"""Numeric condition fields are numbers or None in the output (never the
string "9.0" the LLM sometimes writes), at inheritance and in the final pass.
"""
from src.adjudication.post_processor import (
    NUMERIC_CONDITION_FIELDS,
    _scalar,
    _to_float,
    inherit_conditions,
)
from src.adjudication.local_vars_builder import LocalVarsBuilder


def test_to_float_parses_numeric_strings_and_rejects_the_rest():
    assert _to_float("9.0") == 9.0
    assert _to_float(" -78 ") == -78.0
    assert _to_float("−78") == -78.0          # unicode minus
    assert _to_float("1,200") == 1200.0
    assert _to_float("20 [src=paddleocr]") == 20.0  # provenance tag stripped
    assert _to_float("n.d.") is None
    assert _to_float("rt") is None
    assert _to_float("missing") is None
    assert _to_float("") is None
    assert _to_float(None) is None
    assert _to_float(True) is None
    assert _to_float({"step1": 9.0}) is None
    assert _to_float(3) == 3 and isinstance(_to_float(3), int)
    assert _to_float(float("nan")) is None


def test_scalar_coerces_numeric_fields_only():
    assert _scalar("9.0", "residence_time_2_s") == 9.0
    assert _scalar("rt", "temperature_C") is None
    assert _scalar("THF", "solvent") == "THF"
    assert _scalar("THF [src=vlm]", "solvent") == "THF"
    assert _scalar(9.0) == 9.0
    assert set(NUMERIC_CONDITION_FIELDS) >= {"temperature_C", "residence_time_s", "residence_time_2_s"}


def test_inherited_string_residence_time_becomes_float():
    rec = {"conditions": {}}
    local = {"fixed_conditions": {"residence_time_2_s": "9.0", "temperature_C": "-78", "solvent": "THF"}}
    out = inherit_conditions(rec, local, {}, {"source_local": 0, "paper_global": 0})
    c = out["conditions"]
    assert c["residence_time_2_s"] == 9.0 and isinstance(c["residence_time_2_s"], float)
    assert c["temperature_C"] == -78.0
    assert c["solvent"] == "THF"
    assert out["conditions_provenance"]["residence_time_2_s"] == "source_local"


def test_record_string_time_is_parsed_before_the_quote_guard():
    # a value on the record is kept only when it is one of the quoted statements
    local = {"fixed_conditions": {}, "time_guard": {"residence_time_s": {"column": False, "allowed": [0.5]}}}
    kept = inherit_conditions({"conditions": {"residence_time_s": "0.5"}}, local, {}, {"source_local": 0, "paper_global": 0})
    assert kept["conditions"]["residence_time_s"] == 0.5
    dropped = inherit_conditions({"conditions": {"residence_time_s": "9.0"}}, local, {}, {"source_local": 0, "paper_global": 0})
    assert dropped["conditions"]["residence_time_s"] is None
    assert dropped["conditions_provenance"]["residence_time_s"] == "removed_unquoted_residence_time"


def _guard(value, cands):
    result = {"fixed_conditions": {"residence_time_2_s": value, "reactor_type": "flow microreactor"}}
    return LocalVarsBuilder._enforce_time_candidates(result, cands, csv_head="Entry,Yield")["fixed_conditions"]


def test_time_guard_treats_string_and_number_alike():
    quoted = [{"value_s": 9.0, "step": 2, "scope": "near", "quote": "t_R2 = 9.0 s"}]
    assert _guard("9.0", quoted)["residence_time_2_s"] == 9.0
    assert _guard(9.0, quoted)["residence_time_2_s"] == 9.0
    # not a quoted statement: cleared whether written as number or string
    assert _guard("9.0", [])["residence_time_2_s"] is None
    assert _guard(9.0, [])["residence_time_2_s"] is None
    # a non-numeric string in a numeric field is cleared
    assert _guard("varied", quoted)["residence_time_2_s"] is None
