"""Residence-time statements: regex extraction with verbatim quotes, sweep detection."""
from src.adjudication.time_statements import find_residence_time_statements, fixed_candidates


def test_finds_parenthetical_and_equals_forms_with_quotes():
    text = ("The flow rate is also very important. We carried out the present reaction with the micro flow "
            "system shown in Scheme 10 (−78 °C, Rt = 0.82 s). With flow rates of more than 3 mL/min the best "
            "result was obtained (Table 8). The mixture was passed through R2 (tR = 2.3 s) and quenched. "
            "Under the optimized conditions (T = -68 °C, tR1 = 0.5 s), high yields were obtained.")
    st = find_residence_time_statements(text)
    vals = [s["value_s"] for s in st]
    assert vals == [0.82, 2.3, 0.5]
    assert st[0]["quote"].startswith("We carried out") and "Rt = 0.82 s" in st[0]["quote"]
    assert not any(s["varied"] for s in st)


def test_units_and_sweeps():
    text = ("The residence time in R1 was 55 ms. The reaction was carried out varying the temperature and the "
            "residence time (tR1) from 0.05 to 6.3 s. Reaction time: 60 min in the flask. "
            "High yields were obtained even at 0 °C (tR = 0.057 s).")
    st = find_residence_time_statements(text)
    by = {s["value_s"]: s for s in st}
    assert abs(by[0.055]["value_s"] - 0.055) < 1e-9 and by[0.055]["varied"] is False
    assert 3600.0 not in by                                # "Reaction time: 60 min" is not a residence time
    assert by[0.057]["varied"] is False
    cands = fixed_candidates(st)
    assert [c["value_s"] for c in cands] == [0.055, 0.057]  # the sweep ("from 0.05 to 6.3 s") is never a candidate
    sweep = find_residence_time_statements("The residence time was varied (tR = 2.3 s to 9.8 s).")
    assert sweep and all(s["varied"] for s in sweep) and fixed_candidates(sweep) == []


def test_empty_and_no_match():
    assert find_residence_time_statements("") == []
    assert find_residence_time_statements("The residence time was varied.") == []
    assert fixed_candidates([]) == []
