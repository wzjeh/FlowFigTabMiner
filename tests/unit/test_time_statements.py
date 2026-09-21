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


def test_various_substituents_is_not_a_sweep():
    st = find_residence_time_statements("The reactions of the styrene oxides bearing various substituents were carried out (tR = 23.8 s).")
    assert [c["value_s"] for c in fixed_candidates(st)] == [23.8]
    st = find_residence_time_statements("The residence time was varied between 0.5 and 6 s. Yields at tR = 2 s were highest.")
    assert [c["value_s"] for c in fixed_candidates(st)] == [2.0]


def test_temperature_time_pair_shorthand():
    st = find_residence_time_statements("Using the optimized conditions (-78 °C, 0.8 s), the reactions with various electrophiles were examined.")
    assert [c["value_s"] for c in fixed_candidates(st)] == [0.8]


def test_value_then_symbol_in_parentheses():
    st = find_residence_time_statements("anion 8 reacted with 2 after only 30 ms (τ1) of its formation, and was quenched after an additional 2.5 ms (τ2).")
    assert [c["value_s"] for c in fixed_candidates(st)] == [0.03, 0.0025]


def test_empty_and_no_match():
    assert find_residence_time_statements("") == []
    assert find_residence_time_statements("The residence time was varied.") == []
    assert fixed_candidates([]) == []


def test_step_number_from_symbol():
    st = find_residence_time_statements("The mixture was passed through R1 (tR1 = 0.05 s) and then R2 (tR2 = 2.3 s); τ2 was 30 ms in a later run.")
    assert [(c["value_s"], c["step"]) for c in fixed_candidates(st)] == [(0.05, 1), (2.3, 2), (0.03, 2)]


def test_step_number_from_in_reactor_phrase():
    st = find_residence_time_statements("electrophile (2.0 equivalent in THF), residence time in R2=0.003 s. Isolated yields.")
    assert [(c["value_s"], c["step"]) for c in fixed_candidates(st)] == [(0.003, 2)]


def test_residence_time_of_reactor_form():
    st = find_residence_time_statements("optimized conditions: temperature: 20 °C, residence time of R1: 0.01 s, residence time of R2: 2.3 s (See the supporting information).")
    assert [(c["value_s"], c["step"]) for c in fixed_candidates(st)] == [(0.01, 1), (2.3, 2)]


def test_step_from_enclosing_reactor_parenthesis():
    text = ("microtube reactor: R1 (ø = 500 μm, l = 3.5 cm (tR: 0.057 s)), R2 (ø = 1000 μm, l = 200 cm (tR: 9.8 s)), R3 (ø = 1000 μm (tR: 1.8 s)). "
            "The mixture was passed through R2 (f=1000 mm, L=50 cm, Rt=2.31 s). At 70 °C the maximum was at Rt=0.8 s.")
    assert [(c["value_s"], c["step"]) for c in fixed_candidates(find_residence_time_statements(text))] == \
        [(0.057, 1), (9.8, 2), (1.8, 3), (2.31, 2), (0.8, None)]


def test_bare_time_closing_a_reactor_parenthesis():
    st = find_residence_time_statements("The resulting mixture was passed through R2 (f=1000 mm, L=50 cm, 2.2 s). The reaction temperature (T) was controlled.")
    assert [(c["value_s"], c["step"]) for c in fixed_candidates(st)] == [(2.2, 2)]
    assert find_residence_time_statements("R1 (f=1000 mm, L=50 cm) at 20 °C for 5 min") == []
