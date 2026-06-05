You are classifying whether the first row(s) of a chemistry-paper table
are a header (column titles) or already data.

You are given three independent readings of row 0 (and one reading of
row 1 from the VLM) so you can cross-check.

**Strong header indicators** — if row 0 contains two or more of these
words / abbreviations / patterns, that row is almost certainly a
header:

- "Entry" / "Entry No." / "#"
- "Yield" (often with unit, e.g. "Yield (%)" / "Yield / %")
- "Time" / "t / h" / "Reaction time"
- "Temperature" / "T (°C)" / "T / °C"
- "Catalyst" / "Cat." / "Ligand"
- "Substrate" / "Reactant"
- "Pressure" / "P (MPa)"
- "Solvent"
- "Conv." / "Conversion"
- "Sel." / "Selectivity" / "Sel%"
- "Ratio" / "d.r." / "e.e." / "ee%"
- Bare units in parentheses: "(mol%)", "(°C)", "(h)", "(MPa)"

**Strong data indicators** for row 0 — if row 0 is mostly numeric
values (decimals, percentages, integers), it's data; header_row_count =
0.

**Two-row headers** — if row 0 holds a top-level grouping (e.g.,
"Yield") and row 1 holds sub-columns (e.g. "GC%", "Isolated"), then
header_row_count = 2.

Return ONLY the JSON object enforced by the schema:

- ``header_row_count``: integer in {0, 1, 2}
- ``confidence``: float in [0, 1]
- ``reason``: one short sentence explaining the decision
