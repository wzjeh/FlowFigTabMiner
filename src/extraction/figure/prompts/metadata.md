You are reading a scientific figure from a chemistry paper.

Your job is to transcribe the **text annotations on the figure** — the
labels that are *visibly printed* on or around the chart.  You must NOT
infer anything that is not literally written, and you must NOT extract
data points or numerical values from the plot area.

Return ONLY a JSON object that conforms to the schema; the SDK is
already configured to enforce it.

Rules:

1. **Only what is printed.** If a unit or label is not visible, set it to
   ``null`` rather than guessing.
2. **Series names are chemical labels**, not colour descriptions.  If
   the legend says "3,4-dichloroaniline" emit that exact string.  Never
   emit colour-based labels like "Red Triangle" or "Cyan Circle".  If
   the legend is illegible, emit ``null`` for that series rather than
   inventing a name.
3. **Order in ``legend_series_names`` follows the legend layout**, top
   to bottom (or left to right when arranged horizontally).
4. **Axis units** are returned separately from labels.  Examples:
   - Axis text "Pressure (MPa)" → label="Pressure", unit="MPa"
   - Axis text "Yield / %"       → label="Yield",    unit="%"
   - Axis text "Temperature"      → label="Temperature", unit=null
5. **Do NOT extract data points or numerical values from the plot
   area.**  This call is *metadata only*.  Data points are extracted by
   a separate pipeline that reads pixel coordinates directly.
6. **Footnote** is any small text below the chart explaining symbols /
   conditions; transcribe verbatim if present, else ``null``.
