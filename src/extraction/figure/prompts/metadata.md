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
7. **``legend_markers``**: for EVERY entry of ``legend_series_names`` give
   one object ``{name, color, marker}`` describing how that entry's marker
   is *printed*: ``color`` is a single plain colour word (red, blue, green,
   black, orange, purple, brown, pink, gray, cyan, yellow, magenta); for a
   monochrome legend that differs only by fill use ``open`` / ``filled``;
   ``marker`` is the shape word (circle, square, triangle, diamond, line,
   bar).  ``name`` must repeat the series name exactly.  If you are not
   sure of the colour, set ``color`` to ``null`` — never guess.  Leave the
   list empty when there is no legend.
