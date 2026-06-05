You are transcribing a table from a chemistry paper.

The table grid has been pre-measured by a structural model: it has
exactly **{m} rows** and **exactly {n} columns**.

Your job is to transcribe every cell, including the header row(s).
Output a dense 2-D array of strings — **exactly {m} rows, each with
exactly {n} entries**.

Strict rules:

1. **Never skip a column.** Visually empty cells must be emitted as the
   empty string `""`.  Do not collapse them, do not shift later cells
   left.  The position of every value must match its column index in
   the printed table.
2. **Header rows count too.** If the table has a header row, transcribe
   it as the first row of ``rows``.  Whether row 0 is data or header
   will be decided by a separate downstream step; here you just
   transcribe what's visibly present.
3. **Cell text exactly as printed**, including units, parentheses,
   ``±`` markers, and footnote indicators like ``[a]`` or ``*``.
4. **Molecular structures** drawn inside a cell should be transcribed
   as canonical SMILES.  If the SMILES is uncertain, append ``[?]``.
   Do NOT write placeholders like ``"[structure]"``.
5. The output schema is enforced; the SDK will only accept JSON with the
   ``rows`` field set to a list of length {m}, each inner list of length
   {n}.  Anything else fails the request.
6. If a cell value spans multiple physical rows due to row-merging,
   repeat the value in each row it occupies.  Similarly for column
   spans.

Return ONLY the JSON object.
