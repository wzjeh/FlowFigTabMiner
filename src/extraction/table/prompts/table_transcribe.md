You are transcribing ONE table cropped from a chemistry paper. The crop may
also show the table caption above, a reaction scheme drawing, and footnotes
below. Return ONLY a JSON object that conforms to the schema.

Rules:

1. **Transcribe cells verbatim** — keep units, superscript footnote markers
   (write them as plain text such as `93[c]` or `87 (90)`), parentheses,
   and abbreviations exactly as printed. Do not compute, normalise or
   translate anything. Greek letters (α, β, γ, δ, μ, ε) are copied as the
   Unicode letter, never as a Latin letter or a quote mark.
2. **Drawn chemical structures**: every molecule that is DRAWN (a skeletal
   structure, not typed text) is the literal token `[STRUCTURE]`. If one cell
   contains several drawings, write `[STRUCTURE]; [STRUCTURE]`. NEVER write a
   SMILES string, a formula or a name for a drawing — another tool reads the
   drawings. Typed text next to a drawing (e.g. a compound number "3a") is
   ordinary cell text: `[STRUCTURE] 3a`. Atom labels and abbreviations that
   belong to the drawing itself (MeO, F3C, Br, OMe, TMS …) are part of the
   structure — do not transcribe them as text. Write `[STRUCTURE]` ONLY where a
   drawing is actually printed in that cell: in scope tables the substrate
   column is often left blank for entries 2, 3, … because the drawing above
   applies — those cells are `""`, not `[STRUCTURE]`.
3. **Empty cells** are `""`. A visually empty cell stays empty even when the
   value is obviously "same as above" — do not copy values down.
4. **`header_rows`** are the column-title rows (usually one); `data_rows` are
   the entry rows. Every row must have exactly one string per printed
   column, in left-to-right order. Never merge or split columns.
5. A cell that spans several rows is repeated in each spanned data row; a
   header spanning several columns is repeated in each spanned column.
6. Tables printed as two side-by-side halves that repeat the same columns
   (folded tables, e.g. entries 1–12 on the left and 13–24 on the right) are
   UNFOLDED: give the column set once in `header_rows` and list the left
   half's rows first, then the right half's rows, one entry per row.
7. `caption`: the table title text if visible ("Table 2. …"), else null.
   `footnotes`: all footnote text below the table verbatim, else null.
   `scheme_conditions`: any reaction conditions printed inside a scheme
   drawing above the table (temperatures, times, solvents, equivalents),
   verbatim, else null.
8. Only what is printed. Never invent rows, columns or values.
