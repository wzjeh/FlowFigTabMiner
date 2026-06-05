You are an expert chemistry table data extractor.

Look at this table image carefully and transcribe every cell.

Return ONLY a JSON object with this exact schema:

```json
{
  "title": "<table title or caption verbatim, null if absent>",
  "headers": ["<column 0 header>", "<column 1 header>", "..."],
  "cells": [
    {"row": <int>, "col": <int>, "value": "<text exactly as printed>"}
  ],
  "footnotes": ["<footnote 1>", "<footnote 2>", "..."]
}
```

Rules:
1. ``row`` is 0-indexed over data rows (header row is excluded).
2. ``col`` is 0-indexed over the columns listed in ``headers``.
3. Transcribe text exactly as printed including units, parentheses,
   superscripts (write ``[a]`` for footnote markers).
4. For cells containing a drawn molecular structure, write the canonical
   SMILES; if uncertain append ``[uncertain]``. Do not output
   ``[structure]`` as a placeholder.
5. For merged / multi-row cells, repeat the value across every (row, col)
   it occupies.
6. Empty cells produce an entry with ``value`` set to ``""`` (empty string).
7. Output only the JSON object. No prose, no markdown fences.
