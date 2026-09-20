You are transcribing small text crops taken from a chemistry chart.

The image is a grid of numbered cells. Each cell shows ONE crop: an axis
tick label (a number such as "-78", "20", "0.5", or a log-scale label such
as "10^-1.5") or a data label printed inside the chart (a number such as
"43" or "86"). The index printed in red above each crop ("#0", "#1", …) is
the cell's index.

Return ONLY a JSON object of the form
{"readings": [{"index": 0, "text": "<transcription>", "kind": "tick|value|empty|other"}, ...]}
with exactly one entry per cell, in index order.

Rules:
1. Transcribe EXACTLY what is printed in the crop. Do not infer, complete a
   sequence, or use neighbouring cells to guess.
2. Log-scale labels: write the base and exponent as ``10^<exponent>`` and keep
   the sign and decimals exactly, e.g. a printed 10 with superscript -1.5 →
   "10^-1.5"; 10 with superscript 0 → "10^0".
3. Minus signs must be kept ("-40", not "40"). A number with a space inside
   ("-10 0") is two things — transcribe the largest complete number you can
   read and set kind to "other".
4. If the crop is empty, blurred beyond reading, or not a number at all, set
   text to "" and kind to "empty" (unreadable) or "other" (words / symbols).
5. kind: "tick" for an axis tick label, "value" for a number printed inside
   the plot area (heatmap cell / point label). If unsure, use "tick".
