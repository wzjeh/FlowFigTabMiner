You are reading the labels printed next to drawn chemical structures in a reaction scheme.

The image is a grid of numbered cells. Each cell shows ONE drawn structure with the text
printed around it (usually below it). The red "#0", "#1", … above a cell is its index.

Return ONLY a JSON object of the form
{"readings": [{"index": 0, "label": "3", "variants": []}, ...]}
with exactly one entry per cell, in index order.

Rules:
1. "label" is the compound label printed for THAT structure, copied exactly as printed:
   "3", "1a", "c-9", "t-16", "2b'". Do not invent a label; if none is printed, use null.
2. If the structure is drawn with an R group and the text lists several labels with their R,
   e.g. "c-8 (R = Me)" and "c-10 (R = Ph)", put each pair in "variants" as
   {"label": "c-8", "r_group": "Me"} and leave "label" null.
3. Copy printed text only. Do not name the compound and do not write SMILES.
4. Ignore reagents written over arrows (sBuLi, MeI, t min) and condition text.
