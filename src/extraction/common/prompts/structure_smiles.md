The image shows ONE hand-drawn chemical structure cut from a paper.

Return ONLY a JSON object {"smiles": "<SMILES>"}.

Rules:
1. Write every atom the drawing shows. Expand printed abbreviations to their groups:
   Ph = phenyl, Me = methyl, Et = ethyl, i-Pr = isopropyl, t-Bu = tert-butyl, Bu = n-butyl,
   TMS / SiMe3 = trimethylsilyl, Bus = tert-butylsulfonyl, Boc = tert-butoxycarbonyl,
   OMe = methoxy, Bn = benzyl, Ac = acetyl, NC / CN on a carbon = nitrile.
2. An R, R1, Ar or X group with no definition in the crop is written as * .
3. Copy the ring sizes and substituent positions exactly as drawn. Do not add, remove or
   move a substituent; do not "correct" the chemistry.
4. Ignore compound labels, yields, arrows and reagent text around the structure.
5. If the crop is not a chemical structure, return {"smiles": ""}.
