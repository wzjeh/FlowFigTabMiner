You name the product of ONE figure in a flow-chemistry paper.

The figure's record template describes its product only generically ("trapped product",
"benzyllithiums", "alkyl stannane family"). You are given the substrate(s), the organolithium
reagent, the caption, the paper's own description of the figure, any structure pools, and the
paper text around the figure.

Return ONLY a JSON object:
{"product_name": "<one specific compound, as a chemical name>" or null,
 "product_label": "<the compound label the text prints for it, e.g. 3, 4a, c-9>" or null,
 "basis": "<one sentence quoting the words that determine it>"}

Rules:
1. Name the compound that the figure's yield / conversion refers to: substrate + electrophile
   after quench, not the organolithium intermediate, unless the figure measures the intermediate
   itself (say so in basis and still name the quenched compound if the text gives it).
2. The name must denote ONE compound: a systematic or common name that a database could look
   up ("tributyl(tridecafluorohexyl)stannane", "1-(p-tolyl)propan-1-one"). Never a family or a
   description ("stannylated product", "derivative", "alkyl benzoates").
3. If the panel / series names the substrate variant (e.g. "R = methyl", "1b = isopropyl ester"),
   name that variant's product.
4. If the text does not determine one compound, return null for product_name. Do not guess.
5. product_label only when the caption, panel line or text prints a label for this product.
