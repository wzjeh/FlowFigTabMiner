# Supporting Information

## FlowFigTabMiner: Multimodal Extraction of Structured Flow Chemistry Data from Figures, Tables, and Text Enables Organolithium Lifetime Prediction

Wenyuan Zhao, Xianzhu Zhong, Yosuke Ashikari, Hiromichi V. Miyagishi, Mohmmad S. Qenawy, Simeng Wang, Jouffery Xiaer, Yirui Peng, and Aiichiro Nagaki*

Department of Chemistry, Faculty of Science, Hokkaido University, Sapporo 060-0810, Japan

*E-mail: nagaki@sci.hokudai.ac.jp

---

## Table of Contents

- S1. Flow Chemistry Paper Filtering Keywords
- S2. YOLO Model Training Curves and Detection Examples
- S3. Tab-Scheme-Seg Model Training Results
- S4. LLM Prompts Used in the Pipeline
  - S4.1 Local Variable Builder — Figure Prompt
  - S4.2 Local Variable Builder — Table Prompt
  - S4.3 Global Assembly (Final Adjudication) Prompt
- S5. VLM Benchmark Prompts for Figure and Table Extraction
  - S5.1 Heatmap Figure Extraction Prompt
  - S5.2 Table Extraction Prompt
- S6. Three-Way Benchmark Detailed Results
- S7. Complete List of 125 Papers in the Organolithium Corpus (F1–F10, T1–T10)

---

## S1. Flow Chemistry Paper Filtering Keywords

The following 15 keywords are used to filter relevant papers by checking the first 3 pages of each PDF. A paper is included in the corpus if any keyword is found as a case-insensitive substring match.

| # | Keyword |
|---|---------|
| 1 | flow chemistry |
| 2 | continuous flow |
| 3 | continuous-flow |
| 4 | microreactor |
| 5 | micro-reactor |
| 6 | flow reactor |
| 7 | plug flow |
| 8 | tubular reactor |
| 9 | micropacked |
| 10 | micro-packed |
| 11 | packed bed reactor |
| 12 | coil reactor |
| 13 | microfluidic reactor |
| 14 | flow synthesis |
| 15 | flow process |

Implementation: `src/preprocessing/paper_filter.py`

---

## S2. YOLO Model Training Curves and Detection Examples

### S2.1 Training Loss and Metric Curves (4 main models)

**Figure S1.** Training curves for the four primary YOLO models. Each panel shows training/validation box loss, cls loss, dfl loss, precision, recall, mAP50, and mAP50-95 across epochs. (a) v11m-fig-seg (150 epochs). (b) v11m-fig-sca (200 epochs). (c) v11m-tab-seg (200 epochs). (d) v11s-tab-mol (200 epochs).

![Figure S1](figures/supporting%20information/yolo/fig-yolo-training.png)

### S2.2 Per-Class Detection Results (4 main models)

**Figure S2.** Per-class precision-recall curves, F1-confidence curves, and normalized confusion matrices for the four primary YOLO models. (a–c) v11m-fig-seg. (d–f) v11m-fig-sca. (g–i) v11m-tab-seg. (j–l) v11s-tab-mol.

![Figure S2](figures/supporting%20information/yolo/fig-yolo-results.png)

### S2.3 Per-Class Performance Summary

**Table S1.** Per-class detection metrics for all five YOLO models.

**v11m-fig-seg** (6 classes, 150 epochs)

| Class | Precision | Recall | mAP50 |
|-------|-----------|--------|-------|
| caption | 92.8% | — | — |
| legend | 81.8% | — | — |
| subfigure_marker | 83.9% | — | — |
| target_image | 98.1% | — | — |
| x_axis_title | 96.4% | — | — |
| y_axis_title | 95.3% | — | — |

**v11m-fig-sca** (4 classes, 200 epochs)

| Class | Precision | Recall | mAP50 |
|-------|-----------|--------|-------|
| data_point | 92.0% | — | — |
| data_value | 99.5% | — | — |
| x_tick_label | 99.9% | — | — |
| y_tick_label | 87.6% | — | — |

**v11m-tab-seg** (4 classes, 200 epochs)

| Class | Precision | Recall | mAP50 |
|-------|-----------|--------|-------|
| table_body | 98.0% | — | — |
| table_caption | 95.4% | — | — |
| table_note | 87.8% | — | — |
| table_scheme | 69.5% | — | — |

**v11s-tab-mol** (1 class, 200 epochs)

| Class | Precision | Recall | mAP50 |
|-------|-----------|--------|-------|
| Structure | 95.6% | 95.6% | 95.6% |

---

## S3. Tab-Scheme-Seg Model Training Results

The v11n-tab-scheme-seg model detects 4 classes (arrow, molecule, table-condition, table-mark) within reaction scheme diagrams, enabling automatic construction of compound pools for LLM adjudication.

**Training configuration:** YOLOv11n backbone, 250 epochs (patience=50), AdamW optimizer (lr=5×10⁻⁴, weight_decay=0.05), image size 640 px, rect=True, mosaic=1.0, mixup=0.1, batch=auto.

### S3.1 Overall Performance

**Table S2.** Tab-scheme-seg validation results.

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|--------|-----------|-----------|--------|-------|----------|
| all | 18 | 186 | 93.9% | 94.3% | 96.5% | 72.1% |
| arrow | 16 | 19 | 100% | 88.4% | 93.7% | 63.4% |
| molecule | 17 | 67 | 97.0% | 96.8% | 99.1% | 91.1% |
| table-condition | 15 | 31 | 85.5% | 95.1% | 96.1% | 73.7% |
| table-mark | 15 | 69 | 93.0% | 96.7% | 97.1% | 60.1% |

Inference speed: 8.2 ms preprocess, 30.5 ms inference, 6.3 ms postprocess per image.

### S3.2 Training Curves

**Figure S3.** Training loss and validation metric curves for the v11n-tab-scheme-seg model across 250 epochs.

![Figure S3](figures/supporting%20information/tab-scheme-seg/results.png)

### S3.3 Confusion Matrix

**Figure S4.** Normalized confusion matrix for the v11n-tab-scheme-seg model.

![Figure S4](figures/supporting%20information/tab-scheme-seg/confusion_matrix_normalized.png)

### S3.4 Precision-Recall and F1 Curves

**Figure S5.** (a) Precision-recall curve. (b) F1-confidence curve for the v11n-tab-scheme-seg model.

PR curve:
![Figure S5a](figures/supporting%20information/tab-scheme-seg/PR_curve.png)

F1 curve:
![Figure S5b](figures/supporting%20information/tab-scheme-seg/F1_curve.png)

### S3.5 Validation Examples

**Figure S6.** Validation batch examples. (a) Ground-truth labels. (b) Model predictions.

Ground truth:
![Figure S6a](figures/supporting%20information/tab-scheme-seg/val_batch0_labels.jpg)

Predictions:
![Figure S6b](figures/supporting%20information/tab-scheme-seg/val_batch0_pred.jpg)

---

## S4. LLM Prompts Used in the Pipeline

The FlowFigTabMiner pipeline uses LLM-based adjudication at two stages:
1. **Local Variable Builder** — analyzes individual figures/tables to determine axis semantics, column mappings, and fixed conditions.
2. **Global Assembly** — merges all evidence (figures, tables, text, compound pools) into final structured reaction records.

All prompts use Qwen-3.5-Plus with temperature = 0.1.

### S4.1 Local Variable Builder — Figure Prompt

**System prompt:**

```
You are an expert flow chemistry data analyst. Analyze a single extracted
figure from a flow chemistry paper and produce a structured JSON
sub-variable library documenting what it measures and how to interpret each
axis and series.
Rules:
(1) No SMILES.
(2) Only state fixed_conditions explicitly mentioned in the context.
(3) Output valid JSON only, no markdown fences.
```

**User prompt template:**

```
=== FIGURE EVIDENCE ===
Source ID: {source_id}
Figure type: {figure_type}
X-axis label: {x_title}
Y-left-axis label: {yl_title}
Y-right-axis label: {yr_title}
Legend text: {legend_texts}
Chart text: {chart_texts}
Data point count: {len(raw_data)}
X range: {x_range}
Y_Left range: {yl_range}
Unique series: {unique_series}

=== RELEVANT PAPER TEXT CONTEXT (4000 chars) ===
{text_window}

=== OUTPUT SCHEMA ===
Output a single valid JSON object (no markdown):
{
  "source_id": "...",
  "source_type": "figure",
  "figure_type": "<scatter|line|heatmap|bar|other>",
  "reaction_context": "<one-sentence description>",
  "axis_semantics": {
    "x_axis": {"raw_label": "...", "semantic_meaning": "...",
               "maps_to_field": "..."},
    "y_left_axis": {"raw_label": "...", "semantic_meaning": "...",
                    "maps_to_field": "..."},
    "y_right_axis": null
  },
  "series_semantics": {
    "<series_name>": {"role": "...", "metric": "...", "description": "..."}
  },
  "fixed_conditions": {
    "temperature_C": null, "solvent": null, "catalyst": null,
    "reactor_type": null, "notes": "..."
  },
  "data_interpretation_notes": "..."
}
maps_to_field must be one of: conditions.temperature_C,
conditions.residence_time_s, conditions.flow_rate_mL_min,
conditions.solvent, conditions.catalyst, conditions.pressure_bar,
conditions.reactor_type, yield_pct, conversion_pct, selectivity_pct,
ee_pct, other_metrics.<name>
```

### S4.2 Local Variable Builder — Table Prompt

**System prompt:**

```
You are an expert flow chemistry data analyst. Analyze a single extracted
table from a flow chemistry paper and produce a structured JSON
sub-variable library documenting what it measures and how to interpret
each column.
Rules:
(1) No SMILES.
(2) For fixed_conditions: search BOTH the table caption/note AND the paper
    text context for conditions that apply uniformly to ALL rows of this
    table (e.g. temperature stated in the caption, solvent mentioned in
    surrounding text, reactor type described in the experimental section).
    Fill fixed_conditions even if the condition is only mentioned in the
    paper text, not the CSV.
(3) Output valid JSON only, no markdown fences.
```

**User prompt template:**

```
=== TABLE EVIDENCE ===
Source ID: {source_id}
Caption: {caption}
Table note: {note}
Extracted cell count: {num_extracted}
CSV preview (first 6 rows):
{csv_head}

=== RELEVANT PAPER TEXT CONTEXT (4000 chars) ===
{text_window}

=== SCHEME CONDITIONS (from reaction scheme image) ===
{scheme_conditions}

=== OUTPUT SCHEMA ===
Output a single valid JSON object (no markdown):
{
  "source_id": "...",
  "source_type": "table",
  "reaction_context": "<one-sentence description>",
  "column_semantics": {
    "<column_header>": "<semantic meaning and output field mapping>"
  },
  "fixed_conditions": {
    "temperature_C": null, "residence_time_s": null,
    "solvent": null, "catalyst": null,
    "reactor_type": null, "notes": "..."
  },
  "data_interpretation_notes": "..."
}
```

### S4.3 Global Assembly (Final Adjudication) Prompt

**System prompt:**

```
You are an expert flow chemistry data extractor. Your sole task is to
extract structured reaction data from flow chemistry papers. You must
be precise, grounded, and never hallucinate values not present in the
provided text or tables.
```

**User prompt template** (abbreviated; full template is ~300 lines):

```
You are extracting reaction data from a flow chemistry paper.

=== PAPER TEXT (truncated) ===
{full_text[:50000]}

=== EXTRACTED TABLES (CSV content) ===
{table_data}

=== EXTRACTED FIGURES (coordinate data + axis labels) ===
{figure_data}

=== REACTANT STRUCTURE POOL (from scheme, left of reaction arrow) ===
{reactant_pool}

=== PRODUCT STRUCTURE POOL (from scheme, right of reaction arrow) ===
{product_pool}

=== SCHEME CONDITIONS (apply to all records unless table overrides) ===
{scheme_conditions}

=== FLOW CHEMISTRY DOMAIN KNOWLEDGE ===
- For organolithium flow chemistry papers: if reactor_type is not
  explicitly stated, it is typically a "T-shaped micromixer + capillary
  /coil reactor" setup.
- Solvent abbreviations: THF=tetrahydrofuran, Et2O=diethyl ether, ...

=== LOCAL VARIABLE LIBRARIES ===
Each table and figure may contain a "local_vars" field. USE it to:
- Map axes/columns to the correct output fields
- Apply fixed_conditions to ALL records from that source
- Use reaction_context for reactant/product identification

=== TASK ===
Extract ALL reaction records from the tables and figures above.
Every row in a table and every data point in figure raw_data must
become one record in the output — do NOT skip or merge any.

For each reaction record, output one JSON object with fields:
{
  "reactant1_smiles": "...",
  "reactant1_name": "...",
  "reactant2_smiles": "...",
  "reactant2_name": "...",
  "product_smiles": "...",
  "product_name": "...",
  "product_label": "...",
  "entry_number": null,
  "yield_pct": null,
  "yield_type": null,
  "batch_yield_pct": null,
  "conversion_pct": null,
  "selectivity_pct": null,
  "ee_pct": null,
  "diastereomeric_ratio": null,
  "stoichiometry": null,
  "reaction_class": "...",
  "paper_doi": null,
  "conditions": {
    "temperature_C": null,
    "residence_time_s": null,
    "flow_rate_mL_min": null,
    "solvent": null,
    "catalyst": null,
    "catalyst_metal": null,
    "catalyst_loading_pct": null,
    "ligand": null,
    "additive": null,
    "pressure_bar": null,
    "reactor_type": null
  },
  "source_table_or_figure": "...",
  "notes": null
}

=== RULES (15 rules) ===
1. SUBSTRATE SCOPE TABLES: Each row → one record.
2. OPTIMIZATION TABLES: Each condition set → one record.
3. FIGURES: Every data point → one record. Count check enforced.
4. CONDITIONS: Shared conditions propagated to all records.
5. SMILES: Only from provided data, never invented.
6. SOURCE RESTRICTION: Only extract from provided tables/figures.
7-8. OTHER METRICS in dedicated field. No hallucination.
9. OUTPUT: Valid JSON array only.
10. SMILES LOOKUP: Match labels against compound pools.
11. SCHEME CONDITIONS: Use as fallback for missing fields.
12. LOCAL VARS: Override interpretation when present.
13. REACTION CLASS: Same class for all records per paper.
14. NOTES: Only for info not captured elsewhere.
15. CATALYST SPLITTING: Separate catalyst/ligand/additive.

Output a JSON array of all extracted reaction records:
```

---

## S5. VLM Benchmark Prompts for Figure and Table Extraction

The following prompts were used for the VLM benchmark comparison (Gemini 3.1 Pro and Claude Sonnet 4.6). Each VLM received the same prompt along with the corresponding figure or table image.

### S5.1 Heatmap Figure Extraction Prompt

```
You are extracting data from a flow chemistry heatmap figure. The figure
shows the effect of residence time (tR) and temperature (T) on reaction
yield (%).

## What to extract

1. **Axes**:
   - X-axis: title and tick values (usually tR in seconds, often log
     scale like 10^-2, 10^-1, 10^0)
   - Y-axis: title and tick values (usually Temperature in °C)
   - Identify if each axis is "log" or "linear" scale

2. **Data grid**: For every labeled data point (yield percentage printed
   on the heatmap), record:
   - The tR value (X-axis, in seconds — convert from scientific notation
     if needed, e.g. 10^-1 = 0.1)
   - The temperature value (Y-axis, in °C)
   - The yield value (the number printed at that grid point, in %)

3. **Caption**: The full text of the figure caption

4. **Context** (from caption, sub-figure labels, or molecule structures):
   - Sub-figure label if present (e.g., "a", "b", "c", "d")
   - Reaction description
   - Any compound IDs mentioned
   - Any molecule structure descriptions visible in the figure

## Output format

Return a single JSON object:
{
  "x_axis": {
    "title": "tR1/s",
    "scale": "log",
    "tick_values": [0.01, 0.03, 0.1, 0.3, 1.0]
  },
  "y_axis": {
    "title": "T/°C",
    "scale": "linear",
    "tick_values": [-78, -60, -40, -20]
  },
  "data_points": [
    {"tR_s": 0.01, "T_C": -78, "yield_pct": 24},
    ...
  ],
  "caption": "Figure 3. Effects of ...",
  "context": {
    "sub_figure_label": "a",
    "reaction_description": "...",
    "compound_ids": ["2"],
    "electrophile_or_substrate": "...",
    "molecule_in_figure": "..."
  }
}

## Important rules

- For log-scale X-axis: convert tick labels to actual seconds
  (10^-2 = 0.01, 10^-1 = 0.1, 10^0 = 1.0, etc.)
- Read every yield number visible on the heatmap grid — do not skip any
- If a data point has no visible yield number, omit it (do not guess)
- If the caption mentions multiple sub-figures, only extract data for
  the sub-figure shown in THIS image
- Preserve negative temperature values
- Return ONLY the JSON object, no other text
```

### S5.2 Table Extraction Prompt

The original prompt was written in Chinese and is translated below:

```
Extract the complete table content from this table image. Output strict
JSON format, no other text.

Requirements:
1. Identify all column headers (including merged multi-row headers)
2. Extract all content row by row, cell by cell
3. If a cell contains a chemical molecular structure image (not text),
   attempt to write its SMILES representation
4. Keep original data unmodified (including superscript/subscript markers
   such as [a], [b] footnote marks)

Output format:
{
  "columns": ["Column 1", "Column 2", ...],
  "rows": [
    ["Cell 1", "Cell 2", ...],
    ...
  ]
}

Notes:
- Empty cells represented as ""
- Keep numeric values as-is (e.g., "70 (82)" should not be split)
- For molecular structure images, write SMILES (if unrecognizable,
  write "[structure]")
```

---

## S6. Three-Way Benchmark Detailed Results

### S6.1 Per-Task Precision, Recall, and F1

**Table S5.** Detailed three-way comparison: FlowFigTabMiner vs. VLMs on 5 figures and 5 tables, showing precision, recall, and F1 for each extraction task.

| Task | Metric | FlowFigTabMiner | Gemini 3.1 Pro | Claude Sonnet 4.6 |
|------|--------|-----------------|----------------|-----------------|
| Figure (F1–F5) | Precision | **0.876** | 0.531 | 0.513 |
| | Recall | **0.932** | 0.659 | 0.651 |
| | F1 | **0.892** | 0.575 | 0.561 |
| Table (T1–T5) | Precision | 0.843 | 0.98 | **0.99** |
| | Recall | 0.814 | 0.98 | **0.99** |
| | F1 | 0.827 | 0.98 | **0.99** |
| SMILES (T1–T5) | Precision | 0.925 | **0.944** | 0.893 |
| | Recall | 0.697 | **0.944** | 0.843 |
| | F1 | 0.795 | **0.944** | 0.867 |
| **Overall** | Precision | **0.881** | 0.818 | 0.799 |
| | Recall | 0.814 | **0.861** | 0.828 |
| | F1 | **0.838** | 0.833 | 0.806 |

### S6.2 Per-Sample F1 Scores — Figure Extraction

**Table S6.** Per-sample F1 scores for figure extraction (F1–F10).

| Sample | FlowFigTabMiner | Gemini 3.1 Pro | Claude Sonnet 4.6 |
|--------|-----------------|----------------|-----------------|
| F1 | — | — | — |
| F2 | — | — | — |
| F3 (heatmap) | — | — | — |
| F4 | — | — | — |
| F5 (dense scatter) | — | — | — |
| F6–F10 | — | N/A | N/A |

### S6.3 Per-Sample F1 Scores — Table Extraction

**Table S7.** Per-sample F1 scores for table extraction (T1–T10).

| Sample | FlowFigTabMiner | Gemini 3.1 Pro | Claude Sonnet 4.6 |
|--------|-----------------|----------------|-----------------|
| T1 | — | — | — |
| T2 | — | — | — |
| T3 | — | — | — |
| T4 | — | — | — |
| T5 | — | — | — |
| T6–T10 | — | N/A | N/A |

*Note: Per-sample scores to be filled from evaluation results.*

---

## S7. Complete List of Papers in the Organolithium Corpus

The papers were selected by keyword filtering (Section S1) from flow chemistry literature spanning 2005–2024. The complete DOI list with dataset membership flags is deposited on Zenodo as `corpus_dois.csv`: [DOI: 10.5281/zenodo.19436601](https://doi.org/10.5281/zenodo.19436601).

The full datasets (kinetics: 1,470 rows; scope tables: 1,267 rows) and documentation are also available at the same Zenodo record.
