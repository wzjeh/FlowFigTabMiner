# FlowFigTabMiner

**Automated multimodal mining of structured data locked in the figures and tables of flow chemistry literature.**

Much of the quantitative process data in chemistry papers sits inside figures and tables, where
molecular structures are drawn as images and yields are plotted rather than typed. FlowFigTabMiner
turns these figures and tables into structured reaction records. It chains five custom-trained YOLO
models with chemistry-specific tools (Florence-2/TF-ID, TATR, MolNexTR, PP-OCR via RapidOCR) and uses Gemini
only for table cells and the final record assembly. The figure track is deterministic (YOLO + RANSAC
axis mapping), so the same PDF gives the same output on every run.

> Zhao *et al.*, Automated multimodal mining of structured data locked in the figures and tables of
> flow chemistry literature. In press.

## How it works

```
PDF → [0] flow-chem pre-filter → [1] TF-ID detect figures/tables
    → figures: YOLO segment → YOLO detect points → RANSAC axis mapping → legend match
    → tables:  YOLO segment → TATR structure → Gemini cells + MolNexTR + RapidOCR
    → [3.5] scheme parsing → compound pool   → [4.5] per-source variable libraries
    → [5] per-source Gemini assembly → JSON records → [6] normalize + validate → dataset
```

## Install

Python **3.9**. Run everything from the project root.

```bash
git clone https://github.com/wzjeh/FlowFigTabMiner.git
cd FlowFigTabMiner
python3.9 -m venv flowfigtabminer && source flowfigtabminer/bin/activate
pip install -e .                       # installs deps (requirements.txt) + the CLI
```

**Models** (git-ignored, under `models/`). One Hugging Face repository holds the five YOLO
weights, the MolNexTR checkpoint and the PP-OCR models (ONNX, for RapidOCR), laid out as `config.yaml` expects:

```bash
pip install huggingface_hub
hf download wyzhaoc/FlowFigTabMiner-models --local-dir models/ --exclude "paddlex/*"
```

TF-ID, TATR and docling weights download on first run.

**API key.** The LLM steps use Google Gemini (`gemini-2.5-flash`, temperature 0):

```bash
echo "GEMINI_API_KEY=your_key" > .env     # key from https://aistudio.google.com/apikey
```

### Docker

One CPU image runs the whole pipeline. Weights are fetched into the mounted `models/` volume on
first run (about 1.4 GB, once). The table stage needs about 10 GB of RAM.

```bash
mkdir -p models data/input/my_pdfs          # put PDFs in data/input/my_pdfs
docker run --rm -e GEMINI_API_KEY=your_key \
    -v "$PWD/models:/app/models" -v "$PWD/data:/app/data" \
    ghcr.io/wzjeh/flowfigtabminer --dir data/input/my_pdfs
```

Results appear in `data/final_output/`. The image is built from `Dockerfile.full` by the
`docker-full` GitHub Actions workflow.

## Usage

```bash
set -a; source .env; set +a               # load GEMINI_API_KEY

flowfigtabminer path/to/paper.pdf         # single PDF, full pipeline
flowfigtabminer --dir data/input/corpus   # every PDF in a folder, one subprocess each
flowfigtabminer paper.pdf --skip-tfid     # reuse existing crops
flowfigtabminer paper.pdf --force-assembly# re-run only the LLM assembly

python scripts/batch_all.py --pdf-dir data/input/corpus --resume   # resumable batch
python scripts/reassemble.py paper.pdf    # re-run only the adjudication stages on existing crops
```

Flags: `--dir` · `--skip-tfid` · `--force-assembly` · `--no-prefilter` · `--no-smiles-lookup` · `--no-vlm`.
Module form: `python -m src.pipeline.main paper.pdf`.

## Output

```
data/intermediate/<paper>/   crops, evidence JSONs, compound_pool.json, timing.json
  layout.json, context/      crop geometry, figure/table labels, verbatim captions,
                             footnotes and in-box PDF text (from the PDF text layer)
  global_vars.json           paper-level defaults, each with a verbatim quote and scope
  local_vars/, status/       per-source semantics; per-source outcome (ok / filtered / failed)
data/final_output/<paper>_normalized.json   ← primary result (also .xlsx)
```

Each record is one reaction observation: `reactant1/2_smiles+name`, `product_smiles+name+label`,
`yield_pct` / `conversion_pct` / `ee_pct`, a nested `conditions` block (temperature, residence time,
solvent, reactor type, …) and `reaction_class`. SMILES are never guessed. They come from MolNexTR or a
deterministic label/name backfill. Records with neither identity nor outcome are flagged `is_hollow`
and kept; `has_outcome` marks records with a measured yield, conversion, selectivity or ee. Figure
records are built by code from the extracted data points, so numbers are copied, never re-typed.
Condition fields inherited from the paper-level pool are tagged in `conditions_provenance`.

## Configuration

`config.yaml` holds model paths, detection thresholds and LLM settings (`llm.adjudication.model`,
`llm.max_concurrent`, …), each key documented inline. `keywords.yaml` lists table-relevance keywords.
Runtime knobs as environment variables: `OCR_UPSCALE_MAX_SIDE` (default 1600),
`MOLNEXTR_NUM_WORKERS` (default 1), `OCR_PROFILE=1` (timing), `FFTM_OCR_ENGINE=apple` (Apple Vision OCR on macOS).

## Web tool

A hosted version for trial runs, with a per-user allowance:
https://flowfigtabminer-903119038444.asia-northeast1.run.app

## Data and models

Zenodo **[10.5281/zenodo.19436601](https://doi.org/10.5281/zenodo.19436601)**:

| File | Content |
|---|---|
| `organolithium_dataset_R1.xlsx` | 2,357 curated yield-vs-residence-time data points from 25 publications; the 5,351 raw pipeline records for the same 25 publications; per-publication audit |
| `ALL_organolithium_normalized_3195_records.csv` | 3,195 reaction records mined from the 114-publication corpus |
| `corpus_114_publications.csv` | the 114 publications (DOI, year, journal, records contributed) |
| `flowchem-bench-v1.0.4.tar.gz` | extraction benchmark: figure and table crops, ground truth, the outputs of every evaluated system, scoring code |

Custom YOLO weights are also mirrored at **[wyzhaoc/YOLO11](https://huggingface.co/wyzhaoc/YOLO11)**:

| Model | Task | mAP50 |
|-------|------|-------|
| FigSeg (YOLO11m) | figure macro segmentation | 91.7% |
| DataDet (YOLO11m) | figure micro detection | 92.5% |
| TabSeg (YOLO11m) | table segmentation | 94.8% |
| TabMol (YOLO11s) | molecule detection | 95.6% |
| SchemeParse (YOLO11n) | reaction-scheme parsing | 96.5% |

## Benchmark (30 figures, 30 tables)

Extraction F1 on flowchem-bench v1.0.4: 30 figures (971 data points) and 30 tables (2,438 cells)
from 45 flow chemistry articles and supporting-information files. A dash means no component for the
task and counts as 0 in the overall score.

| | FlowFigTabMiner | Gemini 3.1 Pro | Claude Sonnet 4.6 | DePlot | UniChart | Docling |
|---|---|---|---|---|---|---|
| Figures | **0.815** | 0.719 | 0.621 | 0.069 | 0.124 | – |
| Tables | 0.932 | **0.978** | 0.935 | – | – | 0.683 |
| **Overall** | **0.874** | 0.848 | 0.778 | 0.035 | 0.062 | 0.342 |

The pipeline output is byte-identical across repeated runs. The published scores were produced with
PaddleOCR; RapidOCR runs the same PP-OCR models and gives the same scores on this benchmark. The benchmark archive, with protocol and
scoring code, is in the Zenodo record above.

## License

[PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0): free for academic, research and personal use, including modification and redistribution. Commercial use is not permitted without permission from the authors. See `LICENSE`.
