# FlowFigTabMiner

**Automated multimodal mining of structured data locked in chemistry-literature figures and tables.**

Chemistry papers hide much of their quantitative process data — including the low-yield and
failed reactions ML most needs — inside *figures* and *tables* that render molecular structures
as images. Plain text mining can't read these; general-purpose vision–language models read them
inaccurately and at a per-image cost. FlowFigTabMiner mines them into structured reaction records
by chaining **five custom-trained YOLO models** with chemistry-specific tools (Florence-2/TF-ID,
TATR, MolNexTR, PaddleOCR) and a **lightweight Gemini adjudication step**. On a 10-figure /
10-table benchmark it reaches an extraction **F1 of 0.838**, with a decisive lead on figures.

> 📄 *Automated multimodal mining of structured data locked in chemistry literature figures and
> tables* — Zhao et al., 2025 (manuscript submitted; preprint on ChemRxiv).

## How it works

```
PDF → [0] flow-chem pre-filter → [1] TF-ID detect figures/tables
    → figures: YOLO segment → YOLO detect points → RANSAC axis mapping → legend match
    → tables:  YOLO segment → TATR structure → Gemini cells + MolNexTR + PaddleOCR
    → [3.5] scheme parsing → compound pool   → [4.5] per-source variable libraries
    → [5] per-source Gemini assembly → JSON records → [6] normalize + validate → dataset
```

The figure coordinate track is intentionally **non-VLM** (RANSAC + YOLO) — it beats VLMs on
figure extraction (F1 0.892 vs ~0.57). Gemini is used only for table cells, optional inspection,
and the final record assembly.

## Install

Python **3.9**. Run everything from the project root.

```bash
git clone https://github.com/wzjeh/FlowFigTabMiner.git
cd FlowFigTabMiner
python3.9 -m venv flowfigtabminer && source flowfigtabminer/bin/activate
pip install -e .                       # installs deps (requirements.txt) + the CLI
```

**Models** (git-ignored, place under `models/`). One repository holds the
custom YOLO weights, the MolNexTR checkpoint and the PaddleOCR models, laid
out exactly as `config.yaml` expects:

```bash
pip install huggingface_hub
hf download wyzhaoc/FlowFigTabMiner-models --local-dir models/
```

TF-ID, TATR and docling weights auto-download on first run.

### Docker (no local setup)

A single CPU image runs the whole pipeline; weights are fetched into the
mounted `models/` volume on first run (about 1.4 GB, once):

```bash
mkdir -p models data/input/my_pdfs          # put PDFs in data/input/my_pdfs
docker run --rm -e GEMINI_API_KEY=your_key \
    -v "$PWD/models:/app/models" -v "$PWD/data:/app/data" \
    ghcr.io/wzjeh/flowfigtabminer --dir data/input/my_pdfs
```

Results appear in `data/final_output/`. The container needs about 10 GB of
RAM for the table stage. `scripts/docker_smoke.py` loads every model once
without an API key (`docker run --rm --entrypoint python -v "$PWD/models:/app/models"
ghcr.io/wzjeh/flowfigtabminer scripts/docker_smoke.py`). The image is built
from `Dockerfile.full` by the `docker-full` GitHub Actions workflow.

**API key** — the LLM steps use Google Gemini:

```bash
echo "GEMINI_API_KEY=your_key" > .env     # key from https://aistudio.google.com/apikey
```

## Usage

```bash
set -a; source .env; set +a               # load GEMINI_API_KEY

flowfigtabminer path/to/paper.pdf         # single PDF, full pipeline
flowfigtabminer paper.pdf --skip-tfid     # reuse existing crops
flowfigtabminer paper.pdf --force-assembly# re-run only the LLM assembly

python scripts/batch_all.py --pdf-dir data/input/corpus   # batch (resumable)
```

Flags: `--dir` (batch a folder) · `--skip-tfid` · `--force-assembly` ·
`--no-prefilter` · `--no-smiles-lookup` · `--no-vlm`.
(Equivalent module form: `python -m src.pipeline.main paper.pdf`.)

## Output

```
data/intermediate/<paper>/   crops, evidence JSONs, compound_pool.json, timing.json
  layout.json, context/      crop geometry + real figure/table labels, verbatim captions,
                             footnotes and in-box PDF text (deterministic, from the PDF text layer)
  global_vars.json           paper-level defaults (each value with a verbatim quote + scope)
  local_vars/, status/       per-source semantics; per-source outcome (ok / filtered / failed + reason)
data/final_output/<paper>_normalized.json   ← primary result (also .xlsx)
```

Each record is one reaction observation: `reactant1/2_smiles+name`, `product_smiles+name+label`,
`yield_pct`/`conversion_pct`/`ee_pct`, a nested `conditions` block (temperature, residence time,
solvent, reactor type, …), and `reaction_class`. **SMILES are never guessed** — they come from
MolNexTR or a deterministic label/name backfill. Records with neither identity nor outcome are
flagged `is_hollow` (kept, not dropped) for explicit downstream filtering; `has_outcome` marks records
that carry a measured yield/conversion/selectivity/ee. Figure records are synthesised by code from the
extracted data points after a single per-figure LLM template call (`adjudication.figure_record_synthesis`),
so numbers are copied, never re-typed; empty condition fields are inherited from the paper-level pool and
tagged in `conditions_provenance` (`llm_source` / `source_local` / `paper_global`).

To re-run only the adjudication stages on existing crops (e.g. after a prompt change):
`python scripts/reassemble.py [--rebuild-vars] paper.pdf`. Intermediates produced by older versions can be
upgraded without re-running TF-ID with `python scripts/backfill_layout.py "data/input/corpus/*.pdf"`.

## Configuration

`config.yaml` holds model paths, detection thresholds, and LLM/VLM settings
(`llm.adjudication.model: gemini-2.5-flash`, `llm.max_concurrent: 5`, …) — each key is
documented inline. `keywords.yaml` lists table-relevance keywords. A few runtime knobs are
environment variables: `OCR_UPSCALE_MAX_SIDE` (default 1600), `MOLNEXTR_NUM_WORKERS` (default 1),
`OCR_PROFILE=1` (timing), `USE_EASYOCR=1` (OCR fallback).

## Datasets & models

Extracted organolithium datasets — Zenodo **[10.5281/zenodo.19436601](https://doi.org/10.5281/zenodo.19436601)**:
`organolithium_tr_subdataset_vlm_enriched.csv` (1,470 rows, yield-vs-tR kinetics),
`organolithium_scope_table_dataset.csv` (1,267 rows, scope tables), `corpus_dois.csv` (93 DOIs).

Custom YOLO weights — **[wyzhaoc/YOLO11](https://huggingface.co/wyzhaoc/YOLO11)**:

| Model | Task | mAP50 |
|-------|------|-------|
| fig-seg (YOLOv11m) | figure macro segmentation | 91.7% |
| fig-sca (YOLOv11m) | figure micro detection | 92.5% |
| tab-seg (YOLOv11m) | table segmentation | 94.8% |
| tab-mol (YOLOv11s) | molecule detection | 95.6% |
| tab-scheme-seg (YOLOv11n) | reaction-scheme parsing | 96.5% |

## Benchmark (10 figures + 10 tables)

| | FlowFigTabMiner | Gemini 3.1 Pro | Claude Sonnet 4.6 |
|---|---|---|---|
| Figure F1 | **0.892** | 0.575 | 0.561 |
| Table F1 | 0.827 | 0.98 | **0.99** |
| SMILES F1 | 0.795 | **0.944** | 0.867 |
| **Overall** | **0.838** | 0.833 | 0.806 |

A four-service Cloud Run deployment is in `deploy/` (`./deploy_all.sh`).

## Citation

```bibtex
@misc{zhao2025flowfigtabminer,
  title  = {Automated multimodal mining of structured data locked in
            chemistry literature figures and tables},
  author = {Zhao, Wenyuan and Zhong, Xianzhu and Jouffroy, Xavier and
            Ashikari, Yosuke and Miyagishi, Hiromichi V. and Qenawy, Mohmmad S. and
            Wang, Simeng and Peng, Yirui and Nagaki, Aiichiro},
  year   = {2025},
  note   = {Manuscript submitted; preprint on ChemRxiv}
}
```

## License

Apache 2.0
