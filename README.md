# FlowFigTabMiner

Multimodal pipeline for extracting structured reaction data from flow chemistry literature PDFs. Processes figures (scatter plots, heatmaps), tables (with embedded molecular structures), and text into normalized reaction databases.

**Paper**: *FlowFigTabMiner: Multimodal Extraction of Structured Flow Chemistry Data from Figures, Tables, and Text Enables Organolithium Lifetime Prediction*

## Pipeline Overview

```
PDF ──→ TF-ID (Florence-2) ──→ Figure crops + Table crops + Text
              │                        │                │
              │                ┌───────┴────────┐       │
              │                │                │       │
         FigureDataMiner  TableDataMiner    PyMuPDF
         (YOLO + OCR +    (YOLO + TATR +   (full text)
          coord mapping)   MolNexTR + OCR)      │
              │                │                │
              │                │       LLM: Global/Local
              │                │       Parameters Pool
              └────────┬───────┘                │
                       │                        │
                  LLM Adjudication (Qwen-3.5-Plus)
                       │
                  JSON Records → Manual Inspection → Dataset
```

### Six-Step Pipeline

| Step | Name | Method |
|------|------|--------|
| 1 | TF-ID Detection | Florence-2 detects figures/tables in PDF pages |
| 2 | Macro Segmentation | YOLOv11m isolates chart regions from captions/legends |
| 3 | Micro Detection | YOLOv11m detects data points, tick labels |
| 3T | Table Extraction | YOLO seg → TATR structure → PaddleOCR + MolNexTR |
| 3.5 | Scheme Parsing | YOLOv11n parses reaction schemes → compound pool |
| 4 | Coordinate Mapping | RANSAC pixel→physical transform, legend matching |
| 4.5 | Local Variables | LLM builds axis semantics + fixed conditions per source |
| 5 | Global Assembly | LLM merges all evidence → structured JSON records |
| 6 | Post-Processing | Unit normalization, SMILES validation, deduplication |

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/wzjeh/FlowFigTabMiner.git
cd FlowFigTabMiner
python -m venv flowfigtabminer
source flowfigtabminer/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

### 2. Download Models

Download the following models and place them in the `models/` directory:

**Custom YOLO Models** (from [HuggingFace](https://huggingface.co/wyzhaoc/YOLO11)):

```bash
# Install huggingface-cli if needed: pip install huggingface_hub
huggingface-cli download wyzhaoc/YOLO11 --local-dir models/hf_yolo11

# Copy to expected paths:
mkdir -p models/yolo11m-fig-seg-0207-nobreaknocharttext/runs/detect/train/weights/
mkdir -p models/yolo11m-fig-scatter-0208/runs/detect/train/weights/
mkdir -p models/yolo11m-tab-seg-0209-white/runs/detect/train/weights/
mkdir -p models/yolo11s-tab-molecule-0207/runs/detect/train/weights/
mkdir -p models/tab-scheme-seg/

cp models/hf_yolo11/fig-seg/best.pt models/yolo11m-fig-seg-0207-nobreaknocharttext/runs/detect/train/weights/best.pt
cp models/hf_yolo11/fig-sca/best.pt models/yolo11m-fig-scatter-0208/runs/detect/train/weights/best.pt
cp models/hf_yolo11/tab-seg/best.pt models/yolo11m-tab-seg-0209-white/runs/detect/train/weights/best.pt
cp models/hf_yolo11/tab-mol/best.pt models/yolo11s-tab-molecule-0207/runs/detect/train/weights/best.pt
cp models/hf_yolo11/tab-scheme-seg/best.pt models/tab-scheme-seg/best.pt
```

**Third-Party Models** (auto-downloaded on first run):

| Model | Source | Purpose | Auto-download |
|-------|--------|---------|---------------|
| TF-ID (Florence-2) | [yifeihu/TF-ID-base](https://huggingface.co/yifeihu/TF-ID-base) | PDF figure/table detection | Yes (HuggingFace) |
| TATR | [microsoft/table-transformer-structure-recognition-v1.1-all](https://huggingface.co/microsoft/table-transformer-structure-recognition-v1.1-all) | Table structure recognition | Yes (HuggingFace) |
| PaddleOCR | [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | Text recognition (PP-OCRv4) | Yes (PaddleX) |

**MolNexTR** (manual download required):

Download `molnextr_model_best.pth` (1.06 GB) from [CYF2000127/MolNexTR](https://github.com/CYF2000127/MolNexTR) and place in `models/`.

```bash
# Also download the Swin backbone:
# swin_base_char_aux_1m680k.pth → models/
```

### 3. Set Up API Key

The LLM adjudication step (Step 5) requires a Qwen API key:

```bash
cp .env.example .env
# Edit .env and add your key:
# QWEN_API_KEY=your_dashscope_api_key
```

Get a key at [DashScope Console](https://dashscope.console.aliyun.com/).

### 4. Run the Pipeline

```bash
# Process a single PDF
python -m src.pipeline.main path/to/paper.pdf

# Skip TF-ID if intermediate crops already exist
python -m src.pipeline.main path/to/paper.pdf --skip-tfid

# Force re-run LLM assembly
python -m src.pipeline.main path/to/paper.pdf --force-assembly
```

Output is saved to:
- `data/intermediate/{paper_name}/` — cropped figures, tables, evidence JSONs
- `data/final_output/{paper_name}_final.json` — LLM-assembled records
- `data/final_output/{paper_name}_normalized.json` — post-processed records

## Project Structure

```
FlowFigTabMiner/
├── src/
│   ├── pipeline/           # Main pipeline orchestration
│   │   ├── main.py          # Entry point (6-step pipeline)
│   │   └── figure_pipeline.py
│   ├── parsing/             # Step 1-2: PDF parsing & YOLO detection
│   │   ├── active_area_detector.py  # TF-ID (Florence-2)
│   │   ├── yolo_detector.py         # Macro segmentation
│   │   └── stage2_detector.py       # Micro detection
│   ├── extraction/
│   │   ├── figure/          # Step 3-4: Figure coordinate extraction
│   │   │   ├── coordinate_mapper.py  # RANSAC axis transform
│   │   │   └── legend_matcher.py     # Series-to-legend matching
│   │   ├── table/           # Step 3T: Table structure extraction
│   │   │   ├── pipeline.py           # Table extraction workflow
│   │   │   ├── structure.py          # TATR cell detection
│   │   │   └── scheme_seg_parser.py  # Reaction scheme → compound pool
│   │   └── common/          # Shared: OCR, MolNexTR
│   │       ├── ocr_backend.py        # PaddleOCR rec-only wrapper
│   │       ├── content_recognizer.py # Cell text/molecule recognition
│   │       └── molnextr/             # MolNexTR submodule
│   ├── adjudication/        # Step 4.5-6: LLM assembly & post-processing
│   │   ├── global_assembly.py    # Step 5: Evidence fusion
│   │   ├── local_vars_builder.py # Step 4.5: Sub-variable libraries
│   │   ├── post_processor.py     # Step 6: Normalization
│   │   └── llm_engine.py         # LLM API wrapper
│   ├── assembly/             # Evidence JSON construction
│   ├── services/             # Cloud Run FastAPI services
│   └── utils/                # Config loader
├── models/                   # Model weights (not in git, see setup)
├── config.yaml               # Pipeline configuration
├── deploy/                   # Cloud Run deployment configs
├── eval/                     # Evaluation scripts & benchmarks
├── data/
│   ├── input/                # Input PDFs
│   ├── intermediate/         # Extracted crops & evidence (runtime)
│   ├── final_output/         # Output JSON/CSV datasets
│   └── ml_lifetime/          # Kinetics analysis data
├── pub/                      # Paper & Supporting Information
├── requirements.txt          # Python dependencies
└── .env                      # API keys (not in git)
```

## Datasets

The extracted organolithium datasets are deposited on Zenodo:

**[DOI: 10.5281/zenodo.19436601](https://doi.org/10.5281/zenodo.19436601)**

| Dataset | Rows | Description |
|---------|------|-------------|
| `organolithium_tr_subdataset_vlm_enriched.csv` | 1,470 | Kinetics data from yield-vs-tR heatmaps (14 intermediates, 20 papers) |
| `organolithium_scope_table_dataset.csv` | 1,267 | Scope table data (97 papers, reaction conditions + yields) |
| `corpus_dois.csv` | 93 | DOI list of all source papers |

## Trained Models

Custom YOLO models are available on HuggingFace:

**[wyzhaoc/YOLO11](https://huggingface.co/wyzhaoc/YOLO11)**

| Model | Backbone | Task | mAP50 |
|-------|----------|------|-------|
| fig-seg | YOLOv11m | Figure macro segmentation (6 classes) | 91.7% |
| fig-sca | YOLOv11m | Figure micro detection (4 classes) | 92.5% |
| tab-seg | YOLOv11m | Table segmentation (4 classes) | 94.8% |
| tab-mol | YOLOv11s | Molecular structure detection (1 class) | 95.6% |
| tab-scheme-seg | YOLOv11n | Reaction scheme parsing (4 classes) | 96.5% |

## Cloud Deployment

The pipeline is deployed as 4 microservices on Google Cloud Run:

```bash
cd deploy
./deploy_all.sh  # Build & deploy all services
```

| Service | Purpose | Memory |
|---------|---------|--------|
| tfid-service | PDF → figure/table crops | 16 GiB (GPU) |
| figure-service | Figure → coordinate CSV | 8 GiB |
| table-service | Table → structured CSV + SMILES | 16 GiB |
| frontend-service | Web UI | 512 MiB |

## Benchmark Results

Three-way comparison on 5 figures + 5 tables:

| Task | FlowFigTabMiner | Gemini 3.1 Pro | Claude Sonnet 4.6 |
|------|-----------------|----------------|-------------------|
| Figure F1 | **0.892** | 0.575 | 0.561 |
| Table F1 | 0.827 | 0.98 | **0.99** |
| SMILES F1 | 0.795 | **0.944** | 0.867 |
| **Overall F1** | **0.838** | 0.833 | 0.806 |

## Citation

```bibtex
@article{zhao2025flowfigtabminer,
  title={FlowFigTabMiner: Multimodal Extraction of Structured Flow Chemistry Data from Figures, Tables, and Text Enables Organolithium Lifetime Prediction},
  author={Zhao, Wenyuan and Zhong, Xianzhu and Ashikari, Yosuke and Miyagishi, Hiromichi V. and Qenawy, Mohmmad S. and Wang, Simeng and Xiaer, Jouffery and Peng, Yirui and Nagaki, Aiichiro},
  year={2025}
}
```

## License

Apache 2.0
