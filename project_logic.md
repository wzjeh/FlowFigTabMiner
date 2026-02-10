# FlowFigTabMiner 项目逻辑文档

本文档详细描述了 FlowFigTabMiner 项目的数据提取流程和逻辑结构。

## 1. 核心流程概览

整个项目的目标是从 PDF 论文中提取结构化的数据，主要分为以下几个步骤：

1.  **PDF 预处理与区域检测 (TF-ID)**
2.  **Figure (图片) 数据提取**
3.  **Table (表格) 数据提取**
4.  **全局组装与 LLM 推理**

---

## 2. 详细步骤说明

### 第一步：TF-ID (Table/Figure Identification)
**目标**: 从 PDF 页面中切割出包含图片和表格的区域。
- **输入**: 原始 PDF 文件。
- **工具**: `ActiveAreaDetector` (基于 Transformers/Object Detection)。
- **输出**: 
    - 图片裁剪: 保存至 `data/intermediate/{pdf_name}/figures/`
    - 表格裁剪: 保存至 `data/intermediate/{pdf_name}/tables/` (通常是包含标题的原始区域)

### 第二步部分 A：Figure 提取流水线
**目标**: 从散点图 (Scatter Plot) 等图片中提取数据点。
1.  **Macro Cleaning (宏观清洗)**
    - **工具**: `YoloDetector` (模型: `models/yolo11m-fig-seg-0207-nobreaknocharttext`)
    - **逻辑**: 识别并去除图片中的非数据元素（如标题、轴标签），得到 clean 的绘图区域。
    - **输出**: `data/intermediate/{pdf_name}/macro_cleaned/` 中的清理后图片。
2.  **Micro Detection (微观检测)**
    - **工具**: `Stage2Detector` (模型: `models/yolo11m-fig-scatter-0208`)
    - **逻辑**: 在清理后的图片上检测数据点 (markers) 和图例 (legends)。
3.  **Legend Matching (图例匹配)**
    - **工具**: `LegendMatcher`
    - **逻辑**: 将检测到的图例与数据点进行匹配，确定每组数据的类别。
4.  **Coordinate Mapping (坐标映射)**
    - **工具**: `CoordinateMapper`
    - **逻辑**: 识别 X/Y 轴的刻度值，建立像素坐标到数据坐标的映射关系，将像素点转化为实际数值。
5.  **Assembly (组装)**
    - **工具**: `EvidenceAssembler`
    - **逻辑**: 结合 OCR 识别的图片标题 (Caption) 和提取的数据点，生成 JSON Evidence。

### 第二步部分 B：Table 提取流水线
**目标**: 从表格图片中提取结构化数据和化学分子信息。
1.  **Segmentation (分割)**
    - **工具**: `TableFilter` (模型: `models/yolo11m-tab-seg-0208`)
    - **逻辑**: 将表格区域分割为 Header, Body, Footer 等部分。仅保留 Body 部分用于后续处理。
2.  **Molecule Detection & Replacement (分子检测与替换)**
    - **工具**: `MoleculeProcessor` (模型: `models/yolo11s-tab-molecule-0207`) + `MolScribe`
    - **逻辑**: 
        - 使用 YOLO 检测表格中的化学分子结构。
        - 使用 MolScribe 将分子图片转化为 SMILES 字符串。
        - **关键优化**: 在原图中用 SMILES 文本替换分子图片，以便后续 OCR 识别为文本。
3.  **Structure Recognition (结构识别)**
    - **工具**: `TableStructureRecognizer` (模型: `microsoft/table-transformer-structure-recognition-v1.1-all`)
    - **逻辑**: 识别表格的行、列和单元格 (Cell) 结构。
4.  **Cell Content Recognition (单元格内容识别)**
    - **工具**: `ContentRecognizer` (CRNN/PaddleOCR)
    - **逻辑**: 对每个单元格进行 OCR 识别。如果单元格位置与之前的分子检测重叠，则直接使用 SMILES 字符串。
5.  **Assembly (组装)**
    - **逻辑**: 将识别出的文本按行/列组织成 DataFrame，并结合 OCR 识别的 Caption 和 Note，生成 JSON Evidence。

### 第三步：全局组装 (Global Assembly)
**目标**: 汇总所有提取的信息，利用 LLM 进行最终的语义理解和数据清洗。
- **输入**: 
    - PDF 全文文本 (截断以适应 Context Window)。
    - Figure Evidence JSONs (数据点)。
    - Table Evidence JSONs (表格数据)。
- **逻辑**: 
    - 使用 `scripts/step5_global_single.py`。
    - 构建 Prompt，包含全文背景和所有提取的数据。
    - 调用 LLM (如 Qwen/OpenAI) 补充上下文信息（如实验条件、反应物完整名称等），将分散的数据点组装成最终的结构化数据集。
- **输出**: `data/final_output/{pdf_name}_final_summary.json`

---

## 3. 代码结构映射

- **Pipeline 入口**: `scripts/run_pipeline_steps_1_to_4.py` (处理 Figure), `scripts/run_batch_tables.py` (处理 Table 批量)
- **Figure 逻辑**: `src/pipeline/figure_pipeline.py`
- **Table 逻辑**: `src/pipeline/batch_table_pipeline.py` (优化版), `src/extraction/table_pipeline.py` (核心类)
- **LLM 逻辑**: `src/adjudication/llm_engine.py`, `scripts/step5_global_single.py`
