# FlowFigTabMiner (FlowDevMiner) - 系统架构与逻辑

本文档描述了重构后的 **FlowDevMiner** 4层架构体系，旨在提供高内聚、低耦合的科学文献（PDF）图表数据自动化提取方案。

---

## 1. 系统核心架构 (4-Layer Architecture)

项目采用分层架构设计，各层职责明确，方便扩展与维护：

### **第一层：输入层 (Input Layer)**
- **核心组件**: `src/flow_dev_miner/input_layer/`
- **职能**: 负责 PDF 解析、页面渲染、以及初始的布局分析。
- **关键模型**: Florence-2 (布局检测)，用于快速分割页面中的 `Figure` 和 `Table` 区域。

### **第二层：处理层 (Processing Layer)**
- **核心组件**: `src/flow_dev_miner/processing_layer/`
- **职能**: 针对不同类型的媒体（图、表、文本）进行深度特征提取。
  - **FigureProcessor**: 处理散点图/柱状图。包含 Macro Cleaning (YOLO 分割)、Micro Detection (数据点检测)、Legend Matching 和 Coordinate Mapping (像素到数值转换)。
  - **TableProcessor**: 处理化学/科学表格。包含 Table Segmentation (YOLO)、**MolNexTR** (化学分子结构到 SMILES 转换)、Structure Recognition (TATR 表格变换器) 和 Cell OCR。
  - **TextProcessor**: 负责全文文本的清洗与段落提取。

### **第三层：算法与推理层 (Algorithm Layer)**
- **核心组件**: `src/flow_dev_miner/algorithm_layer/`
- **职能**: “系统的大脑”，处理语义逻辑。
  - **SemanticAlgorithmCore**: 负责将处理层产出的原始数据（像素坐标、CSV 单元格）与论文全文背景进行语义对齐。
  - **Global Variable Pool (GVP)**: 提取论文中的全局变量（如通用温度、压力、催化剂缩写定义）。
  - **Context Resolver**: 利用 LLM 技术解决数据点与其上下文（Caption/Note）的逻辑关系。

### **第四层：输出与可视化层 (Output Layer)**
- **核心组件**: `src/flow_dev_miner/output_layer/`
- **职能**: 数据的序列化与可视化验证。
  - **DataExporter**: 生成标准化的 `final_summary.json`。
  - **VisualAssembler**: 生成可视化评估报告（将提取结果与原图对比拼接），方便人工审计。

---

## 2. 关键技术细节

### **化学分子识别 (MolNexTR Integration)**
- 采用 **MolNexTR** 技术替代了早期的 MolScribe，提供了更强大的 MPS (Apple Silicon) 加速支持。
- **流程**: YOLO 检测单元格分子 -> 裁剪 -> MolNexTR 推理 -> 生成 SMILES。

### **数据召回逻辑 (Evidence Harvesting)**
- 每个图表在处理后会生成一个自包含的 `*_evidence.json` 数据包。
- 只有包含关键词（如 `yield`, `selectivity`, `conversion` 等）的图表才会被判定为 `is_relevant: true` 并进入最终的数据组装环节。

---

## 3. 主要入口说明

- **全量流水线**: `scripts/run_pipeline_api.py` (推荐入口)
- **可视化 Web 应用**: `scripts/viz_app.py` (Streamlit 界面)
- **模型路径配置**: `config.yaml`
- **关键依赖/环境**: `flowfigtabminer` 虚拟环境。

---

## 4. 目录映射

```text
src/
├── flow_dev_miner/
│   ├── input_layer/        # PDF 管理与布局
│   ├── processing_layer/   # 图表核心识别引擎
│   ├── algorithm_layer/    # LLM 组装与语义推理
│   └── output_layer/       # 导出与可视化
├── models/                 # 全局权重文件 (*.pth, *.pt)
└── scripts/                # API 脚本与应用入口
```
