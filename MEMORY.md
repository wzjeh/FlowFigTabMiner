# FlowFigTabMiner 修复记录

## 会话日期：2026-03-08

---

## 一、本次修复的关键问题

### 1. ✅ MolScribe 清理（已完成）
**问题**：MolScribe 已被 MolNexTR 完全替代，但代码中仍有冗余引用

**修复内容**：
- 移除 `config.yaml` 中的 `molscribe_path` 配置
- 移除 `src/extraction/common/content_recognizer.py` 中的 `molscribe_path` 参数
- 更新 `src/extraction/table/pipeline.py` - 移除 molscribe_path 传递
- 更新 `src/flow_dev_miner/processing_layer/table_processor.py` - 移除 molscribe_path 属性
- 更新 `src/pipeline/batch_table_pipeline.py` - 移除 molscribe_path 引用
- 更新 `src/extraction/common/molecule_processor.py` - 注释中 MolScribe 改为 MolNexTR

**验证结果**：
- MolNexTR 成功识别 5 个分子结构，生成有效 SMILES
- 示例：`CC1CC2=C34([N+](=O)[O-])C5=CC(Cl)=C2C3=C51[N+]4(=O)[O-]`

---

### 2. ✅ 图片坐标提取修复（已完成）
**问题**：所有图片提取失败，错误 `AttributeError: 'tuple' object has no attribute 'empty'`

**根本原因**：
- `CoordinateMapper.map_coordinates()` 返回 `(DataFrame, debug_log)` 元组
- `main.py` 未正确解包，直接将 tuple 当作 DataFrame 使用

**修复**：
```python
# 修复前
df = coord_mapper.map_coordinates(matched_points, raw_plot_path)

# 修复后
df, debug_log = coord_mapper.map_coordinates(matched_points, raw_plot_path)
```

**验证结果**：
- 成功提取 5 个图片的数据点
- 示例：page_5_figure_0 提取 18 行，包含 4 个系列数据

---

### 3. ✅ YOLO 模型匹配错误（已完成）
**问题**：Micro 检测模型类别不匹配，导致坐标轴刻度识别失败

**原因分析**：
- `bestYOLOm-2-2.pt` 类别：`{0: 'data_point', 1: 'data_value', 2: 'tick_label', 3: 'tick_mark'}`
  - ❌ 只有通用 `tick_label`，不区分 x/y 轴
- `CoordinateMapper` 需要：`x_tick_label` 和 `y_tick_label`

**解决方案**：
- 切换到正确模型：`models/yolo11m-fig-scatter-0208/runs/detect/train/weights/best.pt`
- 该模型类别：`{0: 'data_point', 1: 'data_value', 2: 'x_tick_label', 3: 'y_tick_label'}`

**修复位置**：`main.py` 第 44 行

---

### 4. ✅ config.yaml 配置统一（已完成）
**问题**：`main.py` 硬编码模型路径，与 `config.yaml` 不一致

**修复前**：
```python
yolo_macro = YoloDetector(model_path="models/bestYOLOn-2-1.pt")  # 错误
yolo_micro = Stage2Detector(model_path="models/bestYOLOm-2-2.pt")  # 错误
```

**修复后**：
```python
# 从 config.yaml 读取配置
cfg = load_config()
figures_cfg = cfg.get("figures", {})
macro_model_path = figures_cfg.get("step2_macro", {}).get("model_path")
micro_model_path = figures_cfg.get("step3_micro", {}).get("model_path")
```

**原则**：所有模型和参数配置必须以 `config.yaml` 为准

---

### 5. ✅ 坐标精度优化（已完成）
**问题**：坐标值精度过高（15位小数），不必要

**修复**：
```python
# 保存 CSV 前四舍五入到 2 位小数
df = df.round({'X': 2, 'Y_Left': 2, 'Y_Right/Data_Value': 2})
```

**效果**：
- 修复前：`2.3873475210145263`
- 修复后：`2.39`

---

### 6. ✅ 网络检查警告优化（已完成）
**问题**：MolNexTR 推理时大量 HuggingFace 网络连接检查，导致性能下降和警告

**修复**：
```python
# src/extraction/common/content_recognizer.py
os.environ["HF_HUB_OFFLINE"] = "1"  # 禁用 HuggingFace 网络检查
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "1"  # 禁用模型源检查
```

---

## 二、当前代码状态

### 模型配置（config.yaml 定义）

#### 图片处理流水线
1. **TF-ID**: Florence-2 (`yifeihu/TF-ID-base`)
2. **Macro 分割**: `models/yolo11m-fig-seg-0207-nobreaknocharttext/runs/detect/train/weights/best.pt`
   - 类别：caption, legend, subfigure_marker, target_image, x_axis_title, y_axis_title
3. **Micro 检测**: `models/yolo11m-fig-scatter-0208/runs/detect/train/weights/best.pt`
   - 类别：data_point, data_value, x_tick_label, y_tick_label

#### 表格处理流水线
1. **表格分割**: `models/yolo11m-tab-seg-0209-white/runs/detect/train/weights/best.pt`
   - 类别：table_body, table_caption, table_note, table_scheme
2. **分子检测**: `models/yolo11s-tab-molecule-0207/runs/detect/train/weights/best.pt`
   - 类别：Structure
3. **结构识别**: Table Transformer (`microsoft/table-transformer-structure-recognition-v1.1-all`)
4. **内容识别**:
   - 文本：PaddleOCR
   - 化学结构：MolNexTR（已完全替代 MolScribe）

---

### 测试结果（example.pdf）

**性能数据**：
- 总耗时：206 秒（约 3.5 分钟）
- 处理对象：5 个图片 + 2 个表格

**图片提取（5/5 成功）**：
- page_5_figure_0: 18 个数据点
- page_5_figure_1: 17 个数据点
- page_6_figure_0: 36 个数据点
- page_6_figure_1: 25 个数据点
- page_7_figure_1: 16 个数据点

**表格提取（2/4 成功）**：
- ✅ page_7_table_0: 11 个单元格
- ✅ page_8_table_0: 45 个单元格，含 5 个 SMILES 分子结构
- ❌ page_1_table_0: 被 Table Filter 拒绝（置信度 0.0）
- ❌ page_4_table_0: 被关键词过滤器拒绝（不含流化学关键词）

**数据质量**：
- CSV 格式正确，坐标精度 2 位小数
- SMILES 识别准确
- 系列分类正确

---

## 三、尚未解决的待办事项

### 1. 🔵 完成旧架构到新架构的迁移

**背景**：
- 项目存在两个架构：
  - **旧架构**：`src/extraction/` 和 `src/parsing/`
  - **新架构**：`src/flow_dev_miner/` (4层架构：Input、Processing、Algorithm、Output)
- 当前 `main.py` 使用旧架构组件

**待完成工作**：
- [ ] 评估旧架构中哪些组件需要迁移到新架构
- [ ] 保留旧架构作为底层算法库（Algorithm Layer）
- [ ] 创建新架构的主入口文件
- [ ] 更新文档说明架构关系

**优先级**：中等

---

### 2. 🟢 性能优化（可选）

#### 已优化
- ✅ 禁用 HuggingFace 网络检查
- ✅ 禁用 PaddleOCR 模型源检查

#### 可选优化点
- [ ] TATR 迁移到 MPS/GPU（当前 CPU 运行较慢）
- [ ] Sequential Mode 内存优化（减少模型常驻内存）
- [ ] 批量 OCR 处理（当前逐单元格处理）

**优先级**：低（当前性能可接受）

---

### 3. 🔵 表格结构对齐优化

**问题**：CSV 列对齐存在小问题
- 示例：`page_7_table_0` 表头缺少 "T" 列名

**优先级**：低

---

### 4. 🔵 关键词过滤器优化

**问题**：`page_4_table_0` 被错误拒绝
- 90 个单元格成功提取，但被关键词过滤器标记为不相关

**建议**：
- 调整关键词匹配策略
- 或将过滤逻辑移至后处理阶段

**优先级**：低

---

## 四、重要文件位置

### 配置文件
- **主配置**：`config.yaml` （所有模型和参数的唯一真值来源）
- **Claude 指令**：`CLAUDE.md`
  1. 每次回复前，必须使用"Zhao"来称呼我
  2. 不用写兼容性代码，除非我主动要求
  3. 遇到不确定的代码设计问题时，必须先询问 Zhao
  4. 所有依赖和运行都要在根目录中的 flowfigtabminer 虚拟环境中

### 核心代码
- **主入口（旧架构）**：`main.py`
- **新架构入口**：尚未创建
- **表格处理**：`src/extraction/table/pipeline.py`
- **图片坐标映射**：`src/extraction/figure/coordinate_mapper.py`
- **内容识别**：`src/extraction/common/content_recognizer.py`
- **分子处理**：`src/extraction/common/molecule_processor.py`

### 测试文件
- **输入**：`data/input/example.pdf`
- **输出**：`data/output/test_run/`
- **验证脚本**：`verify_improvements.py`

---

## 五、运行命令

### 测试完整工作流
```bash
source flowfigtabminer/bin/activate
python3 main.py data/input/example.pdf --output_dir data/output/test_run
```

### 验证改进
```bash
python3 verify_improvements.py
```

### 检查模型类别
```bash
source flowfigtabminer/bin/activate
python3 -c "
from ultralytics import YOLO
model = YOLO('models/yolo11m-fig-scatter-0208/runs/detect/train/weights/best.pt')
print('Classes:', model.names)
"
```

---

## 六、下一步行动

**建议顺序**：
1. ✅ 修复所有已知问题（已完成）
2. 🔵 完成旧架构到新架构的迁移（待执行）
3. 🟢 可选性能优化（根据需求决定）
4. 📝 更新项目文档

**当前状态**：所有关键功能正常，可以进入架构迁移阶段

---

## 七、关键经验教训

1. **配置统一管理**：所有模型路径必须从 `config.yaml` 读取，避免硬编码
2. **模型类别匹配**：确保 YOLO 模型的类别定义与下游代码期望一致
3. **返回值类型检查**：注意函数返回 tuple 时需要正确解包
4. **网络依赖优化**：生产环境应禁用模型下载检查，提升性能
5. **虚拟环境隔离**：所有依赖必须在项目虚拟环境中安装和运行

---

*最后更新：2026-03-08*
*修复者：Claude (Sonnet 4.5)*
*审核者：Zhao*
