# FlowFigTabMiner Windows Runbook

本文件总结了当前项目在 Windows 本地机器上成功运行所需的环境特点、关键改动、验证结果和批量运行建议。

## 1. 当前推荐环境

- 推荐虚拟环境：`C:\Users\18811\.conda\envs\myenv`
- 不推荐作为主运行环境：`openchemie`
- 原因：
  - `myenv` 已确认可用 CUDA，能识别本机 `RTX 4070 SUPER`
  - `openchemie` 上的 `torch` 为旧 CPU 版本，不适合当前批处理流程

已确认可用的关键基础版本：

- `torch 2.7.0+cu126`
- `torchvision 0.22.0`
- `torchaudio 2.7.0`
- `timm 0.4.12`
- `OpenNMT-py 2.2.0`
- `albumentations 1.1.0`
- `easyocr`
- `openpyxl`

## 2. Windows 运行特点

本项目原本可在 macOS 上运行，但迁移到 Windows 后，存在以下关键差异：

### 2.1 OCR 路线

- Windows 上推荐使用 `EasyOCR`
- 不建议默认走 PaddleOCR 主链
- 原因：
  - Paddle/PaddleX 在当前 Windows 环境下会涉及模型下载、临时目录和权限问题
  - `USE_EASYOCR=1` 后，流程更稳定

运行前建议设置：

```powershell
$env:USE_EASYOCR='1'
```

### 2.2 CUDA 设备选择

项目中多个 YOLO / 表格 / 分子相关模块原本没有很好适配 Windows CUDA 选择逻辑。

目前已统一修复为：

- 优先 `cuda`
- 其次 `mps`
- 最后 `cpu`

因此在 Windows + NVIDIA GPU 场景下，会优先使用显卡。

### 2.3 本地模型优先

Windows 下受网络和权限影响，更适合优先使用仓库内已有模型缓存。

当前已修复为本地优先加载的模型包括：

- TF-ID / Florence-2
- Table Transformer 结构识别模型

相关缓存目录位于：

- `models\hub\`

### 2.4 运行时缓存与第三方目录重定向

为避免第三方库把缓存、配置或模型写到系统目录导致权限问题，已新增运行时环境重定向逻辑：

- YOLO 配置目录
- Paddle/PaddleX 缓存目录
- Hugging Face 缓存目录
- EasyOCR 模型目录

相关代码：

- [runtime_env.py](C:\Users\18811\Desktop\2025-NEW\FlowFigTabMiner\FlowFigTabMiner\src\utils\runtime_env.py)

## 3. 已做的关键 Windows 适配改动

### 3.1 批处理脚本适配

文件：

- [batch_all.py](C:\Users\18811\Desktop\2025-NEW\FlowFigTabMiner\FlowFigTabMiner\scripts\batch_all.py)

改动：

- 不再写死 mac 虚拟环境路径
- 自动使用当前 Python 解释器
- 新增 `--limit`
- 支持 `--resume`
- 跳过 mac 复制过来的 `._*.pdf` 假文件

### 3.2 表格结构模型离线加载

文件：

- [structure.py](C:\Users\18811\Desktop\2025-NEW\FlowFigTabMiner\FlowFigTabMiner\src\extraction\table\structure.py)

改动：

- 优先使用仓库内 `models\hub` 的 Hugging Face snapshot
- `local_files_only=True`

### 3.3 TF-ID 本地加载

文件：

- [active_area_detector.py](C:\Users\18811\Desktop\2025-NEW\FlowFigTabMiner\FlowFigTabMiner\src\parsing\active_area_detector.py)

改动：

- 使用本地 snapshot
- `local_files_only=True`
- 兼容新版本 transformers 的 Florence-2 行为

### 3.4 OCR 后端切换

文件：

- [ocr_backend.py](C:\Users\18811\Desktop\2025-NEW\FlowFigTabMiner\FlowFigTabMiner\src\extraction\common\ocr_backend.py)
- [content_recognizer.py](C:\Users\18811\Desktop\2025-NEW\FlowFigTabMiner\FlowFigTabMiner\src\extraction\common\content_recognizer.py)
- [scheme_seg_parser.py](C:\Users\18811\Desktop\2025-NEW\FlowFigTabMiner\FlowFigTabMiner\src\extraction\table\scheme_seg_parser.py)

改动：

- 引入统一 OCR backend
- `USE_EASYOCR=1` 时走 EasyOCR
- 不再强依赖 PaddleOCR 作为唯一 OCR 路径

### 3.5 MolNexTR Windows 修复

文件：

- [chemical.py](C:\Users\18811\Desktop\2025-NEW\FlowFigTabMiner\FlowFigTabMiner\src\extraction\common\molnextr\chemical.py)

Windows 下出现过：

- `PermissionError: [WinError 5] 拒绝访问`

根因：

- MolNexTR 后处理在 Windows 上调用 `multiprocessing.Pool`
- 底层 `multiprocessing.connection.Pipe()` 被系统拒绝

当前修复：

- Windows 自动退回单进程后处理
- Linux/macOS 仍保留原并行逻辑

该修复后，Windows 上的表格分子结构识别已能正常输出 SMILES。

### 3.6 LLM / Qwen 海外端点修复

文件：

- [llm_engine.py](C:\Users\18811\Desktop\2025-NEW\FlowFigTabMiner\FlowFigTabMiner\src\adjudication\llm_engine.py)
- [config.yaml](C:\Users\18811\Desktop\2025-NEW\FlowFigTabMiner\FlowFigTabMiner\config.yaml)

当前已验证可用模型：

- `qwen-plus`
- `qwen-plus-2025-12-01`
- `qwen-turbo`

当前默认批处理轮换顺序为：

- `qwen3-max-2026-01-23`
- `qwen3.5-plus-2026-02-15`
- `qwen3.5-plus`

另外已修复：

- DashScope `max_tokens` 兼容性
- Global Assembly 的 JSON 解析问题
- 顶层 JSON 数组被错误截断的问题
- 当当前模型出现额度耗尽、配额不足、模型不可用等错误时，会自动切换到下一个模型
- 三个模型都失败后停止继续切换

### 3.7 编码统一

Windows 默认编码会导致部分 JSON 文件读取失败。

当前已修复：

- 配置文件按 `utf-8` 读取
- `final.json` 按 `utf-8` 写入
- `normalized.json` 按 `utf-8` 读写

相关文件：

- [config.py](C:\Users\18811\Desktop\2025-NEW\FlowFigTabMiner\FlowFigTabMiner\src\utils\config.py)
- [global_assembly.py](C:\Users\18811\Desktop\2025-NEW\FlowFigTabMiner\FlowFigTabMiner\src\adjudication\global_assembly.py)
- [post_processor.py](C:\Users\18811\Desktop\2025-NEW\FlowFigTabMiner\FlowFigTabMiner\src\adjudication\post_processor.py)

## 4. 已验证结果

已在 Windows 本机上成功试跑真实文献：

- 输入：`data\input\organolithium\80.pdf`

最终验收副本输出：

- [zz_codex_smoke_80_finalcheck_final.json](C:\Users\18811\Desktop\2025-NEW\FlowFigTabMiner\FlowFigTabMiner\data\final_output\zz_codex_smoke_80_finalcheck_final.json)
- [zz_codex_smoke_80_finalcheck_final.xlsx](C:\Users\18811\Desktop\2025-NEW\FlowFigTabMiner\FlowFigTabMiner\data\final_output\zz_codex_smoke_80_finalcheck_final.xlsx)
- [zz_codex_smoke_80_finalcheck_normalized.json](C:\Users\18811\Desktop\2025-NEW\FlowFigTabMiner\FlowFigTabMiner\data\final_output\zz_codex_smoke_80_finalcheck_normalized.json)
- [zz_codex_smoke_80_finalcheck_normalized.xlsx](C:\Users\18811\Desktop\2025-NEW\FlowFigTabMiner\FlowFigTabMiner\data\final_output\zz_codex_smoke_80_finalcheck_normalized.xlsx)

当前验证结果：

- `final.json`：55 条记录
- `normalized.json`：55 条记录
- Excel 导出成功
- MolNexTR 可输出 SMILES
- Qwen 海外 API 可正常返回

## 5. 当前推荐运行方式

### 5.1 跑单篇

```powershell
$env:USE_EASYOCR='1'
& 'C:\Users\18811\.conda\envs\myenv\python.exe' src\pipeline\main.py "data\input\organolithium\80.pdf"
```

### 5.2 跑批量 100 篇

```powershell
$env:USE_EASYOCR='1'
& 'C:\Users\18811\.conda\envs\myenv\python.exe' scripts\batch_all.py --limit 100
```

### 5.3 中断后续跑

```powershell
$env:USE_EASYOCR='1'
& 'C:\Users\18811\.conda\envs\myenv\python.exe' scripts\batch_all.py --resume --limit 100
```

### 5.4 夜间只做提取，不跑 LLM

如果想在晚上尽量利用 GPU 做视觉提取，同时避免消耗 Qwen 免费额度，推荐只跑 Steps 1-3.5：

```powershell
$env:USE_EASYOCR='1'
& 'C:\Users\18811\.conda\envs\myenv\python.exe' scripts\batch_all.py --stop-before-llm --limit 100
```

说明：

- 该模式会运行 Steps 1-3.5
- 会跳过 Step 4.5 `local_vars`
- 会跳过 Step 5 `Global Assembly`
- 会跳过 Step 6 `PostProcessor`

这是因为当前 Step 4.5 本身也依赖 LLM，不适合放到“纯提取”阶段执行

### 5.5 第二天只跑 LLM 组合阶段

在已经完成 Steps 1-3.5 的前提下，第二天可统一运行：

```powershell
$env:USE_EASYOCR='1'
& 'C:\Users\18811\.conda\envs\myenv\python.exe' scripts\batch_step5_6.py --limit 100
```

当前该脚本已补齐：

- Step 4.5 `local_vars`
- Step 5 `Global Assembly`
- Step 6 `PostProcessor`

## 6. 仍需注意的问题

以下问题不阻止当前 Windows 运行，但仍值得后续继续优化：

- 某些 `local_vars` 构建时仍可能遇到 `gbk` 编码相关异常
- HeaderCorrector 的 VLM 返回列数有时和预期不一致，会回退到原始表头
- 个别图表或表格的语义理解仍然依赖 LLM 输出质量

## 7. 建议

在正式跑 100 篇前，建议按以下顺序执行：

1. 先用 1-3 篇真实 PDF 小批量验证输出质量
2. 如果担心夜间耗尽免费额度，优先采用“两段式”
3. 晚上跑 `batch_all.py --stop-before-llm`
4. 白天跑 `batch_step5_6.py`
5. 若中途中断，纯提取阶段优先使用 `--resume`

## 8. 一句话结论

当前项目已经可以在 Windows + `myenv` + `RTX 4070 SUPER` + `EasyOCR` + `qwen-plus` 组合下稳定跑通单篇文献，并产出非空结构化 JSON/Excel 结果，可进入批量运行阶段。
