# FlowFigTabMiner — Web 端架构总结

> 最后更新：2026-03-20
> 负责人：Nagaki Lab, Hokkaido University

---

## 一、整体架构

```
用户浏览器
    │
    ▼
[frontend-service]  ←── FastAPI Web UI（密码保护）
    │
    ├─ POST /extract          上传图片 → 调用 figure/table service
    ├─ POST /extract-example  使用内置示例 → 调用 figure/table service
    └─ POST /feedback         收集错误反馈 → 保存至 GCS
    │
    ├──────────────────────────────────────────────────┐
    ▼                                                  ▼
[figure-service]                              [table-service]
YOLO macro clean                              YOLO filter
YOLO micro detect                             TATR structure
Coord mapping (EasyOCR)                       PaddleOCR + MolNexTR
Legend matching (EasyOCR)                     CSV output
CSV output
    │                                                  │
    └─────────────────┬────────────────────────────────┘
                      ▼
             [GCS: flowfigtabminer-data]
             frontend/{job_id}/image.png    ← 原始上传图
             {job_id}/result.csv            ← 提取结果
             feedback/error/{job_id}__...   ← 用户标记的错误图片

             [GCS: flowfigtabminer-models]  ← 模型权重（FUSE挂载）
```

---

## 二、Cloud Run 服务一览

| 服务 | URL | 内存 | 镜像基础 |
|------|-----|------|----------|
| frontend-service | https://frontend-service-903119038444.us-central1.run.app | 512Mi | python:3.10-slim |
| figure-service   | https://figure-service-903119038444.us-central1.run.app   | 8Gi   | python:3.10-slim |
| table-service    | https://table-service-903119038444.us-central1.run.app    | 16Gi  | python:3.10-slim |
| tfid-service     | https://tfid-service-903119038444.us-central1.run.app     | 16Gi  | pytorch/pytorch (GPU) |

- **GCP 项目 ID**：`gen-lang-client-0080522548`
- **Region**：`us-central1`
- **超时**：所有服务 3600s（Cloud Run 最大值）
- **访问**：`--allow-unauthenticated`（前端自带密码保护）

---

## 三、前端服务（frontend-service）

### 文件
- `src/services/frontend_service.py` — 全部逻辑（单文件）
- `src/services/examples/example_figure.png` — 示例图表
- `src/services/examples/example_table.png` — 示例表格
- `Dockerfile.frontend` — 基于 python:3.10-slim
- `requirements-frontend.txt` — fastapi, uvicorn, httpx, google-cloud-storage, Pillow

### 路由
| 路由 | 方法 | 功能 |
|------|------|------|
| `/` | GET | 返回完整 HTML 页面 |
| `/login` | POST | 密码验证，返回 HMAC token |
| `/extract` | POST | 接收上传图片，存 GCS，调用后端服务 |
| `/extract-example` | POST | 使用内置示例图，调用后端服务 |
| `/feedback` | POST | 把错误图片复制到 GCS feedback/error/ |
| `/examples/*` | GET (static) | 静态文件服务（示例图） |

### 关键设计
- **认证**：可选密码门（由环境变量 `FRONTEND_PASSWORD` 控制）→ 通过密码后发放 HMAC-SHA256 token → Cookie（24h）
- 默认 `FRONTEND_PASSWORD` 为空，即**开放访问**；如需限制，部署时通过 Secret Manager 注入该环境变量
- 图片上传后先存到 `gs://flowfigtabminer-data/frontend/{job_id}/filename`，再传给后端服务
- 后端服务处理完后返回 `csv_gcs_uri`，前端下载并渲染为表格
- 后端调用失败时自动重试一次（间隔 5s）

### 反馈功能
- 提取成功有数据时，表格下方显示 **"✗ Report Errors"** 按钮
- 点击后调用 `/feedback`，把原始图片从 `frontend/{job_id}/` 复制到 `feedback/error/`
- GCS 路径格式：`gs://flowfigtabminer-data/feedback/error/frontend__{job_id}__filename.png`

---

## 四、Figure 服务（figure-service）

### 处理流程
```
输入图片（GCS URI）
  → Step 1: YOLO macro 清洗（去除非图表区域）
  → Step 2: YOLO micro 检测（坐标点、轴标签、legend）
  → Step 3: 坐标映射 + EasyOCR 识别轴刻度/标签
  → Step 4: Legend 匹配 + EasyOCR 识别系列名称
  → Step 5: 关键词过滤（yield/selectivity/conversion 等）
  → 输出：CSV（series, x, y）存入 GCS
```

### 关键文件
- `src/services/figure_service.py` — FastAPI 入口
- `src/pipeline/figure_pipeline.py` — 主流程
- `src/extraction/figure/coordinate_mapper.py` — 坐标映射
- `src/extraction/figure/legend_matcher.py` — Legend 识别
- `src/assembly/evidence_assembler.py` — 证据组装
- `src/extraction/common/ocr_backend.py` — OCR 后端切换（EasyOCR / PaddleOCR）

### OCR 架构（重要）
- **Cloud Run 上**：`USE_EASYOCR=1` → 使用 EasyOCR（PyTorch，无 PaddlePaddle PIR 问题）
- **本地**：不设置该变量 → 使用 PaddleOCR（本地逻辑不变）
- `ocr_backend.py` 的 `_EasyOCRWrapper` 把 EasyOCR 输出转换为 PaddleOCR 格式，所有下游代码无需修改

### 模型
- YOLO macro: `models/macro_best.pt`
- YOLO micro: `models/micro_best.pt`
- EasyOCR: 英文模型（首次请求时从网络下载，后续缓存）
- 模型通过 GCS FUSE 挂载到 `/models/models/`，entrypoint.sh 建立 `/app/models` 软链接

### 已知行为
- Heatmap 自动检测：legend 文字匹配 `\d+~\d+%` 或 `[<>]\d+%` 时触发
- `_is_relevant_chart` 返回 `(bool, text_evidence)` 元组（已知 bug，不影响主流程，勿动）

---

## 五、Table 服务（table-service）

### 处理流程
```
输入图片（GCS URI）
  → YOLO 过滤（确认是表格）
  → TATR 结构识别（行列检测）
  → PaddleOCR 文字识别
  → MolNexTR SMILES 提取（化学结构）
  → 关键词过滤（软过滤，仍输出 CSV）
  → 输出：CSV 存入 GCS
```

### 关键文件
- `src/services/table_service.py` — FastAPI 入口
- `src/extraction/table/pipeline.py` — 主流程（关键词过滤在 ~386 行）
- `src/extraction/common/content_recognizer.py` — PaddleOCR 封装
- `src/extraction/common/molnextr/` — MolNexTR 化学结构识别

### MolNexTR 加载优化
- 模型文件 1.06 GiB，GCS FUSE 随机读取极慢
- 修复：先用 `shutil.copy2` 把模型复制到 `/tmp/`，再 `torch.load`（顺序读取，速度大幅提升）
- 冷启动总时长约 25 分钟（含 TATR、PaddleOCR、MolNexTR 加载）

---

## 六、部署流程

### 构建镜像
```bash
# 正确方式：用 --config 指定 cloudbuild yaml（不能用 -f 标志）
gcloud builds submit --config deploy/cloudbuild-frontend.yaml --project gen-lang-client-0080522548 .
gcloud builds submit --config deploy/cloudbuild-figure.yaml   --project gen-lang-client-0080522548 .
gcloud builds submit --config deploy/cloudbuild-table.yaml    --project gen-lang-client-0080522548 .
gcloud builds submit --config deploy/cloudbuild-tfid.yaml     --project gen-lang-client-0080522548 .
```

> ⚠️ `gcloud builds submit --tag ... -f Dockerfile.xxx` 的 `-f` 标志无效，会静默失败，
> 导致部署旧镜像。必须用 `--config` 方式。

### 部署服务
```bash
gcloud run deploy frontend-service \
  --image gcr.io/gen-lang-client-0080522548/frontend-service:latest \
  --region us-central1 --project gen-lang-client-0080522548

# figure / table / tfid 同理，替换服务名即可
```

### 一键部署脚本
```bash
bash deploy/deploy_all.sh
```

---

## 七、GCS 存储结构

```
gs://flowfigtabminer-models/
└── models/
    ├── best.pt                        # YOLO 权重（软链接 /app/models）
    ├── macro_best.pt
    ├── micro_best.pt
    ├── molnextr_model_best.pth        # MolNexTR 1.06 GiB
    ├── hub/                           # HuggingFace 缓存（Florence-2, TATR）
    └── paddlex/official_models/       # PaddleOCR 3.x 模型缓存

gs://flowfigtabminer-data/
├── frontend/
│   └── {job_id}/
│       └── uploaded_image.png         # 用户上传的原始图片
├── {job_id}/
│   └── result.csv                     # 提取结果
└── feedback/
    └── error/
        └── frontend__{job_id}__*.png  # 用户标记为错误的图片
```

---

## 八、环境变量汇总

| 变量 | 服务 | 说明 |
|------|------|------|
| `FRONTEND_PASSWORD` | frontend | 可选网页访问密码（默认空 = 开放访问） |
| `FRONTEND_SECRET` | frontend | HMAC 签名密钥 |
| `FIGURE_SERVICE_URL` | frontend | figure 服务地址 |
| `TABLE_SERVICE_URL` | frontend | table 服务地址 |
| `USE_EASYOCR` | figure | `1` = 使用 EasyOCR，不设置 = PaddleOCR |
| `FLAGS_use_mkldnn` | figure/table | `0` 禁用 OneDNN（PaddlePaddle 3.x 必须） |
| `FLAGS_pir_apply_mkldnn_pass` | figure/table | `0` |
| `FLAGS_enable_pir_api` | figure/table | `0` |
| `PADDLE_PDX_CACHE_HOME` | figure/table | PaddleX 模型缓存目录 |
| `HF_HOME` | tfid/table | HuggingFace 缓存目录 |
| `HF_HUB_OFFLINE` | tfid/table | `1` 禁止网络下载 |
| `DISABLE_MODEL_SOURCE_CHECK` | figure | 禁止 PaddleX 启动时网络检查 |

---

## 九、本地开发

```bash
# 激活虚拟环境（所有命令在此环境下运行）
source flowfigtabminer/bin/activate

# 本地运行前端（需要 ADC 认证访问 GCS）
gcloud auth application-default login
uvicorn src.services.frontend_service:app --host 0.0.0.0 --port 8080

# 本地跑完整 pipeline
python src/pipeline/main.py --input path/to/paper.pdf
```

---

## 十、待办 / 未来方向

- [ ] 自定义域名绑定（如 `flowfigtabminer.com`）—— 需先购买域名
- [ ] EasyOCR 模型预烘焙到 Docker 镜像（避免冷启动下载）
- [ ] 利用 `feedback/error/` 数据优化 YOLO / OCR 模型
- [ ] 支持 PDF 直接上传（目前只支持图片）
- [ ] table-service 冷启动优化（目前约 25 min）
