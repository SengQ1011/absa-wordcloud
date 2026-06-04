# ABSA Pipeline — CodePulse 使用者回饋情感分析

針對 CodePulse 教學平台的使用者回饋，以 Aspect-Based Sentiment Analysis (ABSA) 抽取四元組：

```
(aspect_term, aspect_category, opinion_term, polarity)
```

範例輸出：`("動畫", "ANIMATION", "流暢", "pos")`

---

## 專案架構

```
五階段 Pipeline
Stage 1  ATE  →  Aspect Term Extraction     (方法 A1–A4)
Stage 2  ACD  →  Aspect Category Detection  (方法 B1–B3 + Combo)
Stage 3  OTE  →  Opinion Term Extraction    (方法 C1–C3)
Stage 4  AOPE →  Aspect-Opinion Pair        (方法 D1–D5)
Stage 5  ASC  →  Sentiment Classification   (方法 F1–F3)

兩條軌道
Track A  rule-based  CKIP + Stanza + SBERT + DistilBERT   Quad F1 = 0.3282
Track B  zero-shot   Gemini 3.1 Flash Lite                Quad F1 = 0.5352
```

---

## 環境需求

| 項目 | 版本 |
|------|------|
| Python | 3.14.0 |
| OS | Windows / macOS / Linux |
| 硬體 | CPU-only（無需 GPU） |

---

## 安裝步驟

### 1. 建立虛擬環境（建議）

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. 安裝套件

```bash
pip install -r requirements.txt
```

> `torch==2.11.0` 預設安裝 CPU 版本，若需 CUDA 請改用官方安裝指令：
> https://pytorch.org/get-started/locally/

### 3. 下載 Stanza 中文模型（首次執行必做）

```bash
python -c "import stanza; stanza.download('zh-hant')"
```

CKIP 與 HuggingFace 模型會在首次呼叫時自動下載：
- **SBERT**：`paraphrase-multilingual-MiniLM-L12-v2`
- **DistilBERT**：`lxyuan/distilbert-base-multilingual-cased-sentiments-student`
- **CKIP**：`ckiplab/bert-base-chinese-ws`、`ckiplab/bert-base-chinese-pos`

### 4. 設定 Gemini API 金鑰（Track B 必做）

在專案根目錄建立 `.env`：

```
GEMINI_3.1_FLASH_LITE_API_KEY=your_api_key_here
```

> API 金鑰可至 [Google AI Studio](https://aistudio.google.com/) 取得。

---

## 資料準備

原始問卷 Excel 放在 `data/raw/`，執行前處理：

```bash
python step1_clean_data.py
```

輸出 `data/cleaned_texts.csv`（約 57 筆有效回饋）。

若已有 `data/cleaned_texts.csv` 與 `data/ground_truth_absa.csv` 可略過此步驟。

---

## 執行方式

### Track A — 規則式 Pipeline

```bash
# 完整執行（首次，需下載模型，約 5–10 分鐘）
python pipeline_track_a.py

# 使用快取（CKIP / Stanza 結果已存 cache/）
python pipeline_track_a.py --from-cache

# 執行後存快取
python pipeline_track_a.py --save-cache

# 詳細輸出
python pipeline_track_a.py --from-cache -v
```

輸出：`output/pipeline_output.csv`

### Track B — Gemini Zero-Shot

```bash
# 完整執行（會呼叫 API，速率限制 ~10 RPM）
python pipeline_track_b.py

# 使用 API 快取（避免重複計費）
python pipeline_track_b.py --from-cache

# 清除快取重新呼叫
python pipeline_track_b.py --reset
```

輸出：`output/track_b_output.csv`

### 評估 Track A / B

```bash
# 評估 Track A
python eval_track.py --track a

# 評估 Track B
python eval_track.py --track b

# 儲存量化指標
python eval_track.py --track a --save-metrics
python eval_track.py --track b --save-metrics
```

---

## 單一 Stage 評估

可分別評估各 Stage 的方法變體：

```bash
# Stage 1 — Aspect Term Extraction
python stage1_aspect_term_extraction/eval.py --from-cache --save-metrics

# Stage 2 — Aspect Category Detection（含 threshold sweep）
python stage2_aspect_category_detection/eval.py --from-cache --sweep --save-metrics

# Stage 3 — Opinion Term Extraction
python stage3_opinion_term_extraction/eval.py --save-cache --save-metrics

# Stage 4 — Aspect-Opinion Pair Extraction
python stage4_aspect_opinion_pair_extraction/eval.py --save-cache --save-metrics

# Stage 5 — Sentiment Classification
python stage5_aspect_sentiment_classification/eval.py --save-metrics
```

---

## 報表與視覺化

```bash
# 生成學術風格圖表（輸出至 reports/charts_academic/）
python generate_report_charts_academic.py

# 生成 Web Viewer JSON
python generate_viewer_data.py --track a
python generate_viewer_data.py --track b

# 生成合成訓練資料（需 Gemini API）
python generate_synthetic_data.py --target 500
```

---

## 目錄結構

```
ABSA/
├── .env                          # Gemini API 金鑰（不進版控）
├── requirements.txt
├── step1_clean_data.py           # 原始資料前處理
├── pipeline_track_a.py           # Track A 主程式
├── pipeline_track_b.py           # Track B 主程式
├── eval_track.py                 # Track A/B 評估比較
├── generate_synthetic_data.py    # 合成資料生成
├── generate_report_charts_academic.py
├── generate_viewer_data.py
├── shared/                       # 共用模組（常數、指標、工具）
├── stage1_aspect_term_extraction/
├── stage2_aspect_category_detection/
├── stage3_opinion_term_extraction/
├── stage4_aspect_opinion_pair_extraction/
├── stage5_aspect_sentiment_classification/
├── data/
│   ├── raw/                      # 原始問卷 Excel
│   ├── cleaned_texts.csv
│   └── ground_truth_absa.csv     # 人工標注 Gold Standard
├── cache/                        # CKIP / API 快取
├── output/                       # Pipeline 預測結果
├── reports/                      # 評估指標與圖表
├── viewer/                       # Web Viewer JSON
├── history/                      # 執行記錄
└── docs/                         # 設計文件、投影片
```

---

## 效能指標

| 指標 | Track A | Track B |
|------|---------|---------|
| Quad Partial F1 | 0.3282 | 0.5352 |
| Pair Partial F1 | 0.5128 | — |
| Sentiment Accuracy | 0.9000 | — |

---

## 常見問題

**Q: CKIP 初始化很慢？**
A: 首次執行會從 HuggingFace 下載模型（約 400 MB），後續使用本機快取。執行後加 `--save-cache`，下次用 `--from-cache` 跳過 tokenization。

**Q: Track B 執行中斷怎麼辦？**
A: 已完成的 API 呼叫儲存在 `cache/track_b_cache.json`，直接重跑 `python pipeline_track_b.py` 會繼續未完成的部分。

**Q: `GEMINI_3.1_FLASH_LITE_API_KEY not found` 錯誤？**
A: 確認 `.env` 檔案在專案根目錄，且 key 名稱完全符合 `GEMINI_3.1_FLASH_LITE_API_KEY`。
