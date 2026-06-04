# Stage 5 — Sentiment Classification  ITERATION_LOG

## 方案說明

| 方案 | 核心技術 | 特點 |
|------|----------|------|
| F1 | opinion_term → DistilBERT | 最簡單，不看語境 |
| F2 | "[text] [SEP] opinion_term" → DistilBERT | BERT-SPC style，帶語境 |
| F3 | polarity_hint（規則）→ fallback F1 | 規則優先，模型補漏 |

**Model**: `lxyuan/distilbert-base-multilingual-cased-sentiments-student`

## 評估設計

- **Sentiment Accuracy**：D4 Pair-TPs（29 個）中，polarity 正確的比例
- **Quadruplet Partial F1**：aspect + opinion + polarity 三者同時正確
- **D4 baseline**：Pair Partial F1 = 0.5370（TP=29/FP=29/FN=21）

## polarity_hint 覆蓋分析（2026-06-01）

eval_cache.json 中 92 pairs 的 polarity_hint 分布：
- `None`：76 / 92（83%）→ 需要模型預測
- `"neg"`：10 / 92（11%）→ 規則已知（否定前綴 / PC 規則）
- `"pos"`：6 / 92（6%）→ 規則已知（V_2 有）

F3 可直接使用規則覆蓋 16/92 pairs，其餘 76 仍需 F1 fallback。

---

## F1 / F2 / F3 橫向比較（2026-06-01，D4 整合前）

- **Problem**: 建立三種情感方法的 baseline
- **Method**: 見方案說明（model: lxyuan/distilbert-base-multilingual-cased-sentiments-student）
- **Result**（eval_cache.json 為 D4 整合前版本，Pred=92 pairs，51 rows）：

| Method | Pair F1 | Senti Acc (TPs) | Quad Partial F1 | Quad Exact F1 | QuadTP/FP/FN |
|--------|---------|-----------------|-----------------|---------------|--------------|
| F1     | 0.3915  | **0.8919** (33/37) | 0.3492       | 0.2963        | 33/56/67     |
| F2     | 0.3915  | **0.8919** (33/37) | 0.3492       | 0.3069        | 33/56/67     |
| F3     | 0.3915  | **0.8919** (33/37) | 0.3492       | 0.2963        | 33/56/67     |

> 注意：Pair F1=0.3915（37/100 GT tuples 匹配），比 D4 Stage 4 的 0.537 低，
> 因為 Stage 4 只評估 50 explicit-explicit pairs，Stage 5 評估全部 100 GT tuples。
> Sentiment Accuracy 是在匹配到的 37 個 pair-TP 上計算。

- **4 個錯誤案例分析**（F1/F3 共有，F2 修正了 row 18 但在 row 40 引入新錯誤）：

| row | aspect    | opinion   | 錯誤 | 根本原因 |
|-----|-----------|-----------|------|----------|
| 8   | implicit  | 深        | pred=pos, GT=neg | 「深」=「深奧」脈絡依賴，模型不懂 |
| 18  | implicit  | 單調      | pred=pos, GT=neg | 「單調」常見負面詞，模型誤判（F1/F3 錯，F2 正確） |
| 29  | 難度      | 高        | pred=pos, GT=neg | 「高」本身中性，需要搭配「難度」才知道是負面 |
| 30  | 畫面      | 單調      | pred=pos, GT=neg | 同 row 18，F2 改用語境仍錯（不同 pair） |

---

## D4 整合後（2026-06-01，eval_cache.json 更新）

D4 Hybrid pipeline 整合進 `eval_cache.json`（95 pairs，50 rows）後重跑：

| Method | Pair F1 | Senti Acc (TPs) | Quad Partial F1 | Quad Exact F1 | QuadTP/FP/FN |
|--------|---------|-----------------|-----------------|---------------|--------------|
| F1     | **0.5128** | **0.9000** (45/50) | 0.3282     | 0.2564        | 32/63/68     |
| F2     | **0.5128** | 0.8200 (41/50)     | 0.2769     | 0.2256        | 27/68/73     |
| F3     | **0.5128** | **0.9000** (45/50) | 0.3282     | 0.2564        | 32/63/68     |

- Pair F1: 0.3915 → **0.5128**（+0.121，D4 整合後 PairTP 37→50）
- Quad Partial F1: 0.3492 → 0.3282（**−0.021**，下降原因：新增 pair 含更多 implicit/跨類別 case，ACD category 準確率下降）
- F2 在較大 pair set 下表現反而變差（Senti Acc 0.89→0.82），原因：implicit pair 更多，text+opinion 語境混淆模型
- F1 = F3 仍然相同（rule 不覆蓋新增 pair 中的錯誤）

---

## 重構驗收（2026-06-03）

程式碼重構為 `ASC` class（`ASC.py` + `eval.py`），category enrichment 改由 `ACD.predict_combo()` 實作。

| 項目 | 說明 |
|------|------|
| F1/F2/F3 邏輯 | `ASC.predict_f1/f2/f3` = `eval_stage5.py` predict_f1/f2/f3 **完全一致** ✓ |
| Category 邏輯 | `ACD.predict_combo()` = 原 `_predict_category` Combo B2→B1→B3 **完全一致** ✓ |
| CATEGORY_SEEDS | `shared/constants.py` = `eval_stage5.py` 內嵌版本（均為 B2+ 版本）✓ |
| _term_match 行為 | implicit==implicit → True；混合 → False；邏輯等效 ✓ |

新跑分（20260603_114947）= D4 整合後版本（20260601_205653）**完全一致**，重構本身無任何影響。

**結論**：邏輯 100% 一致，重構驗收通過。

---

## 最優方案（更新後）

- **最優方案**：**F3**（polarity_hint 規則優先 → fallback F1）
- **Sentiment Accuracy**：**90.0%**（45/50 matched pairs，D4 整合後）
- **Pair Partial F1**：**0.5128**（D4 整合後，PairTP=50）
- **Quad Partial F1**：**0.3282**（aspect + opinion + category + polarity 全對）
- **決策理由**：
  - F1 = F3 在 Senti Acc 上最優（90.0%），F2 語境模式反而增加錯誤
  - F3 優先使用規則（否定前綴 / PC / V_2），符合 Track A「可解釋 Pipeline」特性
  - F3 = 確定性（規則）+ 覆蓋（模型）的最佳組合
- **瓶頸**：Quad F1 主要受 ACD category 準確率限制（implicit pair 難分類），非 sentiment 本身
