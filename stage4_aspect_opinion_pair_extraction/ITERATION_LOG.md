# Stage 4 — Aspect-Opinion Pairing  ITERATION_LOG

## 設計基準

- **Input 固定**：A3 aspects（Stage 1 最優）+ C2 opinions（Stage 3 最優）
- **GT 範圍**：GT 中 aspect_term ≠ implicit AND opinion_term ≠ implicit 的 pairs（50 pairs / 29 rows）
- **評估指標**：Pair F1（exact + partial），greedy matching
- **Stage 依賴鏈**：用 stage1_cache.json（CKIP）重跑 A3+C2，不用 Gold 輸入

---

## 基線 — D1/D2/D3 橫向比較（2026-06-01）

- **Problem**: 建立三種配對策略的 baseline
- **Method**:
  - D1: 線性最近鄰（每個 opinion 配 token 距離最近的 aspect）
  - D2: 子句邊界（逗號分隔，同子句內最近；fallback D1）
  - D3: Stanza nsubj/amod/advmod 依存樹，raw pairs 過濾至 A3+C2
- **Result**:

| Method | Exact F1 | Partial F1 | TP/FP/FN |
|--------|----------|------------|----------|
| D1     | 0.3208   | 0.4528     | 24/32/26 |
| D2     | 0.3396   | 0.4717     | 25/31/25 |
| D3     | 0.2857   | 0.4000     | 14/6/36  |

- **Impact**: D2 > D1（子句邊界有效），D3 高 Precision（0.70）但 Recall 僅 0.28（stanza nsubj 漏連多）

---

## D2+ — find_span_start 否定合成詞修正 + D4 Hybrid（2026-06-01）

- **Problem**: C2 產生否定合成詞（如「不明確」= 不 + skip 太 + 明確），`find_span_start` 無法定位（tokens: 不/太/明確），導致 row 3、9、40 否定詞完全跳過配對
- **Method**:
  1. `find_span_start` 新增 negation-with-intensifier-skip 路徑：
     單字否定前綴（不/沒/非/未）+ 跳過 NEGATION_INTENSIFIERS + base 詞 → 返回前綴 index
  2. D4 Hybrid：D3 有配對的 opinion 用 D3（高精度）；其餘 opinion 用 D2（高覆蓋）
- **Result**:

| Method | Exact F1 | Partial F1 | TP/FP/FN | Δ Partial F1 |
|--------|----------|------------|----------|-------------|
| D2     | 0.3303   | 0.4954     | 27/32/23 | +0.024      |
| D4     | 0.3670   | 0.5321     | 29/30/21 | **+0.060**  |

- **Impact**: Row 3 D2 從 1→2 TP（不明確 配到指引）；Row 29 D4 4 TP（Manual Sort/好玩 from D3 + 程式/簡單/基礎 from D2）；Row 40 D1/D2 完整 2 TP

---

## D4+ — Opinion-only 詞排除 A3 aspect 提取（2026-06-01）

- **Problem**: CUSTOM_OPINION_WORDS（直觀/牛逼/卡/掛/便利）+ VHC_WHITELIST 若 CKIP 標 Na，會被 A3 提取為 aspect；導致 (直觀, 深)、(牛逼, 牛逼) 等無意義 FP pairs
- **Method**: `extract_a3` 加一道 CUSTOM_OPINION_WORDS ∪ VHC_WHITELIST 排除過濾
- **Result**:

| Method | Exact F1 | Partial F1 | TP/FP/FN | Δ vs prev |
|--------|----------|------------|----------|-----------|
| D4     | 0.3704   | **0.5370** | 29/29/21 | +0.005    |

- **Impact**: Row 8 FP (直觀, 深) 消除；Row 41 FP (牛逼, 牛逼) 消除；TP 不變（29）

---

## D5 — 跨子句 Cosine 配對（PC rule implicit 改善嘗試）（2026-06-01）

- **Problem**: PC rule 產生 aspect="implicit" 的 pairs（看不懂/不知道 等），D4 透過 D2 fallback 已給予 explicit aspect，但可能配到錯誤的 aspect（如 row 15：D4 配 `測驗/不知道`，GT = `內容/不知道`）
- **Method**: D5 兩層演算法：
  - 層一：PC opinion 同子句有 A3 aspects → 取最近者（D2-style）
  - 層二：跨子句展開，跳過「已有配對子句」，cosine(MiniLM) 評分，超過 threshold 才配對
  - Threshold sweep：0.20–0.80 步進 0.05（共 13 個值）
- **Result**:

| Method | Partial F1 | TP/FP/FN | Δ vs D4 |
|--------|------------|----------|---------|
| D5@best(0.20) | **0.5370** | 29/29/21 | **0.0000** |

所有 threshold 結果完全相同，D5 無任何新增配對。

- **Root Cause**:
  1. GT 中僅 5 個 PC opinions（看不懂×3、不知道×2）
  2. 全部已被 D4 的 D2 fallback 配對 → `op in d4_op_set` 全部跳過
  3. 唯一 FN（row 15：測驗/不知道 → 應為 內容/不知道）：兩者在**同一子句**，D5 層一也是 proximity，仍選 `測驗`（較近）；層二 cosine 永遠不觸發
- **Impact**: D5 對此資料集無效，問題本質是子句內語意角色辨別（非跨子句距離）

---

## 重構驗收（2026-06-03）

程式碼重構為 `AOPE` class（`AOPE.py` + `eval.py`），constants 移至 `shared/constants.py`，配對邏輯與重構前 `stage4_eval.py` **完全一致**（D1/D2/D3/D4/D5、`_find_span_start`、`_resolve`、`_clause_ranges`、`_stanza_raw_pairs` 全部對等）。

唯一常數差異：`stage4_eval.py` 的 `CKIP_ASPECT_POS` 含 Nf（量詞），`shared/constants.py` 不含；Nf 幾乎不出現於 aspect term，D3 TP 仍改善（14→15），無負面影響。

| Method | LOG 記錄（D4+ 最優期）| 新跑分 | TP/FP/FN | Δ Partial F1 |
|--------|-----------------------|--------|----------|-------------|
| D1     | 0.4528                | **0.5556** | 30/28/20 | +0.1028 |
| D2     | 0.4954                | **0.5741** | 31/27/19 | +0.0787 |
| D3     | 0.4000                | **0.4225** | 15/6/35  | +0.0225 |
| D4     | 0.5370 (29/29/21)     | **0.6111** | 33/25/17 | **+0.0741** |
| D5     | 0.5370（= D4）        | 0.6111（= D4）| — | 仍無新增配對 |

**分數改善根本原因**：新版 eval 使用重構後 `ATE.extract_a3` 作為 Stage 4 輸入，Stage 1 驗收已確認 A3 Partial F1 從 0.7273 提升至 0.7351（+0.0078），aspect 覆蓋改善後**級聯影響配對品質**。C2 opinion 輸出與舊版完全一致（Stage 3 驗收確認：63/10/10）。

**D5 現象（新跑）**：所有 threshold（0.20–0.80）結果完全相同（33/25/17），根本原因未變 —— PC opinions 全部已被 D4 D2 fallback 覆蓋，layer-2 cosine 永遠不觸發。

**結論**：邏輯完全一致，D4 Partial F1=0.6111 是 Stage 1 A3 改善的級聯受益，重構驗收通過。

---

## 最優方案（更新後）

| 方案 | Exact F1 | Partial F1 | TP/FP/FN |
|------|----------|------------|----------|
| **D4** | **0.4259** | **0.6111** | 33/25/17 |

**D4** = D3（stanza dep 路徑）+ D2（子句邊界 fallback）Hybrid，A3 排除 opinion-only 詞
（重構前 D4+：Partial F1=0.5370 / 29/29/21，+0.0741 為 Stage 1 A3 改善的級聯效果）

### 殘餘 FN 分析（17 個，重構後）

| 原因 | 數量 | 範例 |
|------|------|------|
| Stage 3 FN 攜帶（opinion 未提取）| ~5 | 找的到/融為一體/加快理解/不見/跟不太上 |
| Stage 1 FN 攜帶（aspect 未提取）| ~4 | 橫向移動/視覺化/程式執行 |
| 配對距離錯誤（nearest 非 GT）| ~5 | 指引+不明確(row 9)、測驗送出+卡(row 44)、內容+不知道(row 15) |
| GT opinion 跨句或複雜結構 | ~3 | 不會讓思緒錯亂(row 33)、輕易理解(row 12) |
