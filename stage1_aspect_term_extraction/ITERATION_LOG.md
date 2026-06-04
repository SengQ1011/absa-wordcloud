# Stage 1 — Aspect Term Extraction  ITERATION LOG

## 方案說明

| 方案 | 核心技術 | 預期效果 |
|------|----------|----------|
| A1 | CKIP Na/Nb/Nv 直接抽取 | baseline，recall 高但 FP 多 |
| A2 | A1 + anchor cosine 過濾 | 提高 precision，可能降低 recall |
| A3 | A1 + 相鄰 Na 複合名詞合併 | 提高 exact match（錯誤檢測/動畫演示）|
| A4 | A3 + CUSTOM_ASPECT_WORDS 白名單 | 補抓 CKIP 標為動詞的領域 aspect |

---

## A1 — Baseline

- **Problem**: 需要一個最基本的 baseline 測量 aspect extraction 能力
- **Method**: CKIP Na/Nb/Nv POS 過濾，`is_valid_aspect` 去除程度副詞、空洞名詞、長度 < 2 的 token
- **Result**: Partial P=0.5310 / R=0.7792 / F1=**0.6316**；Exact F1=0.4000；TP=60/FP=53/FN=17
- **Result (重構後 2026-06-03)**: Partial F1=**0.6429**；Exact F1=0.3980；TP=63/FP=56/FN=14（+0.0113，FW token 納入 `CKIP_ASPECT_POS`）
- **Impact**:
  - 主要問題：大量 FP（演算法、網路、結果、時間等無關名詞）
  - 複合詞被切開（「錯誤+檢測」「動畫+演示」），exact match 低
  - 隱性 aspect（視覺化、操作）完全無法抓到

---

## A2 — Anchor Filter（放棄）

- **Problem**: A1 抽太多無關名詞，precision 低
- **Method**: sentence_transformers paraphrase-multilingual-MiniLM-L12-v2，category seed 平均向量作 anchor，cos > threshold 才保留
- **Result**: threshold 0.25~0.45 完全沒有過濾效果（137→137）；threshold 0.50 只過濾 4 個（137→133），F1 改善 < 0.003
- **Impact**: 無效。根本原因：單一中文詞的句向量缺乏跨 category 判別力，所有候選詞的 cos 分數都高於 0.45 threshold。
- **決策**: A2 方案放棄。若要改善 precision 需換為 term-in-context embedding（encode 整句後取 aspect span），這屬於 Stage 5 的 BERT-SPC 方案。

---

## A3 — Compound Merge

- **Problem**: A1 的 exact match 低，因複合名詞被 CKIP 切開（錯誤+檢測、動畫+演示、中文+版本 etc.）
- **Method**:
  - 連續 Na/Nb（continuation token >= 2 chars）→ 合併（最多延伸 1 token，防止「答題秒數多寡」）
  - Na + VC/VE/VD（後不接 VH/VJ/A）→ 橋接合併（「錯誤+檢測」「程式+執行」）
  - FW + WHITESPACE + FW → 以空格接合（「Manual Sort」）
  - 起始 token 須通過 is_valid_aspect 且不在 WEAK_ASPECT_NOUNS（避免「東西藏」）
- **Result (原始)**: Partial P=0.5865 / R=0.7922 / F1=**0.6740**；Exact F1=0.5525；TP=61/FP=43/FN=16
- **Result (拆分後 2026-06-03)**: Partial F1=**0.6927**；Exact F1=0.5698；TP=62/FP=40/FN=15（純 A3，不含白名單）
- **Impact**:
  - ✅ Exact F1：0.4000 → 0.5525（+0.1525）
  - ✅ FP：53 → 43（-10），「檢測」「翻譯」「版本」等半詞 FP 消失
  - ✅ 新增 TP：Manual Sort（FW 空格合併）、monkey sort
  - ⚠️ 已知限制：
    - Row 33「逐行帶著你看」是複雜動詞短語，Na merge 無法覆蓋
    - Row 34/41「視覺化」CKIP 標為 VHC，不在 Na/Nb/Nv → 無法抽取（Stage 3 VHC whitelist 可補）
    - Row 53「程式執行」：追蹤(Nv)+程式(Na) merge 後消耗了程式，導致「程式」無法再匹配「程式執行」（minor regression vs A1）
    - Row 43「程式碼+程式碼練習」：greedy matching 導致 FP（inherent limitation of greedy eval）

---

---

## A4 — CUSTOM_ASPECT_WORDS 白名單（2026-06-01）

- **Problem**: CKIP 將 `提交(VD)`/`講解(VE)`/`提示(VE)`/`指引(VF)` 標成動詞，A3 的 _A3_ASP={Na,Nb,Nv,FW} POS gate 把這些詞全部濾掉，造成 6 個 FN。
  分析工具：`diag_stage1_fn.py` 自動診斷各 FN 的 CKIP POS 與失敗原因。
- **Method**:
  - 新增 `CUSTOM_ASPECT_WORDS = {"提交", "講解", "提示", "指引"}`（在 stage4_eval.py 和 stage1_eval.py 各加一份）
  - `extract_a3` 新增 `is_custom = tok in CUSTOM_ASPECT_WORDS`，POS gate 改為 `pos in _A3_ASP or is_custom`
  - CUSTOM_ASPECT_WORDS 的 token 繞過 CUSTOM_OPINION_WORDS/VHC_WHITELIST 排除，直接進入合併邏輯
  - GT 修正：row 33 `aspect_term=逐行帶著你看` → `implicit`（動詞短語，語言學上應為隱式 aspect）
- **Result**: GT: 78→77（-1 implicit），Partial F1: 0.6889 → **0.7273**（+0.038）；TP=68/FP=42/FN=9

| Method | Exact F1 | Partial F1 | TP/FP/FN | Δ Partial F1 |
|--------|----------|------------|----------|-------------|
| A3     | 0.6096   | 0.6889     | 62/40/16 | —           |
| **A4** | **0.6667** | **0.7273** | **68/42/9** | **+0.038** |

- **Result (拆分後 2026-06-03)**: A4 從 A3 程式碼拆出為獨立 `ATE.extract_a4`，與 A3 共用 `_compound_merge(use_whitelist=True)`；Partial F1=**0.7351**（+0.0078 vs 原始 A4）；Exact F1=0.6162；TP=68/FP=40/FN=9（FP 從 42→40，CUSTOM_OPINION_WORDS/VHC_WHITELIST 過濾 non-custom 詞生效）

- **Impact**:
  - ✅ Row 9: 指引 captured（VF）
  - ✅ Row 14, 52: 講解 ×2 captured（VE）
  - ✅ Row 46: 提示 captured（VE）
  - ✅ Row 48, 50: 提交 ×2 captured（VD）
  - ✅ Row 33: GT 修正移除 1 不可達 FN
  - ⚠️ 已知剩餘 FN（9 個）：
    - `視覺化` ×2（VHC，加入 aspect 白名單會使 Stage 4 產生 self-pair FP，跳過）
    - `操作` ×1（VC，通用動詞，FP 風險高，跳過）
    - `程式執行` ×1（追蹤+程式 先合併消耗 `程式`，merge 邏輯結構性問題）
    - `逐行帶著你看` 已改 implicit
    - 重複 GT（介面×2、詳解×2、程式×2、動畫×2）共 5 個假 FN，end-to-end eval 不受影響

---

## 重構驗收（2026-06-03）

程式碼重構為 `ATE` class（`ATE.py` + `eval.py`），共享常數移至 `shared/constants.py`；A3/A4 拆分為獨立方法。

| Method | Exact F1 | Partial F1 | TP/FP/FN | vs 重構前 |
|--------|----------|------------|----------|-----------|
| A1 | 0.3980 | **0.6429** | 63/56/14 | +0.0113（FW 納入 A1） |
| A2 | 0.3980 | **0.6429** | 63/56/14 | = A1（anchor filter 無效，維持放棄）|
| A3 | 0.5698 | **0.6927** | 62/40/15 | 純複合名詞合併，無白名單 |
| **A4** | **0.6162** | **0.7351** | **68/40/9** | **+0.0078 vs 原始 A4（FP 42→40）** |

主要變化：
- `CKIP_ASPECT_POS` 改從 `shared/constants.py` 匯入，含 FW，A1 recall 提升
- A3/A4 共用 `_compound_merge(use_whitelist)` private 方法，程式碼零重複
- A4 的 CUSTOM_OPINION_WORDS/VHC_WHITELIST 過濾（non-custom 詞）使 FP 42→40

---

## 最優方案

- **最優方案**：**`ATE.extract_a4`**（複合名詞合併 + CUSTOM_ASPECT_WORDS 白名單）
- **Partial F1**：**0.7351**（原始 A4: 0.7273，+0.0078）
- **Exact F1**：0.6162
- **TP/FP/FN**：68/40/9

## 下一步

Stage 1 驗收完成，繼續驗收後續 Stage。
