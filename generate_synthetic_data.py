#!/usr/bin/env python3
"""
ABSA 合成資料生成器 — 方案 B TODO: 尚未測試
透過 Gemini API 批量生成合成 ABSA 標注資料，用於訓練 Span-based 模型

使用方式：
  set GEMINI_API_KEY=你的 key
  python generate_synthetic_data.py [--target 500] [--batch 20] [--dry-run]

輸出：
  synthetic_data/synthetic_raw.json    ← LLM 原始輸出（含無效項目）
  synthetic_data/synthetic_valid.json  ← 通過驗證的資料
  synthetic_data/synthetic_gt.csv      ← 與 ground_truth_absa.csv 格式相同，可直接 concat
  synthetic_data/generation_report.md  ← 每批統計報告
"""

import os
import sys
import io
import json
import time
import argparse
import re
import csv
from pathlib import Path
from datetime import datetime

# Windows console 編碼修正（避免繁體中文輸出錯誤）
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from google import genai
from google.genai import types

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "gemini-2.0-flash"
OUTPUT_DIR = Path(__file__).parent / "synthetic_data"

VALID_CATEGORIES = frozenset({"UI_UX", "ANIMATION", "CONTENT", "QUIZ", "PERFORMANCE", "GAMIFICATION", "OTHERS"})
VALID_POLARITY   = frozenset({"pos", "neg"})

# ─────────────────────────────────────────────────────────────────────────────
# System Instruction（告訴 LLM 背景、規則、範例）
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_INSTRUCTION = """
你是一個 ABSA（Aspect-Based Sentiment Analysis）資料標注與生成專家。

【平台背景】
CodePulse 是一個大學生使用的「資料結構與演算法視覺化教學平台」，主要功能：
- 演算法視覺化動畫（Bubble Sort、Insertion Sort、Quick Sort、Merge Sort、
  Binary Search、BFS、DFS、Stack、Queue、Tree 等多種演算法）
- 沙箱模式（學生可自行操作陣列、調整資料，即時看動畫反應）
- 演算法實驗室（時間複雜度對比，多種演算法同時執行顯示）
- 測驗系統（選擇題、程式填空，自動評分、詳細題目解析）
- 錯誤檢測與學習指引（即時提示錯誤原因）
- 遊戲化互動：打地鼠遊戲（反應速度訓練）、Manual Sort（手動排序挑戰）、monkey sort
- 虛擬碼（pseudocode）與實際程式碼對照顯示
- 程式摘要生成（AI 自動說明每段程式碼意義）
- 學習進度與紀錄追蹤
- 數據分析圖表頁面

【ABSA 任務定義】
從一段學生問卷回饋（text）中，抽取所有 (aspect_term, aspect_category, opinion_term, polarity) 四元組。

問卷格式說明：
- text = 正面意見 + "，" + 改進意見（有時只有一面向）
- 正面意見描述「喜歡/滿意的地方」，改進意見描述「希望改善的地方」

【欄位定義】
aspect_term    → 文本中被評論的名詞/名詞組（1–7字），或 "implicit"（無明確名詞時填）
aspect_category → 必須是以下 7 類之一（見下方定義）
opinion_term   → 文本中的核心意見詞（2–6字），或 "implicit"（無明確意見詞時填）
polarity       → 只能填 "pos"（正面）或 "neg"（負面）

【7 個 Aspect Category 定義】
UI_UX        → 介面設計、操作體驗、視覺佈局、圖表（非動畫類）、按鈕、頁面導航
ANIMATION    → 演算法動畫、執行視覺化、沙箱動畫效果、動態呈現
CONTENT      → 教學內容品質、詳解說明、虛擬碼、程式說明、中文翻譯、程式摘要
QUIZ         → 測驗系統、題目難度、答題流程、自動評分、錯誤檢測
PERFORMANCE  → 系統速度、卡頓、提交失敗、程式崩潰、當掉
GAMIFICATION → 打地鼠遊戲、Manual Sort、monkey sort 等遊戲化元素
OTHERS       → 整體感受、無法確定對象的一般形容詞（棒/不錯/新奇）

【標注規則（嚴格遵守，違反則資料無效）】
R1. aspect_term 和 opinion_term 不可同時為 "implicit"（至少一個必須是明確詞）
R2. aspect_term = "implicit" 的情況：文本無明確名詞，只有形容詞描述整體感受
R3. opinion_term = "implicit" 的情況：文本提到功能/需求，但無明確意見詞
    例：「中文翻譯」= 只提到需求，沒說好不好 → opinion = "implicit", polarity = neg（隱含希望改善）
R4. opinion_term 只寫核心意見詞（2–6字），不可寫完整子句
    ✗ 錯誤："使用起來很容易理解"  ✓ 正確："容易理解"
R5. 否定詞一定要連同基礎詞一起寫進 opinion_term
    ✗ 錯誤 opinion="明確" polarity="neg"  ✓ 正確 opinion="不太明確" polarity="neg"
R6. 一個 text 可以有 1–5 個 quadruplet（每個 aspect-opinion pair 各標一個）
R7. 同一 text 中，不同 quadruplet 的 aspect_term 可以相同（同個 aspect 有多個意見）

【implicit aspect 的 category 判斷規則】
- opinion 是視覺/美感詞（單調/好看/乾淨/複雜/整齊/美觀/醜）→ UI_UX
- opinion 是效能詞（卡/失敗/掛/慢/當掉/不穩定）→ PERFORMANCE
- opinion 是學習相關詞（清楚/看不懂/難懂）且語境是教學內容 → CONTENT
- opinion 是學習相關詞且語境是操作介面 → UI_UX
- opinion 是一般感受詞（棒/不錯/新奇/有趣/好/讚）→ OTHERS

【語言風格要求】
- 繁體中文，口語化（像大學生用手機填問卷）
- 句子短（10–60字），可省略主語，可有不完整句
- 可混入英文演算法術語（Bubble Sort / BFS / Stack / Binary Search / Quick Sort 等）
- 自然出現否定修飾詞（不太/稍微/有點/感覺/好像/還算）
- 語氣多元：有輕鬆、有建議、有讚美、有抱怨

【輸出格式（必須嚴格遵守）】
輸出合法 JSON array，不要任何其他文字、不要 markdown code block：
[
  {
    "text": "學生回饋原文（繁體中文）",
    "quadruplets": [
      {
        "aspect_term": "名詞或 implicit",
        "aspect_category": "7類之一",
        "opinion_term": "意見詞或 implicit",
        "polarity": "pos 或 neg"
      }
    ]
  }
]

【True Examples（學習這些模式的語言風格和標注方式）】
[
  {
    "text": "介面乾淨不會複雜",
    "quadruplets": [
      {"aspect_term": "介面", "aspect_category": "UI_UX", "opinion_term": "乾淨", "polarity": "pos"},
      {"aspect_term": "介面", "aspect_category": "UI_UX", "opinion_term": "不會複雜", "polarity": "pos"}
    ]
  },
  {
    "text": "整體介面清楚好懂，答案解析也非常好懂，題目有點看不懂",
    "quadruplets": [
      {"aspect_term": "介面", "aspect_category": "UI_UX", "opinion_term": "清楚", "polarity": "pos"},
      {"aspect_term": "答案解析", "aspect_category": "CONTENT", "opinion_term": "好懂", "polarity": "pos"},
      {"aspect_term": "題目", "aspect_category": "QUIZ", "opinion_term": "看不懂", "polarity": "neg"}
    ]
  },
  {
    "text": "Manual Sort很好玩，難度還是有點高，看看能不能做更簡單更基礎的程式之類的?",
    "quadruplets": [
      {"aspect_term": "Manual Sort", "aspect_category": "GAMIFICATION", "opinion_term": "好玩", "polarity": "pos"},
      {"aspect_term": "難度", "aspect_category": "CONTENT", "opinion_term": "高", "polarity": "neg"},
      {"aspect_term": "程式", "aspect_category": "CONTENT", "opinion_term": "簡單", "polarity": "neg"},
      {"aspect_term": "程式", "aspect_category": "CONTENT", "opinion_term": "基礎", "polarity": "neg"}
    ]
  },
  {
    "text": "打地鼠遊戲幽默風趣，希望測驗送出時有更即時的反應時間，現在會卡",
    "quadruplets": [
      {"aspect_term": "打地鼠遊戲", "aspect_category": "GAMIFICATION", "opinion_term": "幽默風趣", "polarity": "pos"},
      {"aspect_term": "測驗送出", "aspect_category": "PERFORMANCE", "opinion_term": "卡", "polarity": "neg"}
    ]
  },
  {
    "text": "動畫優質，操作直觀點，有些東西藏好深哦",
    "quadruplets": [
      {"aspect_term": "動畫", "aspect_category": "ANIMATION", "opinion_term": "優質", "polarity": "pos"},
      {"aspect_term": "操作", "aspect_category": "UI_UX", "opinion_term": "直觀", "polarity": "pos"},
      {"aspect_term": "implicit", "aspect_category": "UI_UX", "opinion_term": "深", "polarity": "neg"}
    ]
  },
  {
    "text": "可以快速找到進度，中文翻譯",
    "quadruplets": [
      {"aspect_term": "implicit", "aspect_category": "UI_UX", "opinion_term": "快速", "polarity": "pos"},
      {"aspect_term": "中文翻譯", "aspect_category": "CONTENT", "opinion_term": "implicit", "polarity": "neg"}
    ]
  },
  {
    "text": "作答到最後結果出來的詳解解釋得很清楚，容易懂，教學模式的地方第一次點開時會看不懂怎麼操作，需要花一些時間理解",
    "quadruplets": [
      {"aspect_term": "詳解", "aspect_category": "CONTENT", "opinion_term": "清楚", "polarity": "pos"},
      {"aspect_term": "詳解", "aspect_category": "CONTENT", "opinion_term": "容易懂", "polarity": "pos"},
      {"aspect_term": "教學模式", "aspect_category": "CONTENT", "opinion_term": "看不懂", "polarity": "neg"}
    ]
  },
  {
    "text": "視覺化牛逼，我要monkey sort",
    "quadruplets": [
      {"aspect_term": "視覺化", "aspect_category": "ANIMATION", "opinion_term": "牛逼", "polarity": "pos"},
      {"aspect_term": "monkey sort", "aspect_category": "GAMIFICATION", "opinion_term": "implicit", "polarity": "neg"}
    ]
  },
  {
    "text": "很新奇，有點太單調",
    "quadruplets": [
      {"aspect_term": "implicit", "aspect_category": "OTHERS", "opinion_term": "新奇", "polarity": "pos"},
      {"aspect_term": "implicit", "aspect_category": "UI_UX", "opinion_term": "太單調", "polarity": "neg"}
    ]
  },
  {
    "text": "對於演算法中較為抽象的部分，確實能加快理解還有測驗可以驗證你是否理解，有時候會有提交失敗的問題",
    "quadruplets": [
      {"aspect_term": "演算法", "aspect_category": "CONTENT", "opinion_term": "加快理解", "polarity": "pos"},
      {"aspect_term": "測驗", "aspect_category": "QUIZ", "opinion_term": "implicit", "polarity": "pos"},
      {"aspect_term": "提交", "aspect_category": "PERFORMANCE", "opinion_term": "失敗", "polarity": "neg"}
    ]
  }
]
"""

# ─────────────────────────────────────────────────────────────────────────────
# Batch Profiles（輪流使用，確保各 category 均衡分布）
# ─────────────────────────────────────────────────────────────────────────────
BATCH_PROFILES = [
    {
        "name": "UI_UX + ANIMATION 為主",
        "category_dist": "UI_UX 6筆, ANIMATION 5筆, CONTENT 4筆, QUIZ 2筆, OTHERS 3筆",
        "extra_requirements": (
            "   - 至少 2 筆描述沙箱動畫效果（可快可慢）\n"
            "   - 至少 2 筆描述介面佈局、按鈕或頁面導航問題\n"
            "   - 至少 1 筆動畫速度相關（快/慢/跟不太上）"
        ),
    },
    {
        "name": "QUIZ + PERFORMANCE 為主",
        "category_dist": "QUIZ 5筆, PERFORMANCE 5筆, UI_UX 4筆, CONTENT 3筆, OTHERS 3筆",
        "extra_requirements": (
            "   - 至少 3 筆含效能問題（卡/掛/失敗/當掉/不穩定）\n"
            "   - 至少 2 筆含測驗評分或題目難度問題\n"
            "   - 至少 1 筆描述提交失敗或紀錄消失"
        ),
    },
    {
        "name": "CONTENT + GAMIFICATION 為主",
        "category_dist": "CONTENT 6筆, GAMIFICATION 4筆, ANIMATION 4筆, UI_UX 3筆, OTHERS 3筆",
        "extra_requirements": (
            "   - 至少 2 筆涉及虛擬碼或程式碼對照\n"
            "   - 至少 2 筆涉及遊戲化功能（Manual Sort / 打地鼠 / monkey sort）\n"
            "   - 至少 1 筆涉及中文翻譯或程式摘要需求"
        ),
    },
    {
        "name": "均衡分布（7類齊全）",
        "category_dist": "UI_UX 3筆, ANIMATION 3筆, CONTENT 3筆, QUIZ 3筆, PERFORMANCE 2筆, GAMIFICATION 2筆, OTHERS 4筆",
        "extra_requirements": (
            "   - 所有 7 個 category 都必須出現\n"
            "   - 每個 category 至少 2 筆"
        ),
    },
    {
        "name": "多 quadruplet 複雜句（每筆至少 2 個）",
        "category_dist": "各 category 均可",
        "extra_requirements": (
            "   - 所有 20 筆都必須有至少 2 個 quadruplet\n"
            "   - 至少 7 筆有 3 個或以上的 quadruplet\n"
            "   - 至少 3 筆有 4 個 quadruplet（描述多個 aspect 的複雜回饋）"
        ),
    },
]


def make_user_prompt(batch_num: int, batch_profile: dict) -> str:
    """生成每批的 User Message。"""
    return f"""請生成 {BATCH_SIZE} 筆全新的 CodePulse 平台學生回饋資料（第 {batch_num} 批）。

【本批主題：{batch_profile["name"]}】

【本批 Category 分布要求】
{batch_profile["category_dist"]}

【本批必須包含的模式】
   - 至少 3 筆含否定詞的 opinion_term（不太/看不懂/不會/不太明顯/沒有等）
   - 至少 3 筆含 implicit aspect_term
   - 至少 2 筆含 implicit opinion_term
   - 至少 2 筆混入英文術語（演算法名稱或程式詞彙）
{batch_profile["extra_requirements"]}

【其他要求】
   - 20 筆文本內容不可重複或過度相似
   - polarity 比例約 60% pos，40% neg
   - 嚴格遵守所有標注規則 R1–R7
   - 不可複製 Examples，要生成全新內容

直接輸出 JSON array，不要任何其他文字。"""


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────
def validate_item(item: dict) -> tuple[bool, str]:
    """
    驗證一筆生成資料是否符合標注規則。
    Returns (is_valid, reason_if_invalid)
    """
    if not isinstance(item, dict):
        return False, "not a dict"
    if "text" not in item or "quadruplets" not in item:
        return False, "missing required fields: text or quadruplets"
    if not item["text"] or not isinstance(item["text"], str) or len(item["text"].strip()) < 3:
        return False, f"text too short or empty: {item.get('text', '')!r}"
    if not isinstance(item["quadruplets"], list) or len(item["quadruplets"]) == 0:
        return False, "quadruplets must be a non-empty list"
    if len(item["quadruplets"]) > 6:
        return False, f"too many quadruplets: {len(item['quadruplets'])} (max 6)"

    for i, q in enumerate(item["quadruplets"]):
        if not isinstance(q, dict):
            return False, f"quadruplet[{i}] is not a dict"

        for field in ("aspect_term", "aspect_category", "opinion_term", "polarity"):
            if field not in q:
                return False, f"quadruplet[{i}] missing field: {field}"
            if not isinstance(q[field], str) or not q[field].strip():
                return False, f"quadruplet[{i}].{field} is empty"

        # Category check
        if q["aspect_category"] not in VALID_CATEGORIES:
            return False, f"quadruplet[{i}] invalid category: {q['aspect_category']!r}"

        # Polarity check
        if q["polarity"] not in VALID_POLARITY:
            return False, f"quadruplet[{i}] invalid polarity: {q['polarity']!r}"

        # R1: not both implicit
        if q["aspect_term"] == "implicit" and q["opinion_term"] == "implicit":
            return False, f"quadruplet[{i}] violates R1: both aspect_term and opinion_term are implicit"

        # Basic sanity: aspect_term in text (loose check, not enforced for implicit)
        if q["aspect_term"] != "implicit" and len(q["aspect_term"]) > 15:
            return False, f"quadruplet[{i}] aspect_term suspiciously long: {q['aspect_term']!r}"
        if q["opinion_term"] != "implicit" and len(q["opinion_term"]) > 12:
            return False, f"quadruplet[{i}] opinion_term suspiciously long: {q['opinion_term']!r}"

    return True, ""


def extract_json_from_response(raw_text: str) -> list | None:
    """
    從 LLM 回應中提取 JSON array。
    處理 LLM 可能包裹在 markdown code block 的情況。
    """
    # 先嘗試直接 parse
    try:
        parsed = json.loads(raw_text.strip())
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # 嘗試從 markdown code block 提取
    patterns = [
        r"```json\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```",
        r"(\[[\s\S]*\])",
    ]
    for pattern in patterns:
        m = re.search(pattern, raw_text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(1).strip())
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                continue

    return None


# ─────────────────────────────────────────────────────────────────────────────
# CSV Export（格式與 ground_truth_absa.csv 一致）
# ─────────────────────────────────────────────────────────────────────────────
def export_to_csv(valid_items: list[dict], output_path: Path) -> None:
    """
    把 valid_items 轉存成與 ground_truth_absa.csv 格式一致的 CSV。
    row_id 格式：syn_001, syn_002, ...
    """
    rows = []
    for idx, item in enumerate(valid_items, start=1):
        row_id = f"syn_{idx:03d}"
        text = item["text"]
        for q in item["quadruplets"]:
            rows.append({
                "row_id": row_id,
                "text": text,
                "aspect_term": q["aspect_term"],
                "aspect_category": q["aspect_category"],
                "opinion_term": q["opinion_term"],
                "polarity": q["polarity"],
            })

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["row_id", "text", "aspect_term", "aspect_category", "opinion_term", "polarity"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  → CSV exported: {output_path} ({len(rows)} rows, {len(valid_items)} texts)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(target_total: int, batch_size: int, dry_run: bool) -> None:
    global BATCH_SIZE
    BATCH_SIZE = batch_size

    # ── API setup
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key and not dry_run:
        raise ValueError("請設定環境變數 GEMINI_API_KEY，例如: set GEMINI_API_KEY=你的key")

    client = None
    if not dry_run:
        client = genai.Client(api_key=api_key)

    # ── Output dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path    = OUTPUT_DIR / f"synthetic_raw_{timestamp}.json"
    valid_path  = OUTPUT_DIR / "synthetic_valid.json"
    csv_path    = OUTPUT_DIR / "synthetic_gt.csv"
    report_path = OUTPUT_DIR / f"generation_report_{timestamp}.md"

    all_raw:   list[dict] = []
    all_valid: list[dict] = []

    # ── Load existing valid data if any (resume mode)
    if valid_path.exists():
        with open(valid_path, "r", encoding="utf-8") as f:
            all_valid = json.load(f)
        print(f"[Resume] 已載入 {len(all_valid)} 筆既有合成資料")

    report_lines: list[str] = [
        f"# Synthetic Data Generation Report\n",
        f"Generated: {timestamp}  |  Model: {MODEL_NAME}  |  Target: {target_total}\n",
        "---\n",
    ]

    batch_num = 0
    fail_streak = 0  # 連續失敗計數，避免無限重試

    while len(all_valid) < target_total and fail_streak < 5:
        profile = BATCH_PROFILES[batch_num % len(BATCH_PROFILES)]
        batch_num += 1

        print(f"\n[Batch {batch_num}] {profile['name']} | 目前合成資料: {len(all_valid)}/{target_total}")

        if dry_run:
            # Dry run: 只印 prompt，不真的呼叫 API
            prompt = make_user_prompt(batch_num, profile)
            print("── System Instruction（前 300 字）──")
            print(SYSTEM_INSTRUCTION[:300].strip(), "...")
            print("\n── User Prompt ──")
            print(prompt)
            print("\n[Dry Run] 跳過 API 呼叫")
            break

        # ── API call
        user_prompt = make_user_prompt(batch_num, profile)
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.85,
                    response_mime_type="application/json",
                ),
            )
            raw_text = response.text
        except Exception as e:
            print(f"  X API error: {e}")
            fail_streak += 1
            time.sleep(10)
            continue

        # ── Parse
        items = extract_json_from_response(raw_text)
        if items is None:
            print(f"  ✗ JSON 解析失敗，跳過這批")
            report_lines.append(f"## Batch {batch_num} — {profile['name']}\n❌ JSON parse failed\n\n")
            fail_streak += 1
            time.sleep(5)
            continue

        # ── Validate
        batch_valid: list[dict] = []
        batch_invalid: list[tuple[dict, str]] = []
        for item in items:
            ok, reason = validate_item(item)
            if ok:
                batch_valid.append(item)
            else:
                batch_invalid.append((item, reason))

        all_raw.extend(items)
        all_valid.extend(batch_valid)
        fail_streak = 0  # 重置

        # ── Statistics
        n_total   = len(items)
        n_valid   = len(batch_valid)
        n_invalid = len(batch_invalid)

        # Category distribution
        cat_counts: dict[str, int] = {c: 0 for c in VALID_CATEGORIES}
        for item in batch_valid:
            for q in item["quadruplets"]:
                cat_counts[q["aspect_category"]] += 1

        print(f"  ✓ 有效: {n_valid}/{n_total}  |  累計: {len(all_valid)}")
        print(f"  Category: { {k: v for k, v in cat_counts.items() if v > 0} }")
        if batch_invalid:
            print(f"  ✗ 無效項目原因（前 3 個）:")
            for item, reason in batch_invalid[:3]:
                print(f"    - {reason!r}")

        # ── Report
        report_lines.append(
            f"## Batch {batch_num} — {profile['name']}\n"
            f"- Valid/Total: {n_valid}/{n_total}\n"
            f"- Category dist: {cat_counts}\n"
            f"- Cumulative valid: {len(all_valid)}\n"
        )
        if batch_invalid:
            report_lines.append(f"- Invalid examples:\n")
            for item, reason in batch_invalid[:3]:
                report_lines.append(f"  - `{reason}`: {str(item)[:120]}\n")
        report_lines.append("\n")

        # ── Save incrementally
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(all_raw, f, ensure_ascii=False, indent=2)
        with open(valid_path, "w", encoding="utf-8") as f:
            json.dump(all_valid, f, ensure_ascii=False, indent=2)

        time.sleep(SLEEP_BETWEEN_BATCHES)

    # ── Final outputs
    print(f"\n{'='*60}")
    print(f"生成完成！合成資料: {len(all_valid)} 筆")

    if all_valid:
        # CSV export
        export_to_csv(all_valid, csv_path)

        # Final stats
        total_tuples = sum(len(item["quadruplets"]) for item in all_valid)
        all_cats: dict[str, int] = {c: 0 for c in VALID_CATEGORIES}
        all_pol: dict[str, int] = {"pos": 0, "neg": 0}
        for item in all_valid:
            for q in item["quadruplets"]:
                all_cats[q["aspect_category"]] += 1
                all_pol[q["polarity"]] += 1
        impl_asp = sum(1 for item in all_valid for q in item["quadruplets"] if q["aspect_term"] == "implicit")
        impl_op  = sum(1 for item in all_valid for q in item["quadruplets"] if q["opinion_term"] == "implicit")

        summary = [
            f"\n## Final Summary\n",
            f"- Total texts: {len(all_valid)}\n",
            f"- Total tuples: {total_tuples}\n",
            f"- Avg tuples/text: {total_tuples/len(all_valid):.2f}\n",
            f"- Category distribution: {all_cats}\n",
            f"- Polarity distribution: {all_pol}\n",
            f"- Implicit aspect_term: {impl_asp} ({impl_asp/total_tuples*100:.1f}%)\n",
            f"- Implicit opinion_term: {impl_op} ({impl_op/total_tuples*100:.1f}%)\n",
            f"\nOutput files:\n",
            f"- Valid JSON: {valid_path}\n",
            f"- GT CSV: {csv_path}\n",
            f"- Raw JSON: {raw_path}\n",
        ]
        report_lines.extend(summary)
        print("".join(summary))

    # Write report
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    print(f"Report: {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ABSA 合成資料生成器（Gemini API）")
    parser.add_argument("--target",    type=int, default=500, help="目標生成筆數（預設 500）")
    parser.add_argument("--batch",     type=int, default=20,  help="每次 API call 生成筆數（預設 20）")
    parser.add_argument("--dry-run",   action="store_true",   help="只印 Prompt，不真的呼叫 API")
    args = parser.parse_args()

    main(target_total=args.target, batch_size=args.batch, dry_run=args.dry_run)
