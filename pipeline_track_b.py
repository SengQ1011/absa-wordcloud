"""
track_b_pipeline.py — Track B: Gemini Zero-Shot ABSA Quadruplet Pipeline

呼叫 Gemini 3.1 Flash Lite，以 zero-shot prompt 對每筆 feedback 輸出：
  (aspect_term, aspect_category, opinion_term, polarity)

polarity 規則：優點 → pos；缺點 / 建議 → neg
implicit 規則：aspect/opinion 在文字中無可直接 span 的詞時標 "implicit"

Usage:
  python track_b_pipeline.py              # 呼叫 API，存 track_b_output.csv
  python track_b_pipeline.py --from-cache # 從 track_b_cache.json 略過 API
  python track_b_pipeline.py -v           # 印每筆 quadruplet 細節
"""

import argparse
import csv
import json
import os
import time
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR    = Path(__file__).parent
CACHE_PATH  = ROOT_DIR / "cache" / "track_b_cache.json"
DATA_PATH   = ROOT_DIR / "data" / "cleaned_texts.csv"
OUTPUT_PATH = ROOT_DIR / "output" / "track_b_output.csv"
HISTORY_DIR = ROOT_DIR / "history" / "track_b"

sys.path.insert(0, str(ROOT_DIR))
from shared.utils import Tee, append_timing_log
from shared.constants import CATEGORIES as _CATS

CATEGORIES = _CATS
MODEL_ID   = "gemini-3.1-flash-lite"
RATE_DELAY = 6.0   # seconds between calls (~10 RPM, within 11 RPM limit)
MAX_RETRIES = 3


# ── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """你是一位繁體中文 NLP 分析專家，專門對 CodePulse 教育平台的使用者回饋進行 Aspect-Based Sentiment Analysis (ABSA)。

請分析輸入的回饋文字，抽取所有 quadruplet，每個 quadruplet 格式如下：
{
  "aspect_term":     string,  // 被評論的具體詞；若文字中無明確詞彙則填 "implicit"
  "aspect_category": string,  // 見下方六大面向定義
  "opinion_term":    string,  // 評論詞；若文字中無明確評論詞則填 "implicit"
  "polarity":        string   // "pos" 或 "neg"
}

【六大面向定義】
- UI_UX:        介面、畫面、按鈕、顏色、排版、手機版、舒適、乾淨、單調、設計、好看、外觀、美觀、版面、導航、頁面、佈局、操作、指引、功能、提示、觀感、圖表
- ANIMATION:    動畫、演示、逐步、逐行、視覺化、步驟、播放、動態、逐格、動態演示、圖示化、步步
- CONTENT:      中文、翻譯、解釋、程式碼、涵義、概念、虛擬碼、摘要、時間複雜度、教學、講解、課程、說明
- QUIZ:         測驗、題目、難度、答題、解析、錯誤檢測、作答、考卷、成績、分數、答案
- PERFORMANCE:  卡頓、等待、延遲、Bug、失敗、提交、紀錄、即時、卡、同步、錯誤、回應、當掉
- GAMIFICATION: 遊戲、互動、彩蛋、好玩、闖關、挑戰、沙箱、玩、幽默、排行、打地鼠、排行榜、積分、成就
若不屬於以上任一面向，用 OTHERS。

【polarity 規則】
- "pos"：明確的優點、好評、正面描述
- "neg"：缺點、問題、建議改進（建議也視為 neg）

【implicit 規則】
- aspect_term = "implicit"：文字只提到面向的感受，但沒有出現可直接摘取的名詞
  例：「有點卡卡的」→ aspect=implicit（沒說是哪個功能），opinion=「卡卡的」
- opinion_term = "implicit"：文字提到了評論對象但未給出明確評論詞
  例：「測驗系統」→ aspect=「測驗」，opinion=implicit（若沒有明確評論詞）

【輸出格式】
只回傳 JSON array，不要任何說明文字。若無任何 quadruplet，回傳 []。

範例：
輸入：「動畫很流暢，但測驗送出會卡頓」
輸出：[
  {"aspect_term":"動畫","aspect_category":"ANIMATION","opinion_term":"流暢","polarity":"pos"},
  {"aspect_term":"測驗送出","aspect_category":"QUIZ","opinion_term":"卡頓","polarity":"neg"}
]"""


def build_prompt(text: str) -> str:
    return f"{SYSTEM_PROMPT}\n\n輸入：「{text}」"


# ── Gemini client ─────────────────────────────────────────────────────────────

def init_client():
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("GEMINI_3.1_FLASH_LITE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_3.1_FLASH_LITE_API_KEY not found in .env")
    from google import genai
    return genai.Client(api_key=api_key)


def call_gemini(client, text: str) -> list[dict]:
    prompt = build_prompt(text)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
            )
            raw = resp.text.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(
                    l for l in lines
                    if not l.startswith("```")
                ).strip()
            quads = json.loads(raw)
            if not isinstance(quads, list):
                quads = []
            return _validate(quads)
        except json.JSONDecodeError as e:
            print(f"    [JSON error attempt {attempt+1}] {e}  raw={raw[:80]!r}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
        except Exception as e:
            print(f"    [API error attempt {attempt+1}] {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(4)
    return []


def _validate(quads: list) -> list[dict]:
    result = []
    for q in quads:
        if not isinstance(q, dict):
            continue
        cat = q.get("aspect_category", "OTHERS")
        if cat not in CATEGORIES:
            cat = "OTHERS"
        pol = q.get("polarity", "neg")
        if pol not in ("pos", "neg"):
            pol = "neg"
        result.append({
            "aspect_term":     str(q.get("aspect_term", "implicit")).strip() or "implicit",
            "aspect_category": cat,
            "opinion_term":    str(q.get("opinion_term", "implicit")).strip() or "implicit",
            "polarity":        pol,
        })
    return result


# ── Data ─────────────────────────────────────────────────────────────────────

def load_rows() -> list[dict]:
    rows = []
    with open(DATA_PATH, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("status", "").strip() == "skip":
                continue
            text = r.get("text", "").strip()
            if not text:
                continue
            rows.append({"row_id": r["row_id"].strip(), "text": text})
    return rows


# ── Run pipeline ──────────────────────────────────────────────────────────────

def run(args, timings: list) -> list[dict]:
    rows = load_rows()
    print(f"Input: {len(rows)} rows from {DATA_PATH.name}")

    cache: dict = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH, encoding="utf-8") as f:
            cache = json.load(f)
        print(f"Cache loaded: {len(cache)} entries")

    if args.from_cache:
        print("--from-cache: skipping API calls")
    else:
        client = init_client()
        print(f"Model: {MODEL_ID}  rate_delay={RATE_DELAY}s")
        new_entries = 0
        t0 = time.perf_counter()
        for i, row in enumerate(rows):
            if row["row_id"] in cache:
                print(f"  [{i+1:02d}/{len(rows)}] row {row['row_id']} (cached)")
                continue
            print(f"  [{i+1:02d}/{len(rows)}] row {row['row_id']} ...", end=" ", flush=True)
            quads = call_gemini(client, row["text"])
            cache[row["row_id"]] = quads
            new_entries += 1
            print(f"{len(quads)} quads")
            if i < len(rows) - 1:
                time.sleep(RATE_DELAY)
        timings.append((f"Gemini API ({new_entries} new calls)", time.perf_counter() - t0))

        if new_entries > 0:
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            print(f"Cache saved -> {CACHE_PATH.name}  ({new_entries} new entries)")

    # Build output records
    texts_by_id = {r["row_id"]: r["text"] for r in rows}
    records: list[dict] = []
    for row in rows:
        rid  = row["row_id"]
        text = texts_by_id[rid]
        for q in cache.get(rid, []):
            records.append({
                "row_id":          rid,
                "text":            text,
                "aspect_term":     q["aspect_term"],
                "aspect_category": q["aspect_category"],
                "opinion_term":    q["opinion_term"],
                "polarity":        q["polarity"],
            })

    return records


# ── Output ────────────────────────────────────────────────────────────────────

_CSV_FIELDS = ["row_id", "text", "aspect_term", "aspect_category", "opinion_term", "polarity"]


def write_csv(records: list[dict], path: Path):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    print(f"CSV saved -> {path.name}")


def print_summary(records: list[dict], verbose: bool = False):
    from collections import defaultdict
    by_row: dict = defaultdict(list)
    for r in records:
        by_row[r["row_id"]].append(r)

    sep = "=" * 64
    print(f"\n{sep}")
    print("# Track B Pipeline Output Summary")
    print(sep)
    print(f"Total quadruplets : {len(records)}")
    print(f"Rows with output  : {len(by_row)}")

    cat_counts: dict = defaultdict(int)
    pol_counts: dict = defaultdict(int)
    impl_asp = impl_op = 0
    for r in records:
        cat_counts[r["aspect_category"]] += 1
        pol_counts[r["polarity"]] += 1
        if r["aspect_term"]  == "implicit": impl_asp += 1
        if r["opinion_term"] == "implicit": impl_op  += 1

    print(f"\nCategory breakdown:")
    for cat in sorted(cat_counts):
        print(f"  {cat:<14} {cat_counts[cat]}")
    print(f"\nPolarity: pos={pol_counts['pos']}  neg={pol_counts['neg']}")
    print(f"Implicit: aspect={impl_asp}  opinion={impl_op}")

    if verbose:
        print(f"\n{sep}")
        print("## Per-Row Quadruplets\n")
        for row_id in sorted(by_row.keys(), key=int):
            pairs = by_row[row_id]
            print(f"[row {row_id}]  {pairs[0]['text'][:55]}")
            for p in pairs:
                print(f"  ({p['aspect_term']}, {p['aspect_category']}, "
                      f"{p['opinion_term']}, {p['polarity']})")
            print()


# ── Timing ───────────────────────────────────────────────────────────────────

def print_timing(timings: list, total: float):
    sep = "=" * 64
    print(f"\n{sep}")
    print("## Timing")
    if timings:
        w = max(len(label) for label, _ in timings)
        for label, s in timings:
            print(f"  {label:<{w}} : {s:6.1f}s")
        print(f"  {'─' * (w + 10)}")
        print(f"  {'Total':<{w}} : {total:6.1f}s")
    else:
        print(f"  Total : {total:6.1f}s")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Track B: Gemini Zero-Shot ABSA Pipeline")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--from-cache", action="store_true",
                      help="Skip API calls, rebuild CSV from track_b_cache.json")
    mode.add_argument("--reset", action="store_true",
                      help="Clear cache and re-call Gemini API for all rows (requires confirmation)")
    parser.add_argument("-o", "--output", default=str(OUTPUT_PATH),
                        help="Output CSV path (default: track_b_output.csv)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print per-row quadruplets")
    args = parser.parse_args()

    if args.reset:
        if CACHE_PATH.exists():
            with open(CACHE_PATH, encoding="utf-8") as f:
                n = len(json.load(f))
            print(f"Warning: this will delete {CACHE_PATH.name} ({n} entries) and re-call Gemini API.")
        else:
            print("Cache does not exist. All rows will be sent to Gemini API.")
        confirm = input("Confirm reset? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            sys.exit(0)
        if CACHE_PATH.exists():
            CACHE_PATH.unlink()
            print(f"Cache cleared: {CACHE_PATH.name}\n")

    HISTORY_DIR.mkdir(exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = HISTORY_DIR / f"track_b_{timestamp}.md"

    with open(report_path, "w", encoding="utf-8") as f:
        sys.stdout = Tee(f)
        try:
            t_total     = time.perf_counter()
            timings:list = []
            print(f"# Track B run {timestamp}\n")
            records = run(args, timings)
            write_csv(records, Path(args.output))
            print_summary(records, verbose=args.verbose)
            total_s = time.perf_counter() - t_total
            print_timing(timings, total_s)
            append_timing_log("b", timings, total_s, timestamp)
            print(f"\nReport saved -> {report_path}")
        finally:
            sys.stdout = sys.stdout.stdout


if __name__ == "__main__":
    main()
