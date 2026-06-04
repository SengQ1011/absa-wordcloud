"""
pipeline_track_a.py — ABSA 完整 Pipeline（A4 + Combo + C2+ + D4 + F3）

最優組合（各 Stage 驗證後的 argmax）：
  Stage 1 A4   : CKIP Na/Nb/Nv/FW + 複合名詞合併（max_merges=1）+ CUSTOM_ASPECT_WORDS 白名單
  Stage 2 Combo: B2→B1→B3（explicit）；B1-imp（implicit）；exp_thr=0.30
  Stage 3 C2+  : VH/VJ/A + negation + PC + CUSTOM{直觀,便利}
  Stage 4 D4   : D3 stanza nsubj/amod + D2 子句邊界 fallback
  Stage 5 F3   : polarity_hint 規則優先 → fallback DistilBERT（opinion_term）

評估結果（GT 100 tuples, 50 rows）：
  Pair Partial F1 = 0.5128  |  Senti Acc = 0.9000  |  Quad Partial F1 = 0.3282

Usage:
  python pipeline_track_a.py                  # 完整跑（CKIP+stanza+SBERT+DistilBERT）
  python pipeline_track_a.py --from-cache     # 跳過 CKIP+stanza，直接從 cache/eval_cache.json
  python pipeline_track_a.py --save-cache     # 額外存 D4 pairs 到 cache/eval_cache.json
  python pipeline_track_a.py -o results.csv   # 指定輸出路徑
  python pipeline_track_a.py -v               # 額外印每筆 pair 細節
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT_DIR    = Path(__file__).parent
CACHE_PATH  = ROOT_DIR / "cache" / "eval_cache.json"
DATA_PATH   = ROOT_DIR / "data" / "cleaned_texts.csv"
HISTORY_DIR = ROOT_DIR / "history" / "track_a"

sys.path.insert(0, str(ROOT_DIR))

from shared.utils import Tee, init_ckip, ckip_tokenize, init_stanza, load_texts, append_timing_log
from stage1_aspect_term_extraction.ATE import ATE
from stage2_aspect_category_detection.ACD import ACD
from stage3_opinion_term_extraction.OTE import OTE
from stage4_aspect_opinion_pair_extraction.AOPE import AOPE
from stage5_aspect_sentiment_classification.ASC import ASC


# ── Pair extraction ───────────────────────────────────────────────

def build_pairs(rows: list[dict], args, timings: list) -> dict:
    """
    Returns { row_id: [pair_dict, ...] } with aspect/opinion/aspect_type/polarity_hint.
    """
    if args.from_cache:
        print(f"Load pairs from cache: {CACHE_PATH.name}")
        with open(CACHE_PATH, encoding="utf-8") as f:
            cache = json.load(f)
        print(f"  {sum(len(v) for v in cache.values())} pairs, {len(cache)} rows")
        return cache

    t0 = time.perf_counter()
    print("Load CKIP...")
    ws, pos_tagger = init_ckip()
    timings.append(("Load CKIP", time.perf_counter() - t0))

    t0 = time.perf_counter()
    texts = [r["text"] for r in rows]
    print(f"CKIP tokenize ({len(texts)} rows)...")
    ws_res, pos_res = ckip_tokenize(texts, ws, pos_tagger)
    timings.append(("CKIP tokenize", time.perf_counter() - t0))

    t0 = time.perf_counter()
    print("Load stanza...")
    nlp  = init_stanza()
    timings.append(("Load Stanza", time.perf_counter() - t0))

    ate  = ATE()
    ote  = OTE()
    aope = AOPE(stanza_nlp=nlp)

    t0 = time.perf_counter()
    results: dict = {}
    for i, row in enumerate(rows):
        pairs = aope.pair_pipeline(ws_res[i], pos_res[i], ate, ote)
        results[row["row_id"]] = [
            {k: v for k, v in p.items() if k != "opinion_idx"}
            for p in pairs
        ]
    timings.append(("Pair extraction (D4)", time.perf_counter() - t0))

    if args.save_cache:
        CACHE_PATH.parent.mkdir(exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Cache saved -> {CACHE_PATH.name}")

    return results


# ── Enrich: category + sentiment ─────────────────────────────────

def enrich(pairs_by_row: dict, texts_by_id: dict, timings: list) -> list[dict]:
    """Add aspect_category + polarity to each pair dict."""
    t0 = time.perf_counter()
    acd = ACD()
    acd.load_sbert()
    print(f"Anchors built for categories")
    timings.append(("Load SBERT (ACD)", time.perf_counter() - t0))

    t0 = time.perf_counter()
    asc = ASC()
    asc.load_distilbert()
    timings.append(("Load DistilBERT (ASC)", time.perf_counter() - t0))

    t0 = time.perf_counter()
    output: list[dict] = []
    for row_id in sorted(pairs_by_row.keys(), key=int):
        pairs = pairs_by_row[row_id]
        text  = texts_by_id.get(row_id, "")
        for p in pairs:
            output.append({
                "row_id":          row_id,
                "text":            text,
                "aspect_term":     p["aspect"],
                "aspect_category": acd.predict_combo(p["aspect"], p["opinion"], text),
                "opinion_term":    p["opinion"],
                "polarity":        asc.predict_pipeline(p, text),
                "aspect_type":     p.get("aspect_type", ""),
                "polarity_hint":   p.get("polarity_hint") or "",
            })
    timings.append(("ACD + ASC inference", time.perf_counter() - t0))
    return output


# ── Output ────────────────────────────────────────────────────────

_CSV_FIELDS = ["row_id", "text", "aspect_term", "aspect_category",
               "opinion_term", "polarity", "aspect_type", "polarity_hint"]


def write_csv(records: list[dict], path: Path):
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    print(f"CSV saved -> {path}")


def print_summary(records: list[dict], verbose: bool = False):
    by_row = defaultdict(list)
    for r in records:
        by_row[r["row_id"]].append(r)
    sep = "=" * 64
    print(f"\n{sep}")
    print("# Pipeline Output Summary")
    print(sep)
    print(f"Total quadruplets : {len(records)}")
    print(f"Rows with output  : {len(by_row)}")
    cat_counts: dict = defaultdict(int)
    pol_counts: dict = defaultdict(int)
    for r in records:
        cat_counts[r["aspect_category"]] += 1
        pol_counts[r["polarity"]] += 1
    print("\nCategory breakdown:")
    for cat in sorted(cat_counts):
        print(f"  {cat:<14} {cat_counts[cat]}")
    print(f"\nPolarity: pos={pol_counts['pos']}  neg={pol_counts['neg']}")
    if verbose:
        print(f"\n{sep}")
        print("## Per-Row Quadruplets\n")
        for row_id in sorted(by_row.keys(), key=int):
            pairs = by_row[row_id]
            print(f"[row {row_id}]  {pairs[0]['text'][:55]}")
            for p in pairs:
                print(f"  ({p['aspect_term']}, {p['aspect_category']}, "
                      f"{p['opinion_term']}, {p['polarity']})  [{p['aspect_type']}]")
            print()


# ── Timing ────────────────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="ABSA Pipeline (A4+Combo+C2++D4+F3)")
    parser.add_argument("--from-cache", action="store_true",
                        help="Skip CKIP+stanza, load cache/eval_cache.json")
    parser.add_argument("--save-cache", action="store_true",
                        help="Save D4 pairs to cache/eval_cache.json")
    parser.add_argument("-o", "--output",
                        default=str(ROOT_DIR / "output" / "pipeline_output.csv"))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = HISTORY_DIR / f"track_a_{timestamp}.md"

    with open(report_path, "w", encoding="utf-8") as f:
        sys.stdout = Tee(f)
        try:
            t_total     = time.perf_counter()
            timings:list = []
            print(f"# Pipeline run {timestamp}\n")
            rows        = load_texts(DATA_PATH)
            texts_by_id = {r["row_id"]: r["text"] for r in rows}
            print(f"Input: {len(rows)} rows from {DATA_PATH.name}")

            pairs_by_row = build_pairs(rows, args, timings)
            records      = enrich(pairs_by_row, texts_by_id, timings)
            write_csv(records, Path(args.output))
            print_summary(records, verbose=args.verbose)
            total_s = time.perf_counter() - t_total
            print_timing(timings, total_s)
            append_timing_log("a", timings, total_s, timestamp)
            print(f"\nReport saved -> {report_path}")

        finally:
            sys.stdout = sys.stdout.stdout


if __name__ == "__main__":
    main()
