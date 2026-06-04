"""
Stage 4 — Aspect-Opinion Pairing  橫向比較
D1: 線性最近鄰
D2: 子句邊界配對（fallback D1）
D3: Stanza nsubj/amod/advmod 依存配對，過濾至 A3+C2 已抽取的詞
D4: D3 + D2 fallback（main eval baseline）
D5: D4 + cross-clause cosine pairing for PC opinions（sweep only）

Input 固定為 Stage 1 最優（A4）+ Stage 3 最優（C2）。

執行：
  python eval.py --save-cache    # 首次跑（需 stanza）
  python eval.py --from-cache    # 讀取 stage4_cache.json
  python eval.py -v              # per-row 錯誤細節
  python eval.py --save-metrics  # 更新 reports/stage_metrics.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
import csv

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.constants import GT_PATH, CACHE_DIR
from shared.utils import Tee, init_stanza
from shared.metrics import greedy_pair_match, prf
from stage1_aspect_term_extraction.ATE import ATE
from stage3_opinion_term_extraction.OTE import OTE
from stage4_aspect_opinion_pair_extraction.AOPE import AOPE

HISTORY_DIR  = Path(__file__).parent / "history"
CACHE_PATH   = CACHE_DIR / "stage4_cache.json"
STAGE1_CACHE = CACHE_DIR / "stage1_cache.json"

_SEP = "=" * 64


# ── Data loading ──────────────────────────────────────────────────

def load_gt_pairs() -> dict:
    """{ row_id: [(asp_term, op_term), ...] } — explicit-explicit only."""
    from collections import defaultdict
    result: dict = defaultdict(list)
    with open(GT_PATH, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            asp = r["aspect_term"].strip()
            op  = r["opinion_term"].strip()
            if asp != "implicit" and op != "implicit":
                result[r["row_id"]].append((asp, op))
    return {rid: pairs for rid, pairs in result.items() if pairs}


# ── Evaluation ────────────────────────────────────────────────────

def eval_pairs(gt_by_row: dict, pred_by_row: dict) -> dict:
    counters = {
        "exact":   {"tp": 0, "fp": 0, "fn": 0},
        "partial": {"tp": 0, "fp": 0, "fn": 0},
    }
    per_row: dict = {}
    for row_id, gt_pairs in gt_by_row.items():
        pred_pairs = pred_by_row.get(row_id, [])
        row_data: dict = {"gt": gt_pairs, "pred": pred_pairs}
        for level, c in counters.items():
            tp, fp, fn, ok_gi, miss_gi, fp_pi = greedy_pair_match(pred_pairs, gt_pairs, level)
            c["tp"] += tp
            c["fp"] += fp
            c["fn"] += fn
            if level == "partial":
                row_data["ok"]   = [gt_pairs[i]   for i in ok_gi]
                row_data["miss"] = [gt_pairs[i]   for i in miss_gi]
                row_data["fp"]   = [pred_pairs[i] for i in fp_pi]
        per_row[row_id] = row_data
    return {"exact": prf(**counters["exact"]), "partial": prf(**counters["partial"]),
            "per_row": per_row}


# ── Reporting ─────────────────────────────────────────────────────

def print_report(results: dict, gt_count: int):
    print(f"\n{_SEP}")
    print("# Stage 4 Eval — Aspect-Opinion Pairing")
    print(_SEP)
    print(f"\nGT explicit-explicit pairs: {gt_count} across all rows\n")
    header = (f"{'Method':<12} {'Exact P':>8} {'Exact R':>8} {'Exact F1':>9}  "
              f"{'Partial P':>10} {'Partial R':>10} {'Partial F1':>11}   {'TP/FP/FN (partial)'}")
    print(header)
    print("-" * len(header))
    for method, res in results.items():
        e = res["exact"]
        p = res["partial"]
        print(
            f"{method:<12} "
            f"{e['P']:>8.4f} {e['R']:>8.4f} {e['F1']:>9.4f}  "
            f"{p['P']:>10.4f} {p['R']:>10.4f} {p['F1']:>11.4f}   "
            f"{p['TP']}/{p['FP']}/{p['FN']}"
        )


def _print_d5_sweep(d5_results: dict, thresholds: list):
    print(f"\n{_SEP}")
    print("## D5 Threshold Sweep (Partial F1)\n")
    print(f"{'Threshold':>10} {'P':>8} {'R':>8} {'F1':>8}   {'TP/FP/FN'}")
    print("-" * 50)
    for thr in thresholds:
        r = d5_results[thr]["partial"]
        marker = " ◀ best" if thr == max(thresholds, key=lambda t: d5_results[t]["partial"]["F1"]) else ""
        print(f"{thr:>10.2f} {r['P']:>8.4f} {r['R']:>8.4f} {r['F1']:>8.4f}   "
              f"{r['TP']}/{r['FP']}/{r['FN']}{marker}")


def print_per_row(results: dict, gt_by_row: dict):
    print(f"\n{_SEP}")
    print("## Per-Row Detail (partial, rows with FN or FP)\n")
    for row_id in sorted(gt_by_row.keys(), key=int):
        rows_data = {m: results[m]["per_row"].get(row_id, {}) for m in results}
        if not any(d.get("miss") or d.get("fp") for d in rows_data.values()):
            continue
        print(f"[row {row_id}]  GT: {gt_by_row[row_id]}")
        for method, d in rows_data.items():
            print(f"  {method}: pred={d.get('pred', [])}")
            print(f"       ok={d.get('ok', [])}  miss={d.get('miss', [])}  FP={d.get('fp', [])}")
        print()


# ── Main ──────────────────────────────────────────────────────────

def _run(args):
    gt_by_row = load_gt_pairs()
    gt_count  = sum(len(v) for v in gt_by_row.values())
    print(f"GT: {gt_count} explicit-explicit pairs, {len(gt_by_row)} rows")

    with open(STAGE1_CACHE, encoding="utf-8") as f:
        ckip_data = json.load(f)
    print(f"CKIP cache: {len(ckip_data)} rows")

    stanza_cache: dict = {}
    nlp = None
    if args.from_cache:
        with open(CACHE_PATH, encoding="utf-8") as f:
            stanza_cache = json.load(f)
        print(f"Stage4 cache loaded: {len(stanza_cache)} rows")
    else:
        nlp = init_stanza()

    print("Load SentenceTransformer for D5...")
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    ate  = ATE()
    ote  = OTE()
    aope = AOPE(stanza_nlp=nlp)

    d1_by_row: dict       = {}
    d2_by_row: dict       = {}
    d3_by_row: dict       = {}
    d4_by_row: dict       = {}
    d5_scored_by_row: dict = {}

    for row_id, data in ckip_data.items():
        if row_id not in gt_by_row:
            continue
        tok, pos = data["tokens"], data["pos_tags"]
        aspects  = ate.extract_a4(tok, pos)
        opinions = ote.extract_c2(tok, pos)

        d1_by_row[row_id] = aope.pair_d1(tok, aspects, opinions)
        d2_by_row[row_id] = aope.pair_d2(tok, pos, aspects, opinions)

        if args.from_cache and row_id in stanza_cache:
            raw_pairs = [tuple(p) for p in stanza_cache[row_id]["raw_pairs"]]
        elif nlp is not None:
            try:
                doc  = nlp([tok])
                sent = doc.sentences[0]
                raw_pairs = aope._stanza_raw_pairs(sent, tok, pos)
                stanza_cache[row_id] = {"raw_pairs": [list(p) for p in raw_pairs]}
            except Exception as e:
                print(f"  [stanza err row {row_id}] {e}")
                raw_pairs = []
                stanza_cache[row_id] = {"raw_pairs": []}
        else:
            raw_pairs = []

        d3_by_row[row_id] = aope.pair_d3_from_raw(raw_pairs, aspects, opinions)
        d4_pairs           = aope.pair_d4(tok, pos, aspects, opinions, raw_pairs)
        d4_by_row[row_id] = d4_pairs
        d5_scored_by_row[row_id] = aope._d5_scored_additions(tok, pos, aspects, d4_pairs, st_model)

    if args.save_cache:
        CACHE_DIR.mkdir(exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(stanza_cache, f, ensure_ascii=False, indent=2)
        print(f"Cache saved -> {CACHE_PATH.name}")

    results = {
        "D1": eval_pairs(gt_by_row, d1_by_row),
        "D2": eval_pairs(gt_by_row, d2_by_row),
        "D3": eval_pairs(gt_by_row, d3_by_row),
        "D4": eval_pairs(gt_by_row, d4_by_row),
    }

    thresholds = [round(0.20 + 0.05 * i, 2) for i in range(13)]
    d5_sweep: dict = {}
    for thr in thresholds:
        d5_by_row = {
            rid: AOPE.pair_d5_threshold(d4_by_row[rid], d5_scored_by_row[rid], thr)
            for rid in d4_by_row
        }
        d5_sweep[thr] = eval_pairs(gt_by_row, d5_by_row)

    best_thr      = max(thresholds, key=lambda t: d5_sweep[t]["partial"]["F1"])
    d5_best_name  = f"D5@{best_thr}"
    d5_best_by_row = {
        rid: AOPE.pair_d5_threshold(d4_by_row[rid], d5_scored_by_row[rid], best_thr)
        for rid in d4_by_row
    }
    results[d5_best_name] = d5_sweep[best_thr]

    print_report(results, gt_count)
    _print_d5_sweep(d5_sweep, thresholds)
    if args.verbose:
        print_per_row(results, gt_by_row)

    print(f"\n{_SEP}")
    print("## Pairing Stats\n")
    for m, d in [("D1", d1_by_row), ("D2", d2_by_row), ("D3", d3_by_row),
                 ("D4", d4_by_row), (d5_best_name, d5_best_by_row)]:
        print(f"  {m} total pairs: {sum(len(v) for v in d.values())}")

    best = max(results.keys(), key=lambda m: results[m]["partial"]["F1"])
    print(f"\n  Best (partial F1): {best} = {results[best]['partial']['F1']:.4f}")

    CACHE_DIR.mkdir(exist_ok=True)
    (CACHE_DIR / "stage4_metrics.json").write_text(
        json.dumps({
            "pair_partial_f1": results["D4"]["partial"]["F1"],
            "pair_tp":         results["D4"]["partial"]["TP"],
        }, indent=2),
        encoding="utf-8",
    )
    print(f"  Metrics saved -> cache/stage4_metrics.json")

    if args.save_metrics:
        from shared.metrics_store import update_iterations, update_pipeline_standalone
        updates = {}
        for eval_id in ("D1", "D2", "D3", "D4"):
            if eval_id not in results:
                continue
            p = results[eval_id]["partial"]
            updates[eval_id] = {
                "f1": round(p["F1"], 4),
                "tp": p["TP"], "fp": p["FP"], "fn": p["FN"],
            }
        update_iterations("stage4", updates)
        update_pipeline_standalone("stage4", results["D4"]["partial"]["F1"])


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Stage 4 Aspect-Opinion Pairing Eval")
    parser.add_argument("--from-cache",    action="store_true")
    parser.add_argument("--save-cache",    action="store_true")
    parser.add_argument("--save-metrics",  action="store_true",
                        help="Update reports/stage_metrics.json with this run's results")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    HISTORY_DIR.mkdir(exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = HISTORY_DIR / f"{timestamp}_D1D2D3D4D5.md"

    with open(report_path, "w", encoding="utf-8") as f:
        sys.stdout = Tee(f)
        try:
            print(f"# Stage 4 Eval  {timestamp}\n")
            _run(args)
        finally:
            sys.stdout = sys.stdout.stdout

    print(f"\nReport saved -> {report_path}")


if __name__ == "__main__":
    main()
