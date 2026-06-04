"""
Stage 5 — Sentiment Classification
Compare F1 / F2 / F3 polarity prediction methods.

F1: opinion_term → DistilBERT
F2: text [SEP] opinion_term → DistilBERT
F3: polarity_hint (rule) → fallback F1

Metrics:
  Sentiment Accuracy : of D4 pair-TPs, fraction with correct polarity
  Quadruplet F1      : aspect + opinion + category + polarity all correct

Usage:
  python eval.py                         # 跑所有方法（F1 / F2 / F3）
  python eval.py --method F3             # 僅跑指定方法（可多選，e.g. --method F1 F2）
  python eval.py -v                      # per-row 情感錯誤細節
  python eval.py --save-metrics          # 更新 reports/stage_metrics.json + E2E 結果
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.constants import GT_PATH, CACHE_DIR, DATA_PATH
from shared.utils import Tee
from stage2_aspect_category_detection.ACD import ACD
from stage5_aspect_sentiment_classification.ASC import ASC

HISTORY_DIR = Path(__file__).parent / "history"
CACHE_PATH  = CACHE_DIR / "eval_cache.json"

def _load_d4_baseline() -> tuple[float, int]:
    p = CACHE_DIR / "stage4_metrics.json"
    if p.exists():
        m = json.loads(p.read_text(encoding="utf-8"))
        return m["pair_partial_f1"], m["pair_tp"]
    return 0.6111, 33  # fallback to last known values

D4_PAIR_PARTIAL_F1, D4_TP = _load_d4_baseline()

_SEP = "=" * 64


# ── Data loading ──────────────────────────────────────────────────

def load_gt() -> dict:
    gt: dict = defaultdict(list)
    with open(GT_PATH, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            gt[row["row_id"]].append({
                "aspect_term":     row["aspect_term"].strip(),
                "aspect_category": row["aspect_category"].strip(),
                "opinion_term":    row["opinion_term"].strip(),
                "polarity":        row["polarity"].strip(),
            })
    return dict(gt)


def load_pred() -> dict:
    with open(CACHE_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_texts() -> dict:
    texts = {}
    with open(DATA_PATH, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["status"] != "skip" and row["text"].strip():
                texts[row["row_id"]] = row["text"].strip()
    return texts


# ── Term matching (handles implicit) ─────────────────────────────

def _term_match(pred: str, gt: str, level: str) -> bool:
    p, g = pred.strip(), gt.strip()
    if p == g:
        return True
    if level == "exact":
        return False
    if "implicit" in (p, g):
        return False
    if p in g or g in p:
        return True
    shorter = min(len(p), len(g))
    overlap = len(set(p) & set(g))
    return shorter > 0 and (overlap / shorter) >= 0.7


def _prf(tp: int, fp: int, fn: int) -> dict:
    p  = tp / (tp + fp) if tp + fp > 0 else 0.0
    r  = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def greedy_pair_match(pred_pairs: list, gt_tuples: list, level: str) -> tuple:
    """Match on aspect + opinion only (pair level)."""
    used: set = set()
    pair_tps, pair_fns = [], []
    for gi, gt_t in enumerate(gt_tuples):
        found = False
        for pi, pp in enumerate(pred_pairs):
            if pi in used:
                continue
            if (_term_match(pp["aspect"], gt_t["aspect_term"], level)
                    and _term_match(pp["opinion"], gt_t["opinion_term"], level)):
                pair_tps.append((gi, pi))
                used.add(pi)
                found = True
                break
        if not found:
            pair_fns.append(gi)
    pair_fps = [pi for pi in range(len(pred_pairs)) if pi not in used]
    return pair_tps, pair_fns, pair_fps


def greedy_quad_match(pred_pairs: list, gt_tuples: list, level: str) -> tuple:
    """Match on aspect + opinion + category + polarity."""
    used: set = set()
    quad_tps, quad_fns = [], []
    for gi, gt_t in enumerate(gt_tuples):
        found = False
        for pi, pp in enumerate(pred_pairs):
            if pi in used:
                continue
            if (_term_match(pp["aspect"], gt_t["aspect_term"], level)
                    and _term_match(pp["opinion"], gt_t["opinion_term"], level)
                    and pp.get("aspect_category") == gt_t["aspect_category"]
                    and pp.get("polarity") == gt_t["polarity"]):
                quad_tps.append((gi, pi))
                used.add(pi)
                found = True
                break
        if not found:
            quad_fns.append(gi)
    quad_fps = [pi for pi in range(len(pred_pairs)) if pi not in used]
    return quad_tps, quad_fns, quad_fps


# ── Category enrichment ───────────────────────────────────────────

def enrich_with_category(pred_raw: dict, texts: dict) -> dict:
    acd = ACD()
    acd.load_sbert()
    result = {}
    for row_id, pairs in pred_raw.items():
        text = texts.get(row_id, "")
        result[row_id] = [
            {**p, "aspect_category": acd.predict_combo(
                p["aspect"], p["opinion"], text)}
            for p in pairs
        ]
    print("Category enrichment done.")
    return result


# ── Evaluation ────────────────────────────────────────────────────

def evaluate_method(method_name: str, pred_with_polarity: dict, gt_by_row: dict) -> dict:
    pair_c  = {"tp": 0, "fp": 0, "fn": 0}
    quad_pe = {k: {"tp": 0, "fp": 0, "fn": 0} for k in ("partial", "exact")}
    senti_correct = 0
    senti_total   = 0
    per_row: dict = {}

    for row_id, gt_tuples in sorted(gt_by_row.items(), key=lambda x: int(x[0])):
        pred_pairs = pred_with_polarity.get(row_id, [])
        p_tps, p_fns, p_fps = greedy_pair_match(pred_pairs, gt_tuples, "partial")
        pair_c["tp"] += len(p_tps)
        pair_c["fp"] += len(p_fps)
        pair_c["fn"] += len(p_fns)

        row_correct = 0
        row_wrong   = []
        for gi, pi in p_tps:
            senti_total += 1
            gt_pol   = gt_tuples[gi]["polarity"]
            pred_pol = pred_pairs[pi].get("polarity", "pos")
            if gt_pol == pred_pol:
                senti_correct += 1
                row_correct   += 1
            else:
                row_wrong.append({
                    "aspect":  pred_pairs[pi]["aspect"],
                    "opinion": pred_pairs[pi]["opinion"],
                    "pred":    pred_pol,
                    "gt":      gt_pol,
                })

        for level in ("partial", "exact"):
            q_tps, q_fns, q_fps = greedy_quad_match(pred_pairs, gt_tuples, level)
            quad_pe[level]["tp"] += len(q_tps)
            quad_pe[level]["fp"] += len(q_fps)
            quad_pe[level]["fn"] += len(q_fns)

        per_row[row_id] = {
            "pair_tp": len(p_tps), "pair_fn": len(p_fns), "pair_fp": len(p_fps),
            "senti_correct": row_correct, "senti_wrong": row_wrong,
        }

    senti_acc = round(senti_correct / senti_total, 4) if senti_total > 0 else 0.0
    return {
        "method":       method_name,
        "pair_partial": _prf(**pair_c),
        "senti_acc":    senti_acc,
        "senti_tp":     senti_correct,
        "senti_total":  senti_total,
        "quad_partial": _prf(**quad_pe["partial"]),
        "quad_exact":   _prf(**quad_pe["exact"]),
        "per_row":      per_row,
    }


# ── Output ────────────────────────────────────────────────────────

def print_results(results: list):
    print(f"\n{_SEP}")
    print("# Stage 5 — Sentiment Classification")
    print(f"# D4 baseline: Pair Partial F1 = {D4_PAIR_PARTIAL_F1}  (TP={D4_TP})")
    print(_SEP)
    print(f"\n{'Method':<6}  {'Pair F1':>8}  {'SentiAcc':>9}  {'Quad P-F1':>10}  "
          f"{'Quad E-F1':>10}  {'QuadTP/FP/FN'}")
    print("-" * 75)
    for r in results:
        pp = r["pair_partial"]
        qp = r["quad_partial"]
        qe = r["quad_exact"]
        print(
            f"{r['method']:<6}  {pp['f1']:>8.4f}  {r['senti_acc']:>9.4f}  "
            f"{qp['f1']:>10.4f}  {qe['f1']:>10.4f}  {qp['tp']}/{qp['fp']}/{qp['fn']}"
        )
    print("\n## Sentiment Accuracy Detail")
    for r in results:
        print(f"\n[{r['method']}] Senti correct: {r['senti_tp']} / {r['senti_total']}")


def print_verbose(results: list):
    print(f"\n{_SEP}")
    print("## Per-Row Sentiment Errors (per method)")
    for r in results:
        print(f"\n### {r['method']}")
        for row_id, data in sorted(r["per_row"].items(), key=lambda x: int(x[0])):
            if data["senti_wrong"]:
                print(f"  [row {row_id}]")
                for w in data["senti_wrong"]:
                    print(f"    ({w['aspect']}, {w['opinion']}) pred={w['pred']}  GT={w['gt']}")


# ── Main ──────────────────────────────────────────────────────────

def _run(args):
    gt        = load_gt()
    pred_raw  = load_pred()
    texts     = load_texts()
    total_gt   = sum(len(v) for v in gt.values())
    total_pred = sum(len(v) for v in pred_raw.values())
    print(f"GT: {total_gt} tuples, {len(gt)} rows")
    print(f"Pred cache: {total_pred} pairs, {len(pred_raw)} rows")

    pred_base = enrich_with_category(pred_raw, texts)

    methods = args.method if args.method else ["F1", "F2", "F3"]
    asc     = ASC()
    asc.load_distilbert()
    results = []

    for method in methods:
        print(f"\n--- Predicting {method} ---")
        pred_with_polarity: dict = {}
        for row_id, pairs in pred_base.items():
            text = texts.get(row_id, "")
            enriched = []
            for p in pairs:
                p2 = dict(p)
                if method == "F1":
                    p2["polarity"] = asc.predict_f1(p, text)
                elif method == "F2":
                    p2["polarity"] = asc.predict_f2(p, text)
                elif method == "F3":
                    p2["polarity"] = asc.predict_f3(p, text)
                enriched.append(p2)
            pred_with_polarity[row_id] = enriched

        result = evaluate_method(method, pred_with_polarity, gt)
        results.append(result)
        print(f"  Senti Acc={result['senti_acc']:.4f}  "
              f"Quad Partial F1={result['quad_partial']['f1']:.4f}")

    print_results(results)
    if args.verbose:
        print_verbose(results)

    if args.save_metrics:
        from shared.metrics_store import update_e2e, update_pipeline_standalone
        best_result = max(results, key=lambda r: r["quad_partial"]["f1"])
        update_e2e(best_result["quad_partial"]["f1"])
        update_pipeline_standalone("stage5", best_result["senti_acc"])


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Stage 5 Sentiment eval")
    parser.add_argument("--method", nargs="+", choices=["F1", "F2", "F3"])
    parser.add_argument("--save-metrics",  action="store_true",
                        help="Update reports/stage_metrics.json with this run's results")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    HISTORY_DIR.mkdir(exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = HISTORY_DIR / f"stage5_{timestamp}.md"

    with open(report_path, "w", encoding="utf-8") as f:
        sys.stdout = Tee(f)
        try:
            print(f"# Stage 5 run {timestamp}\n")
            _run(args)
        finally:
            sys.stdout = sys.stdout.stdout

    print(f"\nReport saved -> {report_path}")


if __name__ == "__main__":
    main()
