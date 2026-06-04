"""
eval_track.py — Track A / Track B Quadruplet F1 Evaluation

Usage:
  python eval_track.py               # 同時評估 Track A + Track B 並對比（預設）
  python eval_track.py --track a     # 僅評估 Track A (output/pipeline_output.csv)
  python eval_track.py --track b     # 僅評估 Track B (output/track_b_output.csv)
  python eval_track.py -v            # 加上 per-row 錯誤細節
  python eval_track.py --save-metrics  # 更新 reports/stage_metrics.json
"""

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT_DIR    = Path(__file__).parent
GT_PATH     = ROOT_DIR / "data" / "ground_truth_absa.csv"
TRACK_A_CSV = ROOT_DIR / "output" / "pipeline_output.csv"
TRACK_B_CSV = ROOT_DIR / "output" / "track_b_output.csv"
HIST_A_DIR       = ROOT_DIR / "history" / "track_a"
HIST_B_DIR       = ROOT_DIR / "history" / "track_b"
HIST_COMPARE_DIR = ROOT_DIR / "history" / "compare"

sys.path.insert(0, str(ROOT_DIR))
from shared.utils import Tee
from shared.constants import CATEGORIES as _CATS
from shared.metrics import prf

CATEGORIES = _CATS


# ── Matching helpers ──────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return s.strip()


def term_match(pred: str, gt: str) -> dict:
    """{"exact": bool, "partial": bool}"""
    p, g = _norm(pred), _norm(gt)
    if p == "implicit" or g == "implicit":
        eq = (p == g)
        return {"exact": eq, "partial": eq}
    if p == g:
        return {"exact": True, "partial": True}
    if p in g or g in p:
        return {"exact": False, "partial": True}
    overlap = len(set(p) & set(g))
    shorter = min(len(p), len(g))
    partial = shorter > 0 and (overlap / shorter) >= 0.7
    return {"exact": False, "partial": partial}


def greedy_match(pred_pairs: list, gt_tuples: list, level: str) -> tuple:
    """aspect + opinion both match → TP"""
    used: set = set()
    matched_gt:   list = []
    unmatched_gt: list = []
    for gi, gt_t in enumerate(gt_tuples):
        found = False
        for pi, pred_p in enumerate(pred_pairs):
            if pi in used:
                continue
            asp_ok = term_match(pred_p["aspect_term"], gt_t["aspect_term"])[level]
            op_ok  = term_match(pred_p["opinion_term"], gt_t["opinion_term"])[level]
            if asp_ok and op_ok:
                matched_gt.append(gi)
                used.add(pi)
                found = True
                break
        if not found:
            unmatched_gt.append(gi)
    unmatched_pred = [pi for pi in range(len(pred_pairs)) if pi not in used]
    return matched_gt, unmatched_gt, unmatched_pred


def _single_side_f1(gt_terms: list, pred_terms: list, level: str) -> dict:
    used: set = set()
    tp = fn = 0
    for gt_t in gt_terms:
        found = False
        for pi, pred_t in enumerate(pred_terms):
            if pi not in used and term_match(pred_t, gt_t)[level]:
                used.add(pi)
                found = True
                break
        if found:
            tp += 1
        else:
            fn += 1
    fp = max(0, len(pred_terms) - len(used))
    return prf(tp, fp, fn)


# ── Quadruplet / triplet matching ─────────────────────────────────────────────

def quad_match(pred_pairs: list, gt_tuples: list, level: str) -> tuple:
    """aspect + opinion + category + polarity → TP"""
    used: set = set()
    matched_gt:   list = []
    unmatched_gt: list = []
    for gi, gt_t in enumerate(gt_tuples):
        found = False
        for pi, pred_p in enumerate(pred_pairs):
            if pi in used:
                continue
            asp_ok = term_match(pred_p["aspect_term"], gt_t["aspect_term"])[level]
            op_ok  = term_match(pred_p["opinion_term"], gt_t["opinion_term"])[level]
            cat_ok = pred_p["aspect_category"] == gt_t["aspect_category"]
            pol_ok = pred_p["polarity"] == gt_t["polarity"]
            if asp_ok and op_ok and cat_ok and pol_ok:
                matched_gt.append(gi)
                used.add(pi)
                found = True
                break
        if not found:
            unmatched_gt.append(gi)
    unmatched_pred = [pi for pi in range(len(pred_pairs)) if pi not in used]
    return matched_gt, unmatched_gt, unmatched_pred


def _pair_match_detailed(pred_pairs: list, gt_tuples: list, level: str) -> tuple:
    used: set = set()
    matched:      list = []
    unmatched_gt: list = []
    for gi, gt_t in enumerate(gt_tuples):
        found = False
        for pi, pred_p in enumerate(pred_pairs):
            if pi in used:
                continue
            if (term_match(pred_p["aspect_term"],  gt_t["aspect_term"])[level] and
                    term_match(pred_p["opinion_term"], gt_t["opinion_term"])[level]):
                matched.append((gi, pi))
                used.add(pi)
                found = True
                break
        if not found:
            unmatched_gt.append(gi)
    unmatched_pred = [pi for pi in range(len(pred_pairs)) if pi not in used]
    return matched, unmatched_gt, unmatched_pred


def cat_triplet_match(pred_pairs: list, gt_tuples: list, level: str) -> tuple:
    """aspect + opinion + category → TP"""
    used: set = set()
    matched_gt:   list = []
    unmatched_gt: list = []
    for gi, gt_t in enumerate(gt_tuples):
        found = False
        for pi, pred_p in enumerate(pred_pairs):
            if pi in used:
                continue
            if (term_match(pred_p["aspect_term"],  gt_t["aspect_term"])[level] and
                    term_match(pred_p["opinion_term"], gt_t["opinion_term"])[level] and
                    pred_p["aspect_category"] == gt_t["aspect_category"]):
                matched_gt.append(gi)
                used.add(pi)
                found = True
                break
        if not found:
            unmatched_gt.append(gi)
    unmatched_pred = [pi for pi in range(len(pred_pairs)) if pi not in used]
    return matched_gt, unmatched_gt, unmatched_pred


def pol_triplet_match(pred_pairs: list, gt_tuples: list, level: str) -> tuple:
    """aspect + opinion + polarity → TP"""
    used: set = set()
    matched_gt:   list = []
    unmatched_gt: list = []
    for gi, gt_t in enumerate(gt_tuples):
        found = False
        for pi, pred_p in enumerate(pred_pairs):
            if pi in used:
                continue
            if (term_match(pred_p["aspect_term"],  gt_t["aspect_term"])[level] and
                    term_match(pred_p["opinion_term"], gt_t["opinion_term"])[level] and
                    pred_p["polarity"] == gt_t["polarity"]):
                matched_gt.append(gi)
                used.add(pi)
                found = True
                break
        if not found:
            unmatched_gt.append(gi)
    unmatched_pred = [pi for pi in range(len(pred_pairs)) if pi not in used]
    return matched_gt, unmatched_gt, unmatched_pred


# ── Data loading ──────────────────────────────────────────────────────────────

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


def load_pred(csv_path: Path) -> dict:
    pred: dict = defaultdict(list)
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            pred[row["row_id"]].append({
                "aspect_term":     row["aspect_term"].strip(),
                "aspect_category": row["aspect_category"].strip(),
                "opinion_term":    row["opinion_term"].strip(),
                "polarity":        row["polarity"].strip(),
            })
    return dict(pred)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(gt_by_row: dict, pred_by_row: dict) -> dict:
    counters = {
        k: {"tp": 0, "fp": 0, "fn": 0}
        for k in ("pair_partial", "pair_exact",
                  "cat_partial", "pol_partial",
                  "quad_partial", "quad_exact",
                  "aspect_partial", "opinion_partial")
    }
    cat_cls = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CATEGORIES}
    pol_cls = {"pos": {"tp": 0, "fp": 0, "fn": 0},
               "neg": {"tp": 0, "fp": 0, "fn": 0}}
    errors = {
        "total_gt": 0, "total_pred": 0,
        "asp_found_op_missed": 0,
        "op_found_asp_missed": 0,
        "both_missed": 0,
        "implicit_asp_missed": 0,
        "explicit_asp_missed": 0,
    }
    per_row: dict = {}

    for row_id, gt_tuples in sorted(gt_by_row.items(), key=lambda x: int(x[0])):
        pred_pairs = pred_by_row.get(row_id, [])
        errors["total_gt"]   += len(gt_tuples)
        errors["total_pred"] += len(pred_pairs)

        for level in ("partial", "exact"):
            m_gt, fn_gt, fp_pred = greedy_match(pred_pairs, gt_tuples, level)
            counters[f"pair_{level}"]["tp"] += len(m_gt)
            counters[f"pair_{level}"]["fp"] += len(fp_pred)
            counters[f"pair_{level}"]["fn"] += len(fn_gt)

            qm_gt, qfn_gt, qfp_pred = quad_match(pred_pairs, gt_tuples, level)
            counters[f"quad_{level}"]["tp"] += len(qm_gt)
            counters[f"quad_{level}"]["fp"] += len(qfp_pred)
            counters[f"quad_{level}"]["fn"] += len(qfn_gt)

            if level == "partial":
                pred_aspects  = [p["aspect_term"]  for p in pred_pairs]
                pred_opinions = [p["opinion_term"] for p in pred_pairs]
                for gi in fn_gt:
                    t = gt_tuples[gi]
                    if t["aspect_term"] == "implicit":
                        errors["implicit_asp_missed"] += 1
                    else:
                        errors["explicit_asp_missed"] += 1
                    asp_found = any(term_match(pa, t["aspect_term"])["partial"] for pa in pred_aspects)
                    op_found  = any(term_match(po, t["opinion_term"])["partial"] for po in pred_opinions)
                    if asp_found and not op_found:
                        errors["asp_found_op_missed"] += 1
                    elif op_found and not asp_found:
                        errors["op_found_asp_missed"] += 1
                    else:
                        errors["both_missed"] += 1

        m, fn, fp = cat_triplet_match(pred_pairs, gt_tuples, "partial")
        counters["cat_partial"]["tp"] += len(m)
        counters["cat_partial"]["fp"] += len(fp)
        counters["cat_partial"]["fn"] += len(fn)

        m, fn, fp = pol_triplet_match(pred_pairs, gt_tuples, "partial")
        counters["pol_partial"]["tp"] += len(m)
        counters["pol_partial"]["fp"] += len(fp)
        counters["pol_partial"]["fn"] += len(fn)

        pm, _, _ = _pair_match_detailed(pred_pairs, gt_tuples, "partial")
        for gi, pi in pm:
            gc = gt_tuples[gi]["aspect_category"]
            pc = pred_pairs[pi]["aspect_category"]
            gp = gt_tuples[gi]["polarity"]
            pp = pred_pairs[pi]["polarity"]
            for c in CATEGORIES:
                if gc == c and pc == c:  cat_cls[c]["tp"] += 1
                elif pc == c:            cat_cls[c]["fp"] += 1
                elif gc == c:            cat_cls[c]["fn"] += 1
            for pol in ("pos", "neg"):
                if gp == pol and pp == pol:  pol_cls[pol]["tp"] += 1
                elif pp == pol:              pol_cls[pol]["fp"] += 1
                elif gp == pol:              pol_cls[pol]["fn"] += 1

        gt_asp = [t["aspect_term"]  for t in gt_tuples]
        pr_asp = [p["aspect_term"]  for p in pred_pairs]
        m = _single_side_f1(gt_asp, pr_asp, "partial")
        counters["aspect_partial"]["tp"] += m["TP"]
        counters["aspect_partial"]["fp"] += m["FP"]
        counters["aspect_partial"]["fn"] += m["FN"]

        gt_op = [t["opinion_term"] for t in gt_tuples]
        pr_op = [p["opinion_term"] for p in pred_pairs]
        m = _single_side_f1(gt_op, pr_op, "partial")
        counters["opinion_partial"]["tp"] += m["TP"]
        counters["opinion_partial"]["fp"] += m["FP"]
        counters["opinion_partial"]["fn"] += m["FN"]

        per_row[row_id] = {"gt": gt_tuples, "pred": pred_pairs}

    metrics     = {k: prf(**v) for k, v in counters.items()}
    cat_metrics = {c: prf(**v) for c, v in cat_cls.items()}
    pol_metrics = {p: prf(**v) for p, v in pol_cls.items()}
    cat_macro   = round(sum(v["F1"] for v in cat_metrics.values()) / len(cat_metrics), 4)
    pol_macro   = round(sum(v["F1"] for v in pol_metrics.values()) / len(pol_metrics), 4)
    return {
        "metrics":      metrics,
        "cat_cls":      cat_metrics,
        "pol_cls":      pol_metrics,
        "cat_macro_f1": cat_macro,
        "pol_macro_f1": pol_macro,
        "errors":       errors,
        "per_row":      per_row,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

_SEP = "=" * 68


def print_results(res: dict, label: str):
    m  = res["metrics"]
    er = res["errors"]
    _L = "-" * 67

    print(f"\n{_SEP}")
    print(f"# {label} — Quadruplet Evaluation")
    print(_SEP)
    print(f"\nTotal GT tuples  : {er['total_gt']}")
    print(f"Total Pred tuples: {er['total_pred']}\n")

    print(f"  {'Metric':<24} {'P':>7} {'R':>7} {'F1':>7}   {'TP':>4} {'FP':>4} {'FN':>4}")
    print(f"  {_L}")
    for name, key in [
        ("Aspect Partial F1",     "aspect_partial"),
        ("Opinion Partial F1",    "opinion_partial"),
        ("Pair    (asp+op)",      "pair_partial"),
        ("Pair    (exact)",       "pair_exact"),
        ("Cat     (asp+op+cat)",  "cat_partial"),
        ("Polar   (asp+op+pol)",  "pol_partial"),
        ("Quad    (partial)",     "quad_partial"),
        ("Quad    (exact)",       "quad_exact"),
    ]:
        v = m[key]
        print(f"  {name:<24} {v['P']:>7.4f} {v['R']:>7.4f} {v['F1']:>7.4f}"
              f"   {v['TP']:>4} {v['FP']:>4} {v['FN']:>4}")

    print(f"\n  ★ Quad Partial F1 = {m['quad_partial']['F1']:.4f}")

    print(f"\n## Category Classification  "
          f"(on Pair-TPs, macro F1={res['cat_macro_f1']:.4f})\n")
    print(f"  {'Category':<14} {'P':>7} {'R':>7} {'F1':>7}  {'TP':>3} {'FP':>3} {'FN':>3}")
    print(f"  {'-'*14} {'-'*7} {'-'*7} {'-'*7}  {'-'*3} {'-'*3} {'-'*3}")
    for cat, v in res["cat_cls"].items():
        print(f"  {cat:<14} {v['P']:>7.4f} {v['R']:>7.4f} {v['F1']:>7.4f}"
              f"  {v['TP']:>3} {v['FP']:>3} {v['FN']:>3}")

    print(f"\n## Polarity Classification  "
          f"(on Pair-TPs, macro F1={res['pol_macro_f1']:.4f})\n")
    print(f"  {'Polarity':<10} {'P':>7} {'R':>7} {'F1':>7}  {'TP':>3} {'FP':>3} {'FN':>3}")
    print(f"  {'-'*10} {'-'*7} {'-'*7} {'-'*7}  {'-'*3} {'-'*3} {'-'*3}")
    for pol, v in res["pol_cls"].items():
        print(f"  {pol:<10} {v['P']:>7.4f} {v['R']:>7.4f} {v['F1']:>7.4f}"
              f"  {v['TP']:>3} {v['FP']:>3} {v['FN']:>3}")

    fn = er['total_gt'] - m['pair_partial']['TP']
    if fn:
        print(f"\nFN breakdown (Pair Partial):")
        print(f"  asp_found_op_missed  : {er['asp_found_op_missed']:3d}"
              f"  ({er['asp_found_op_missed']/fn*100:.0f}%)")
        print(f"  op_found_asp_missed  : {er['op_found_asp_missed']:3d}"
              f"  ({er['op_found_asp_missed']/fn*100:.0f}%)")
        print(f"  both_missed          : {er['both_missed']:3d}"
              f"  ({er['both_missed']/fn*100:.0f}%)")
        print(f"  implicit_asp_missed  : {er['implicit_asp_missed']:3d}")
        print(f"  explicit_asp_missed  : {er['explicit_asp_missed']:3d}")


def print_comparison(res_a: dict, res_b: dict):
    ma = res_a["metrics"]
    mb = res_b["metrics"]
    print(f"\n{_SEP}")
    print("# Track A vs Track B — 對比")
    print(_SEP)
    print(f"\n  {'Metric':<24} {'Track A':>10} {'Track B':>10} {'Δ':>8}")
    print(f"  {'-'*24} {'-'*10} {'-'*10} {'-'*8}")
    keys = [
        ("Aspect Partial F1",    "aspect_partial"),
        ("Opinion Partial F1",   "opinion_partial"),
        ("Pair Partial F1",      "pair_partial"),
        ("Cat  (asp+op+cat)",    "cat_partial"),
        ("Polar(asp+op+pol)",    "pol_partial"),
        ("Quad Partial F1",      "quad_partial"),
    ]
    for name, key in keys:
        fa = ma[key]["F1"]
        fb = mb[key]["F1"]
        delta = fb - fa
        marker = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
        print(f"  {name:<24} {fa:>10.4f} {fb:>10.4f} {marker}{abs(delta):>7.4f}")

    print(f"\n  {'Category macro F1':<24} {res_a['cat_macro_f1']:>10.4f} {res_b['cat_macro_f1']:>10.4f}"
          f" {'▲' if res_b['cat_macro_f1'] > res_a['cat_macro_f1'] else '▼'}"
          f"{abs(res_b['cat_macro_f1'] - res_a['cat_macro_f1']):>7.4f}")
    print(f"  {'Polarity macro F1':<24} {res_a['pol_macro_f1']:>10.4f} {res_b['pol_macro_f1']:>10.4f}"
          f" {'▲' if res_b['pol_macro_f1'] > res_a['pol_macro_f1'] else '▼'}"
          f"{abs(res_b['pol_macro_f1'] - res_a['pol_macro_f1']):>7.4f}")


def print_errors(res: dict, label: str):
    print(f"\n{_SEP}")
    print(f"## {label} — Per-Row Error Cases\n")
    for row_id, data in sorted(res["per_row"].items(), key=lambda x: int(x[0])):
        gt    = data["gt"]
        pred  = data["pred"]
        m_gt, fn_gt, fp_pred = greedy_match(pred, gt, "partial")
        if not fn_gt and not fp_pred:
            continue
        print(f"[row {row_id}]")
        if fn_gt:
            for i in fn_gt:
                t = gt[i]
                print(f"  FN  ({t['aspect_term']}, {t['aspect_category']}, "
                      f"{t['opinion_term']}, {t['polarity']})")
        if fp_pred:
            for i in fp_pred:
                p = pred[i]
                print(f"  FP  ({p['aspect_term']}, {p['aspect_category']}, "
                      f"{p['opinion_term']}, {p['polarity']})")
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Track A/B Quadruplet F1 Evaluation")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--track", choices=["a", "b"],
                       help="僅評估單一 track（預設: 兩個都跑並對比）")
    parser.add_argument("--save-metrics",  action="store_true",
                        help="Update reports/stage_metrics.json with this run's results")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="顯示 per-row 錯誤細節")
    args = parser.parse_args()

    do_a = args.track in (None, "a")
    do_b = args.track in (None, "b")

    # History path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.track == "a":
        hist_dir  = HIST_A_DIR
        hist_name = f"eval_a_{timestamp}.md"
    elif args.track == "b":
        hist_dir  = HIST_B_DIR
        hist_name = f"eval_b_{timestamp}.md"
    else:
        hist_dir  = HIST_COMPARE_DIR
        hist_name = f"eval_compare_{timestamp}.md"

    hist_dir.mkdir(parents=True, exist_ok=True)
    report_path = hist_dir / hist_name

    with open(report_path, "w", encoding="utf-8") as f:
        sys.stdout = Tee(f)
        try:
            print(f"# eval_track  {timestamp}\n")

            gt_by_row = load_gt()
            print(f"GT: {sum(len(v) for v in gt_by_row.values())} tuples, "
                  f"{len(gt_by_row)} rows")

            res_a = res_b = None

            if do_a:
                pred_a = load_pred(TRACK_A_CSV)
                print(f"Track A pred: {sum(len(v) for v in pred_a.values())} tuples, "
                      f"{len(pred_a)} rows")
                res_a = evaluate(gt_by_row, pred_a)

            if do_b:
                pred_b = load_pred(TRACK_B_CSV)
                print(f"Track B pred: {sum(len(v) for v in pred_b.values())} tuples, "
                      f"{len(pred_b)} rows")
                res_b = evaluate(gt_by_row, pred_b)

            if res_a:
                print_results(res_a, label="Track A")
            if res_b:
                print_results(res_b, label="Track B")
            if res_a and res_b:
                print_comparison(res_a, res_b)

            if args.verbose:
                if res_a:
                    print_errors(res_a, label="Track A")
                if res_b:
                    print_errors(res_b, label="Track B")

            if args.save_metrics:
                from shared.metrics_store import update_track_comparison
                if res_a:
                    update_track_comparison("track_a", res_a)
                if res_b:
                    update_track_comparison("track_b", res_b)

            print(f"\nReport saved -> {report_path}")
        finally:
            sys.stdout = sys.stdout.stdout


if __name__ == "__main__":
    main()
