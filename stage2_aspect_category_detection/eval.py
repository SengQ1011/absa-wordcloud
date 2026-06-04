"""
Stage 2 — Aspect Category Mapping  橫向比較
B2:     Seed 詞典 substring match（explicit only）
B1:     aspect_term [SEP] text → cosine(anchor)（explicit only）
B3:     text → cosine(anchor)（all tuples）
B1-imp: opinion_term [SEP] text → cosine(anchor)（implicit only）
Combo:  explicit → B2→B1→B3；  implicit → B1-imp

執行：
  python eval.py                        # full run
  python eval.py --save-cache           # 儲存 raw scores cache
  python eval.py --from-cache           # 讀取快取（略過模型載入）
  python eval.py --from-cache --sweep   # threshold grid search
  python eval.py --no-others            # 排除 OTHERS category 的 tuple
  python eval.py -v                     # per-tuple 錯誤細節
  python eval.py --save-metrics         # 更新 reports/stage_metrics.json
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.constants import GT_PATH, CACHE_DIR
from shared.utils import Tee
from stage2_aspect_category_detection.ACD import ACD, OTHERS_THRESHOLD_EXP, OTHERS_THRESHOLD_IMP

HISTORY_DIR = Path(__file__).parent / "history"
CACHE_PATH  = CACHE_DIR / "stage2_cache.json"

_SEP = "=" * 68
SWEEP_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]


# ── Data loading ──────────────────────────────────────────────────

def load_gt() -> list[dict]:
    rows = []
    with open(GT_PATH, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append({
                "row_id":   r["row_id"].strip(),
                "text":     r["text"].strip(),
                "a_term":   r["aspect_term"].strip(),
                "category": r["aspect_category"].strip(),
                "o_term":   r["opinion_term"].strip(),
                "polarity": r["polarity"].strip(),
            })
    return rows


# ── Evaluation ────────────────────────────────────────────────────

def compute_metrics(preds_gt: list[tuple]) -> dict:
    if not preds_gt:
        return {"accuracy": 0.0, "macro_f1": 0.0, "n": 0, "correct": 0, "per_class": {}}
    tp: dict = defaultdict(int)
    fp: dict = defaultdict(int)
    fn: dict = defaultdict(int)
    for pred, gt in preds_gt:
        if pred == gt:
            tp[gt] += 1
        else:
            fp[pred] += 1
            fn[gt]   += 1
    correct = sum(tp.values())
    acc     = correct / len(preds_gt)
    gt_cats = sorted({g for _, g in preds_gt})
    f1s, per_cls = [], {}
    for cat in gt_cats:
        p_ = tp[cat] / (tp[cat] + fp[cat]) if (tp[cat] + fp[cat]) > 0 else 0.0
        r_ = tp[cat] / (tp[cat] + fn[cat]) if (tp[cat] + fn[cat]) > 0 else 0.0
        f1_ = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) > 0 else 0.0
        f1s.append(f1_)
        per_cls[cat] = {"P": round(p_, 4), "R": round(r_, 4), "F1": round(f1_, 4),
                        "TP": tp[cat], "FP": fp[cat], "FN": fn[cat]}
    return {"accuracy": round(acc, 4), "macro_f1": round(sum(f1s) / len(f1s), 4),
            "n": len(preds_gt), "correct": correct, "per_class": per_cls}


def build_combo(tuples: list[dict], all_b2, all_b1, all_b3, all_b1_imp) -> list[str]:
    combos = []
    for t, b2, b1, b3, b1_imp in zip(tuples, all_b2, all_b1, all_b3, all_b1_imp):
        if t["a_term"] == "implicit":
            combos.append(b1_imp)
        else:
            if b2 is not None and b2 != "OTHERS":
                combos.append(b2)
            elif b1 is not None:
                combos.append(b1)
            else:
                combos.append(b3)
    return combos


def build_method_results(tuples, preds_b2, preds_b1, preds_b3, preds_b1imp, preds_cbo) -> dict:
    gt_cats = [t["category"] for t in tuples]
    is_exp  = [t["a_term"] != "implicit" for t in tuples]
    exp_gt  = [g for g, e in zip(gt_cats, is_exp) if e]
    imp_gt  = [g for g, e in zip(gt_cats, is_exp) if not e]
    _empty  = {"accuracy": 0.0, "macro_f1": 0.0, "n": 0, "correct": 0, "per_class": {}}

    def split(lst):
        return ([x for x, e in zip(lst, is_exp) if e],
                [x for x, e in zip(lst, is_exp) if not e])

    exp_b2, _         = split(preds_b2)
    exp_b1, _         = split(preds_b1)
    exp_b3, imp_b3    = split(preds_b3)
    _, imp_b1imp      = split(preds_b1imp)
    exp_cbo, imp_cbo  = split(preds_cbo)

    return {
        "B2":    {"all": compute_metrics(list(zip(exp_b2, exp_gt))),
                  "explicit": compute_metrics(list(zip(exp_b2, exp_gt))),
                  "implicit": {**_empty, "note": "N/A"}},
        "B1":    {"all": compute_metrics(list(zip(exp_b1, exp_gt))),
                  "explicit": compute_metrics(list(zip(exp_b1, exp_gt))),
                  "implicit": {**_empty, "note": "N/A"}},
        "B3":    {"all": compute_metrics(list(zip(preds_b3, gt_cats))),
                  "explicit": compute_metrics(list(zip(exp_b3, exp_gt))),
                  "implicit": compute_metrics(list(zip(imp_b3, imp_gt)))},
        "B1-imp":{"all": compute_metrics(list(zip(imp_b1imp, imp_gt))),
                  "explicit": {**_empty, "note": "N/A"},
                  "implicit": compute_metrics(list(zip(imp_b1imp, imp_gt)))},
        "Combo": {"all": compute_metrics(list(zip(preds_cbo, gt_cats))),
                  "explicit": compute_metrics(list(zip(exp_cbo, exp_gt))),
                  "implicit": compute_metrics(list(zip(imp_cbo, imp_gt)))},
    }


# ── Threshold sweep ────────────────────────────────────────────────

def run_threshold_sweep(tuples: list[dict], cache: dict):
    print(f"\n{_SEP}")
    print("# Threshold Sweep — Combo macro F1 (all / explicit / implicit)")
    print(_SEP)
    is_imp = [t["a_term"] == "implicit" for t in tuples]
    gt_all = [t["category"] for t in tuples]
    b2_preds = []; b1_raws = []; b3_raws = []; b1imp_raws = []
    for t in tuples:
        key  = f"{t['row_id']}|{t['a_term']}"
        c    = cache.get(key, {})
        b2_preds.append(c.get("b2"))
        b1_raws.append(c.get("b1_raw"))
        b3_raws.append(c.get("b3_raw"))
        b1imp_raws.append(c.get("b1_imp_raw"))
    best_f1  = -1
    best_cfg = (OTHERS_THRESHOLD_EXP, OTHERS_THRESHOLD_IMP)
    print(f"\n{'exp\\imp':>8}", end="")
    for it in SWEEP_THRESHOLDS:
        print(f"  {it:.2f}", end="")
    print()
    print("-" * (8 + 6 * len(SWEEP_THRESHOLDS)))
    for et in SWEEP_THRESHOLDS:
        print(f"   {et:.2f} ", end="")
        for it in SWEEP_THRESHOLDS:
            b1_p    = [ACD.apply_threshold_from_raw(r, et) if r else None for r in b1_raws]
            b3_p    = [ACD.apply_threshold_from_raw(r, et) for r in b3_raws]
            b1imp_p = [ACD.apply_threshold_from_raw(r, it) if r else "OTHERS" for r in b1imp_raws]
            combo = []
            for i, t in enumerate(tuples):
                if is_imp[i]:
                    combo.append(b1imp_p[i])
                else:
                    b2 = b2_preds[i]
                    b1 = b1_p[i]
                    b3 = b3_p[i]
                    if b2 is not None and b2 != "OTHERS":
                        combo.append(b2)
                    elif b1 is not None:
                        combo.append(b1)
                    else:
                        combo.append(b3)
            m  = compute_metrics(list(zip(combo, gt_all)))
            f1 = m["macro_f1"]
            marker = "*" if f1 > best_f1 else " "
            if f1 > best_f1:
                best_f1  = f1
                best_cfg = (et, it)
            print(f" {f1:.4f}{marker}", end="")
        print()
    print(f"\n  * = new best;  Best config: exp_thr={best_cfg[0]:.2f}, imp_thr={best_cfg[1]:.2f}"
          f"  →  Combo F1={best_f1:.4f}")


# ── Reporting ─────────────────────────────────────────────────────

def print_summary(res: dict, n_exp: int, n_imp: int):
    print(f"\n{_SEP}")
    print("# Stage 2 Eval — Aspect Category Mapping")
    print(_SEP)
    print(f"\nNote: B2/B1 explicit only (n={n_exp});  B1-imp implicit only (n={n_imp});  "
          f"B3/Combo all (n={n_exp + n_imp})\n")
    hdr = (f"{'Method':<8} {'Acc(all)':>9} {'F1(all)':>8}  "
           f"{'Acc(exp)':>9} {'F1(exp)':>8}  "
           f"{'Acc(imp)':>9} {'F1(imp)':>8}")
    print(hdr)
    print("-" * len(hdr))
    for method in ("B2", "B1", "B3", "B1-imp", "Combo"):
        r = res[method]
        a = r["all"]
        e = r["explicit"]
        i = r["implicit"]
        def fmt(d, key):
            return f"{d[key]:.4f}" if d.get("n", 0) > 0 else "   N/A"
        print(f"{method:<8} {fmt(a,'accuracy'):>9} {fmt(a,'macro_f1'):>8}  "
              f"{fmt(e,'accuracy'):>9} {fmt(e,'macro_f1'):>8}  "
              f"{fmt(i,'accuracy'):>9} {fmt(i,'macro_f1'):>8}")


def print_per_class(res: dict):
    for method in ("B2", "B1", "B3", "B1-imp", "Combo"):
        pc = res[method]["all"].get("per_class", {})
        if not pc:
            continue
        print(f"\n{_SEP}")
        print(f"## {method} — Per-Class\n")
        print(f"  {'Category':<14} {'P':>7} {'R':>7} {'F1':>7}  TP/FP/FN")
        for cat, m in pc.items():
            print(f"  {cat:<14} {m['P']:>7.4f} {m['R']:>7.4f} {m['F1']:>7.4f}"
                  f"  {m['TP']}/{m['FP']}/{m['FN']}")


def print_error_detail(tuples: list[dict], preds: dict):
    print(f"\n{_SEP}")
    print("## Combo — Error Cases (GT ≠ prediction)\n")
    errors = [
        (t, preds["b2"][i], preds["b1"][i], preds["b3"][i],
         preds["b1_imp"][i], preds["combo"][i])
        for i, t in enumerate(tuples)
        if preds["combo"][i] != t["category"]
    ]
    for t, b2, b1, b3, b1_imp, combo in errors:
        kind = "IMP" if t["a_term"] == "implicit" else "EXP"
        print(f"  [{kind} row {t['row_id']}] term={t['a_term']!r:<20}"
              f"  GT={t['category']:<14}  combo={combo:<14}")
        if kind == "IMP":
            print(f"    B1-imp={b1_imp:<14}  B3={str(b3):<14}  op={t['o_term']!r}")
        else:
            print(f"    B2={str(b2):<14}  B1={str(b1):<14}  B3={str(b3):<14}")
        print(f"    text: {t['text'][:70]!r}")
    print(f"\n  Total errors: {len(errors)}")


# ── Main ──────────────────────────────────────────────────────────

def _run(args):
    tuples = load_gt()
    if args.no_others:
        before = len(tuples)
        tuples = [t for t in tuples if t["category"] != "OTHERS"]
        print(f"--no-others: dropped {before - len(tuples)} OTHERS tuples")
    n_exp = sum(1 for t in tuples if t["a_term"] != "implicit")
    n_imp = sum(1 for t in tuples if t["a_term"] == "implicit")
    print(f"GT: {len(tuples)} tuples ({n_exp} explicit, {n_imp} implicit)")
    print(f"OTHERS_THR_EXP={OTHERS_THRESHOLD_EXP}  OTHERS_THR_IMP={OTHERS_THRESHOLD_IMP}")

    acd = ACD()
    if args.from_cache:
        with open(CACHE_PATH, encoding="utf-8") as f:
            cache = json.load(f)
        print(f"Cache loaded: {len(cache)} entries from {CACHE_PATH.name}")
    else:
        acd.load_sbert()
        cache = {}

    all_b2     = []
    all_b1     = []
    all_b3     = []
    all_b1_imp = []
    raw_cache_out: dict = {}

    for t in tuples:
        key = f"{t['row_id']}|{t['a_term']}"
        c   = cache.get(key, {})

        if args.from_cache and c and "b1_raw" in c:
            b2         = c.get("b2")
            b1_raw     = c.get("b1_raw")
            b3_raw     = c.get("b3_raw")
            b1_imp_raw = c.get("b1_imp_raw")
        else:
            b2 = acd.predict_b2(t["a_term"])
            _, b1_raw     = acd.predict_b1(t["text"], t["a_term"])
            _, b3_raw     = acd.predict_b3(t["text"])
            _, b1_imp_raw = (
                acd.predict_b1_implicit(t["text"], t["o_term"])
                if t["a_term"] == "implicit"
                else (None, None)
            )
            raw_cache_out[key] = {
                "b2": b2, "b1_raw": b1_raw,
                "b3_raw": b3_raw, "b1_imp_raw": b1_imp_raw,
            }

        b1     = ACD.apply_threshold_from_raw(b1_raw,     OTHERS_THRESHOLD_EXP) if b1_raw     else None
        b3     = ACD.apply_threshold_from_raw(b3_raw,     OTHERS_THRESHOLD_EXP)
        b1_imp = ACD.apply_threshold_from_raw(b1_imp_raw, OTHERS_THRESHOLD_IMP) if b1_imp_raw else "OTHERS"

        all_b2.append(b2)
        all_b1.append(b1)
        all_b3.append(b3)
        all_b1_imp.append(b1_imp)

    if args.save_cache:
        CACHE_DIR.mkdir(exist_ok=True)
        merged = {**cache, **raw_cache_out}
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"Cache saved -> {CACHE_PATH.name}")

    fill_b2   = [p if p is not None else "OTHERS" for p in all_b2]
    fill_b1   = [p if p is not None else "OTHERS" for p in all_b1]
    all_combo = build_combo(tuples, all_b2, all_b1, all_b3, all_b1_imp)

    method_results = build_method_results(
        tuples, fill_b2, fill_b1, all_b3, all_b1_imp, all_combo,
    )
    print_summary(method_results, n_exp, n_imp)
    print_per_class(method_results)
    if args.verbose:
        print_error_detail(
            tuples,
            {"b2": all_b2, "b1": all_b1, "b3": all_b3, "b1_imp": all_b1_imp, "combo": all_combo},
        )
    if args.sweep:
        loaded_cache = cache if args.from_cache else {**cache, **raw_cache_out}
        run_threshold_sweep(tuples, loaded_cache)

    print(f"\n{_SEP}")
    print("## Summary\n")
    for m, r in method_results.items():
        if r["all"].get("n", 0) > 0:
            print(f"  {m}: F1={r['all']['macro_f1']:.4f}  Acc={r['all']['accuracy']:.4f}  "
                  f"(n={r['all']['n']}, correct={r['all']['correct']})")

    if args.save_metrics:
        from shared.metrics_store import update_iterations, update_stage2_uiux, update_pipeline_standalone
        # B2 (原始 seed, f1=0.5913) and B2+ standalone (0.6577) are historical baselines —
        # current code uses B2+ seeds, so they cannot be regenerated and must stay as-is.
        # Combo iterations entry stores Explicit F1 (not All F1) for chart consistency.
        updates = {
            "B1":    {"f1": method_results["B1"]["explicit"]["macro_f1"]},
            "Combo": {"f1": method_results["Combo"]["explicit"]["macro_f1"]},
        }
        update_iterations("stage2", updates)

        # uiux_recall["B2"] (0.4348, original seeds) is historical — do not overwrite.
        # uiux_recall["B2+"] reflects current B2 code (which uses B2+ seeds).
        uiux_updates = {}
        b2_pc = method_results["B2"]["all"].get("per_class", {})
        if "UI_UX" in b2_pc:
            uiux_updates["B2+"] = b2_pc["UI_UX"]["R"]
        if uiux_updates:
            update_stage2_uiux(uiux_updates)

        best_f1 = method_results["Combo"]["all"]["macro_f1"]
        update_pipeline_standalone("stage2", best_f1)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Stage 2 Category Mapping Eval")
    parser.add_argument("--from-cache",    action="store_true")
    parser.add_argument("--save-cache",    action="store_true")
    parser.add_argument("--sweep",         action="store_true")
    parser.add_argument("--no-others",     action="store_true")
    parser.add_argument("--save-metrics",  action="store_true",
                        help="Update reports/stage_metrics.json with this run's results")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    HISTORY_DIR.mkdir(exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = HISTORY_DIR / f"{timestamp}_B1B2B3Combo.md"

    with open(report_path, "w", encoding="utf-8") as f:
        sys.stdout = Tee(f)
        try:
            print(f"# Stage 2 Eval  {timestamp}\n")
            _run(args)
        finally:
            sys.stdout = sys.stdout.stdout

    print(f"\nReport saved -> {report_path}")


if __name__ == "__main__":
    main()
