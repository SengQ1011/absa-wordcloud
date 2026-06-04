"""
Stage 3 — Opinion Term Extraction  橫向比較
C1: CKIP VH/VJ/A baseline（+ VHC whitelist + custom words）
C2: C1 + attach_negation（否定前綴合併）+ PC rule
C3: C2 + stanza amod（捕捉 is_modifier 過濾掉的形容詞修飾語）

執行：
  python eval.py --save-cache    # 首次跑（CKIP 沿用 stage1_cache，stanza 新算）
  python eval.py --from-cache    # 讀取 stage3_cache.json（略過 stanza）
  python eval.py -v              # per-row 錯誤細節
  python eval.py --save-metrics  # 更新 reports/stage_metrics.json
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
from shared.utils import Tee, init_stanza
from shared.metrics import greedy_match, prf
from stage3_opinion_term_extraction.OTE import OTE

HISTORY_DIR  = Path(__file__).parent / "history"
CACHE_PATH   = CACHE_DIR / "stage3_cache.json"
STAGE1_CACHE = CACHE_DIR / "stage1_cache.json"

_SEP = "=" * 64


# ── Data loading ──────────────────────────────────────────────────

def load_gt_opinions() -> dict:
    seen_per_row: dict = defaultdict(set)
    result: dict       = defaultdict(list)
    with open(GT_PATH, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rid  = r["row_id"].strip()
            term = r["opinion_term"].strip()
            if term != "implicit" and term not in seen_per_row[rid]:
                seen_per_row[rid].add(term)
                result[rid].append(term)
    return dict(result)


# ── Evaluation ────────────────────────────────────────────────────

def eval_opinions(gt_by_row: dict, pred_by_row: dict) -> dict:
    counters = {
        "exact":   {"tp": 0, "fp": 0, "fn": 0},
        "partial": {"tp": 0, "fp": 0, "fn": 0},
    }
    per_row: dict = {}
    for row_id, gt_terms in gt_by_row.items():
        pred_terms = pred_by_row.get(row_id, [])
        row_data: dict = {"gt": gt_terms, "pred": pred_terms}
        for level, c in counters.items():
            tp, fp, fn, ok_gi, miss_gi, fp_pi = greedy_match(pred_terms, gt_terms, level)
            c["tp"] += tp
            c["fp"] += fp
            c["fn"] += fn
            if level == "partial":
                row_data["ok"]   = [gt_terms[i]   for i in ok_gi]
                row_data["miss"] = [gt_terms[i]   for i in miss_gi]
                row_data["fp"]   = [pred_terms[i] for i in fp_pi]
        per_row[row_id] = row_data
    return {"exact": prf(**counters["exact"]), "partial": prf(**counters["partial"]),
            "per_row": per_row}


# ── Reporting ─────────────────────────────────────────────────────

def print_report(results: dict, gt_count: int):
    print(f"\n{_SEP}")
    print("# Stage 3 Eval — Opinion Term Extraction")
    print(_SEP)
    print(f"\nGT explicit opinions: {gt_count} terms across all rows\n")
    header = (f"{'Method':<6} {'Exact P':>8} {'Exact R':>8} {'Exact F1':>9}  "
              f"{'Partial P':>10} {'Partial R':>10} {'Partial F1':>11}   {'TP/FP/FN (partial)'}")
    print(header)
    print("-" * len(header))
    for method in ("C1", "C2", "C3"):
        if method not in results:
            continue
        e = results[method]["exact"]
        p = results[method]["partial"]
        print(
            f"{method:<6} "
            f"{e['P']:>8.4f} {e['R']:>8.4f} {e['F1']:>9.4f}  "
            f"{p['P']:>10.4f} {p['R']:>10.4f} {p['F1']:>11.4f}   "
            f"{p['TP']}/{p['FP']}/{p['FN']}"
        )


def print_per_row(results: dict, gt_by_row: dict):
    print(f"\n{_SEP}")
    print("## Per-Row Detail (partial, rows with FN or FP)\n")
    for row_id in sorted(gt_by_row.keys(), key=int):
        rows_data = {m: results[m]["per_row"].get(row_id, {}) for m in results}
        if not any(d.get("miss") or d.get("fp") for d in rows_data.values()):
            continue
        gt_terms = gt_by_row[row_id]
        print(f"[row {row_id}]  GT: {gt_terms}")
        for method, d in sorted(rows_data.items()):
            print(f"  {method}: pred={d.get('pred', [])}")
            print(f"       ok={d.get('ok', [])}  miss={d.get('miss', [])}  FP={d.get('fp', [])}")
        print()


# ── Main ──────────────────────────────────────────────────────────

def _run(args):
    gt_by_row = load_gt_opinions()
    gt_count  = sum(len(v) for v in gt_by_row.values())
    print(f"GT: {gt_count} explicit opinion terms, {len(gt_by_row)} rows")

    with open(STAGE1_CACHE, encoding="utf-8") as f:
        ckip_data = json.load(f)
    print(f"CKIP cache: {len(ckip_data)} rows")

    stanza_cache: dict = {}
    nlp = None
    if args.from_cache:
        with open(CACHE_PATH, encoding="utf-8") as f:
            stanza_cache = json.load(f)
        print(f"Stage3 cache loaded: {len(stanza_cache)} rows")
    else:
        nlp = init_stanza()

    ote = OTE(stanza_nlp=nlp)

    c1_by_row: dict = {}
    c2_by_row: dict = {}
    c3_by_row: dict = {}

    for row_id, data in ckip_data.items():
        if row_id not in gt_by_row:
            continue
        tok, pos = data["tokens"], data["pos_tags"]
        c1_by_row[row_id] = ote.extract_c1(tok, pos)
        c2_by_row[row_id] = ote.extract_c2(tok, pos)

        if args.from_cache and row_id in stanza_cache:
            c3_by_row[row_id] = stanza_cache[row_id]["c3"]
        elif nlp is not None:
            try:
                doc  = nlp([tok])
                sent = doc.sentences[0]
                c3_by_row[row_id] = ote.extract_c3(tok, pos, sent)
                stanza_cache[row_id] = {"c3": c3_by_row[row_id]}
            except Exception:
                c3_by_row[row_id] = list(c2_by_row[row_id])
        else:
            c3_by_row[row_id] = list(c2_by_row[row_id])

    if args.save_cache:
        CACHE_DIR.mkdir(exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(stanza_cache, f, ensure_ascii=False, indent=2)
        print(f"Cache saved -> {CACHE_PATH.name}")

    results = {
        "C1": eval_opinions(gt_by_row, c1_by_row),
        "C2": eval_opinions(gt_by_row, c2_by_row),
        "C3": eval_opinions(gt_by_row, c3_by_row),
    }

    print_report(results, gt_count)
    if args.verbose:
        print_per_row(results, gt_by_row)

    print(f"\n{_SEP}")
    print("## Extraction Stats\n")
    for m, d in [("C1", c1_by_row), ("C2", c2_by_row), ("C3", c3_by_row)]:
        print(f"  {m} total: {sum(len(v) for v in d.values())}")

    best = max(results, key=lambda m: results[m]["partial"]["F1"])
    print(f"\n  Best (partial F1): {best} = {results[best]['partial']['F1']:.4f}")

    if args.save_metrics:
        from shared.metrics_store import update_iterations, update_pipeline_standalone
        # C1→"C1", C2→"C2+" (refactored C2 = C2+ best method with CUSTOM/NEUTRAL)
        # C3 (amod regression) is an internal experiment, not part of the iteration narrative.
        # JSON "C2" (original negation+PC, 0.8258) is a historical value; eval no longer has a
        # separate code path for it, so it is left unchanged.
        id_map = {"C1": "C1", "C2": "C2+"}
        updates = {}
        for eval_id, json_id in id_map.items():
            if eval_id not in results:
                continue
            p = results[eval_id]["partial"]
            updates[json_id] = {
                "f1": round(p["F1"], 4),
                "tp": p["TP"], "fp": p["FP"], "fn": p["FN"],
            }
        update_iterations("stage3", updates)
        best_f1 = results[best]["partial"]["F1"]
        update_pipeline_standalone("stage3", best_f1)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Stage 3 Opinion Extraction Eval")
    parser.add_argument("--from-cache",    action="store_true")
    parser.add_argument("--save-cache",    action="store_true")
    parser.add_argument("--save-metrics",  action="store_true",
                        help="Update reports/stage_metrics.json with this run's results")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    HISTORY_DIR.mkdir(exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = HISTORY_DIR / f"{timestamp}_C1C2C3.md"

    with open(report_path, "w", encoding="utf-8") as f:
        sys.stdout = Tee(f)
        try:
            print(f"# Stage 3 Eval  {timestamp}\n")
            _run(args)
        finally:
            sys.stdout = sys.stdout.stdout

    print(f"\nReport saved -> {report_path}")


if __name__ == "__main__":
    main()
