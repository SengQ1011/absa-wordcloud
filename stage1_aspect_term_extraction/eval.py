"""
Stage 1 — Aspect Term Extraction  橫向比對
A1: CKIP Na/Nb/Nv 直接抽取（baseline）
A2: A1 + Anchor cosine 過濾（sentence_transformers, threshold=0.25）
A3: A1 + 相鄰 Na/Nb/Nv/(VC/VE/VD) 複合名詞合併

執行：
  python eval.py                      # full run
  python eval.py --from-cache         # skip CKIP re-tokenize，讀取 stage1_cache.json
  python eval.py --save-cache         # 儲存 CKIP tokenize 結果
  python eval.py --skip-a2            # 略過 A2 方法
  python eval.py --threshold 0.3      # A2 cosine threshold（預設 0.25）
  python eval.py -v                   # per-row detail
  python eval.py --save-metrics       # 更新 reports/stage_metrics.json
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.constants import GT_PATH, DATA_PATH, CACHE_DIR
from shared.utils import Tee, init_ckip, ckip_tokenize, load_texts
from shared.metrics import greedy_match, prf
from stage1_aspect_term_extraction.ATE import ATE

HISTORY_DIR = Path(__file__).parent / "history"
CACHE_PATH  = CACHE_DIR / "stage1_cache.json"

_SEP = "=" * 64


# ── GT loading ────────────────────────────────────────────────────

def load_gt_aspects() -> dict:
    all_rows: set = set()
    explicit: dict = defaultdict(list)
    with open(GT_PATH, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            row_id = row["row_id"]
            all_rows.add(row_id)
            term = row["aspect_term"].strip()
            if term != "implicit":
                explicit[row_id].append(term)
    return {rid: explicit.get(rid, []) for rid in sorted(all_rows, key=int)}


# ── Evaluation ────────────────────────────────────────────────────

def eval_aspects(gt_by_row: dict, pred_by_row: dict) -> dict:
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

def print_report(results: dict, gt_count: int, a2_threshold: float):
    print(f"\n{_SEP}")
    print("# Stage 1 Eval — Aspect Term Extraction")
    print(_SEP)
    print(f"\nGT explicit aspects: {gt_count} terms across all rows\n")
    header = f"{'Method':<6} {'Exact P':>8} {'Exact R':>8} {'Exact F1':>9}  "
    header += f"{'Partial P':>10} {'Partial R':>10} {'Partial F1':>11}   {'TP/FP/FN (partial)'}"
    print(header)
    print("-" * len(header))
    for method, res in sorted(results.items()):
        e = res["exact"]
        p = res["partial"]
        tag = f"(thr={a2_threshold})" if method == "A2" else ""
        print(
            f"{method:<6} "
            f"{e['P']:>8.4f} {e['R']:>8.4f} {e['F1']:>9.4f}  "
            f"{p['P']:>10.4f} {p['R']:>10.4f} {p['F1']:>11.4f}   "
            f"{p['TP']}/{p['FP']}/{p['FN']}  {tag}"
        )


def print_per_row(results: dict, gt_by_row: dict):
    print(f"\n{_SEP}")
    print("## Per-Row Detail (partial match, rows with FN or FP)\n")
    for row_id in sorted(gt_by_row.keys(), key=int):
        rows_data = {m: results[m]["per_row"].get(row_id, {}) for m in results}
        if not any(d.get("miss") or d.get("fp") for d in rows_data.values()):
            continue
        gt_terms = gt_by_row[row_id]
        print(f"[row {row_id}]  GT explicit: {gt_terms}")
        for method, d in sorted(rows_data.items()):
            print(f"  {method}: pred={d.get('pred', [])}")
            print(f"       ok={d.get('ok', [])}  miss={d.get('miss', [])}  FP={d.get('fp', [])}")
        print()


# ── Main ──────────────────────────────────────────────────────────

def _run(args):
    gt_by_row = load_gt_aspects()
    gt_count  = sum(len(v) for v in gt_by_row.values())
    print(f"GT: {gt_count} explicit aspect terms, {len(gt_by_row)} rows")

    if args.from_cache:
        with open(CACHE_PATH, encoding="utf-8") as f:
            ckip_data = json.load(f)
        print(f"Cache: {len(ckip_data)} rows loaded")
    else:
        rows  = load_texts(DATA_PATH)
        ws, pos_model = init_ckip()
        texts = [r["text"] for r in rows]
        print(f"CKIP tokenize ({len(texts)} rows)...")
        ws_r, pos_r = ckip_tokenize(texts, ws, pos_model)
        ckip_data = {
            r["row_id"]: {"tokens": ws_r[i], "pos_tags": pos_r[i]}
            for i, r in enumerate(rows)
        }
        if args.save_cache:
            CACHE_DIR.mkdir(exist_ok=True)
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(ckip_data, f, ensure_ascii=False, indent=2)
            print(f"Cache saved -> {CACHE_PATH.name}")

    # Load sentence_transformers for A2
    sbert = None
    if not args.skip_a2:
        from sentence_transformers import SentenceTransformer
        print("Load sentence_transformers (paraphrase-multilingual-MiniLM-L12-v2)...")
        sbert = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        print(f"A2 threshold={args.threshold}")

    ate = ATE(sbert_model=sbert)

    # Extraction
    a1_by_row: dict = {}
    a2_by_row: dict = {}
    a3_by_row: dict = {}
    a4_by_row: dict = {}
    for row_id, data in ckip_data.items():
        tok, pos = data["tokens"], data["pos_tags"]
        a1_by_row[row_id] = ate.extract_a1(tok, pos)
        a3_by_row[row_id] = ate.extract_a3(tok, pos)
        a4_by_row[row_id] = ate.extract_a4(tok, pos)
        if sbert:
            a2_by_row[row_id] = ate.extract_a2(tok, pos, args.threshold)

    results: dict = {
        "A1": eval_aspects(gt_by_row, a1_by_row),
        "A3": eval_aspects(gt_by_row, a3_by_row),
        "A4": eval_aspects(gt_by_row, a4_by_row),
    }
    if sbert:
        results["A2"] = eval_aspects(gt_by_row, a2_by_row)

    print_report(results, gt_count, args.threshold)
    if args.verbose:
        print_per_row(results, gt_by_row)

    print(f"\n{_SEP}")
    print("## Extraction Stats\n")
    print(f"  A1 total: {sum(len(v) for v in a1_by_row.values())}")
    print(f"  A3 total: {sum(len(v) for v in a3_by_row.values())}")
    print(f"  A4 total: {sum(len(v) for v in a4_by_row.values())}")
    if sbert:
        print(f"  A2 total: {sum(len(v) for v in a2_by_row.values())} (threshold={args.threshold})")

    best = max(results, key=lambda m: results[m]["partial"]["F1"])
    print(f"\n  Best (partial F1): {best} = {results[best]['partial']['F1']:.4f}")

    if args.save_metrics:
        from shared.metrics_store import update_iterations, update_pipeline_standalone
        updates = {}
        for method_id in ("A1", "A2", "A3", "A4"):
            if method_id not in results:
                continue
            p = results[method_id]["partial"]
            updates[method_id] = {
                "f1": round(p["F1"], 4),
                "tp": p["TP"], "fp": p["FP"], "fn": p["FN"],
            }
        update_iterations("stage1", updates)
        best_f1 = results[best]["partial"]["F1"]
        update_pipeline_standalone("stage1", best_f1)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Stage 1 Aspect Extraction Eval")
    parser.add_argument("--from-cache",    action="store_true")
    parser.add_argument("--save-cache",    action="store_true")
    parser.add_argument("--skip-a2",       action="store_true")
    parser.add_argument("--threshold",     type=float, default=0.25)
    parser.add_argument("--save-metrics",  action="store_true",
                        help="Update reports/stage_metrics.json with this run's results")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    HISTORY_DIR.mkdir(exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = HISTORY_DIR / f"{timestamp}_A1A2A3.md"

    with open(report_path, "w", encoding="utf-8") as f:
        sys.stdout = Tee(f)
        try:
            print(f"# Stage 1 Eval  {timestamp}\n")
            _run(args)
        finally:
            sys.stdout = sys.stdout.stdout

    print(f"\nReport saved -> {report_path}")


if __name__ == "__main__":
    main()
