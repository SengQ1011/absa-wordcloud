"""
Shared evaluation metrics for ABSA stage evals.

Provides:
  term_match(pred, gt, level)           → bool
  greedy_match(pred_terms, gt_terms, level) → tuple
  pair_match(pred_pair, gt_pair, level) → bool
  greedy_pair_match(pred_pairs, gt_pairs, level) → tuple
  prf(tp, fp, fn)                       → dict
"""


def term_match(pred: str, gt: str, level: str = "partial") -> bool:
    p, g = pred.strip(), gt.strip()
    if p == g:
        return True
    if level == "exact":
        return False
    if p in g or g in p:
        return True
    shorter = min(len(p), len(g))
    overlap = len(set(p) & set(g))
    return shorter > 0 and (overlap / shorter) >= 0.7


def greedy_match(pred_terms: list, gt_terms: list, level: str) -> tuple:
    """
    Returns (tp, fp, fn, matched_gt_indices, missed_gt_indices, fp_pred_indices).
    """
    used: set = set()
    tp = fn   = 0
    ok_gi:   list = []
    miss_gi: list = []
    for gi, gt_t in enumerate(gt_terms):
        found = False
        for pi, pred_t in enumerate(pred_terms):
            if pi not in used and term_match(pred_t, gt_t, level):
                used.add(pi)
                found = True
                ok_gi.append(gi)
                break
        if found:
            tp += 1
        else:
            miss_gi.append(gi)
            fn += 1
    fp_pis = [pi for pi in range(len(pred_terms)) if pi not in used]
    return tp, len(fp_pis), fn, ok_gi, miss_gi, fp_pis


def pair_match(pred: tuple, gt: tuple, level: str) -> bool:
    return term_match(pred[0], gt[0], level) and term_match(pred[1], gt[1], level)


def greedy_pair_match(pred_pairs: list, gt_pairs: list, level: str) -> tuple:
    """
    Returns (tp, fp, fn, matched_gt_indices, missed_gt_indices, fp_pred_indices).
    Pairs are (aspect_term, opinion_term) tuples.
    """
    used: set = set()
    tp = fn   = 0
    ok_gi:   list = []
    miss_gi: list = []
    for gi, gt_p in enumerate(gt_pairs):
        found = False
        for pi, pred_p in enumerate(pred_pairs):
            if pi not in used and pair_match(pred_p, gt_p, level):
                used.add(pi)
                found = True
                ok_gi.append(gi)
                break
        if found:
            tp += 1
        else:
            miss_gi.append(gi)
            fn += 1
    fp_pis = [pi for pi in range(len(pred_pairs)) if pi not in used]
    return tp, len(fp_pis), fn, ok_gi, miss_gi, fp_pis


def prf(tp: int, fp: int, fn: int) -> dict:
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"P": round(p, 4), "R": round(r, 4), "F1": round(f1, 4),
            "TP": tp, "FP": fp, "FN": fn}
