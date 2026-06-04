"""
metrics_store.py
Read-modify-write helper for reports/stage_metrics.json.
Each eval script calls the appropriate function with --save-metrics.
"""

import json
from pathlib import Path

_METRICS_PATH = Path(__file__).parent.parent / "reports" / "stage_metrics.json"

_NUMERIC = {"f1", "tp", "fp", "fn", "precision", "recall", "accuracy", "macro_f1"}

_PIPELINE_INDEX = {
    "stage1": 0,
    "stage2": 1,
    "stage3": 2,
    "stage4": 3,
    "stage5": 4,
}


def _load() -> dict:
    with open(_METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _dump(data: dict) -> None:
    _METRICS_PATH.parent.mkdir(exist_ok=True)
    with open(_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def update_iterations(stage: str, updates: dict) -> None:
    """
    Merge numeric metric fields into stage iterations.

    stage   : "stage1" | "stage2" | "stage3" | "stage4"
    updates : { json_iteration_id: { "f1": ..., "tp": ..., "fp": ..., "fn": ... } }

    label / status fields are never touched — only numeric keys in _NUMERIC.
    """
    data    = _load()
    changed = 0
    for it in data[stage]["iterations"]:
        patch = updates.get(it["id"])
        if patch is None:
            continue
        for k, v in patch.items():
            if k in _NUMERIC:
                it[k] = v
                changed += 1
    if changed:
        _dump(data)
        print(f"  [metrics_store] {stage}: {changed} fields updated → stage_metrics.json")
    else:
        print(f"  [metrics_store] {stage}: no matching iteration IDs found, nothing saved")


def update_stage2_uiux(updates: dict) -> None:
    """
    updates: { json_iteration_id: recall_float }
    e.g. {"B2": 0.4348, "B2+": 0.8696}
    """
    data    = _load()
    changed = 0
    for entry in data["stage2"]["uiux_recall"]:
        if entry["id"] in updates:
            entry["recall"] = round(updates[entry["id"]], 4)
            changed += 1
    if changed:
        _dump(data)
        print(f"  [metrics_store] stage2 uiux_recall updated → stage_metrics.json")


def update_pipeline_standalone(stage: str, f1: float) -> None:
    """Update a stage's standalone F1 value in pipeline.stage_standalone."""
    idx  = _PIPELINE_INDEX.get(stage)
    if idx is None:
        return
    data = _load()
    data["pipeline"]["stage_standalone"][idx]["f1"] = round(f1, 4)
    _dump(data)
    print(f"  [metrics_store] pipeline.stage_standalone[{idx}] ({stage}) f1={f1:.4f}")


def update_e2e(quad_f1: float) -> None:
    """Update Track A end-to-end Quad Partial F1 in pipeline and track_comparison."""
    data = _load()
    data["pipeline"]["e2e_quad_f1"] = round(quad_f1, 4)
    data["track_comparison"]["track_a"]["quad_partial_f1"] = round(quad_f1, 4)
    _dump(data)
    print(f"  [metrics_store] pipeline.e2e_quad_f1={quad_f1:.4f} → stage_metrics.json")


def update_track_comparison(track: str, result: dict) -> None:
    """
    Update a full track's entry in track_comparison from eval_track.py output.

    track  : "track_a" or "track_b"
    result : the dict returned by evaluate(), containing:
               metrics      → { aspect_partial, opinion_partial, pair_partial, quad_partial, ... }
               cat_cls      → { CATEGORY: {F1, ...}, ... }
               errors       → { both_missed, op_found_asp_missed, asp_found_op_missed, ... }
    """
    data     = _load()
    m        = result["metrics"]
    cat_cls  = result["cat_cls"]
    errors   = result["errors"]
    tc       = data["track_comparison"][track]

    tc["aspect_partial_f1"]  = m["aspect_partial"]["F1"]
    tc["opinion_partial_f1"] = m["opinion_partial"]["F1"]
    tc["pair_partial_f1"]    = m["pair_partial"]["F1"]
    tc["quad_partial_f1"]    = m["quad_partial"]["F1"]

    tc["fn_breakdown"] = {
        "both_missed":         errors["both_missed"],
        "op_found_asp_missed": errors["op_found_asp_missed"],
        "asp_found_op_missed": errors["asp_found_op_missed"],
    }

    for cat, v in cat_cls.items():
        if cat in tc.get("category_f1", {}):
            tc["category_f1"][cat] = v["F1"]

    if track == "track_a":
        data["pipeline"]["e2e_quad_f1"] = m["quad_partial"]["F1"]

    _dump(data)
    print(f"  [metrics_store] track_comparison.{track} updated → stage_metrics.json")
