"""
generate_viewer_data.py
從 pipeline_output.csv（Track A）或 track_b_output.csv（Track B）
產生各 aspect_category 的 JSON 供 viewer.html 使用。

Track A 輸出：ui_ux_data.json / animation_data.json / ...
Track B 輸出：ui_ux_data_b.json / animation_data_b.json / ...

Usage:
  python generate_viewer_data.py           # Track A（預設）
  python generate_viewer_data.py --track b # Track B
"""

import argparse
import colorsys
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent

CATEGORIES_A = {
    "UI_UX":        {"label": "介面與體驗",  "filename": "ui_ux_data_a.json"},
    "ANIMATION":    {"label": "動畫演示",    "filename": "animation_data_a.json"},
    "CONTENT":      {"label": "教學內容",    "filename": "content_data_a.json"},
    "QUIZ":         {"label": "測驗系統",    "filename": "quiz_data_a.json"},
    "PERFORMANCE":  {"label": "系統效能",    "filename": "performance_data_a.json"},
    "GAMIFICATION": {"label": "遊戲化互動",  "filename": "gamification_data_a.json"},
}

CATEGORIES_B = {
    "UI_UX":        {"label": "介面與體驗",  "filename": "ui_ux_data_b.json"},
    "ANIMATION":    {"label": "動畫演示",    "filename": "animation_data_b.json"},
    "CONTENT":      {"label": "教學內容",    "filename": "content_data_b.json"},
    "QUIZ":         {"label": "測驗系統",    "filename": "quiz_data_b.json"},
    "PERFORMANCE":  {"label": "系統效能",    "filename": "performance_data_b.json"},
    "GAMIFICATION": {"label": "遊戲化互動",  "filename": "gamification_data_b.json"},
}


def pos_neg_color(pos: int, neg: int) -> str:
    total = pos + neg
    if total == 0:
        return "rgb(108,117,125)"
    ratio = pos / total          # 0.0=全 neg → red, 1.0=全 pos → green
    hue   = ratio * 0.33         # HSV hue: 0=red, 0.33=green
    r, g, b = colorsys.hsv_to_rgb(hue, 0.70, 0.72)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


_NEG_PREFIXES = ("沒有", "不", "沒", "未", "非")   # longer first

def find_span(text: str, term: str) -> dict | None:
    if not term or term == "implicit":
        return None
    # Exact match
    idx = text.find(term)
    if idx >= 0:
        return {"start": idx, "end": idx + len(term)}
    # Negation-prefix fallback:
    # pipeline may strip degree adverbs (e.g. "不太明確" → stored as "不明確")
    # Find the base term, then look back up to 5 chars for the negation prefix.
    for prefix in _NEG_PREFIXES:
        if not term.startswith(prefix):
            continue
        base = term[len(prefix):]
        if not base:
            continue
        bidx = text.find(base)
        if bidx < 0:
            continue
        look_from = max(0, bidx - 5)
        pidx = text.rfind(prefix, look_from, bidx)
        if pidx >= 0:
            return {"start": pidx, "end": bidx + len(base)}
        # prefix not nearby — highlight just the base term
        return {"start": bidx, "end": bidx + len(base)}
    return None


def build_review(text: str, asp: str, op: str) -> dict:
    asp_span = find_span(text, asp)
    op_span  = find_span(text, op)
    return {
        "text":          text,
        "aspect_spans":  [asp_span] if asp_span else [],
        "opinion_spans": [op_span]  if op_span  else [],
    }


def main():
    parser = argparse.ArgumentParser(description="Generate viewer JSON data")
    parser.add_argument("--track", choices=["a", "b"], default="a",
                        help="Track A (pipeline_output.csv) or B (track_b_output.csv)")
    args = parser.parse_args()

    if args.track == "b":
        csv_path   = ROOT / "output" / "track_b_output.csv"
        categories = CATEGORIES_B
        print("Track B mode -> reading output/track_b_output.csv")
    else:
        csv_path   = ROOT / "output" / "pipeline_output.csv"
        categories = CATEGORIES_A

    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    by_cat: dict[str, list] = defaultdict(list)
    for r in rows:
        by_cat[r["aspect_category"]].append(r)

    for cat_key, cat_info in categories.items():
        cat_rows = by_cat.get(cat_key, [])

        # key: (word, term_type) → {pos, neg, seen_pairs, reviews}
        # seen_pairs: set of (text, counterpart) to deduplicate reviews
        word_data: dict = defaultdict(lambda: {
            "pos": 0, "neg": 0, "seen_pairs": set(), "reviews": []
        })

        for r in cat_rows:
            text     = r["text"]
            polarity = r["polarity"]   # "pos" | "neg"
            asp      = r["aspect_term"]
            op       = r["opinion_term"]

            pol_key = "pos" if polarity == "pos" else "neg"

            # aspect term bubble (skip implicit)
            if asp and asp != "implicit":
                entry = word_data[(asp, "aspect")]
                entry[pol_key] += 1
                pair_key = (text, op)   # same text + different opinion = different review
                if pair_key not in entry["seen_pairs"]:
                    entry["seen_pairs"].add(pair_key)
                    entry["reviews"].append(build_review(text, asp, op))

            # opinion term bubble (skip implicit)
            if op and op != "implicit":
                entry = word_data[(op, "opinion")]
                entry[pol_key] += 1
                pair_key = (text, asp)  # same text + different aspect = different review
                if pair_key not in entry["seen_pairs"]:
                    entry["seen_pairs"].add(pair_key)
                    entry["reviews"].append(build_review(text, asp, op))

        # Build words list
        words = []
        for (word, term_type), data in word_data.items():
            pos   = data["pos"]
            neg   = data["neg"]
            count = pos + neg
            words.append({
                "word":           word,
                "font_size":      count,          # viewer scales this to bubble radius
                "color":          pos_neg_color(pos, neg),
                "term_type":      term_type,       # "aspect" | "opinion"
                "word_sentiment": {"pos": pos, "neg": neg},
                "reviews":        data["reviews"],
            })

        words.sort(key=lambda x: x["font_size"], reverse=True)

        output = {
            "title":  f"CodePulse — {cat_info['label']}",
            "aspect": cat_key,
            "label":  cat_info["label"],
            "words":  words,
        }

        out_path = ROOT / "viewer" / cat_info["filename"]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"OK {out_path.name}  ({len(words)} words, {len(cat_rows)} quadruplets)")


if __name__ == "__main__":
    main()
