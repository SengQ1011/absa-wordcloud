"""
step1_clean_data.py
把問卷正面欄和改進欄合併為 ABSA 用的 text，過濾無意義回覆。

規則：
  - 兩欄都是噪音 / 空白 → 整筆 skip
  - 只有正面欄有效          → text = pos
  - 只有改進欄有效          → text = imp
  - 兩欄都有效              → text = pos + "，" + imp

輸出：
  cleaned_texts.csv   ← row_id / text_pos / text_imp / text / status
"""

import sys
import re
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
_ROOT = Path(__file__).parent
XLSX  = str(_ROOT / "data" / "raw" / "CodePulse 資料結構與演算法視覺化教學平台 使用者回饋問卷.xlsx")
OUT   = str(_ROOT / "data" / "cleaned_texts.csv")

_NOISE_RE = re.compile(
    r"^\s*("
    r"無|沒有|目前沒有|還沒發現|沒特別|不特別|暫無|尚未|"
    r"不錯|好|可以|還好|none|n/a|na|"
    r"[🈚️\-\.\s]*"          # 空白、emoji 無字符、單純符號
    r")\s*[。，.]*\s*$",
    re.IGNORECASE,
)

def is_noise(text) -> bool:
    if not isinstance(text, str):
        return True
    t = text.strip()
    if not t:
        return True
    if len(t) <= 1:
        return True
    return bool(_NOISE_RE.match(t))


df = pd.read_excel(XLSX)
col_pos = df.columns[30]   # 最滿意的地方
col_imp = df.columns[31]   # 待改進的地方

print(f"讀入 {len(df)} 筆")
print(f"正面欄：{col_pos.strip()[:50]}")
print(f"改進欄：{col_imp.strip()[:50]}\n")

rows = []
for idx, row in df.iterrows():
    row_id   = idx + 1
    raw_pos  = str(row[col_pos]).strip()
    raw_imp  = str(row[col_imp]).strip()
    pos_ok   = not is_noise(raw_pos)
    imp_ok   = not is_noise(raw_imp)

    if not pos_ok and not imp_ok:
        status = "skip"
        text   = ""
    elif pos_ok and not imp_ok:
        status = "pos_only"
        text   = raw_pos
    elif not pos_ok and imp_ok:
        status = "imp_only"
        text   = raw_imp
    else:
        status = "both"
        text   = raw_pos + "，" + raw_imp

    rows.append({
        "row_id":   row_id,
        "text_pos": raw_pos if pos_ok else "",
        "text_imp": raw_imp if imp_ok else "",
        "text":     text,
        "status":   status,
    })

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT, index=False, encoding="utf-8-sig")

print("=" * 55)
total = len(out_df)
for s in ["both", "pos_only", "imp_only", "skip"]:
    n = (out_df["status"] == s).sum()
    print(f"  {s:<10}: {n:>2} 筆")
print(f"  {'合計':<10}: {total:>2} 筆")
print(f"\n有效（需標注）: {(out_df['status'] != 'skip').sum()} 筆")
print(f"跳過         : {(out_df['status'] == 'skip').sum()} 筆")
print(f"\n輸出：{OUT}")
print("=" * 55)

print("\n【預覽 — 前 10 筆有效資料】")
valid = out_df[out_df["status"] != "skip"].head(10)
for _, r in valid.iterrows():
    print(f"  row {r['row_id']:>2} [{r['status']:<8}] {r['text'][:70]}")
