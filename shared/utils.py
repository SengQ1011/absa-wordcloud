import csv
import json
import sys
from pathlib import Path

_TIMING_LOG = Path(__file__).parent.parent / "history" / "timing_log.json"


class Tee:
    """Dual-write stdout to both console and file."""
    def __init__(self, file):
        self.file   = file
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def isatty(self):
        return False


def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save_json(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved -> {path.name}")


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def init_ckip():
    from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger
    print("Load CKIP...")
    ws  = CkipWordSegmenter(model="bert-base", device=-1)
    pos = CkipPosTagger(model="bert-base", device=-1)
    return ws, pos


def ckip_tokenize(texts: list, ws, pos_model) -> tuple:
    ws_r  = ws(texts, show_progress=False)
    pos_r = pos_model(ws_r, show_progress=False)
    return ws_r, pos_r


def init_stanza():
    import stanza
    print("Load stanza zh-hant pipeline...")
    return stanza.Pipeline(
        "zh-hant",
        processors="tokenize,lemma,pos,depparse",
        tokenize_pretokenized=True,
        verbose=False,
    )


def append_timing_log(track: str, timings: list, total: float, timestamp: str):
    """Append one pipeline run's timing to history/timing_log.json."""
    entry = {
        "timestamp": timestamp,
        "track": track,
        "total_s": round(total, 2),
        "stages": {label: round(s, 2) for label, s in timings},
    }
    existing = []
    if _TIMING_LOG.exists():
        with open(_TIMING_LOG, encoding="utf-8") as f:
            existing = json.load(f)
    existing.append(entry)
    with open(_TIMING_LOG, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


def load_texts(data_path: Path) -> list[dict]:
    """Return list of {row_id, text} skipping 'skip' rows."""
    rows = []
    with open(data_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["status"] != "skip" and r["text"].strip():
                rows.append({"row_id": r["row_id"], "text": r["text"].strip()})
    return rows
