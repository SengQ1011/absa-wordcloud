"""
Stage 5 — Aspect Sentiment Classification (ASC)

class ASC:
  predict_f1(pair, text) → str   opinion_term → DistilBERT
  predict_f2(pair, text) → str   '[text] [SEP] [opinion]' → DistilBERT
  predict_f3(pair, text) → str   polarity_hint → fallback F1
  predict_pipeline(pair, text) → str   same as F3 (used in pipeline_absa.py)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.constants import DISTILBERT_MODEL


class ASC:
    """Aspect Sentiment Classification — Stage 5."""

    def __init__(self, distilbert_model=None):
        self._model = distilbert_model
        self._f1_cache: dict = {}

    def load_distilbert(self, model_name: str = DISTILBERT_MODEL):
        from transformers import pipeline
        print(f"Load DistilBERT ({model_name})...")
        self._model = pipeline("text-classification", model=model_name, top_k=None)

    def _classify(self, text: str) -> str:
        results = self._model(text[:512])[0]
        scores  = {r["label"]: r["score"] for r in results}
        return "pos" if scores.get("positive", 0) >= scores.get("negative", 0) else "neg"

    def predict_f1(self, pair: dict, _text: str) -> str:
        """F1: opinion_term → DistilBERT directly."""
        target = pair["opinion"]
        if target == "implicit":
            target = pair["aspect"]
        if not target or target == "implicit":
            return "pos"
        return self._classify(target)

    def predict_f2(self, pair: dict, text: str) -> str:
        """F2: BERT-SPC '[text] [SEP] [opinion]' → DistilBERT."""
        target = pair["opinion"]
        if target == "implicit":
            target = pair["aspect"]
        if not target or target == "implicit":
            return "pos"
        return self._classify(f"{text} [SEP] {target}")

    def predict_f3(self, pair: dict, text: str) -> str:
        """F3: polarity_hint (rule) → fallback F1."""
        hint = pair.get("polarity_hint")
        if hint in ("neg", "pos"):
            return hint
        key = (pair["aspect"], pair["opinion"])
        if key not in self._f1_cache:
            self._f1_cache[key] = self.predict_f1(pair, text)
        return self._f1_cache[key]

    def predict_pipeline(self, pair: dict, _text: str) -> str:
        """
        Pipeline sentiment (F3 variant used in pipeline_absa.py):
        polarity_hint first, then DistilBERT on opinion (or aspect if implicit).
        """
        hint = pair.get("polarity_hint")
        if hint in ("pos", "neg"):
            return hint
        target = pair["opinion"]
        if target == "implicit":
            target = pair["aspect"]
        if not target or target == "implicit":
            return "pos"
        return self._classify(target)
