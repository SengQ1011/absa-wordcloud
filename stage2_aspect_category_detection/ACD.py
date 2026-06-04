"""
Stage 2 — Aspect Category Detection (ACD)

class ACD:
  predict_b2(aspect_term) → str | None
  predict_b1(text, aspect_term, threshold) → tuple[str | None, dict | None]
  predict_b3(text, threshold) → tuple[str, dict]
  predict_b1_implicit(text, opinion_term, threshold) → tuple[str, dict]
  predict_combo(tuple_dict, text) → str
  apply_threshold_from_raw(raw, threshold) → str
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.constants import CATEGORY_SEEDS, SBERT_MODEL

OTHERS_THRESHOLD_EXP = 0.30
OTHERS_THRESHOLD_IMP = 0.30


class ACD:
    """Aspect Category Detection — Stage 2."""

    def __init__(self, sbert_model=None):
        self._sbert  = sbert_model
        self._anchors: dict | None = None
        if sbert_model is not None:
            self._anchors = self._build_anchors()

    def load_sbert(self, model_name: str = SBERT_MODEL):
        from sentence_transformers import SentenceTransformer
        self._sbert   = SentenceTransformer(model_name)
        self._anchors = self._build_anchors()

    # ── Anchor setup ──────────────────────────────────────────────

    def _build_anchors(self) -> dict:
        import numpy as np
        anchors = {}
        for cat, seeds in CATEGORY_SEEDS.items():
            embs = self._sbert.encode(seeds, normalize_embeddings=True, show_progress_bar=False)
            vec  = embs.mean(axis=0)
            norm = float(np.linalg.norm(vec))
            anchors[cat] = vec / norm if norm > 0 else vec
        return anchors

    def _cosine_scores(self, query: str) -> dict:
        import numpy as np
        emb = self._sbert.encode(query, normalize_embeddings=True, show_progress_bar=False)
        return {cat: float(np.dot(emb, vec)) for cat, vec in self._anchors.items()}

    @staticmethod
    def _apply_threshold(cos_scores: dict, threshold: float) -> str:
        best_cat   = max(cos_scores, key=cos_scores.get)
        best_score = cos_scores[best_cat]
        return best_cat if best_score >= threshold else "OTHERS"

    @staticmethod
    def apply_threshold_from_raw(raw: dict | None, threshold: float,
                                 fallback: str = "OTHERS") -> str:
        if raw is None:
            return fallback
        best_cat   = max(raw, key=raw.get)
        best_score = raw[best_cat]
        return best_cat if best_score >= threshold else "OTHERS"

    # ── Prediction methods ────────────────────────────────────────

    def predict_b2(self, aspect_term: str) -> str | None:
        """B2: seed dict substring match. Returns None for implicit."""
        if aspect_term == "implicit":
            return None
        best_cat, best_score = None, 0
        for cat, seeds in CATEGORY_SEEDS.items():
            score = sum(1 for s in seeds if s in aspect_term or aspect_term in s)
            if score > best_score:
                best_score, best_cat = score, cat
        return best_cat if best_score > 0 else "OTHERS"

    def predict_b1(
        self,
        text: str,
        aspect_term: str,
        threshold: float = OTHERS_THRESHOLD_EXP,
    ) -> tuple[str | None, dict | None]:
        """B1: encode('aspect [SEP] text') → cosine. Returns (pred, raw_scores)."""
        if aspect_term == "implicit":
            return None, None
        scores = self._cosine_scores(f"{aspect_term} [SEP] {text}")
        return self._apply_threshold(scores, threshold), scores

    def predict_b3(
        self,
        text: str,
        threshold: float = OTHERS_THRESHOLD_EXP,
    ) -> tuple[str, dict]:
        """B3: encode(text) → cosine. Works for all tuples."""
        scores = self._cosine_scores(text)
        return self._apply_threshold(scores, threshold), scores

    def predict_b1_implicit(
        self,
        text: str,
        opinion_term: str,
        threshold: float = OTHERS_THRESHOLD_IMP,
    ) -> tuple[str, dict]:
        """B1-imp: encode('opinion [SEP] text') → cosine. Symmetric to B1 for implicit."""
        scores = self._cosine_scores(f"{opinion_term} [SEP] {text}")
        return self._apply_threshold(scores, threshold), scores

    def predict_combo(
        self,
        aspect_term: str,
        opinion_term: str,
        text: str,
        exp_thr: float = OTHERS_THRESHOLD_EXP,
        imp_thr: float = OTHERS_THRESHOLD_IMP,
    ) -> str:
        """
        Combo (B2 → B1 → B3) for explicit; B1-imp for implicit.
        Same logic as pipeline_absa.predict_category.
        """
        if aspect_term == "implicit":
            op = opinion_term if opinion_term != "implicit" else ""
            if op:
                scores = self._cosine_scores(f"{op} [SEP] {text}")
                return self._apply_threshold(scores, imp_thr)
            scores = self._cosine_scores(text)
            return self._apply_threshold(scores, exp_thr)

        b2 = self.predict_b2(aspect_term)
        if b2 and b2 != "OTHERS":
            return b2
        _, scores_b1 = self.predict_b1(text, aspect_term, exp_thr)
        if scores_b1:
            b1 = self._apply_threshold(scores_b1, exp_thr)
            if b1 != "OTHERS":
                return b1
        _, scores_b3 = self.predict_b3(text, exp_thr)
        return self._apply_threshold(scores_b3, exp_thr)
