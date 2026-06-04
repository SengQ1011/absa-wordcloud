"""
Stage 1 — Aspect Term Extraction (ATE)

class ATE:
  extract_a1(tokens, pos_tags) → list[str]   baseline CKIP Na/Nb/Nv
  extract_a2(tokens, pos_tags, threshold) → list[str]  + anchor cosine filter
  extract_a3(tokens, pos_tags) → list[str]   + compound noun merging
  extract_a4(tokens, pos_tags) → list[str]   A3 + CUSTOM_ASPECT_WORDS whitelist
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.constants import (
    CKIP_ASPECT_POS, DEGREE_ADV, STOP_WORDS, WEAK_ASPECT_NOUNS,
    CUSTOM_ASPECT_WORDS, CUSTOM_OPINION_WORDS, VHC_WHITELIST,
    A3_COMPOUND_V, A3_OPINION_POS, _PUNCT,
    CATEGORY_SEEDS_A2, SBERT_MODEL,
)


def _is_valid_aspect(text: str) -> bool:
    t = text.strip()
    if not t or len(t) < 2:
        return False
    if all(c in _PUNCT for c in t):
        return False
    if not any("一" <= c <= "鿿" or c.isalnum() for c in t):
        return False
    if t in DEGREE_ADV or t in STOP_WORDS:
        return False
    return True


class ATE:
    """Aspect Term Extraction — Stage 1."""

    def __init__(self, ckip_ws=None, ckip_pos=None, sbert_model=None):
        self._ws    = ckip_ws
        self._pos   = ckip_pos
        self._sbert = sbert_model
        self._anchors: dict | None = None
        if sbert_model is not None:
            self._anchors = self._build_anchors()

    # ── A2: anchor setup ──────────────────────────────────────────

    def _build_anchors(self) -> dict:
        import numpy as np
        anchors = {}
        for cat, seeds in CATEGORY_SEEDS_A2.items():
            embs = self._sbert.encode(seeds, show_progress_bar=False)
            anchors[cat] = embs.mean(axis=0)
        return anchors

    def load_sbert(self, model_name: str = SBERT_MODEL):
        from sentence_transformers import SentenceTransformer
        self._sbert  = SentenceTransformer(model_name)
        self._anchors = self._build_anchors()

    # ── Extraction methods ────────────────────────────────────────

    def extract_a1(self, tokens: list, pos_tags: list) -> list[str]:
        """A1: CKIP Na/Nb/Nv direct extraction with is_valid_aspect filter."""
        aspects: list[str] = []
        seen: set = set()
        for tok, pos in zip(tokens, pos_tags):
            if (pos in CKIP_ASPECT_POS
                    and _is_valid_aspect(tok)
                    and tok not in WEAK_ASPECT_NOUNS
                    and tok not in seen):
                seen.add(tok)
                aspects.append(tok)
        return aspects

    def extract_a2(
        self,
        tokens: list,
        pos_tags: list,
        threshold: float = 0.25,
    ) -> list[str]:
        """A2: A1 + anchor cosine filter, keeps candidates with max cosine ≥ threshold."""
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        if self._sbert is None or self._anchors is None:
            raise RuntimeError("sbert model not loaded — call load_sbert() first")

        candidates = self.extract_a1(tokens, pos_tags)
        if not candidates:
            return []
        embs          = self._sbert.encode(candidates, show_progress_bar=False)
        anchor_matrix = np.stack(list(self._anchors.values()))
        cos_scores    = cosine_similarity(embs, anchor_matrix).max(axis=1)
        return [c for c, score in zip(candidates, cos_scores) if score >= threshold]

    def _compound_merge(self, tokens: list, pos_tags: list, use_whitelist: bool) -> list[str]:
        """Shared compound-noun merging loop used by A3 and A4."""
        MAX_MERGES = 1
        aspects: list[str] = []
        seen: set = set()
        i = 0
        while i < len(tokens):
            tok       = tokens[i]
            is_custom = use_whitelist and (tok in CUSTOM_ASPECT_WORDS)
            if pos_tags[i] not in CKIP_ASPECT_POS and not is_custom:
                i += 1
                continue

            if not is_custom and (tok in CUSTOM_OPINION_WORDS or tok in VHC_WHITELIST):
                i += 1
                continue
            if not _is_valid_aspect(tok) or tok in WEAK_ASPECT_NOUNS:
                i += 1
                continue

            w = tok
            j = i + 1
            merge_count = 0
            while j < len(tokens) and merge_count < MAX_MERGES:
                pj = pos_tags[j]
                if pj in {"Na", "Nb", "Nv"} and len(tokens[j]) >= 2:
                    w += tokens[j]
                    j += 1
                    merge_count += 1
                elif pj == "FW":
                    w += (" " if pos_tags[i] == "FW" else "") + tokens[j]
                    j += 1
                    merge_count += 1
                elif (pj == "WHITESPACE"
                      and pos_tags[i] == "FW"
                      and j + 1 < len(tokens)
                      and pos_tags[j + 1] == "FW"):
                    j += 1
                elif (pj in A3_COMPOUND_V
                      and not (j + 1 < len(tokens) and pos_tags[j + 1] in A3_OPINION_POS)):
                    w += tokens[j]
                    j += 1
                    merge_count += 1
                elif (pj == "De"
                      and tokens[j] == "的"
                      and j + 1 < len(tokens)
                      and pos_tags[j + 1] in CKIP_ASPECT_POS):
                    w += tokens[j] + tokens[j + 1]
                    j += 2
                    merge_count += 1
                else:
                    break

            w = w.strip()
            if _is_valid_aspect(w) and w not in WEAK_ASPECT_NOUNS and w not in seen:
                seen.add(w)
                aspects.append(w)
            i = j

        return aspects

    def extract_a3(self, tokens: list, pos_tags: list) -> list[str]:
        """A3: Compound noun merging (Na/Nb/Nv/FW base + VC/VE/VD bridge, MAX_MERGES=1)."""
        return self._compound_merge(tokens, pos_tags, use_whitelist=False)

    def extract_a4(self, tokens: list, pos_tags: list) -> list[str]:
        """A4: A3 + CUSTOM_ASPECT_WORDS whitelist (domain verbs bypass POS filter)."""
        return self._compound_merge(tokens, pos_tags, use_whitelist=True)
