"""
Stage 3 — Opinion Term Extraction (OTE)

class OTE:
  extract_c1(tokens, pos_tags) → list[str]   CKIP VH/VJ/A baseline
  extract_c2(tokens, pos_tags) → list[str]   C1 + attach_negation + PC rule
  extract_c3(tokens, pos_tags, sent) → list[str]   C2 + stanza amod path
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.constants import (
    CKIP_OPINION_POS, CKIP_NOUN_POS, _MODIFIER_NOUN_POS,
    VHC_WHITELIST, CUSTOM_OPINION_WORDS, NEUTRAL_OPINION_WORDS,
    DEGREE_ADV, NEGATION_PREFIXES, NEGATION_INTENSIFIERS,
    PC_VERB_POS, NEG_COGNITIVE_VERBS, WEAK_ASPECT_NOUNS,
)


def _is_opinion_candidate(token: str, ck_pos: str) -> bool:
    if token in CUSTOM_OPINION_WORDS:
        return True
    if ck_pos in CKIP_OPINION_POS:
        return True
    if ck_pos == "VHC" and token in VHC_WHITELIST:
        return True
    return False


def _is_modifier(tokens: list, pos_tags: list, idx: int) -> bool:
    n = len(tokens)
    if idx + 1 < n and pos_tags[idx + 1] in _MODIFIER_NOUN_POS:
        return True
    if (pos_tags[idx] == "A"
            and idx + 1 < n and tokens[idx + 1] == "的"
            and idx + 2 < n and pos_tags[idx + 2] in _MODIFIER_NOUN_POS):
        return True
    return False


class OTE:
    """Opinion Term Extraction — Stage 3."""

    def __init__(self, ckip_ws=None, ckip_pos=None, stanza_nlp=None):
        self._ws  = ckip_ws
        self._pos = ckip_pos
        self._nlp = stanza_nlp

    # ── C1: CKIP baseline ─────────────────────────────────────────

    def extract_c1(self, tokens: list, pos_tags: list) -> list[str]:
        """VH/VJ/A + VHC whitelist + custom words, filtered by neutral + is_modifier."""
        seen: set    = set()
        result: list = []
        for i, (tok, pos) in enumerate(zip(tokens, pos_tags)):
            if not _is_opinion_candidate(tok, pos):
                continue
            if tok in NEUTRAL_OPINION_WORDS or tok in DEGREE_ADV:
                continue
            if _is_modifier(tokens, pos_tags, i):
                continue
            if tok not in seen:
                seen.add(tok)
                result.append(tok)
        return result

    # ── PC rule ───────────────────────────────────────────────────

    def extract_pc_opinions(self, tokens: list, pos_tags: list) -> list[str]:
        """
        Pattern A: 不 + 認知動詞 → 不知道 / 不理解
        Pattern B: V + 不/得 + V → 看不懂 / 看得懂
        """
        seen: set    = set()
        result: list = []
        i = 0
        while i < len(tokens):
            if (tokens[i] == "不"
                    and pos_tags[i] in {"D", "AD"}
                    and i + 1 < len(tokens)
                    and tokens[i + 1] in NEG_COGNITIVE_VERBS):
                compound = tokens[i] + tokens[i + 1]
                if compound not in seen:
                    seen.add(compound)
                    result.append(compound)
                i += 2
                continue
            if (i + 2 < len(tokens)
                    and pos_tags[i] in PC_VERB_POS
                    and tokens[i + 1] in {"不", "得"}
                    and pos_tags[i + 2] in PC_VERB_POS):
                compound = tokens[i] + tokens[i + 1] + tokens[i + 2]
                if compound not in NEUTRAL_OPINION_WORDS and compound not in seen:
                    seen.add(compound)
                    result.append(compound)
                i += 3
                continue
            i += 1
        return result

    # ── C2: C1 + attach_negation + PC rule ───────────────────────

    def extract_c2(self, tokens: list, pos_tags: list) -> list[str]:
        """C1 + left-scan negation attachment + PC rule."""
        c1_terms = self.extract_c1(tokens, pos_tags)
        seen: set    = set()
        result: list = []

        for tok in c1_terms:
            idx = next((i for i, t in enumerate(tokens) if t == tok), None)
            if idx is None:
                if tok not in seen:
                    seen.add(tok)
                    result.append(tok)
                continue

            j = idx - 1
            while j >= 0 and tokens[j] in NEGATION_INTENSIFIERS:
                j -= 1

            if j >= 0 and tokens[j] in NEGATION_PREFIXES:
                compound = tokens[j] + tok
                term = compound if compound not in seen else tok
            else:
                term = tok

            if term not in seen:
                seen.add(term)
                result.append(term)

        for pc in self.extract_pc_opinions(tokens, pos_tags):
            if pc not in seen:
                seen.add(pc)
                result.append(pc)

        return result

    # ── C3: C2 + stanza amod ─────────────────────────────────────

    def _extract_c3_amod(self, sent, ckip_tokens: list, ckip_pos: list) -> list[str]:
        """Additional opinions via stanza amod: aspect noun → amod child = opinion."""
        if len(sent.words) != len(ckip_tokens):
            return []
        seen: set    = set()
        result: list = []
        for w in sent.words:
            asp_idx = w.id - 1
            if asp_idx >= len(ckip_pos):
                continue
            ck_pos = ckip_pos[asp_idx]
            if ck_pos not in CKIP_NOUN_POS and w.upos not in {"NOUN", "PROPN"}:
                continue
            if w.text in WEAK_ASPECT_NOUNS:
                continue
            if _is_opinion_candidate(w.text, ck_pos):
                continue
            for dep_w in sent.words:
                if dep_w.head != w.id or dep_w.deprel != "amod":
                    continue
                op_idx    = dep_w.id - 1
                op_ck_pos = ckip_pos[op_idx] if op_idx < len(ckip_pos) else ""
                if not _is_opinion_candidate(dep_w.text, op_ck_pos):
                    continue
                if dep_w.text in NEUTRAL_OPINION_WORDS or dep_w.text in DEGREE_ADV:
                    continue
                if dep_w.text not in seen:
                    seen.add(dep_w.text)
                    result.append(dep_w.text)
        return result

    def extract_c3(self, tokens: list, pos_tags: list, sent) -> list[str]:
        """C2 + stanza amod opinions."""
        c2       = self.extract_c2(tokens, pos_tags)
        amod_extra = self._extract_c3_amod(sent, tokens, pos_tags)
        seen     = set(c2)
        result   = list(c2)
        for op in amod_extra:
            if op not in seen:
                seen.add(op)
                result.append(op)
        return result
