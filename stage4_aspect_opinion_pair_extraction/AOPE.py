"""
Stage 4 — Aspect-Opinion Pair Extraction (AOPE)

Eval variants (stage4_eval.py logic):
  pair_d1(tokens, aspects, opinions) → list[tuple]
  pair_d2(tokens, pos_tags, aspects, opinions) → list[tuple]
  pair_d3_from_raw(raw_pairs, aspects, opinions) → list[tuple]
  pair_d4(tokens, pos_tags, aspects, opinions, raw_stanza_pairs) → list[tuple]

Pipeline variant (test_real_data.py + pipeline_absa.py logic):
  extract_v2_pairs(tokens, pos_tags) → (pairs, consumed)
  extract_pc_pairs(tokens, pos_tags) → (pairs, consumed)
  extract_nsubj_pairs(sent, tokens, pos_tags, consumed) → list[dict]
  extract_amod_pairs(sent, tokens, pos_tags, consumed) → list[dict]
  attach_negation(tokens, pairs) → list[dict]
  merge_all_pairs(v2_pairs, pc_pairs, dep_pairs) → list[dict]
  pair_pipeline(tokens, pos_tags, ate, ote) → list[dict]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from collections import defaultdict
from shared.constants import (
    CKIP_ASPECT_POS, CKIP_OPINION_POS, CKIP_NOUN_POS, _MODIFIER_NOUN_POS, _PUNCT,
    DEGREE_ADV, STOP_WORDS, WEAK_ASPECT_NOUNS,
    CUSTOM_ASPECT_WORDS, CUSTOM_OPINION_WORDS, VHC_WHITELIST,
    NEUTRAL_OPINION_WORDS, NEGATION_PREFIXES, NEGATION_INTENSIFIERS,
    NEG_BASE_OPINIONS, NEG_COGNITIVE_VERBS, PC_VERB_POS,
    FIRST_PERSON_PRONOUNS, COGNITIVE_VERBS_EXCLUDE,
    V2_POSITIVE_TOKENS, V2_POSITIVE_POS, V2_NEGATIVE_TOKENS, MODAL_BEFORE_V2,
    NOUN_COLLECT_POS, NOUN_SKIP_POS, MIN_ASPECT_LEN,
)

# ATE POS used in A3 merging
_A3_ASP     = frozenset({"Na", "Nb", "Nv", "FW"})
_A3_COMPV   = frozenset({"VC", "VE", "VD"})
_A3_OPN_POS = frozenset({"VH", "VJ", "VK", "A"})


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _is_valid_aspect(text: str, min_len: int = 2) -> bool:
    t = text.strip()
    if not t or len(t) < min_len:
        return False
    if all(c in _PUNCT for c in t):
        return False
    if not any("一" <= c <= "鿿" or c.isalnum() for c in t):
        return False
    if t in DEGREE_ADV or t in STOP_WORDS:
        return False
    return True


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


def _get_fw_text(tokens: list, pos_tags: list, idx: int) -> str:
    """Handle FW compound like 'Manual Sort'."""
    if idx < len(pos_tags) and pos_tags[idx] == "FW":
        parts = []
        j = idx
        while j < len(tokens) and pos_tags[j] in {"FW", "WHITESPACE"}:
            if pos_tags[j] == "FW":
                parts.append(tokens[j])
            j += 1
        return " ".join(parts) if parts else tokens[idx]
    return tokens[idx] if idx < len(tokens) else ""


def _find_span_start(tokens: list, term: str) -> int:
    """Return starting token index for term (handles compound merges and negation-skip)."""
    if not term:
        return -1
    for i in range(len(tokens)):
        if tokens[i] == term:
            return i
        acc = ""
        for j in range(i, min(i + 8, len(tokens))):
            if tokens[j] in (" ", "　"):
                continue
            acc += tokens[j]
            if acc == term:
                return i
            if len(acc) >= len(term):
                break
    for neg_char in ("不", "沒", "非", "未"):
        if not term.startswith(neg_char):
            continue
        base = term[len(neg_char):]
        if not base:
            continue
        for i, t in enumerate(tokens):
            if t != neg_char:
                continue
            j = i + 1
            while j < len(tokens) and tokens[j] in NEGATION_INTENSIFIERS:
                j += 1
            if j >= len(tokens):
                continue
            acc = ""
            for k in range(j, min(j + 4, len(tokens))):
                acc += tokens[k]
                if acc == base:
                    return i
                if len(acc) > len(base):
                    break
    return -1


def _resolve(raw: str, candidates: list) -> str | None:
    """Return first candidate that partially matches raw."""
    for c in candidates:
        if raw == c or raw in c or c in raw:
            return c
    shorter = min(len(raw), min((len(c) for c in candidates), default=1))
    for c in candidates:
        if shorter > 0 and len(set(raw) & set(c)) / shorter >= 0.7:
            return c
    return None


def _clause_ranges(tokens: list, pos_tags: list) -> list[tuple]:
    ranges: list[tuple] = []
    start = 0
    for i, (tok, pos) in enumerate(zip(tokens, pos_tags)):
        if tok in {"，", "、", "；", "。"} or pos in {"COMMACATEGORY", "PERIODCATEGORY"}:
            ranges.append((start, i))
            start = i + 1
    ranges.append((start, len(tokens)))
    return [r for r in ranges if r[0] < r[1]]


class AOPE:
    """Aspect-Opinion Pair Extraction — Stage 4."""

    def __init__(self, stanza_nlp=None):
        self._nlp = stanza_nlp

    # ── Eval D1/D2/D3/D4 (matches stage4_eval.py exactly) ────────

    def pair_d1(self, tokens: list, aspects: list, opinions: list) -> list[tuple]:
        """D1: linear proximity — each opinion paired with nearest aspect."""
        asp_pos = {a: _find_span_start(tokens, a) for a in aspects}
        op_pos  = {o: _find_span_start(tokens, o) for o in opinions}
        valid_asps = [a for a in aspects if asp_pos[a] >= 0]
        if not valid_asps:
            return []
        pairs: list[tuple] = []
        seen: set = set()
        for op, o_idx in op_pos.items():
            if o_idx < 0:
                continue
            nearest = min(valid_asps, key=lambda a: abs(asp_pos[a] - o_idx))
            pair = (nearest, op)
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
        return pairs

    def pair_d2(self, tokens: list, pos_tags: list,
                aspects: list, opinions: list) -> list[tuple]:
        """D2: clause boundary pairing, fallback to nearest."""
        asp_pos    = {a: _find_span_start(tokens, a) for a in aspects}
        op_pos     = {o: _find_span_start(tokens, o) for o in opinions}
        valid_asps = [a for a in aspects if asp_pos[a] >= 0]
        clauses    = _clause_ranges(tokens, pos_tags)

        pairs: list[tuple] = []
        seen: set = set()
        for op, o_idx in op_pos.items():
            if o_idx < 0:
                continue
            op_clause   = next((r for r in clauses if r[0] <= o_idx < r[1]), None)
            clause_asps = (
                [a for a in valid_asps if op_clause[0] <= asp_pos[a] < op_clause[1]]
                if op_clause else []
            )
            candidates = clause_asps if clause_asps else valid_asps
            if not candidates:
                continue
            nearest = min(candidates, key=lambda a: abs(asp_pos[a] - o_idx))
            pair = (nearest, op)
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
        return pairs

    def _stanza_raw_pairs(self, sent, ckip_tokens: list, ckip_pos: list) -> list[tuple]:
        """Extract (asp_text, op_text) from nsubj, amod, and advmod→obj paths."""
        if len(sent.words) != len(ckip_tokens):
            return []
        children: dict = defaultdict(list)
        for w in sent.words:
            if w.head > 0:
                children[w.head].append(w)
        pairs: list[tuple] = []

        # nsubj + advmod→obj
        for w in sent.words:
            idx    = w.id - 1
            ck_pos = ckip_pos[idx] if idx < len(ckip_pos) else ""
            if not _is_opinion_candidate(w.text, ck_pos):
                continue
            if w.text in NEUTRAL_OPINION_WORDS or w.text in DEGREE_ADV:
                continue
            if _is_modifier(ckip_tokens, ckip_pos, idx):
                continue
            for child in children[w.id]:
                if child.deprel != "nsubj":
                    continue
                if child.text in FIRST_PERSON_PRONOUNS:
                    continue
                c_idx  = child.id - 1
                c_ck   = ckip_pos[c_idx] if c_idx < len(ckip_pos) else ""
                if c_ck not in CKIP_ASPECT_POS and child.upos not in {"NOUN", "PROPN", "X"}:
                    continue
                if not _is_valid_aspect(child.text):
                    continue
                pairs.append((_get_fw_text(ckip_tokens, ckip_pos, c_idx), w.text))

            if w.deprel == "advmod" and w.head > 0:
                head_w = sent.words[w.head - 1]
                if head_w.text not in COGNITIVE_VERBS_EXCLUDE:
                    for sib in children[head_w.id]:
                        if sib.deprel != "obj":
                            continue
                        s_idx = sib.id - 1
                        s_ck  = ckip_pos[s_idx] if s_idx < len(ckip_pos) else ""
                        if s_ck not in CKIP_ASPECT_POS and sib.upos not in {"NOUN", "PROPN", "X"}:
                            continue
                        if not _is_valid_aspect(sib.text):
                            continue
                        pairs.append((_get_fw_text(ckip_tokens, ckip_pos, s_idx), w.text))

        # amod: aspect → amod child = opinion
        for w in sent.words:
            a_idx  = w.id - 1
            ck_pos = ckip_pos[a_idx] if a_idx < len(ckip_pos) else ""
            if ck_pos not in CKIP_ASPECT_POS and w.upos not in {"NOUN", "PROPN"}:
                continue
            if not _is_valid_aspect(w.text):
                continue
            if _is_opinion_candidate(w.text, ck_pos):
                continue
            if w.text in WEAK_ASPECT_NOUNS:
                continue
            asp_text = _get_fw_text(ckip_tokens, ckip_pos, a_idx)
            for dep_w in sent.words:
                if dep_w.head != w.id or dep_w.deprel != "amod":
                    continue
                o_idx = dep_w.id - 1
                o_ck  = ckip_pos[o_idx] if o_idx < len(ckip_pos) else ""
                if not _is_opinion_candidate(dep_w.text, o_ck):
                    continue
                if dep_w.text in NEUTRAL_OPINION_WORDS or dep_w.text in DEGREE_ADV:
                    continue
                pairs.append((asp_text, dep_w.text))

        return pairs

    def pair_d3_from_raw(self, raw_pairs: list, aspects: list,
                         opinions: list) -> list[tuple]:
        """Filter raw stanza pairs to A3 aspects and C2 opinions."""
        pairs: list[tuple] = []
        seen: set = set()
        for raw_asp, raw_op in raw_pairs:
            asp = _resolve(raw_asp, aspects)
            op  = _resolve(raw_op, opinions)
            if asp and op:
                pair = (asp, op)
                if pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)
        return pairs

    def pair_d4(self, tokens: list, pos_tags: list, aspects: list,
                opinions: list, raw_stanza_pairs: list) -> list[tuple]:
        """D3 pairs first; opinions not in D3 → D2 fallback."""
        d3     = self.pair_d3_from_raw(raw_stanza_pairs, aspects, opinions)
        d3_ops = {op for _, op in d3}
        remaining_ops = [o for o in opinions if o not in d3_ops]
        d2_partial    = self.pair_d2(tokens, pos_tags, aspects, remaining_ops)
        seen: set = set()
        result: list = []
        for pair in d3 + d2_partial:
            if pair not in seen:
                seen.add(pair)
                result.append(pair)
        return result

    # ── D5: cross-clause cosine pairing (experimental) ───────────

    def _d5_scored_additions(self, tokens: list, pos_tags: list,
                              a3_aspects: list, d4_pairs: list, model) -> list:
        """Compute (asp, op, score) candidates for PC opinions not already in D4."""
        from stage3_opinion_term_extraction.OTE import OTE
        ote = OTE()
        pc_opinions = ote.extract_pc_opinions(tokens, pos_tags)
        if not pc_opinions or not a3_aspects:
            return []

        asp_pos: dict = {a: _find_span_start(tokens, a) for a in a3_aspects}
        d4_op_set     = {op for _, op in d4_pairs}
        paired_asp_positions: set = set()
        for asp, _ in d4_pairs:
            p = asp_pos.get(asp, _find_span_start(tokens, asp))
            asp_pos[asp] = p
            if p >= 0:
                paired_asp_positions.add(p)

        clauses   = _clause_ranges(tokens, pos_tags)
        additions: list = []
        seen_ops: set   = set()

        for op in pc_opinions:
            if op in d4_op_set or op in seen_ops:
                continue
            seen_ops.add(op)
            op_idx = _find_span_start(tokens, op)
            if op_idx < 0:
                continue
            op_clause = next((r for r in clauses if r[0] <= op_idx < r[1]), None)

            # Layer 1: same clause, distance-based
            same_asps = []
            if op_clause:
                same_asps = [
                    a for a in a3_aspects
                    if op_clause[0] <= asp_pos.get(a, -1) < op_clause[1]
                ]
            if same_asps:
                nearest = min(same_asps, key=lambda a: abs(asp_pos[a] - op_idx))
                additions.append((nearest, op, 2.0))
                continue

            # Layer 2: expand to free clauses, cosine scoring
            free_asps = []
            for a in a3_aspects:
                a_pos = asp_pos.get(a, -1)
                if a_pos < 0:
                    continue
                if op_clause and op_clause[0] <= a_pos < op_clause[1]:
                    continue
                a_clause = next((r for r in clauses if r[0] <= a_pos < r[1]), None)
                if a_clause and any(a_clause[0] <= p < a_clause[1]
                                    for p in paired_asp_positions):
                    continue
                free_asps.append(a)

            if not free_asps:
                continue
            from sentence_transformers import util as st_util
            op_emb   = model.encode(op, convert_to_tensor=True)
            cand_emb = model.encode(free_asps, convert_to_tensor=True)
            scores   = st_util.cos_sim(op_emb, cand_emb)[0].tolist()
            scored   = sorted(zip(free_asps, scores), key=lambda x: x[1], reverse=True)
            if scored:
                best_asp, best_score = scored[0]
                additions.append((best_asp, op, float(best_score)))

        return additions

    @staticmethod
    def pair_d5_threshold(d4_pairs: list, scored_additions: list,
                          threshold: float) -> list:
        """Apply threshold to scored additions and merge with D4 pairs."""
        result = list(d4_pairs)
        seen   = set(d4_pairs)
        for asp, op, score in scored_additions:
            if score >= threshold:
                pair = (asp, op)
                if pair not in seen:
                    seen.add(pair)
                    result.append(pair)
        return result

    # ── Pipeline rules (test_real_data.py logic) ──────────────────

    def extract_pc_pairs(
        self, tokens: list, pos_tags: list
    ) -> tuple[list[dict], set[int]]:
        """
        Pattern A: 不 + 認知動詞 → polarity_hint=neg
        Pattern B: V + 不/得 + V → 看不懂 / 看得懂
        Returns (pairs, consumed_indices).
        """
        pairs:    list[dict] = []
        consumed: set[int]   = set()
        i = 0
        while i < len(tokens):
            if (tokens[i] == "不"
                    and pos_tags[i] in {"D", "AD"}
                    and i + 1 < len(tokens)
                    and tokens[i + 1] in NEG_COGNITIVE_VERBS):
                compound = tokens[i] + tokens[i + 1]
                pairs.append({
                    "aspect":        "implicit",
                    "opinion":       compound,
                    "opinion_idx":   i,
                    "aspect_type":   "pc_rule",
                    "polarity_hint": "neg",
                })
                consumed.update({i, i + 1})
                i += 2
                continue
            if (i + 2 < len(tokens)
                    and pos_tags[i] in PC_VERB_POS
                    and tokens[i + 1] in {"不", "得"}
                    and pos_tags[i + 2] in PC_VERB_POS):
                compound = tokens[i] + tokens[i + 1] + tokens[i + 2]
                polarity  = "neg" if tokens[i + 1] == "不" else "pos"
                from shared.constants import NEUTRAL_OPINION_WORDS as _NOW
                if compound not in _NOW:
                    pairs.append({
                        "aspect":        "implicit",
                        "opinion":       compound,
                        "opinion_idx":   i,
                        "aspect_type":   "pc_rule",
                        "polarity_hint": polarity,
                    })
                consumed.update({i, i + 1, i + 2})
                i += 3
                continue
            i += 1
        return pairs, consumed

    def extract_v2_pairs(
        self, tokens: list, pos_tags: list
    ) -> tuple[list[dict], set[int]]:
        """
        有(V_2)/沒有 + [名詞片語].
        v5: 前接 modal 時跳過.
        Returns (pairs, consumed_indices).
        """
        pairs:    list[dict] = []
        consumed: set[int]   = set()
        i = 0
        while i < len(tokens):
            tok, p = tokens[i], pos_tags[i]
            if tok in V2_POSITIVE_TOKENS and p in V2_POSITIVE_POS:
                if i > 0 and tokens[i - 1] in MODAL_BEFORE_V2:
                    i += 1
                    continue
                polarity_hint = "pos"
            elif tok in V2_NEGATIVE_TOKENS:
                polarity_hint = "neg"
            else:
                i += 1
                continue

            consumed.add(i)
            aspect, next_i, noun_indices = self._collect_noun_phrase(tokens, pos_tags, i + 1)
            consumed.update(noun_indices)
            if len(aspect) >= MIN_ASPECT_LEN:
                pairs.append({
                    "aspect":        aspect,
                    "opinion":       "implicit",
                    "opinion_idx":   None,
                    "aspect_type":   "v2_rule",
                    "polarity_hint": polarity_hint,
                })
            i = next_i
        return pairs, consumed

    @staticmethod
    def _collect_noun_phrase(
        tokens: list, pos_tags: list, start: int
    ) -> tuple[str, int, set[int]]:
        parts:   list[str] = []
        indices: set[int]  = set()
        i = start
        while i < len(tokens):
            p = pos_tags[i]
            if p in NOUN_COLLECT_POS:
                parts.append(tokens[i])
                indices.add(i)
            elif p in NOUN_SKIP_POS:
                pass
            else:
                break
            i += 1
        return "".join(parts), i, indices

    def extract_nsubj_pairs(
        self,
        sent,
        ckip_tokens: list,
        ckip_pos:    list,
        consumed:    set[int] | None = None,
    ) -> list[dict]:
        """
        Path A: opinion nsubj child → explicit aspect.
        Path B: opinion as advmod → head verb's obj = explicit aspect.
        No explicit aspect found → implicit.
        Uses min_len=1 for aspect validity (preserving test_real_data behavior).
        """
        consumed = consumed or set()
        if len(sent.words) != len(ckip_tokens):
            return []

        children: dict = defaultdict(list)
        for w in sent.words:
            if w.head > 0:
                children[w.head].append(w)

        pairs: list[dict] = []
        for w in sent.words:
            idx = w.id - 1
            if idx >= len(ckip_pos) or idx in consumed:
                continue
            ck_pos = ckip_pos[idx]
            if not _is_opinion_candidate(w.text, ck_pos):
                continue
            if w.text in NEUTRAL_OPINION_WORDS or w.text in DEGREE_ADV:
                continue
            if _is_modifier(ckip_tokens, ckip_pos, idx):
                continue

            found = False
            # Path A: nsubj
            for child in children[w.id]:
                if child.deprel != "nsubj":
                    continue
                child_idx = child.id - 1
                if child_idx in consumed:
                    continue
                if child.text in FIRST_PERSON_PRONOUNS:
                    continue
                child_ck = ckip_pos[child_idx] if child_idx < len(ckip_pos) else ""
                if not _is_valid_aspect(child.text, min_len=1):
                    continue
                if child_ck in CKIP_ASPECT_POS or child.upos in {"NOUN", "PROPN", "X"}:
                    pairs.append({
                        "aspect":        _get_fw_text(ckip_tokens, ckip_pos, child_idx),
                        "opinion":       w.text,
                        "opinion_idx":   idx,
                        "aspect_type":   "explicit",
                        "polarity_hint": None,
                    })
                    found = True

            # Path B: advmod → head verb's obj
            if not found and w.deprel == "advmod" and w.head > 0:
                head_w = sent.words[w.head - 1]
                if head_w.text not in COGNITIVE_VERBS_EXCLUDE:
                    for sib in children[head_w.id]:
                        if sib.deprel != "obj":
                            continue
                        sib_idx = sib.id - 1
                        if sib_idx in consumed:
                            continue
                        sib_ck = ckip_pos[sib_idx] if sib_idx < len(ckip_pos) else ""
                        if not _is_valid_aspect(sib.text, min_len=1):
                            continue
                        if sib_ck in CKIP_ASPECT_POS or sib.upos in {"NOUN", "PROPN", "X"}:
                            pairs.append({
                                "aspect":        _get_fw_text(ckip_tokens, ckip_pos, sib_idx),
                                "opinion":       w.text,
                                "opinion_idx":   idx,
                                "aspect_type":   "explicit",
                                "polarity_hint": None,
                            })
                            found = True

            if not found:
                pairs.append({
                    "aspect":        "implicit",
                    "opinion":       w.text,
                    "opinion_idx":   idx,
                    "aspect_type":   "implicit",
                    "polarity_hint": None,
                })
        return pairs

    def extract_amod_pairs(
        self,
        sent,
        ckip_tokens: list,
        ckip_pos:    list,
        consumed:    set[int] | None = None,
    ) -> list[dict]:
        """
        amod path: aspect noun → amod child = opinion.
        e.g. 清楚的介面 → (aspect=介面, opinion=清楚)
        """
        consumed = consumed or set()
        if len(sent.words) != len(ckip_tokens):
            return []
        pairs: list[dict] = []
        for w in sent.words:
            asp_idx = w.id - 1
            if asp_idx >= len(ckip_pos) or asp_idx in consumed:
                continue
            ck_pos = ckip_pos[asp_idx]
            if ck_pos not in CKIP_ASPECT_POS and w.upos not in {"NOUN", "PROPN"}:
                continue
            if not _is_valid_aspect(w.text, min_len=1):
                continue
            if _is_opinion_candidate(w.text, ck_pos):
                continue
            if w.text in WEAK_ASPECT_NOUNS:
                continue
            for dep_w in sent.words:
                if dep_w.head != w.id or dep_w.deprel != "amod":
                    continue
                op_idx    = dep_w.id - 1
                if op_idx in consumed:
                    continue
                op_ck_pos = ckip_pos[op_idx] if op_idx < len(ckip_pos) else ""
                if not _is_opinion_candidate(dep_w.text, op_ck_pos):
                    continue
                if dep_w.text in NEUTRAL_OPINION_WORDS or dep_w.text in DEGREE_ADV:
                    continue
                pairs.append({
                    "aspect":        _get_fw_text(ckip_tokens, ckip_pos, asp_idx),
                    "opinion":       dep_w.text,
                    "opinion_idx":   op_idx,
                    "aspect_type":   "explicit",
                    "polarity_hint": None,
                })
        return pairs

    def attach_negation(self, tokens: list, pairs: list[dict]) -> list[dict]:
        """
        For polarity_hint=None pairs, scan left for negation prefix.
        Double-negative: neg + NEG_BASE_OPINIONS → polarity=pos.
        """
        result: list[dict] = []
        for p in pairs:
            if p.get("polarity_hint") is not None or p["opinion"] == "implicit":
                result.append(p)
                continue
            idx = p.get("opinion_idx")
            if idx is None:
                result.append(p)
                continue
            j = idx - 1
            while j >= 0 and tokens[j] in NEGATION_INTENSIFIERS:
                j -= 1
            if j >= 0 and tokens[j] in NEGATION_PREFIXES:
                polarity = "pos" if p["opinion"] in NEG_BASE_OPINIONS else "neg"
                result.append({**p, "opinion": tokens[j] + p["opinion"],
                                "polarity_hint": polarity})
            else:
                result.append(p)
        return result

    @staticmethod
    def merge_all_pairs(
        v2_pairs: list[dict],
        pc_pairs: list[dict],
        dep_pairs: list[dict],
    ) -> list[dict]:
        """Merge V2 > PC > dep pairs, dedup by (aspect, opinion)."""
        seen:   set[tuple] = set()
        result: list[dict] = []
        for p in v2_pairs + pc_pairs + dep_pairs:
            key = (p["aspect"], p["opinion"])
            if key not in seen:
                seen.add(key)
                result.append(p)
        return result

    # ── Full pipeline pairing ──────────────────────────────────────

    def pair_pipeline(self, tokens: list, pos_tags: list, ate, ote) -> list[dict]:
        """
        Full pipeline D4+: V2/PC rules + stanza nsubj/amod + D2 fallback.
        ate: ATE instance, ote: OTE instance.
        Returns list of pair dicts with aspect/opinion/aspect_type/polarity_hint.
        """
        if self._nlp is None:
            raise RuntimeError("stanza_nlp not loaded — pass nlp to AOPE()")

        pc_pairs, pc_consumed = self.extract_pc_pairs(tokens, pos_tags)
        v2_pairs, v2_consumed = self.extract_v2_pairs(tokens, pos_tags)
        consumed = pc_consumed | v2_consumed

        try:
            doc  = self._nlp([tokens])
            sent = doc.sentences[0]

            if len(sent.words) != len(tokens):
                return self.merge_all_pairs(v2_pairs, pc_pairs, [])

            nsubj_pairs = self.extract_nsubj_pairs(sent, tokens, pos_tags, consumed)
            amod_pairs  = self.extract_amod_pairs(sent, tokens, pos_tags, consumed)
            raw_dep     = nsubj_pairs + amod_pairs

            # Remove duplicate implicit pairs
            explicit_op_idx = {
                p["opinion_idx"] for p in raw_dep
                if p["aspect_type"] == "explicit" and p.get("opinion_idx") is not None
            }
            raw_dep = [
                p for p in raw_dep
                if not (p["aspect_type"] == "implicit"
                        and p.get("opinion_idx") in explicit_op_idx)
            ]
            dep_pairs = self.attach_negation(tokens, raw_dep)

            # D2 fallback for C2 opinions not covered by D3 explicit
            d3_ops = {p["opinion"] for p in dep_pairs if p["aspect_type"] == "explicit"}
            a4     = ate.extract_a4(tokens, pos_tags)
            c2     = ote.extract_c2(tokens, pos_tags)
            uncov  = [op for op in c2 if op not in d3_ops]
            if uncov and a4:
                d2_raw  = self.pair_d2(tokens, pos_tags, a4, uncov)
                d2_rich = self.attach_negation(tokens, [
                    {
                        "aspect":        asp,
                        "opinion":       op,
                        "opinion_idx":   _find_span_start(tokens, op),
                        "aspect_type":   "explicit",
                        "polarity_hint": None,
                    }
                    for asp, op in d2_raw
                ])
                d2_ops    = {p["opinion"] for p in d2_rich}
                dep_pairs = [
                    p for p in dep_pairs
                    if not (p["aspect_type"] == "implicit" and p["opinion"] in d2_ops)
                ] + d2_rich

            return self.merge_all_pairs(v2_pairs, pc_pairs, dep_pairs)

        except Exception as exc:
            print(f"  [AOPE pipeline error] {exc}")
            return []
