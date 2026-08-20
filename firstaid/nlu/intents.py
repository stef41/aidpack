"""Intent classification: utterance -> protocol family.

Hybrid scorer, pure stdlib, built at startup from the KB itself:
  1. weighted keyword/phrase lexicon (exact + fuzzy, typo-tolerant)
  2. char n-gram TF-IDF cosine against per-family exemplar centroids
No training pipeline, no model file: the KB *is* the model, so adding a
protocol automatically extends the NLU.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..kb import KnowledgeBase
from ..text import char_ngrams, cosine, fuzzy_contains, normalize, phrase_in, tokens

KW_WEIGHT = 0.6
VEC_WEIGHT = 0.4
ACCEPT_THRESHOLD = 0.14
CLARIFY_THRESHOLD = 0.07
MARGIN = 0.02
KW_SATURATION = 9.0
# Inputs longer than this are clamped to head+tail tokens before scoring —
# bounds worst-case latency on pathological inputs without losing trailing
# emergencies ("...and then he collapsed").
MAX_CLASSIFY_TOKENS = 170
# Triage fallback families lose ties against specific-cause families.
TRIAGE_FAMILIES = {"general_help", "breathing_difficulty"}
TRIAGE_TIE_BAND = 0.12
# breathing_difficulty only cedes to respiratory-adjacent specifics.
RESP_FAMILIES = {"choking", "asthma", "anaphylaxis", "croup", "panic",
                 "cardiac_arrest", "heart_attack", "carbon_monoxide", "drowning"}


@dataclass
class IntentResult:
    kind: str                     # "intent" | "clarify" | "unknown"
    family: str | None
    score: float
    candidates: list[tuple[str, float]]


class IntentClassifier:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.family_keywords: dict[str, list[tuple[str, float]]] = {}
        # per-language vector spaces: {lang: {family: centroid}} / {lang: idf}
        self.family_centroids: dict[str, dict[str, dict[str, float]]] = {}
        self._idfs: dict[str, dict[str, float]] = {}
        self._build()

    def _build(self) -> None:
        exemplars_by_lang: dict[str, dict[str, list[str]]] = {"en": {}}
        for p in self.kb.protocols.values():
            kws = self.family_keywords.setdefault(p.family, [])
            for phrase, w in p.keywords:
                kws.append((normalize(phrase), w))
            exemplars_by_lang["en"].setdefault(p.family, []).extend(p.exemplars)

        # merge language-pack lexicons: keywords are shared (phrase-anchored,
        # cannot dilute English matching); exemplar vector spaces stay per-language
        # so multilingual data never shifts English cosine geometry.
        from ..i18n import load_packs
        for lang, pack in load_packs().items():
            per = exemplars_by_lang.setdefault(lang, {})
            for fam, lex in pack.lexicon.items():
                if fam not in self.family_keywords:
                    continue
                for phrase, w in lex.get("keywords", []):
                    self.family_keywords[fam].append((normalize(phrase), float(w)))
                per.setdefault(fam, []).extend(lex.get("exemplars", []))

        for lang, family_exemplars in exemplars_by_lang.items():
            doc_bags = []
            for fam, exemplars in family_exemplars.items():
                for ex in exemplars:
                    doc_bags.append(char_ngrams(ex))
            n_docs = max(1, len(doc_bags))
            df: dict[str, int] = {}
            for bag in doc_bags:
                for g in bag:
                    df[g] = df.get(g, 0) + 1
            idf = {g: math.log(n_docs / (1 + c)) + 1.0 for g, c in df.items()}
            self._idfs[lang] = idf

            centroids: dict[str, dict[str, float]] = {}
            for fam, exemplars in family_exemplars.items():
                centroid: dict[str, float] = {}
                for ex in exemplars:
                    bag = char_ngrams(ex)
                    norm = math.sqrt(sum(v * v for v in bag.values())) or 1.0
                    for g, v in bag.items():
                        centroid[g] = centroid.get(g, 0.0) + (v / norm) * idf.get(g, 1.0)
                n = max(1, len(exemplars))
                centroids[fam] = {g: v / n for g, v in centroid.items()}
            self.family_centroids[lang] = centroids

    def _vectorize(self, text: str, lang: str = "en") -> dict[str, float]:
        idf = self._idfs.get(lang) or self._idfs["en"]
        bag = char_ngrams(text)
        return {g: v * idf.get(g, 1.0) for g, v in bag.items()}

    def _keyword_score(self, text_norm: str, toks: list[str], fam: str) -> float:
        total = 0.0
        for phrase, w in self.family_keywords.get(fam, []):
            if " " in phrase:
                if phrase_in(text_norm, phrase):
                    total += w
                else:
                    words = phrase.split()
                    if all(fuzzy_contains(toks, word) for word in words):
                        total += 0.7 * w
            else:
                if phrase in toks:
                    total += w
                elif fuzzy_contains(toks, phrase):
                    total += 0.8 * w
        return min(1.0, total / KW_SATURATION)

    def classify(self, text: str, lang: str = "en") -> IntentResult:
        toks = tokens(text)
        if not toks:
            return IntentResult("unknown", None, 0.0, [])
        if len(toks) > MAX_CLASSIFY_TOKENS:
            half = MAX_CLASSIFY_TOKENS // 2
            toks = toks[:half] + toks[-half:]
            text = " ".join(toks)
        text_norm = normalize(text)
        vec = self._vectorize(text, lang)
        cents = self.family_centroids.get(lang) or self.family_centroids["en"]
        en_cents = self.family_centroids["en"]
        scores: list[tuple[str, float]] = []
        for fam in self.family_keywords:
            kw = self._keyword_score(text_norm, toks, fam)
            cs = cosine(vec, cents.get(fam) or en_cents.get(fam, {}))
            scores.append((fam, KW_WEIGHT * kw + VEC_WEIGHT * cs))
        scores.sort(key=lambda x: -x[1])
        if len(scores) > 1 and scores[0][0] in TRIAGE_FAMILIES \
                and scores[0][1] - scores[1][1] <= TRIAGE_TIE_BAND \
                and scores[1][0] not in TRIAGE_FAMILIES:
            runner_ok = (scores[0][0] == "general_help"
                         or scores[1][0] in RESP_FAMILIES)
            if runner_ok:
                scores[0], scores[1] = scores[1], scores[0]
        top_fam, top = scores[0]
        second = scores[1][1] if len(scores) > 1 else 0.0
        if top >= ACCEPT_THRESHOLD and (top - second) >= MARGIN:
            return IntentResult("intent", top_fam, top, scores[:3])
        if top >= CLARIFY_THRESHOLD:
            return IntentResult("clarify", top_fam, top, scores[:3])
        return IntentResult("unknown", None, top, scores[:3])
