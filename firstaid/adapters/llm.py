"""Optional constrained LLM paraphrase layer (OFF by default).

An on-device LLM (llama.cpp GGUF) may soften the engine's fixed phrasing, but
must never author medical content. `LexicalGroundingValidator` enforces that:
a paraphrase is accepted only if every critical content token of the original
survives and no new medical tokens appear. On any doubt, the original text
ships unchanged — determinism is the default and the fallback.
"""
from __future__ import annotations

import re
from typing import Protocol as TypingProtocol

from ..text import tokens

# Tokens whose presence/absence changes medical meaning.
_CRITICAL_PATTERN = re.compile(
    r"^\d+$|^(?:not|no|never|do|dont|cannot|stop|call|push|press|pinch|tilt|"
    r"lean|cool|warm|hot|cold|ice|water|aspirin|epinephrine|naloxone|inhaler|"
    r"tourniquet|aed|cpr|compressions?|breaths?|thrusts?|blows?|seconds?|"
    r"minutes?|hours?|centimeters?|inches?|milligrams?|degrees?|left|right|"
    r"up|down|back|forward|under|above|below|hard|fast|gently|firmly)$"
)

_MEDICAL_NEW_TOKEN = re.compile(
    r"^(?:dose|doses|pill|pills|tablet|tablets|inject|injection|medicine|"
    r"medication|drug|drugs|antibiotic|antibiotics|morphine|codeine|paracetamol|"
    r"ibuprofen|aspirin|adrenaline|epinephrine|insulin|suck|cut|burn|bleed|"
    r"vomit|swallow|chew|drink|eat|apply|remove|massage|rub)$"
)


# Substances that must never be INTRODUCED by a paraphrase (folk remedies).
_HARMFUL_NEW_SUBSTANCE = re.compile(
    r"^(?:butter|toothpaste|margarine|lard|mayonnaise|urine|tobacco|mud|ipecac|"
    r"peroxide|iodine|kerosene|gasoline|bleach|vinegar|honey|milk|salt|sugar|"
    r"whiskey|vodka|brandy)$"
)


class Paraphraser(TypingProtocol):
    def paraphrase(self, text: str, style_hint: str) -> str: ...


class NullParaphraser:
    """Identity — the shipped default."""

    def paraphrase(self, text: str, style_hint: str = "") -> str:
        return text


class LexicalGroundingValidator:
    def validate(self, original: str, candidate: str) -> bool:
        orig_toks = tokens(original)
        cand_toks = tokens(candidate)
        orig_set, cand_set = set(orig_toks), set(cand_toks)
        # 1. all critical tokens preserved (multiset for numbers)
        orig_crit = [t for t in orig_toks if _CRITICAL_PATTERN.match(t)]
        for t in orig_crit:
            if orig_toks.count(t) > cand_toks.count(t):
                return False
        # 2. no new medical-action tokens introduced
        for t in cand_set - orig_set:
            if _MEDICAL_NEW_TOKEN.match(t) or _CRITICAL_PATTERN.match(t) \
                    or _HARMFUL_NEW_SUBSTANCE.match(t):
                return False
        # 3. length sanity (no runaway generation)
        if len(cand_toks) > int(len(orig_toks) * 1.5) + 8:
            return False
        return True


class ValidatedParaphraser:
    """Wrap any LLM paraphraser with the grounding validator."""

    def __init__(self, inner: Paraphraser):
        self.inner = inner
        self.validator = LexicalGroundingValidator()

    def paraphrase(self, text: str, style_hint: str = "") -> str:
        try:
            candidate = self.inner.paraphrase(text, style_hint)
        except Exception:
            return text
        return candidate if self.validator.validate(text, candidate) else text


LLM_SYSTEM_PROMPT = (
    "You rephrase emergency first-aid instructions to sound calmer and more "
    "natural. You MUST keep every action, number, body location, and warning "
    "EXACTLY as given. Never add, remove, or change any medical content. "
    "Output only the rephrased text."
)
