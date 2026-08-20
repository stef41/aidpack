"""Red-flag scanner: life-threat signals that preempt any conversation state.

Runs on EVERY user input regardless of dialog state. A hit escalates to the
mapped protocol family immediately (same turn), unless the active protocol
already covers it (e.g. already in CPR when user repeats "not breathing").
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class RedFlag:
    pattern: re.Pattern
    target_family: str
    label: str


# NOTE: patterns run on normalized text (lowercase, contractions expanded).
_RED_FLAGS: list[RedFlag] = [
    RedFlag(re.compile(
        r"\b(?:not|is not|stopped|has stopped|no longer|quit) breathing\b|"
        r"\bno breathing\b|\bbreathing stopped\b|\bno pulse\b|\bheart stopped\b|"
        r"\bnot breathe\b|\bcardiac arrest\b|"
        r"\bno respira\b|\bdejo de respirar\b|\bya no respira\b|\bsin pulso\b|"
        r"\bparo cardiaco\b|\bne respire (?:pas|plus)\b|\ba arrete de respirer\b|"
        r"\bpas de pouls\b|\barret cardiaque\b"), "cardiac_arrest", "not breathing"),
    RedFlag(re.compile(
        r"\bgasping\b|\bagonal\b|\bturning blue\b|"
        r"\blips (?:are |went |going |turning )?(?:blue|gray|grey)\b|"
        r"\bgone blue\b|\bface is blue\b|"
        r"\besta morad[oa]\b|\blabios morados\b|\bse pone azul\b|"
        r"\b(?:il|elle) est bleue?\b|\blevres bleues\b|\bdevient bleue?\b"), "cardiac_arrest", "agonal/cyanosis"),
    RedFlag(re.compile(
        r"\bunconscious\b|\bunresponsive\b|\bpassed out\b|\bwill not wake\b|"
        r"\bwont wake\b|\bnot waking\b|\bnot responding\b|\bno response\b|"
        r"\bout cold\b|\bcollapsed\b|\bjust collapsed\b|\bwent limp\b|"
        r"\bgone limp\b|\bnot moving\b|\bstopped moving\b|"
        r"\binconsciente?\b|\bno responde\b|\bno despierta\b|\bse desplomo\b|"
        r"\bno se mueve\b|\bdesmayad[oa] y no\b|"
        r"\binconscient\b|\bne repond pas\b|\bne se reveille pas\b|"
        r"\bs est effondre\b|\bne bouge pas\b"), "unconscious", "unresponsive"),
    RedFlag(re.compile(
        r"\bchoking\b|\bcannot breathe .{0,20}stuck\b|\bsomething stuck in (?:his|her|their|my) throat\b|"
        r"\batragant\w*\b|\bs etouffe\b|\betouffe avec\b"),
        "choking", "choking"),
    RedFlag(re.compile(
        r"\bbleeding (?:heavily|badly|a lot|everywhere|uncontrollably)\b|"
        r"\bblood (?:is )?(?:spurting|gushing|pouring)\b|\bbleeding will not stop\b|"
        r"\bbleeding wont stop\b|\bsoaked in blood\b|\bblood everywhere\b|"
        r"\bsangra(?:ndo)? (?:mucho|muchisimo)\b|\bno para de sangrar\b|"
        r"\bsangre por todos lados\b|\bchorro de sangre\b|"
        r"\bsaigne (?:beaucoup|enormement)\b|\bn arrete pas de saigner\b|"
        r"\bsang partout\b|\bsang (?:qui )?gicle\b"),
        "severe_bleeding", "severe bleeding"),
    RedFlag(re.compile(
        r"\bthroat\b(?:(?!\bnot\b|\bnever\b).){0,20}\b(?:clos|swell|tight)|"
        r"\btongue (?:is )?swell|\banaphyla|"
        r"\bse (?:le |me )?cierra la garganta\b|\blengua hinchada\b|\bgarganta cerrada\b|"
        r"\bgorge (?:qui )?se ferme\b|\blangue (?:qui )?gonfle\b|\banafilax"),
        "anaphylaxis", "anaphylaxis"),
    RedFlag(re.compile(
        r"\bseizure\b|\bconvulsing\b|\bconvulsion\b|\bfitting\b|\bhaving a fit\b|"
        r"\bconvulsion(?:es|ando)?\b|\bconvulsiona(?:ndo)?\b|\bataque epileptico\b|"
        r"\bconvulse\b|\bcrise d epilepsie\b"),
        "seizure", "seizure"),
    RedFlag(re.compile(
        r"\bface (?:is )?droop|\bslurr(?:ed|ing) speech\b|\bone side (?:is )?(?:weak|numb|droop)|"
        r"\bcara (?:torcida|caida)\b|\bderrame cerebral\b|\bembolia\b|"
        r"\bvisage (?:qui )?tombe\b|\bvisage tombant\b|\bbouche de travers\b|\bavc\b"),
        "stroke", "stroke signs"),
    RedFlag(re.compile(
        r"\bcrushing chest\b|\bchest (?:pain|pressure|tightness|crushing)\b|"
        r"\bchest (?:feels|feeling) (?:crushed|tight|heavy|like)\b|"
        r"\bchest is crushing\b|\bcrushing (?:feeling|pain) in (?:my|his|her|the) chest\b|"
        r"\bdolor (?:en el|de) pecho\b|\bopresion en el pecho\b|\binfarto\b|"
        r"\bdouleur (?:a|dans) la poitrine\b|\boppression (?:dans|a) la poitrine\b|"
        r"\bcrise cardiaque\b|\binfarctus\b"),
        "heart_attack", "chest pain"),
]

# Escalation is suppressed when already inside a protocol that handles the flag.
_COVERED_BY: dict[str, set[str]] = {
    "cardiac_arrest": {"cardiac_arrest", "choking"},
    "unconscious": {"cardiac_arrest", "unconscious", "overdose", "alcohol_poisoning",
                    "heat_stroke", "hypothermia", "drowning", "electric_shock",
                    "lightning", "carbon_monoxide", "fainting", "poisoning", "stroke",
                    "seizure", "febrile_seizure", "diabetic", "shock", "choking"},
    "choking": {"choking", "cardiac_arrest"},
    "severe_bleeding": {"severe_bleeding", "amputation", "chest_wound", "abdominal_wound"},
    "anaphylaxis": {"anaphylaxis"},
    "seizure": {"seizure", "febrile_seizure"},
    "stroke": {"stroke"},
    "heart_attack": {"heart_attack", "cardiac_arrest"},
}


_PRONOUNS = {"he", "she", "they", "it", "i", "we", "you", "hes", "shes"}
_NEGATORS = {"no", "not", "without", "denies", "denied", "never", "nobody"}


def _negated(text: str, start: int) -> bool:
    """True if the match at `start` is negated within the last 4 words
    ("no chest pain", "she is not having a seizure", "no one is unconscious").
    A negator followed directly by a pronoun ("wait no he stopped breathing")
    is a self-correction, not a symptom negation."""
    words = text[:start].split()[-4:]
    for i, w in enumerate(words):
        if w in _NEGATORS:
            nxt = words[i + 1] if i + 1 < len(words) else None
            if nxt in _PRONOUNS:
                continue
            return True
    return False


def scan_red_flags(text_norm: str, active_family: str | None) -> RedFlag | None:
    """Return highest-priority uncovered, non-negated red flag in the input."""
    for rf in _RED_FLAGS:
        for m in rf.pattern.finditer(text_norm):
            if _negated(text_norm, m.start()):
                continue
            covered = _COVERED_BY.get(rf.target_family, {rf.target_family})
            if active_family in covered:
                break
            return rf
    return None


def covered_families(flag_family: str) -> set[str]:
    return _COVERED_BY.get(flag_family, {flag_family})


# Generic red-flag families defer to a confident specific classification
# within these sets (e.g. "seizure" signal + fever context -> febrile_seizure).
DEFERRALS: dict[str, set[str]] = {
    "unconscious": covered_families("unconscious"),
    "seizure": {"seizure", "febrile_seizure"},
}
