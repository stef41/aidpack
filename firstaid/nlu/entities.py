"""Entity extraction: age group, consciousness, breathing status.

Regex + lexical-window based; deterministic and port-friendly.
Entities become session *facts* that gate protocol variant selection and
decision nodes, so extraction is deliberately conservative: emit a value
only on clear evidence.
"""
from __future__ import annotations

import re

from ..text import normalize

# --- age group ---------------------------------------------------------------

_INFANT_WORDS = r"(?:baby|babies|infant|newborn|new born|bebe|bebes|recien nacido|nourrisson)"
_CHILD_WORDS = r"(?:child|kid|toddler|son|daughter|boy|girl|preschooler|schoolkid|nino|nina|hijo|hija|enfant|fils|fille|petit|petite|gamin|gamine)"
_ADULT_WORDS = (
    r"(?:man|woman|guy|lady|adult|husband|wife|dad|father|mom|mother|grandma|"
    r"grandmother|grandpa|grandfather|colleague|coworker|neighbor|elderly|"
    r"old man|old woman|teenager|teen|hombre|mujer|senor|senora|esposo|esposa|"
    r"abuelo|abuela|marido|adulto|adulta|homme|femme|monsieur|madame|mari|"
    r"epouse|adulte|grand mere|grand pere)"
)

_AGE_NUM = re.compile(
    r"\b(\d{1,3})\s*(?:and a half\s*)?(year|yr|month|mo|week|wk|day)s?[\s-]*old\b"
    r"|\b(?:de\s+|tiene\s+|a\s+)?(\d{1,3})\s*(anos?|meses?|semanas?|ans?|mois|semaines?)\b"
)
_AGE_NUM_SHORT = re.compile(r"\b(\d{1,3})\s*(?:yo|y o)\b")

_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}
_WORD_AGE = re.compile(
    rf"\b({'|'.join(_NUMBER_WORDS)})\s+(year|month|week|day)s?[\s-]*old\b"
)


def _age_from_number(value: int, unit: str) -> str:
    unit = {"ano": "year", "anos": "year", "an": "year", "ans": "year",
            "mes": "month", "meses": "month", "mois": "month",
            "semana": "week", "semanas": "week", "semaine": "week",
            "semaines": "week"}.get(unit, unit)
    if unit.startswith(("month", "mo", "week", "wk", "day")):
        return "infant" if unit.startswith(("week", "wk", "day")) or value < 12 else "child"
    if value < 1:
        return "infant"
    if value <= 12:
        return "child"
    return "adult"


def extract_age_group(text: str) -> str | None:
    t = normalize(text)
    m = _AGE_NUM.search(t)
    if m:
        if m.group(1):
            return _age_from_number(int(m.group(1)), m.group(2))
        return _age_from_number(int(m.group(3)), m.group(4))
    m = _WORD_AGE.search(t)
    if m:
        return _age_from_number(_NUMBER_WORDS[m.group(1)], m.group(2))
    m = _AGE_NUM_SHORT.search(t)
    if m:
        return _age_from_number(int(m.group(1)), "year")
    if re.search(rf"\b{_INFANT_WORDS}\b", t):
        return "infant"
    if re.search(rf"\bmy {_CHILD_WORDS}\b|\b(?:a|the|this) {_CHILD_WORDS}\b|\b{_CHILD_WORDS} is\b|\blittle (?:one|girl|boy)\b|\btoddler\b|\bkid\b|\bchild\b", t):
        return "child"
    if re.search(rf"\b{_ADULT_WORDS}\b", t):
        return "adult"
    return None


# --- consciousness ------------------------------------------------------------

_UNCONSCIOUS = re.compile(
    r"\bunconscious\b|\bunresponsive\b|\bpassed out\b|\bblacked out\b|"
    r"\bknocked out\b|\bout cold\b|\bwill not wake\b|\bwont wake\b|"
    r"\bnot waking\b|\bdoes not respond\b|\bnot responding\b|\bno response\b|"
    r"\bdoes not react\b|\bnot reacting\b|\blimp and\b|\bwent limp\b|"
    r"\bgone limp\b|\bis floppy\b|\bwent floppy\b|\bfloppy and\b|\blifeless\b|"
    r"\bcollapsed\b|\bnot moving\b|\bstopped moving\b|\bcannot wake (?:him|her|them) up\b|"
    r"\binconsciente?\b|\bno responde\b|\bno reacciona\b|\bno despierta\b|"
    r"\bdesmayad[oa]\b|\bse desplomo\b|\bno se mueve\b|"
    r"\binconscient\b|\bne repond pas\b|\bne reagit pas\b|\bne se reveille pas\b|"
    r"\bevanouie?\b|\bs est effondre\b|\bne bouge pas\b"
)
_CONSCIOUS = re.compile(
    r"\bis awake\b|\bwoke up\b|\bcame (?:to|around|round)\b|\bis alert\b|"
    r"\bis talking\b|\bis crying\b|\bresponds\b|\bis responding\b|\bconscious now\b|"
    r"\bis conscious\b|\bstill awake\b|"
    r"\besta despiert[oa]\b|\bya desperto\b|\besta consciente\b|\bsi responde\b|"
    r"\best reveille\b|\best reveillee\b|\b(?:il|elle) repond\b|\best conscient\b|\best consciente\b"
)


def extract_consciousness(text: str) -> str | None:
    t = normalize(text)
    if _UNCONSCIOUS.search(t):
        return "unresponsive"
    if _CONSCIOUS.search(t):
        return "responsive"
    return None


# --- breathing ----------------------------------------------------------------

_NOT_BREATHING = re.compile(
    r"\b(?:not|is not|stopped|has stopped|no longer|quit|barely|hardly|scarcely) breathing\b|"
    r"\bno breathing\b|\bbreathing stopped\b|\bno breath\b|\bwithout breathing\b|"
    r"\bcannot tell if .{0,20}breathing\b|\bnot breathe\b|\bno pulse\b|"
    r"\bno respira\b|\bdejo de respirar\b|\bno esta respirando\b|\bsin pulso\b|"
    r"\bya no respira\b|\bne respire (?:pas|plus)\b|\ba arrete de respirer\b|"
    r"\bpas de pouls\b|\bplus de pouls\b"
)
# Agonal-type signals always mean abnormal (arrest-adjacent).
_HARD_ABNORMAL = re.compile(
    r"(?<!no )(?<!not )\bgasping\b|\bagonal\b|\bsnoring breaths\b|"
    r"\bgasps? (?:now and then|sometimes|occasionally)\b"
)
# Soft cues (slow/shallow/odd) are overridden by an explicit "normal".
_SOFT_ABNORMAL = re.compile(
    r"\bweird breath\b|\bstrange breathing\b|"
    r"\bbreathing (?:weird|strange|funny|odd|shallow)\b|"
    r"\bbreathing\b[^.]{0,20}\bslow(?:ly)?\b|\bslow(?:ly)? breathing\b|"
    r"\bshallow breathing\b|\birregular breath\b"
)
_EXPLICIT_NORMAL = re.compile(r"\bnormal(?:ly)?\b")
_NOT_NORMAL = re.compile(r"\b(?:not|is not|no) normal\b|\babnormal\b")
_BREATHING_OK = re.compile(
    r"\bis breathing\b|\bbreathing normally\b|\bbreathing fine\b|\bbreathing ok\b|"
    r"\bbreathes normally\b|\bstill breathing\b|\bbreathing on (?:his|her|their) own\b|"
    r"\b(?:but|and) breathing\b|"
    r"\brespira (?:bien|normal)\b|\besta respirando\b|\bsi respira\b|"
    r"\brespire (?:bien|normalement)\b|\b(?:il|elle) respire\b|\bpero respira\b|\bmais (?:il|elle) respire\b"
)


def extract_breathing(text: str) -> str | None:
    t = normalize(text)
    if _NOT_BREATHING.search(t):
        return "not_breathing"
    if _HARD_ABNORMAL.search(t):
        return "abnormal"
    if _SOFT_ABNORMAL.search(t):
        if _EXPLICIT_NORMAL.search(t) and not _NOT_NORMAL.search(t):
            return "breathing"
        return "abnormal"
    if _BREATHING_OK.search(t):
        return "breathing"
    return None


def extract_entities(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    age = extract_age_group(text)
    if age:
        out["age_group"] = age
    consciousness = extract_consciousness(text)
    if consciousness:
        out["consciousness"] = consciousness
    breathing = extract_breathing(text)
    if breathing:
        out["breathing"] = breathing
    # Worst-case inference: someone not breathing (or only gasping) is treated
    # as unresponsive unless stated otherwise — matches resuscitation guidance.
    if out.get("breathing") in ("not_breathing", "abnormal") and "consciousness" not in out:
        out["consciousness"] = "unresponsive"
    return out
