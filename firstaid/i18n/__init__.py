"""Offline multilingual support: language packs + detection.

Design:
- Language packs are pure data (firstaid/i18n/<lang>/): input lexicons extend
  the NLU for ALL protocol families; output packs translate system strings and
  the time-critical (tier-1) protocols. Untranslated protocols fall back to
  English behind a translated notice + translated EMS banner — honest and safe.
- Detection is stopword/marker-based, sticky per session (no mid-emergency
  flip-flopping), and biased toward the current language on short inputs.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

I18N_DIR = os.path.dirname(os.path.abspath(__file__))
SUPPORTED = ("en", "es", "fr")

# Accent-stripped function words / markers (inputs are normalized before scoring).
_MARKERS: dict[str, set[str]] = {
    "es": {"el", "la", "los", "las", "un", "una", "esta", "estoy", "es", "se",
           "mi", "hijo", "hija", "por", "favor", "ayuda", "que", "hago", "tiene",
           "y", "con", "del", "muy", "socorro", "no", "puede", "respira",
           "me", "te", "lo", "le", "pero", "tengo", "duele", "cuando"},
    "fr": {"le", "la", "les", "un", "une", "est", "je", "il", "elle", "mon",
           "ma", "au", "aidez", "moi", "que", "faire", "sil", "vous", "plait",
           "ne", "pas", "plus", "avec", "du", "tres", "secours", "respire",
           "mal", "tombe", "chez", "suis", "sa", "ses", "apres", "dans"},
    "en": {"the", "is", "he", "she", "my", "a", "an", "please", "help", "what",
           "do", "i", "and", "with", "very", "not", "can", "cannot", "breathing",
           "on", "in", "his", "her", "has", "was"},
}
# Words unique enough to be decisive on their own.
_STRONG: dict[str, set[str]] = {
    "es": {"ayuda", "socorro", "respira", "atragantando", "sangrando", "quemadura",
           "convulsiones", "inconsciente", "ahogando", "estoy", "hago",
           "nino", "nina", "bebe", "trago", "tomo", "cayo", "desmayo", "veneno",
           "pecho", "pierna", "herida", "esposo", "esposa", "abuela", "abuelo"},
    "fr": {"aidez", "secours", "respire", "etouffe", "saigne", "brulure",
           "convulsions", "inconscient", "noie", "sil", "plait",
           "avale", "javel", "poitrine", "jambe", "blessure", "mari", "femme",
           "tombee", "effondre", "evanouie", "evanoui"},
    "en": {"help", "breathing", "choking", "bleeding", "burn", "seizure",
           "unconscious", "drowning"},
}


def detect_language(text_norm: str, current: str = "en") -> str:
    """Return the detected language, sticky toward `current` on weak evidence."""
    toks = text_norm.split()
    if not toks:
        return current
    scores = {lang: 0.0 for lang in SUPPORTED}
    for t in toks:
        for lang in SUPPORTED:
            if t in _STRONG[lang]:
                scores[lang] += 3.0
            elif t in _MARKERS[lang]:
                scores[lang] += 1.0
    best = max(scores, key=lambda k: scores[k])
    best_score = scores[best]
    if best == current or best_score == 0.0:
        return current
    # switch only on clear evidence: >= 2 points and strictly ahead of current.
    # (a lone shared token like "no" must never flip an English conversation)
    if best_score >= 2.0 and best_score > scores[current]:
        return best
    return current


@dataclass
class LanguagePack:
    lang: str
    strings: dict[str, str] = field(default_factory=dict)
    scope: dict[str, str] = field(default_factory=dict)
    counters: dict[str, str] = field(default_factory=dict)   # counter-key -> text
    protocols: dict[str, dict] = field(default_factory=dict)  # pid -> {nodes: {nid: {...}}}
    lexicon: dict[str, dict] = field(default_factory=dict)    # family -> {keywords, exemplars}

    def node_field(self, pid: str, nid: str, fld: str):
        return self.protocols.get(pid, {}).get("nodes", {}).get(nid, {}).get(fld)

    def proto_name(self, pid: str):
        return self.protocols.get(pid, {}).get("name")


_PACKS: dict[str, LanguagePack] | None = None


def load_packs() -> dict[str, LanguagePack]:
    global _PACKS
    if _PACKS is not None:
        return _PACKS
    packs: dict[str, LanguagePack] = {}
    for lang in SUPPORTED:
        if lang == "en":
            continue
        base = os.path.join(I18N_DIR, lang)
        if not os.path.isdir(base):
            continue
        pack = LanguagePack(lang)
        for name, attr in (("pack.json", None), ("lexicon.json", "lexicon")):
            path = os.path.join(base, name)
            if not os.path.isfile(path):
                continue
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if attr == "lexicon":
                pack.lexicon = data
            else:
                pack.strings = data.get("strings", {})
                pack.scope = data.get("scope", {})
                pack.counters = data.get("counters", {})
                pack.protocols = data.get("protocols", {})
        packs[lang] = pack
    _PACKS = packs
    return packs
