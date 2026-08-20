"""Text normalization and similarity primitives (pure stdlib, port-friendly).

Everything here is deterministic and locale-independent so that Kotlin/Swift
ports can reproduce results byte-for-byte from the exported test vectors.
"""
from __future__ import annotations

import re
import unicodedata

_CONTRACTIONS = {
    "can't": "cannot", "cant": "cannot", "won't": "will not", "wont": "will not",
    "isn't": "is not", "isnt": "is not", "aren't": "are not", "arent": "are not",
    "doesn't": "does not", "doesnt": "does not", "don't": "do not", "dont": "do not",
    "didn't": "did not", "didnt": "did not", "hasn't": "has not", "hasnt": "has not",
    "haven't": "have not", "havent": "have not", "wasn't": "was not", "wasnt": "was not",
    "he's": "he is", "she's": "she is", "it's": "it is", "there's": "there is",
    "hes": "he is", "shes": "she is", "theres": "there is",
    "they're": "they are", "theyre": "they are", "i'm": "i am", "im": "i am",
    "we're": "we are", "what's": "what is", "whats": "what is",
    "couldn't": "could not", "couldnt": "could not", "shouldn't": "should not",
    "wouldn't": "would not", "ain't": "is not", "aint": "is not",
    "od'd": "overdosed", "odd'd": "overdosed",
}

_SLANG = {
    "omg": "", "omfg": "", "pls": "please", "plz": "please", "thx": "thanks",
    "u": "you", "ur": "your", "r": "are", "n": "and", "w": "with", "thru": "through",
    "rn": "right now", "asap": "immediately", "sos": "help emergency",
    "od": "overdose", "oded": "overdosed", "kiddo": "child", "lil": "little",
    "bp": "blood pressure", "er": "emergency room", "ambo": "ambulance",
    "resus": "resuscitation", "defib": "defibrillator", "sum1": "someone",
    "babby": "baby", "hubby": "husband",
    "breeth": "breathe", "breth": "breathe", "brething": "breathing",
    "breething": "breathing", "choaking": "choking", "seizur": "seizure",
    "hart": "heart", "amublance": "ambulance",
}

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def squeeze_repeats(word: str, max_run: int = 2) -> str:
    """'heeeelp' -> 'heelp' (later fuzzy-matched to 'help')."""
    out: list[str] = []
    run = 0
    prev = ""
    for ch in word:
        if ch == prev:
            run += 1
        else:
            run = 1
            prev = ch
        if run <= max_run:
            out.append(ch)
    return "".join(out)


def normalize(text: str) -> str:
    """Canonical lowercase form used by all NLU components."""
    t = _ZERO_WIDTH_RE.sub("", text)
    t = strip_accents(t).lower()
    t = t.replace("&", " and ").replace("+", " and ").replace("/", " ")
    for k, v in _CONTRACTIONS.items():
        t = re.sub(rf"\b{re.escape(k)}\b", v, t)
    t = t.replace("'", " ").replace("\u2019", " ")  # French elisions: l'eau -> l eau
    words = _WORD_RE.findall(t)
    out: list[str] = []
    for w in words:
        w = squeeze_repeats(w)
        w = _SLANG.get(w, w)
        if w:
            out.extend(w.split())
    return " ".join(out)


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def char_ngrams(text: str, lo: int = 3, hi: int = 5) -> dict[str, int]:
    """Char n-gram bag over ' '-padded normalized text (robust to typos/segmentation)."""
    padded = f" {normalize(text)} "
    bag: dict[str, int] = {}
    for n in range(lo, hi + 1):
        for i in range(len(padded) - n + 1):
            g = padded[i : i + n]
            bag[g] = bag.get(g, 0) + 1
    return bag


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b[k] for k, v in a.items() if k in b)
    if dot == 0.0:
        return 0.0
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    return dot / (na * nb)


def damerau_levenshtein(a: str, b: str, cap: int = 3) -> int:
    """Edit distance with transpositions, early-exit above `cap`."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > cap:
        return cap + 1
    prev2: list[int] = []
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        row_min = i
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
            row_min = min(row_min, cur[j])
        if row_min > cap:
            return cap + 1
        prev2, prev = prev, cur
    return prev[lb]


def fuzzy_contains(haystack_tokens: list[str], word: str, max_dist: int | None = None) -> bool:
    """True if any token is within edit distance of `word` (distance scales with length).
    Simple plural forms always match regardless of length."""
    if max_dist is None:
        max_dist = 0 if len(word) <= 4 else (1 if len(word) <= 7 else 2)
    for t in haystack_tokens:
        if t == word or t == word + "s" or word == t + "s":
            return True
        if max_dist and abs(len(t) - len(word)) <= max_dist:
            if damerau_levenshtein(t, word, cap=max_dist) <= max_dist:
                return True
    return False


def phrase_in(text_norm: str, phrase_norm: str) -> bool:
    """Word-boundary phrase containment on normalized text."""
    return f" {phrase_norm} " in f" {text_norm} "
