#!/usr/bin/env python3
"""Export deterministic test vectors for byte-exact Kotlin/Swift ports.

Produces deploy/test_vectors.json covering: normalization, fuzzy matching,
intent classification, entity extraction, dialog acts, and full conversation
transcripts. A port that reproduces these vectors is behaviorally equivalent.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from firstaid.engine import Session
from firstaid.kb import load_kb
from firstaid.nlu.dialog_acts import detect_dialog_act
from firstaid.nlu.entities import extract_entities
from firstaid.nlu.intents import IntentClassifier
from firstaid.text import char_ngrams, damerau_levenshtein, normalize

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "deploy", "test_vectors.json")

NORM_CASES = [
    "He's NOT breathing!!!", "my babby can't breeth", "OMG she OD'd on pills",
    "héllo wörld", "heeeelp meeee", "i think he's choking rn",
]
FUZZY_CASES = [("choking", "choaking"), ("bleeding", "bleding"), ("seizure", "seizur"),
               ("breathe", "breth"), ("tourniquet", "torniquet")]
INTENT_CASES = [
    "my dad collapsed and hes not breathing", "baby is choking on a grape",
    "deep cut bleeding everywhere", "she is having a seizure", "burned my hand on the stove",
    "wasp stung me and my throat is closing", "grandma slurring one side droopy",
    "help", "i cant breathe", "swallowed bleach",
]
ENTITY_CASES = [
    "my baby is not breathing", "my 8 year old broke his arm",
    "grandma is unconscious but breathing", "he is gasping strangely",
    "she woke up and is talking",
]
ACT_CASES = [("yes", True), ("no not really", True), ("not sure", True),
             ("done", False), ("say that again", False), ("the paramedics are here", False),
             ("ok", True), ("ok", False)]
CONVERSATIONS = [
    ["my husband collapsed and hes not breathing", "called", "done", "done", "done", "no", "done", "done"],
    ["my baby is choking", "no", "done", "done", "no", "done", "yes shes crying"],
    ["i burned my finger on a pan", "done", "done", "no", "done", "done"],
    ["should i put butter on his burn", "done"],
    ["help", "no hes not responding", "i cant tell if hes breathing"],
    ["mi esposo se desplomó y no respira", "ya llamé", "listo", "listo"],
    ["mon bébé s'étouffe", "non elle ne tousse pas", "fait"],
]
LANG_CASES = [
    "mi esposo se desplomó y no respira", "mon bébé s'étouffe",
    "my dad collapsed", "ayuda por favor", "aidez-moi il ne respire plus",
    "me torcí el tobillo corriendo", "no", "help me please",
]


def main() -> None:
    kb = load_kb()
    clf = IntentClassifier(kb)
    vectors: dict = {"version": 1}

    vectors["normalize"] = [{"in": t, "out": normalize(t)} for t in NORM_CASES]
    vectors["ngram_counts"] = [
        {"in": t, "n_grams": len(char_ngrams(t))} for t in NORM_CASES]
    vectors["edit_distance"] = [
        {"a": a, "b": b, "d": damerau_levenshtein(a, b)} for a, b in FUZZY_CASES]
    vectors["intents"] = [
        {"in": t, "kind": r.kind, "family": r.family, "score": round(r.score, 4)}
        for t in INTENT_CASES for r in [clf.classify(t)]]
    vectors["entities"] = [{"in": t, "out": extract_entities(t)} for t in ENTITY_CASES]
    vectors["dialog_acts"] = [
        {"in": t, "awaiting": aw, "act": detect_dialog_act(t, aw)} for t, aw in ACT_CASES]
    from firstaid.i18n import detect_language
    vectors["language_detection"] = [
        {"in": t, "current": "en", "out": detect_language(normalize(t), "en")}
        for t in LANG_CASES]

    convs = []
    for turns in CONVERSATIONS:
        s = Session(kb)
        rec = []
        for u in turns:
            r = s.handle(u)
            rec.append({"user": u, "lang": s.lang, "protocol": r.protocol_id,
                        "node": r.node_id, "kind": r.kind, "text": r.text})
        convs.append(rec)
    vectors["conversations"] = convs

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(vectors, f, indent=1, ensure_ascii=False)
    print(f"wrote {OUT}: {os.path.getsize(OUT)} bytes, "
          f"{sum(len(v) if isinstance(v, list) else 1 for v in vectors.values())} records")


if __name__ == "__main__":
    main()
