"""Dialog-act detection: yes/no/unsure/done/repeat/stop and friends.

Runs before intent classification when the engine awaits an answer.
Must be robust to noise ("yeah i think so", "no not really", "ok done").
"""
from __future__ import annotations

import re

from ..text import normalize, tokens

YES_WORDS = {
    "yes", "yeah", "yep", "yup", "ya", "aye", "correct", "right", "affirmative",
    "definitely", "certainly", "absolutely", "sure", "indeed", "totally", "si", "oui",
    "claro", "correcto", "ouais", "exact",
}
NO_WORDS = {"no", "nope", "nah", "negative", "none", "non"}
UNSURE_PHRASES = [
    "not sure", "i do not know", "i dont know", "dont know", "do not know", "unsure",
    "cannot tell", "cant tell", "hard to tell", "hard to say", "maybe", "possibly",
    "i think so", "kind of", "sort of", "not certain", "no idea", "i guess",
    "perhaps", "difficult to say", "unclear",
    "no se", "no lo se", "no estoy seguro", "no estoy segura", "ni idea",
    "quizas", "tal vez", "je ne sais pas", "sais pas", "aucune idee",
    "peut etre", "pas sur", "pas sure",
]
DONE_PHRASES = [
    "done", "ok done", "did it", "did that", "next", "next step", "ok next",
    "finished", "completed", "what now", "what next", "now what", "then what",
    "ok now what", "i did it", "we did it", "got it done", "that is done", "and now",
    "keep going", "continue", "go on", "whats next", "ready",
    "listo", "lista", "hecho", "ya esta", "siguiente", "que sigue", "continua",
    "fait", "c est fait", "ca y est", "termine", "suivant", "et apres",
]
REPEAT_PHRASES = [
    "repeat", "say that again", "say again", "come again", "what was that",
    "one more time", "did not hear", "didnt hear", "didnt catch", "did not catch",
    "can you repeat", "repeat that", "again please", "sorry what", "pardon",
    "repite", "repita", "otra vez", "de nuevo", "no escuche",
    "repete", "repetez", "encore une fois",
]
STOP_PHRASES = [
    "stop", "cancel", "never mind", "nevermind", "forget it", "quit", "exit",
    "we are done", "all good now", "no more help", "thats all", "that is all",
    "detente", "cancela", "olvidalo", "annule", "laisse tomber",
]
HELP_ARRIVED_PHRASES = [
    "ambulance is here", "paramedics are here", "ems is here", "help arrived",
    "help is here", "the medics arrived", "paramedics arrived", "ambulance arrived",
    "they took over", "doctors have taken over", "hospital now",
    "llego la ambulancia", "llegaron los paramedicos", "ya llegaron",
    "los paramedicos estan aqui", "l ambulance est arrivee",
    "les secours sont la", "les pompiers sont la", "ils sont arrives",
]
GREETING_PHRASES = ["hello", "hi", "hey", "good morning", "good evening", "good afternoon",
                    "hola", "buenos dias", "buenas tardes", "bonjour", "bonsoir", "salut"]
THANKS_PHRASES = ["thank you", "thanks", "thank u", "thx", "cheers", "appreciate it",
                  "gracias", "muchas gracias", "merci", "merci beaucoup"]

CONTINUE_QUESTIONS = [
    "should i keep going", "do i keep going", "should i continue", "keep going?",
    "do i continue", "should i carry on", "am i doing it right", "is this right",
    "like this", "am i doing this right",
]

# Pure emotion, no clinical content — needs reassurance, not a reprompt.
DISTRESS_PHRASES = [
    "oh god", "oh my god", "oh no", "please hurry", "hurry up", "hurry",
    "i am scared", "so scared", "i am panicking", "i am freaking out",
    "i cannot do this", "i cant handle this", "this is bad", "this is so bad",
    "is he going to die", "is she going to die", "is he dying", "is she dying",
    "dont let him die", "dont let her die", "do not let him die", "do not let her die",
    "please please", "help me please", "please help me", "what do i do",
    "i dont know what to do", "i do not know what to do",
    "dios mio", "por dios", "apurate", "date prisa", "que hago",
    "no se que hacer", "se va a morir", "va a morir", "tengo miedo",
    "mon dieu", "oh mon dieu", "depechez vous", "que faire",
    "je ne sais pas quoi faire", "il va mourir", "elle va mourir",
    "j y arrive pas", "j ai peur",
]
# Clinical words that mean a "distress" utterance still carries content.
_CLINICAL_TOKENS = {
    "breathing", "breathe", "breath", "bleeding", "blood", "choking", "burn",
    "burned", "seizure", "chest", "unconscious", "collapsed", "pulse", "heart",
    "swelling", "vomiting", "drowning", "poison", "baby", "pregnant",
    "respira", "sangre", "sangrando", "atraganta", "quemadura", "convulsion",
    "pecho", "inconsciente", "bebe", "respire", "saigne", "etouffe", "brulure",
    "poitrine", "inconscient",
}

_ACK_ONLY = {"ok", "okay", "k", "kk", "alright", "fine", "good", "roger", "understood", "will do", "on it"}
ACK_PHRASES = [
    "i called", "called them", "i have called", "calling now", "on the phone",
    "they are on the way", "on the way", "ambulance is coming", "called 911",
    "called 112", "called an ambulance", "someone is calling", "someone called",
    "called", "ya llame", "llame ya", "estan en camino", "deja llame",
    "j ai appele", "ils arrivent", "deja llamamos", "ya llamamos",
]


def _has_phrase(t: str, phrases: list[str]) -> bool:
    padded = f" {t} "
    return any(f" {p} " in padded for p in phrases)


def detect_dialog_act(text: str, awaiting_answer: bool) -> str | None:
    """Return one of: yes, no, unsure, done, repeat, stop, help_arrived,
    greeting, thanks — or None if the utterance is contentful (route to NLU)."""
    t = normalize(text)
    if not t:
        return "repeat"
    toks = tokens(text)
    n = len(toks)

    if _has_phrase(t, HELP_ARRIVED_PHRASES):
        return "help_arrived"
    if _has_phrase(t, REPEAT_PHRASES):
        return "repeat"
    if n <= 4 and _has_phrase(t, STOP_PHRASES):
        return "stop"

    if n <= 8 and _has_phrase(t, CONTINUE_QUESTIONS):
        return "confirm_continue"
    if n <= 10 and _has_phrase(t, UNSURE_PHRASES):
        return "unsure"
    if n <= 10 and _has_phrase(t, DISTRESS_PHRASES) \
            and not any(w in _CLINICAL_TOKENS for w in toks):
        return "distress"

    if awaiting_answer:
        # Leading token decides first: "yes hes an adult no allergies" => yes.
        if toks and toks[0] in YES_WORDS:
            return "yes"
        if toks and (toks[0] in NO_WORDS or toks[0] == "not"):
            return "no"
        # Negation-first for short answers: "i dont think so" => no.
        has_no = any(w in NO_WORDS for w in toks) or " not " in f" {t} "
        has_yes = any(w in YES_WORDS for w in toks)
        if has_no and n <= 7:
            return "no"
        if has_yes and n <= 7:
            return "yes"
        # Bare acknowledgments are NOT answers to safety questions.
        if n <= 3 and toks and " ".join(toks) in _ACK_ONLY:
            return "ack"
        if n <= 6 and _has_phrase(t, ACK_PHRASES):
            return "ack"

    if _has_phrase(t, DONE_PHRASES) and n <= 6:
        return "done"
    if n <= 3 and toks and " ".join(toks) in _ACK_ONLY:
        return "done"
    if n <= 6 and _has_phrase(t, ACK_PHRASES):
        return "done"

    if n <= 3 and _has_phrase(t, THANKS_PHRASES):
        return "thanks"
    if n <= 3 and _has_phrase(t, GREETING_PHRASES):
        return "greeting"
    return None
