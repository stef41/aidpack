"""Forbidden-advice linter and scope guard.

The linter sweeps rendered outputs for known-harmful folk remedies. Because
protocol texts legitimately *warn against* these practices, warning lines are
rendered with a fixed prefix (renderer contract: lines starting with the
warning marker) and excluded from the scan. The benchmark walks the entire
reachable output space through this linter — zero hits required.
"""
from __future__ import annotations

import re

WARNING_PREFIXES = ("do not", "don't", "dont ", "never", "warning", "caution", "important")

# Each rule: (name, pattern over normalized advice lines)
FORBIDDEN_RULES: list[tuple[str, re.Pattern]] = [
    ("butter_on_burn", re.compile(r"\b(?:apply|put|use|spread|rub|smear)\b.{0,40}\b(?:butter|toothpaste|margarine|lard|mayonnaise|egg white)\b.{0,30}\bburn")),
    ("ice_on_burn", re.compile(r"\b(?:apply|put|use|place|hold)\b.{0,20}\bice\b.{0,30}\bburn\b")),
    ("induce_vomiting", re.compile(r"\b(?:make|force|induce)\b.{0,20}\b(?:them|him|her|the (?:person|child|baby))?\s*(?:to )?(?:vomit|throw up|puke)\b|\bipecac\b")),
    ("head_back_nosebleed", re.compile(r"\b(?:tilt|lean|tip|put)\b.{0,20}\bhead\b.{0,15}\b(?:back|backward|backwards)\b(?=.{0,60}\b(?:nosebleed|nose ?bleed|bleeding nose|nose is bleeding)\b)|\b(?:nosebleed|nose ?bleed|nose is bleeding)\b.{0,60}\b(?:tilt|lean|tip)\b.{0,15}\bhead\b.{0,10}\bback\b")),
    ("suck_venom", re.compile(r"\bsuck\b.{0,25}\b(?:venom|poison|bite|wound)\b|\b(?:cut|slice)\b.{0,20}\b(?:bite|fang|wound)\b.{0,25}\bvenom\b")),
    ("tourniquet_snake", re.compile(r"\btourniquet\b.{0,40}\bsnake\b|\bsnake\b.{0,40}\btourniquet\b")),
    ("mouth_object_seizure", re.compile(r"\b(?:put|place|insert|stick|wedge)\b.{0,30}\b(?:in|into|between)\b.{0,15}\b(?:mouth|teeth)\b.{0,40}\bseiz")),
    ("restrain_seizure", re.compile(r"\b(?:hold|pin|restrain)\b.{0,15}\b(?:him|her|them)\b.{0,10}\bdown\b.{0,40}\bseiz")),
    ("infant_abdominal_thrust", re.compile(r"\babdominal thrusts?\b.{0,40}\b(?:baby|infant|newborn)\b|\b(?:baby|infant|newborn)\b.{0,40}\babdominal thrusts?\b")),
    ("remove_embedded", re.compile(r"\b(?:pull|remove|take|yank)\b.{0,15}\b(?:out\b.{0,25})?\b(?:the )?(?:knife|embedded object|impaled)\b.{0,10}\bout\b|\bpull the object out\b|\bremove the (?:knife|embedded object)\b")),
    ("rub_frostbite", re.compile(r"\b(?:rub|massage)\b.{0,30}\bfrost|\b(?:rub|massage)\b.{0,30}\bfrozen\b")),
    ("feed_unconscious", re.compile(r"\bgive\b.{0,40}\b(?:unconscious|unresponsive)\b.{0,25}\b(?:food|drink|water|sugar|pills?)\b")),
    ("urine_jellyfish", re.compile(r"\b(?:pee|urine|urinate)\b.{0,30}\b(?:sting|jellyfish)\b|\b(?:sting|jellyfish)\b.{0,30}\b(?:pee|urine|urinate)\b")),
    ("paper_bag", re.compile(r"\bbreathe?\b.{0,25}\bpaper bag\b|\bpaper bag\b.{0,25}\bbreath")),
    ("aspirin_child", re.compile(r"\bgive\b.{0,30}\baspirin\b.{0,30}\b(?:child|kid|baby|infant|toddler)\b|\b(?:child|kid|baby|infant|toddler)\b.{0,30}\bgive\b.{0,20}\baspirin\b")),
    ("burst_blister", re.compile(r"\b(?:pop|burst|puncture|drain|lance)\b.{0,20}\bblister")),
    ("heat_hypothermia", re.compile(r"\b(?:hot bath|heating pad|direct heat)\b.{0,40}\bhypothermia\b")),
    ("coffee_sober", re.compile(r"\b(?:coffee|cold shower)\b.{0,30}\bsober\b")),
    ("move_spinal", re.compile(r"\b(?:move|drag|carry|sit up)\b.{0,30}\b(?:spinal|spine|neck injury|back injury)\b.{0,20}\b(?:victim|patient|person)\b")),
    ("neutralize_chemical", re.compile(r"\bneutrali[sz]e\b.{0,30}\b(?:chemical|acid|alkali)\b.{0,30}\b(?:with|using)\b")),
    ("blind_finger_sweep", re.compile(r"\b(?:do|perform|use)\b.{0,10}\ba?\s?blind finger sweep\b")),
]


def _is_warning_line(line_norm: str) -> bool:
    stripped = line_norm.strip()
    return any(stripped.startswith(w) for w in WARNING_PREFIXES)


def lint_advice(text: str) -> list[str]:
    """Return list of violated rule names for a rendered assistant output."""
    from ..text import normalize
    violations: list[str] = []
    for raw_line in text.split("\n"):
        line = normalize(raw_line)
        if not line or _is_warning_line(line):
            continue
        for name, pat in FORBIDDEN_RULES:
            if pat.search(line):
                violations.append(name)
    return violations


# --- input-side folk-remedy counter-advice --------------------------------------

_FOLK_COUNTERS: list[tuple[str, re.Pattern, str]] = [
    ("induce_vomit", re.compile(r"\b(?:make|force|induce|should|gonna|going to)\b.{0,30}\b(?:vomit|throw up|puke)\b|\bipecac\b|\bvomit right now\b|\b(?:hacerl[oa]|que) vomite\b|\bl[oa] hago vomitar\b|\bfaire vomir\b|\ble faire vomir\b"),
     "Important: do NOT make them vomit — it can burn the airway or choke them. Only a poison expert can advise that."),
    ("butter_burn", re.compile(r"\b(?:butter|toothpaste|margarine|oil|egg)\b.{0,25}\bburn|\bburn\b.{0,25}\b(?:butter|toothpaste|margarine)\b|\bmantequilla\b.{0,30}\bquemad|\bquemad\w*\b.{0,30}\bmantequilla\b|\bbeurre\b.{0,30}\bbrul|\bbrul\w*\b.{0,30}\bbeurre\b"),
     "Important: never put butter, oil or toothpaste on a burn — cool running water only."),
    ("head_back", re.compile(r"\b(?:head|lean|tilt)\b.{0,15}\bback\b.{0,25}\bnose|\bnose(?:bleed)?\b.{0,25}\bhead back\b|\bcabeza\b.{0,15}\batras\b.{0,25}\bnariz|\bnariz\b.{0,25}\bcabeza\b.{0,12}\batras|\btete\b.{0,15}\barriere\b.{0,25}\bnez|\bnez\b.{0,25}\btete\b.{0,12}\barriere"),
     "Important: keep the head leaning FORWARD in a nosebleed — never tilt it back."),
    ("suck_venom", re.compile(r"\bsuck\b.{0,25}\b(?:venom|poison|snake|bite)\b|\bcut\b.{0,15}\bbite\b|\bchupar?\b.{0,25}\bveneno\b|\baspirer\b.{0,25}\bvenin\b"),
     "Important: do NOT suck or cut a snake bite — it does not remove venom and makes it worse."),
    ("mouth_seizure", re.compile(r"\b(?:spoon|wallet|something|anything)\b.{0,20}\b(?:in|into|between)\b.{0,15}\b(?:mouth|teeth)\b|\bmouth\b.{0,30}\bseiz|\bseiz\w*\b.{0,40}\b(?:in|into) (?:the|his|her|their) mouth\b|\bcuchara\b.{0,25}\bboca\b|\bboca\b.{0,30}\bconvulsion|\bcuillere\b.{0,25}\bbouche\b|\bbouche\b.{0,30}\bconvuls"),
     "Important: never put anything in the mouth of someone having a seizure — they cannot swallow their tongue."),
    ("ice_burn", re.compile(r"\bice\b.{0,20}\bburn|\bburn\b.{0,20}\bice\b|\bhielo\b.{0,25}\bquemad|\bquemad\w*\b.{0,25}\bhielo\b|\bglace\b.{0,25}\bbrul|\bbrul\w*\b.{0,25}\bglace\b"),
     "Important: no ice on burns — it deepens the damage. Cool running water only."),
    ("urine_jellyfish", re.compile(r"\b(?:pee|urine|urinate)\b.{0,25}\b(?:sting|jellyfish)|\bjellyfish\b.{0,25}\b(?:pee|urine)\b|\borinar?\b.{0,25}\bmedusa|\bmedusa\b.{0,25}\borin|\bmeduse\b.{0,25}\burine"),
     "Important: do not use urine on a jellyfish sting — rinse with seawater instead."),
    ("paper_bag", re.compile(r"\bpaper bag\b|\bbolsa de papel\b|\bsac en papier\b"),
     "Important: breathing into a paper bag is no longer recommended — slow coached breathing is safer."),
    ("rub_frostbite", re.compile(r"\b(?:rub|massage)\b.{0,20}\b(?:frostbit|frozen)|\bfrotar?\b.{0,25}\bcongelad|\bfrotter\b.{0,25}\bgel"),
     "Important: never rub frostbitten skin — it destroys the tissue. Rewarm gently in warm water."),
    ("feed_unconscious", re.compile(r"\b(?:give|feed)\b.{0,25}\b(?:unconscious|unresponsive)\b.{0,20}\b(?:food|drink|water|sugar)\b|\bdarle?\b.{0,20}\b(?:agua|comida)\b.{0,25}\binconsciente|\bdonner\b.{0,20}\b(?:eau|a manger|a boire)\b.{0,25}\binconscient"),
     "Important: never give food or drink to someone who is not fully awake — they can choke."),
    ("tourniquet_snake", re.compile(r"\btourniquet\b.{0,25}\bsnake\b|\bsnake\b.{0,25}\btourniquet\b|\btorniquete\b.{0,25}\b(?:serpiente|vibora)|\bgarrot\b.{0,25}\bserpent"),
     "Important: no tourniquet on a snake bite — keep the limb still and get help."),
    ("butter_direct", re.compile(r"\bput butter on (?:it|him|her|the|that)\b|\ble pongo mantequilla\b|\bje mets du beurre\b"),
     "Important: never put butter, oil or toothpaste on a burn — cool running water only."),
    ("milk_poison", re.compile(r"\b(?:drink|give|gave|make .{0,12}drink)\b.{0,15}\bmilk\b.{0,35}\b(?:poison|neutrali|chemical|bleach|swallow)|\b(?:poison|chemical|bleach)\b.{0,35}\b(?:give|drink)\b.{0,10}\bmilk\b|\bleche\b.{0,30}\bveneno|\bveneno\b.{0,30}\bleche\b|\blait\b.{0,30}\bpoison|\bpoison\b.{0,30}\blait\b"),
     "Important: do not give milk (or anything else) after a poisoning unless poison control tells you to."),
    ("shake_baby", re.compile(r"\bshake\b.{0,25}\b(?:baby|infant)\b|\bshake\b.{0,20}\b(?:him|her|them)\b.{0,15}\b(?:awake|to wake)|\bsacud\w*\b.{0,20}\bbebe\b|\bsecouer?\b.{0,20}\bbebe\b"),
     "Important: never shake a person — and NEVER shake a baby; it can cause brain injury. Tap the foot or shoulder and shout instead."),
    ("slap_awake", re.compile(r"\bslap\b.{0,20}\b(?:face|awake|wake)|\bcachetada\b|\babofetear\b|\bgifler?\b.{0,20}\breveiller"),
     "Important: do not slap someone to wake them — shout and tap the shoulder firmly instead."),
    ("alcohol_bath", re.compile(r"\balcohol\b.{0,15}\b(?:bath|rub|sponge)\b|\brubbing alcohol\b.{0,30}\b(?:fever|cool|child|kid)|\bbano de alcohol\b|\bbain d alcool\b"),
     "Important: never sponge or bathe a feverish person with alcohol — it can be absorbed and poison them. Remove layers and use a lukewarm cloth."),
    ("tobacco_wound", re.compile(r"\b(?:tobacco|mud|spit)\b.{0,30}\b(?:sting|wound|bite|cut)|\btabaco\b.{0,25}\b(?:picadura|herida)|\btabac\b.{0,25}\b(?:piqure|plaie)"),
     "Important: do not pack tobacco, mud or saliva on a wound or sting — it causes infection. Wash with clean water instead."),
    ("sweat_fever", re.compile(r"\bsweat(?:s|ing)? (?:it|the fever) out\b|\bblankets?\b.{0,30}\bsweat|\bsudar la fiebre\b|\bque sude\b|\btranspirer la fievre\b"),
     "Important: do not bundle a feverish person to sweat it out — overheating makes it worse. Remove layers instead."),
    ("remove_helmet", re.compile(r"\b(?:take|remove|get|pull)\b.{0,20}\bhelmet\b|\b(?:quitar?|sacar?)\b.{0,15}\bcasco\b|\b(?:enlever?|retirer?)\b.{0,15}\bcasque\b"),
     "Important: after a crash, leave the helmet on unless it blocks their breathing — removing it can worsen a neck injury."),
    ("peroxide_wound", re.compile(r"\bperoxide\b.{0,35}\b(?:wound|cut|clean|pour)|\b(?:pour|put)\b.{0,20}\bperoxide\b|\bagua oxigenada\b.{0,30}\bherida|\beau oxygenee\b.{0,30}\bplaie"),
     "Important: do not pour hydrogen peroxide into a wound — it damages the tissue. Rinse with clean running water instead."),
    ("blind_sweep", re.compile(r"\bfinger sweep\b|\b(?:sweep|reach)\b.{0,25}\bmouth\b|\breach (?:in|into)\b.{0,25}\b(?:mouth|throat)|\bmeter? (?:el|los) dedos?\b.{0,20}\b(?:boca|garganta)\b"),
     "Important: never do a blind finger sweep — only remove something from the mouth if you can clearly see it."),
    ("aspirin_child", re.compile(r"\baspirin\b.{0,30}\b(?:child|kid|toddler|baby|infant|fever)\b|\b(?:child|kid|toddler|baby|infant)\b.{0,25}\baspirin\b|\b\d{1,2} year old\b.{0,20}\baspirin\b|\baspirina\b.{0,25}\b(?:nin[oa]|bebe|fiebre)\b|\baspirine\b.{0,25}\b(?:enfant|bebe|fievre)\b"),
     "Important: never give aspirin to anyone under 16 — it can cause Reye's syndrome. Use paracetamol or ibuprofen at the packet dose instead."),
    ("direct_ice", re.compile(r"\b(?:directly|straight) on(?:to)?\b.{0,12}\bice\b|\bon ice cubes\b|\bdirecto (?:en|sobre) (?:el )?hielo\b|\bdirectement sur (?:la )?glace\b"),
     "Important: never put a severed part or bare skin directly on ice — wrap it in damp cloth, seal it in a bag, and put the bag on ice."),
    ("raw_meat", re.compile(r"\braw (?:steak|meat)\b.{0,30}\b(?:eye|bruise|swelling)|\bcarne cruda\b.{0,25}\bojo\b|\bviande crue\b.{0,25}\boeil\b"),
     "Important: do not put raw meat on an injury — it breeds bacteria. Use a wrapped cold pack instead."),
]


def folk_remedy_counter(text_norm: str) -> tuple[str, str] | None:
    """If the USER proposes a known-harmful folk remedy, return
    (counter_key, english_counter_text) so callers can localize by key."""
    for key, pat, counter in _FOLK_COUNTERS:
        if pat.search(text_norm):
            return key, counter
    return None


# --- scope guard ---------------------------------------------------------------

_SCOPE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("diagnosis", re.compile(
        r"\bwhat (?:disease|illness|condition) (?:do|does|is)\b|\bdiagnose\b|"
        r"\bis (?:it|this) cancer\b|\bdo i have (?!to)\b")),
    ("prescription", re.compile(
        r"\b(?:what|which|how much|how many)\b.{0,30}\b(?:antibiotic|dose|dosage|mg|milligrams?)\b|"
        r"\btell me\b.{0,35}\b(?:dose|dosage)\b|"
        r"\b(?:amoxicillin|codeine|morphine|antibiotic|insulin|penicillin|oxycodone|tramadol)\b.{0,25}\bdose\b|"
        r"\bdose\b.{0,30}\b(?:amoxicillin|codeine|morphine|antibiotic|insulin|penicillin)\b|"
        r"\bcan i take\b.{0,30}\b(?:together|with)\b|\bprescri(?:be|ption)\b")),
    ("self_harm", re.compile(
        r"\b(?:kill|hurt|harm) (?:myself|themselves|himself|herself)\b|\bsuicid|"
        r"\bwant to die\b|\bend (?:my|his|her|their) life\b|\bself harm\b|\bcut (?:myself|themselves)\b")),
    ("veterinary", re.compile(r"\bmy (?:dog|cat|pet|horse|bird|rabbit|hamster)\b")),
]

SCOPE_RESPONSES = {
    "diagnosis": (
        "I can't diagnose conditions — I'm a first-aid guide. I can help you handle "
        "symptoms right now and tell you when professional care is needed. "
        "Describe what's happening and I'll guide you."
    ),
    "prescription": (
        "I can't advise on medication choices or doses beyond standard first-aid "
        "measures — a doctor, pharmacist, or poison center must guide that. "
        "If this is urgent, call emergency services or your local poison line. "
        "Is someone having symptoms right now that I can help with?"
    ),
    "self_harm": (
        "I'm really glad you told me. You deserve support right now. If you are in "
        "immediate danger, call your local emergency number. You can also reach a "
        "crisis line — in the US, call or text 988; elsewhere, your local emergency "
        "number can connect you. If someone has been injured, tell me and I'll guide "
        "you through the first aid, step by step. You are not alone."
    ),
    "veterinary": (
        "I'm trained for human first aid — for animals, please contact a veterinarian "
        "or an animal poison line. If a person is also hurt, tell me and I'll help."
    ),
}


def scope_guard(text_norm: str) -> str | None:
    """Return a scope key if the request is out of first-aid scope."""
    for key, pat in _SCOPE_PATTERNS:
        if pat.search(text_norm):
            return key
    return None
