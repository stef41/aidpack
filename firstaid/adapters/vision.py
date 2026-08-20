"""Vision adapter: video/camera frames -> findings -> protocol suggestions.

Perception is *advisory* (safety model rule 5): the vision path never issues
medical guidance directly. It maps VLM output onto a closed finding vocabulary
derived from the KB, and the engine starts the suggested protocol whose entry
questions verbally confirm the situation.

On-device backend: SmolVLM2-500M/2.2B or MedGemma-4B GGUF via llama.cpp mtmd,
prompted with `FINDINGS_PROMPT` (constrained output -> `parse_vlm_findings`).
Frame sampling: 1 fps up to 8 frames per clip, keyframe-biased (see deploy/).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol as TypingProtocol

from ..kb import KnowledgeBase


@dataclass
class VisualFinding:
    tag: str
    confidence: float  # 0..1


class VisionAdapter(TypingProtocol):
    def analyze_frames(self, frames: list) -> list[VisualFinding]: ...


class MockVisionAdapter:
    """Deterministic adapter for tests: frames are (tag, confidence) tuples."""

    def analyze_frames(self, frames: list) -> list[VisualFinding]:
        return [VisualFinding(tag, conf) for tag, conf in frames]


def finding_vocabulary(kb: KnowledgeBase) -> set[str]:
    vocab: set[str] = set()
    for p in kb.protocols.values():
        vocab.update(p.visual_findings)
    return vocab


FINDINGS_PROMPT = """You are a first-aid triage camera. Look at the image(s) and report ONLY what you see, using EXACTLY these labels, one per line, with a confidence 0-100. Report at most 4 labels. If nothing matches, output NONE.
Allowed labels:
{vocab}
Format:
label confidence
Example:
heavy_bleeding 85
person_collapsed 60"""


def build_findings_prompt(kb: KnowledgeBase) -> str:
    return FINDINGS_PROMPT.format(vocab="\n".join(sorted(finding_vocabulary(kb))))


_LINE_RE = re.compile(r"^([a-z_]+)\s+(\d{1,3})\s*$")


def parse_vlm_findings(raw: str, kb: KnowledgeBase) -> list[VisualFinding]:
    """Parse constrained VLM output; unknown labels are dropped (closed world)."""
    vocab = finding_vocabulary(kb)
    out: list[VisualFinding] = []
    for line in raw.strip().lower().splitlines():
        m = _LINE_RE.match(line.strip())
        if not m:
            continue
        tag, conf = m.group(1), min(100, int(m.group(2)))
        if tag in vocab:
            out.append(VisualFinding(tag, conf / 100.0))
    return out


class FindingMapper:
    """Map findings -> (family, score) suggestions using KB visual_findings."""

    SUGGEST_THRESHOLD = 0.55
    # Tags whose meaning points at one family regardless of how many protocols
    # list them (bypasses specificity dilution for person-down signals).
    PRIORITY_MAP = {
        "person_collapsed": "unconscious",
        "person_motionless": "unconscious",
        "infant_motionless": "cardiac_arrest",
        "cyanosis": "cardiac_arrest",
    }

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.tag_to_families: dict[str, list[str]] = {}
        for p in kb.protocols.values():
            for tag in p.visual_findings:
                fams = self.tag_to_families.setdefault(tag, [])
                if p.family not in fams:
                    fams.append(p.family)

    def suggest(self, findings: list[VisualFinding]) -> tuple[str, float] | None:
        scores: dict[str, float] = {}
        for f in findings:
            preferred = self.PRIORITY_MAP.get(f.tag)
            if preferred:
                # person-down signals map undamped: the suggested protocols
                # open with verbal checks, so a false positive costs one
                # question while a miss can cost a life
                scores[preferred] = scores.get(preferred, 0.0) + f.confidence
                continue
            fams = self.tag_to_families.get(f.tag, [])
            for fam in fams:
                specificity = 1.0 / len(fams)
                scores[fam] = scores.get(fam, 0.0) + f.confidence * specificity
        if not scores:
            return None
        fam, score = max(scores.items(), key=lambda kv: kv[1])
        if score >= self.SUGGEST_THRESHOLD:
            return fam, score
        return None


# --- caption -> findings (deterministic, portable) -------------------------------
# Small on-device VLMs are far more reliable at free captioning than at
# structured output, and leading prompts induce hallucinations (verified on
# SmolVLM2-500M: a "mention anyone lying down" prompt made it see people in an
# empty landscape). So the shipped pipeline captions with NEUTRAL prompts and
# maps captions onto the closed finding vocabulary with this lexicon.

CAPTION_PROMPTS = [
    "Describe this image briefly.",
    "List the main things you see in this image.",
]

_CAPTION_RULES: list[tuple[re.Pattern, str, float]] = [
    (re.compile(r"\blying (?:down|on the (?:ground|floor|bed|table|street|road|grass))|"
                r"\bcollapsed\b|\bunconscious\b|\bmotionless\b|\bnot moving\b|\blifeless\b"),
     "person_collapsed", 0.50),
    (re.compile(r"\bmedical (?:procedure|treatment|assistance|emergency|attention)\b|"
                r"\bfirst aid\b|\bresuscitat|\bcpr\b|\bparamedic|\bbeing examined\b"),
     "person_collapsed", 0.35),
    (re.compile(r"\ba lot of blood\b|\bpool of blood\b|\bblood everywhere\b|\bbleeding (?:heavily|profusely)\b"),
     "heavy_bleeding", 0.75),
    (re.compile(r"\bblood\b|\bbleeding\b"), "heavy_bleeding", 0.45),
    (re.compile(r"\bburn(?:s|ed|t)?\b|\bscald"), "burned_skin", 0.60),
    (re.compile(r"\bblister"), "blister_skin", 0.55),
    (re.compile(r"\bhives\b"), "hives_rash", 0.60),
    (re.compile(r"\brash\b|\bred and irritated\b|\birritated skin\b|\bredness\b"), "red_skin", 0.40),
    (re.compile(r"\bwound\b|\bgash\b|\blaceration\b|\bdeep cut\b|\binjur"), "open_wound_large", 0.45),
    (re.compile(r"\bswollen\b|\bswelling\b"), "swollen_limb", 0.40),
    (re.compile(r"\bchoking\b|\bholding (?:his|her|their) (?:throat|neck)\b"), "hands_at_throat", 0.60),
    (re.compile(r"\bconvuls|\bseizure\b|\bshaking violently\b|\bjerking\b"), "person_convulsing", 0.60),
    (re.compile(r"\bvomit"), "vomit", 0.50),
    (re.compile(r"\bpale\b"), "pale_skin", 0.40),
    (re.compile(r"\bblue lips\b|\bbluish (?:lips|skin|face)\b|\bcyanotic\b"), "cyanosis", 0.60),
    (re.compile(r"\bsmoke\b|\bon fire\b|\bflames\b"), "smoke_scene", 0.50),
    (re.compile(r"\bin the (?:water|pool|lake|sea|river)\b|\bdrowning\b"), "person_in_water", 0.50),
    (re.compile(r"\bsnake\b(?!\s?-?\s?(?:print|skin|pattern|logo|charm|toy|plush|shaped))"), "snake_visible", 0.60),
    (re.compile(r"\btick\b"), "tick_attached", 0.60),
    (re.compile(r"\bbite marks?\b|\bbitten\b|\banimal bite\b|\bdog bite\b"), "bite_marks", 0.50),
    (re.compile(r"\bsting\b|\bstinger\b"), "sting_site", 0.50),
    (re.compile(r"\bbone (?:is )?(?:visible|exposed|sticking|protruding)\b"), "bone_exposed", 0.65),
    (re.compile(r"\bdeformed\b|\bbent at (?:a|an) (?:odd|strange|wrong|unnatural) angle\b"), "deformed_limb", 0.50),
    (re.compile(r"\bbruis"), "bruise", 0.45),
    (re.compile(r"\bpills?\b|\bmedication bottle\b|\bpill bottle\b"), "pill_bottle", 0.50),
    (re.compile(r"\bbleach\b|\bchemical(?:s| bottle| container)?\b|\bdetergent\b"), "chemical_container", 0.45),
    (re.compile(r"\bnosebleed\b|\bblood (?:coming )?from (?:the |his |her )?nose\b"), "nose_bleeding", 0.60),
]

AGREEMENT_BONUS = 0.25


def caption_findings(captions: list[str]) -> list[VisualFinding]:
    """Map one caption per prompt/frame onto the finding vocabulary.
    A tag seen in more than one caption gets an agreement bonus, which is what
    lets weak single mentions stay below the suggestion threshold."""
    hits: dict[str, tuple[float, int]] = {}
    for cap in captions:
        low = cap.lower()
        seen_this_cap: set[str] = set()
        for pat, tag, conf in _CAPTION_RULES:
            if tag in seen_this_cap:
                continue
            if pat.search(low):
                seen_this_cap.add(tag)
                best, count = hits.get(tag, (0.0, 0))
                hits[tag] = (max(best, conf), count + 1)
    out = []
    for tag, (conf, count) in hits.items():
        if count > 1:
            conf = min(1.0, conf + AGREEMENT_BONUS)
        out.append(VisualFinding(tag, conf))
    out.sort(key=lambda f: -f.confidence)
    return out
