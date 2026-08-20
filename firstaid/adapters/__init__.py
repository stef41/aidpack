"""Pluggable perception/speech/LLM adapters."""
from .llm import LexicalGroundingValidator, NullParaphraser, ValidatedParaphraser
from .speech import shape_for_speech, split_sentences
from .vision import (FindingMapper, MockVisionAdapter, VisualFinding,
                     build_findings_prompt, parse_vlm_findings)

__all__ = [
    "shape_for_speech", "split_sentences", "VisualFinding", "MockVisionAdapter",
    "FindingMapper", "build_findings_prompt", "parse_vlm_findings",
    "NullParaphraser", "ValidatedParaphraser", "LexicalGroundingValidator",
]
