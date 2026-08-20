"""Speech I/O adapters.

The reference implementation ships interfaces plus a pure text-shaping
function. On-device implementations bind to sherpa-onnx (see deploy/):
  ASR: Zipformer streaming transducer int8 (~70 MB) or Whisper-tiny int8
  TTS: Piper/VITS en voice (~25-60 MB)
  VAD: silero-vad (~2 MB)
"""
from __future__ import annotations

import re
from typing import Protocol as TypingProtocol


class ASRAdapter(TypingProtocol):
    def transcribe(self, audio_pcm16: bytes, sample_rate: int) -> str: ...


class TTSAdapter(TypingProtocol):
    def synthesize(self, text: str) -> bytes: ...


_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F02F\u2190-\u21FF\u2B00-\u2BFF]"
)

_SPOKEN_SUBS = [
    (re.compile(r"\bAED\b"), "A E D"),
    (re.compile(r"\bCPR\b"), "C P R"),
    (re.compile(r"\bEMS\b"), "E M S"),
    (re.compile(r"\b911\b"), "nine one one"),
    (re.compile(r"\b112\b"), "one one two"),
    (re.compile(r"\b999\b"), "nine nine nine"),
    (re.compile(r"\bmg\b"), "milligrams"),
    (re.compile(r"\bcm\b"), "centimeters"),
    (re.compile(r"°C\b"), " degrees Celsius"),
    (re.compile(r"\b1-800-222-1222\b"), "1 800, 2 2 2, 1 2 2 2"),
]


def shape_for_speech(text: str) -> str:
    """Turn a rendered chat response into TTS-friendly text.

    - strips emoji/markdown residue
    - expands abbreviations and digit strings that TTS engines mangle
    - converts newlines/dashes into sentence pauses
    """
    t = _EMOJI_RE.sub("", text)
    t = t.replace("\u2014", ", ").replace("\u2013", ", ")  # em/en dashes -> pause
    t = t.replace("\n", ". ")
    t = re.sub(r"\bStep (\d+):", r"Step \1.", t)
    t = t.replace('"', "")
    for pat, sub in _SPOKEN_SUBS:
        t = pat.sub(sub, t)
    t = re.sub(r"\.\s*\.", ".", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def split_sentences(text: str, max_len: int = 220) -> list[str]:
    """Chunk shaped text for incremental TTS synthesis (lower time-to-first-audio)."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    buf = ""
    for p in parts:
        if len(buf) + len(p) + 1 <= max_len:
            buf = f"{buf} {p}".strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks
