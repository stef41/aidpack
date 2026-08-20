"""Real on-device VLM backend: llama.cpp multimodal CLI (mtmd).

Runs SmolVLM2 (or any GGUF VLM with an mmproj) fully offline via subprocess.
Captions with neutral prompts (leading prompts hallucinate on 500M models),
then `caption_findings` maps captions deterministically onto the closed
finding vocabulary. This mirrors exactly what the phone build does through
JNI/xcframework bindings instead of a subprocess.

Asset discovery (first hit wins):
  $AIDPACK_MTMD_CLI, $AIDPACK_VLM_MODEL, $AIDPACK_VLM_MMPROJ
  <repo>/third_party/llama.cpp/build/bin/llama-mtmd-cli
  <repo>/models/*Q8_0.gguf + <repo>/models/mmproj-*.gguf
"""
from __future__ import annotations

import os
import subprocess

from .vision import CAPTION_PROMPTS, VisualFinding, caption_findings

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _default_paths() -> tuple[str | None, str | None, str | None]:
    cli = os.environ.get("AIDPACK_MTMD_CLI") or os.path.join(
        _REPO, "third_party", "llama.cpp", "build", "bin", "llama-mtmd-cli")
    model = os.environ.get("AIDPACK_VLM_MODEL")
    mmproj = os.environ.get("AIDPACK_VLM_MMPROJ")
    models_dir = os.path.join(_REPO, "models")
    if (not model or not mmproj) and os.path.isdir(models_dir):
        for f in sorted(os.listdir(models_dir)):
            if f.endswith(".gguf") and f.startswith("mmproj"):
                mmproj = mmproj or os.path.join(models_dir, f)
            elif f.endswith(".gguf"):
                model = model or os.path.join(models_dir, f)
    return cli, model, mmproj


class LlamaCppVision:
    """VisionAdapter backed by llama-mtmd-cli. Frames are image file paths."""

    def __init__(self, cli: str | None = None, model: str | None = None,
                 mmproj: str | None = None, threads: int = 8, timeout_s: int = 120):
        d_cli, d_model, d_mmproj = _default_paths()
        self.cli = cli or d_cli
        self.model = model or d_model
        self.mmproj = mmproj or d_mmproj
        self.threads = threads
        self.timeout_s = timeout_s

    def available(self) -> bool:
        return all(p and os.path.isfile(p) for p in (self.cli, self.model, self.mmproj))

    def caption(self, image_paths: list[str], prompt: str, max_tokens: int = 80) -> str:
        cmd = [self.cli, "-m", self.model, "--mmproj", self.mmproj,
               "-p", prompt, "--temp", "0", "-n", str(max_tokens),
               "--threads", str(self.threads)]
        for p in image_paths:
            cmd += ["--image", p]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_s)
        if out.returncode != 0:
            raise RuntimeError(f"mtmd-cli failed: {out.stderr[-400:]}")
        return out.stdout.strip()

    def captions(self, image_paths: list[str]) -> list[str]:
        """Caption strategy tuned for 500M-class models:
        - 1-2 images: every neutral prompt on the batch.
        - video frame sets: first/middle/last frame captioned SEPARATELY
          (batching many frames dilutes captions), second prompt on the
          middle frame. Cross-caption voting then rewards agreement."""
        caps: list[str] = []
        if len(image_paths) <= 2:
            plans = [(image_paths, p) for p in CAPTION_PROMPTS]
        else:
            first, mid, last = image_paths[0], image_paths[len(image_paths) // 2], image_paths[-1]
            plans = [([first], CAPTION_PROMPTS[0]), ([mid], CAPTION_PROMPTS[0]),
                     ([last], CAPTION_PROMPTS[0]), ([mid], CAPTION_PROMPTS[1])]
        for paths, prompt in plans:
            try:
                caps.append(self.caption(paths, prompt))
            except (RuntimeError, subprocess.TimeoutExpired):
                continue
        return caps

    def analyze_frames(self, frames: list) -> list[VisualFinding]:
        caps = self.captions(list(frames))
        return caption_findings(caps)
