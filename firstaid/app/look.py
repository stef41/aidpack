"""Point-at-it perception pipeline: camera media -> findings -> guidance.

`perceive()` accepts image and/or video paths (what a phone camera produces),
samples video frames with ffmpeg, captions them with the on-device VLM, maps
captions onto the closed finding vocabulary, and hands the result to the
session. Vision stays advisory: protocols start with a camera notice and their
entry questions verbally confirm before any physical instruction.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

from ..adapters.vision import FindingMapper, VisualFinding, caption_findings
from ..adapters.vlm_llamacpp import LlamaCppVision
from ..engine import Response, Session

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".3gp"}
MAX_VIDEO_FRAMES = 6


def sample_video_frames(video_path: str, out_dir: str,
                        max_frames: int = MAX_VIDEO_FRAMES) -> list[str]:
    """Evenly sample up to `max_frames` frames with ffmpeg (phone builds use
    MediaMetadataRetriever/AVAssetImageGenerator instead)."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found — cannot sample video frames")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True)
    try:
        duration = max(0.5, float(probe.stdout.strip()))
    except ValueError:
        duration = 4.0
    fps = max_frames / duration
    pattern = os.path.join(out_dir, "frame_%02d.jpg")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
         "-vf", f"fps={fps:.4f},scale='min(768,iw)':-2",
         "-frames:v", str(max_frames), pattern],
        check=True, capture_output=True)
    return sorted(
        os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.startswith("frame_"))


@dataclass
class Perception:
    captions: list[str] = field(default_factory=list)
    findings: list[VisualFinding] = field(default_factory=list)
    response: Response | None = None
    error: str | None = None


def perceive(session: Session, media_paths: list[str],
             vlm: LlamaCppVision | None = None) -> Perception:
    vlm = vlm or LlamaCppVision(threads=os.cpu_count() and min(16, os.cpu_count()) or 8)
    result = Perception()
    if not vlm.available():
        result.error = ("Vision model not installed. Run tools/get_vision_assets.sh "
                        "or set AIDPACK_MTMD_CLI / AIDPACK_VLM_MODEL / AIDPACK_VLM_MMPROJ.")
        return result

    with tempfile.TemporaryDirectory(prefix="aidpack_frames_") as tmp:
        frames: list[str] = []
        for path in media_paths:
            ext = os.path.splitext(path)[1].lower()
            if ext in VIDEO_EXTS:
                try:
                    frames.extend(sample_video_frames(path, tmp))
                except (subprocess.CalledProcessError, RuntimeError, OSError):
                    result.error = f"could not read video: {os.path.basename(path)}"
                    return result
            elif ext in IMAGE_EXTS:
                frames.append(path)
            else:
                result.error = f"unsupported media type: {path}"
                return result
        if not frames:
            result.error = "no frames to analyze"
            return result

        result.captions = vlm.captions(frames[:MAX_VIDEO_FRAMES])

    if not result.captions:
        result.error = "vision model produced no captions"
        return result
    result.findings = caption_findings(result.captions)

    resp = session.handle_visual(result.findings)
    if resp is None and result.findings:
        # caption fallback only when the scene showed SOMETHING clinical —
        # otherwise benign scenes (food, streets) leak through the classifier
        resp = session.handle_camera_caption(" ".join(result.captions))
    result.response = resp
    return result


def describe_perception(p: Perception) -> str:
    """Human-readable trace for CLI/debug output."""
    lines = []
    if p.error:
        return f"[vision] {p.error}"
    if p.findings:
        tags = ", ".join(f"{f.tag}({f.confidence:.2f})" for f in p.findings[:5])
        lines.append(f"[vision] findings: {tags}")
    else:
        lines.append("[vision] no emergency findings")
    return "\n".join(lines)
