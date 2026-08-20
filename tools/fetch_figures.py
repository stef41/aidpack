#!/usr/bin/env python3
"""Fetch and package the externally sourced instructional figures.

Eight figures use professionally drawn, openly licensed medical illustrations
from Wikimedia Commons (pinned by URL + sha256). Each is downscaled, embedded
into a 480x360 SVG wrapper with the standard caption band, and stamped with
the required attribution. The remaining four figures are generated locally by
gen_figures.py. figures/sources.json records provenance for every figure and
is enforced by bench gate G21.

Usage: python3 tools/fetch_figures.py [--cache DIR]
Requires: ImageMagick (convert/identify), network on first run only.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "firstaid", "figures")
CACHE = os.path.join(ROOT, ".cache", "figures")
UA = {"User-Agent": "AidPackFigures/1.0 (offline first-aid app asset build; contact: maintainer)"}

W, H = 480, 360
ART_X, ART_Y, ART_W, ART_H = 4, 4, 472, 254
DARK, RED = "#1a1a1a", "#d32f2f"

SOURCES: dict[str, dict] = {
    "cpr_hands": {
        "title": "CPR Adult Chest Compression 2",
        "url": "https://upload.wikimedia.org/wikipedia/commons/9/9a/CPR_Adult_Chest_Compression_2.png",
        "page": "https://commons.wikimedia.org/wiki/File:CPR_Adult_Chest_Compression_2.png",
        "sha256": "618bcfdae6db0647",  # prefix-checked (full recorded in manifest)
        "author": "BruceBlaus (Blausen Medical)",
        "license": "CC BY-SA 4.0",
        "kind": "raster",
        "caption": ("HEEL OF HAND - CENTER OF CHEST", "PUSH HARD: 100-120 / MIN"),
    },
    "cpr_infant": {
        "title": "CPR Infant Chest Compression",
        "url": "https://upload.wikimedia.org/wikipedia/commons/8/8a/CPR_Infant_Chest_Compression.png",
        "page": "https://commons.wikimedia.org/wiki/File:CPR_Infant_Chest_Compression.png",
        "sha256": "30ddccd692547429",
        "author": "BruceBlaus (Blausen Medical)",
        "license": "CC BY-SA 4.0",
        "kind": "raster",
        "caption": ("TWO FINGERS - JUST BELOW NIPPLE LINE", "PUSH DOWN 1/3 OF CHEST DEPTH"),
    },
    "head_tilt": {
        "title": "CPR Adult Airway",
        "url": "https://upload.wikimedia.org/wikipedia/commons/a/aa/CPR_Adult_Airway.png",
        "page": "https://commons.wikimedia.org/wiki/File:CPR_Adult_Airway.png",
        "sha256": "e253979c0ab6d3c0",
        "author": "BruceBlaus (Blausen Medical)",
        "license": "CC BY-SA 4.0",
        "kind": "raster",
        "caption": ("TILT THE HEAD BACK - LIFT THE CHIN", "THIS OPENS THE AIRWAY"),
    },
    "abdominal_thrusts": {
        "title": "Heimlich Adult & Child",
        "url": "https://upload.wikimedia.org/wikipedia/commons/f/f2/Heimlich_Adult_%26_Child.png",
        "page": "https://commons.wikimedia.org/wiki/File:Heimlich_Adult_%26_Child.png",
        "sha256": "c0a7c2d2f048add4",
        "author": "BruceBlaus (Blausen Medical)",
        "license": "CC BY-SA 4.0",
        "kind": "raster",
        "caption": ("FIST JUST ABOVE THE BELLY BUTTON", "PULL SHARPLY IN AND UP"),
    },
    "infant_back_blows": {
        "title": "Heimlich Infant",
        "url": "https://upload.wikimedia.org/wikipedia/commons/4/40/Heimlich_Infant.png",
        "page": "https://commons.wikimedia.org/wiki/File:Heimlich_Infant.png",
        "sha256": "dc1524d325533967",
        "author": "BruceBlaus (Blausen Medical)",
        "license": "CC BY-SA 4.0",
        "kind": "raster",
        "caption": ("FACE DOWN - HEAD LOW - 5 BACK BLOWS", "THEN 5 CHEST THRUSTS - NEVER SHAKE"),
    },
    "back_blows": {
        "title": "Back blows (back slaps) against choking for adult people",
        "url": "https://upload.wikimedia.org/wikipedia/commons/1/16/Back_blows_%28back_slaps%29_against_choking_for_adult_people.jpg",
        "page": "https://commons.wikimedia.org/wiki/File:Back_blows_(back_slaps)_against_choking_for_adult_people.jpg",
        "sha256": "dee36969ba61dc2f",
        "author": "Trakotako",
        "license": "CC BY-SA 4.0",
        "kind": "raster",
        "caption": ("LEAN FORWARD - SUPPORT THE CHEST", "5 SHARP BLOWS BETWEEN SHOULDER BLADES"),
    },
    "recovery_position": {
        "title": "Recovery position 02",
        "url": "https://upload.wikimedia.org/wikipedia/commons/1/1a/Recovery_position_02.svg",
        "page": "https://commons.wikimedia.org/wiki/File:Recovery_position_02.svg",
        "sha256": "9eded4b3fb812d25",
        "author": "Mrmw",
        "license": "CC0",
        "kind": "svg",
        "caption": ("ON THEIR SIDE - HEAD TILTED BACK", "TOP KNEE BENT - WATCH THE BREATHING"),
    },
    "tourniquet": {
        "title": "Tourniquet",
        "url": "https://upload.wikimedia.org/wikipedia/commons/f/f4/Tourniquet.png",
        "page": "https://commons.wikimedia.org/wiki/File:Tourniquet.png",
        "sha256": "c1e56fcb077e687b",
        "author": "Baedr-9439",
        "license": "CC0",
        "kind": "raster",
        "caption": ("TIGHT BAND 5-8 cm ABOVE THE WOUND", "TWIST UNTIL STOPPED - WRITE THE TIME"),
    },
}

# generated locally by gen_figures.py (no acceptable open illustration found)
GENERATED = {
    "epipen_thigh": "generated pictogram (tools/gen_figures.py)",
    "burn_cooling": "generated pictogram (tools/gen_figures.py)",
    "nosebleed": "generated pictogram (tools/gen_figures.py)",
    "fast_stroke": "generated pictogram (tools/gen_figures.py)",
}


def sh(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


def fetch(fid: str, spec: dict) -> str:
    os.makedirs(CACHE, exist_ok=True)
    ext = os.path.splitext(spec["url"].split("?")[0])[1].lower()
    path = os.path.join(CACHE, f"{fid}{ext}")
    if not os.path.exists(path):
        req = urllib.request.Request(spec["url"], headers=UA)
        data = urllib.request.urlopen(req, timeout=60).read()
        with open(path, "wb") as f:
            f.write(data)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if not digest.startswith(spec["sha256"]):
        raise RuntimeError(f"{fid}: sha256 mismatch — source changed upstream "
                           f"(got {digest[:16]}, pinned {spec['sha256']}). Re-review before updating.")
    return path


def caption_svg(line1: str, line2: str) -> str:
    parts = []
    for text, y, color in ((line1, 306, DARK), (line2, 336, RED)):
        size = min(21, int(464 / (0.62 * max(1, len(text)))))
        if size < 14:
            raise ValueError(f"caption too long: {text!r}")
        if not text.isascii():
            raise ValueError(f"caption not ascii: {text!r}")
        parts.append(f'<text x="{W // 2}" y="{y}" font-size="{size}" fill="{color}" '
                     f'text-anchor="middle" font-weight="bold">{text}</text>')
    return "".join(parts)


def attribution_svg(spec: dict) -> str:
    text = f"Art: {spec['author']} - {spec['license']} - Wikimedia Commons"
    return (f'<text x="{W // 2}" y="355" font-size="10" fill="#8a8a8a" '
            f'text-anchor="middle">{text}</text>')


def embed_raster(fid: str, spec: dict, src: str) -> str:
    small = os.path.join(CACHE, f"{fid}_small.jpg")
    sh("convert", src, "-resize", "640x420>", "-background", "white",
       "-alpha", "remove", "-strip", "-quality", "82", small)
    w, h = map(int, sh("identify", "-format", "%w %h", small).split())
    scale = min(ART_W / w, ART_H / h)
    dw, dh = round(w * scale), round(h * scale)
    dx, dy = ART_X + (ART_W - dw) // 2, ART_Y + (ART_H - dh) // 2
    b64 = base64.b64encode(open(small, "rb").read()).decode()
    return (f'<image x="{dx}" y="{dy}" width="{dw}" height="{dh}" '
            f'href="data:image/jpeg;base64,{b64}"/>')


def embed_svg(fid: str, spec: dict, src: str) -> str:
    raw = open(src, encoding="utf-8").read()
    low = raw.lower()
    for banned in ("<script", "onload=", "xlink:href=\"http", "href=\"http", "<foreignobject"):
        if banned in low:
            raise RuntimeError(f"{fid}: unsafe SVG content ({banned})")
    m = re.search(r"<svg\b[^>]*>", raw, re.I | re.S)
    if not m:
        raise RuntimeError(f"{fid}: no <svg> root")
    root_tag = m.group(0)
    vb = re.search(r'viewBox="([\d.\s\-]+)"', root_tag)
    if vb:
        viewbox = vb.group(1)
    else:
        wm = re.search(r'width="([\d.]+)', root_tag)
        hm = re.search(r'height="([\d.]+)', root_tag)
        viewbox = f"0 0 {wm.group(1)} {hm.group(1)}"
    body = raw[m.end():raw.rfind("</svg>")]
    body = re.sub(r"<\?xml[^?]*\?>|<!DOCTYPE[^>]*>", "", body)
    vw, vh = float(viewbox.split()[2]), float(viewbox.split()[3])
    scale = min(ART_W / vw, ART_H / vh)
    dw, dh = round(vw * scale), round(vh * scale)
    dx, dy = ART_X + (ART_W - dw) // 2, ART_Y + (ART_H - dh) // 2
    return (f'<svg x="{dx}" y="{dy}" width="{dw}" height="{dh}" viewBox="{viewbox}" '
            f'preserveAspectRatio="xMidYMid meet">{body}</svg>')


def build_figure(fid: str, spec: dict) -> str:
    src = fetch(fid, spec)
    art = embed_svg(fid, spec, src) if spec["kind"] == "svg" else embed_raster(fid, spec, src)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
            f'viewBox="0 0 {W} {H}" font-family="Arial, Helvetica, sans-serif">'
            f'<rect width="{W}" height="{H}" fill="#ffffff"/>'
            + art + caption_svg(*spec["caption"]) + attribution_svg(spec)
            + "</svg>")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    manifest: dict[str, dict] = {}
    total = 0
    for fid, spec in SOURCES.items():
        svg = build_figure(fid, spec)
        path = os.path.join(OUT, f"{fid}.svg")
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg + "\n")
        size = os.path.getsize(path)
        total += size
        manifest[fid] = {
            "source": "wikimedia-commons", "title": spec["title"], "page": spec["page"],
            "url": spec["url"], "source_sha256_prefix": spec["sha256"],
            "author": spec["author"], "license": spec["license"],
            "figure_license": spec["license"] if spec["license"].startswith("CC BY") else "CC0",
        }
        print(f"{fid}.svg  {size // 1024} KiB  [{spec['license']}: {spec['author']}]")
    for fid, note in GENERATED.items():
        manifest[fid] = {"source": "generated", "note": note,
                         "author": "AidPack", "license": "repository license"}
    with open(os.path.join(OUT, "sources.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(f"external total {total / 1024:.0f} KiB + sources.json ({len(manifest)} entries)")


if __name__ == "__main__":
    sys.exit(main())
