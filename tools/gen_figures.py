#!/usr/bin/env python3
"""Generate the four locally drawn instructional figures (SVG pictograms).
The other eight figures are packaged from openly licensed professional medical
illustrations by tools/fetch_figures.py — run that first; this script only
writes the topics with no acceptable open illustration.
Regenerate with: python3 tools/gen_figures.py

Pictogram rules (learned the hard way — wire-thin stick figures are unreadable):
- Filled silhouettes, two tones: GRAY = the person helped, BLACK = the helper.
- Art lives above the caption band; captions never overlap the art.
- Exactly one red action cue (arrow/zone) per figure; blue only for water.
- Solid triangular arrowheads; limbs are thick round-capped strokes.
- ASCII-only text, explicit width/height attrs (renderer-proof).
"""
from __future__ import annotations

import math
import os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "firstaid", "figures")

W, H = 480, 360
DARK = "#1a1a1a"         # helper / objects
GRAY = "#9a9a9a"         # person being helped
RED = "#d32f2f"          # the one action cue
BLUE = "#1976d2"         # water only

HEAD_SVG = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
            f'viewBox="0 0 {W} {H}" font-family="Arial, Helvetica, sans-serif">'
            f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
TAIL = '</svg>'


def limb(x1, y1, x2, y2, color=DARK, w=14):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{w}" stroke-linecap="round"/>')


def polyline(pts, color=DARK, w=14):
    d = " ".join(f"{x},{y}" for x, y in pts)
    return (f'<polyline points="{d}" fill="none" stroke="{color}" '
            f'stroke-width="{w}" stroke-linecap="round" stroke-linejoin="round"/>')


def head(cx, cy, r, color=DARK):
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}"/>'


def dot(cx, cy, r, color=DARK):
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}"/>'


def ring(cx, cy, r, color=RED, w=6):
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="{w}"/>'


def ground(y=252, x1=20, x2=460):
    return limb(x1, y, x2, y, "#c4c4c4", 5)


def _arrow_head(x, y, ang, size, color):
    p1 = (x + size * math.cos(ang + 2.62), y + size * math.sin(ang + 2.62))
    p2 = (x + size * math.cos(ang - 2.62), y + size * math.sin(ang - 2.62))
    return (f'<polygon points="{x:.0f},{y:.0f} {p1[0]:.0f},{p1[1]:.0f} '
            f'{p2[0]:.0f},{p2[1]:.0f}" fill="{color}"/>')


def arrow(x1, y1, x2, y2, color=RED, w=8, hs=18):
    ang = math.atan2(y2 - y1, x2 - x1)
    bx, by = x2 - (hs - 4) * math.cos(ang), y2 - (hs - 4) * math.sin(ang)
    return (f'<line x1="{x1}" y1="{y1}" x2="{bx:.0f}" y2="{by:.0f}" '
            f'stroke="{color}" stroke-width="{w}" stroke-linecap="round"/>'
            + _arrow_head(x2, y2, ang, hs, color))


def arc_arrow(x0, y0, qx, qy, x1, y1, color=RED, w=7, hs=17):
    ang = math.atan2(y1 - qy, x1 - qx)
    return (f'<path d="M{x0} {y0} Q{qx} {qy} {x1} {y1}" fill="none" '
            f'stroke="{color}" stroke-width="{w}" stroke-linecap="round"/>'
            + _arrow_head(x1 + 6 * math.cos(ang), y1 + 6 * math.sin(ang), ang, hs, color))


def times5(x, y, n=5, color=RED):
    return (f'<text x="{x}" y="{y}" font-size="34" font-weight="bold" '
            f'fill="{color}" text-anchor="middle">x {n}</text>')


def small_label(x, y, text, color=DARK, size=17, anchor="middle"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}" font-weight="bold">{text}</text>')


def caption(line1, line2, c1=DARK, c2=RED):
    out = []
    for text, y, color in ((line1, 310, c1), (line2, 343, c2)):
        size = min(22, int(464 / (0.62 * max(1, len(text)))))
        if size < 15:
            raise ValueError(f"caption too long: {text!r}")
        if not text.isascii():
            raise ValueError(f"caption not ascii: {text!r}")
        out.append(f'<text x="{W // 2}" y="{y}" font-size="{size}" fill="{color}" '
                   f'text-anchor="middle" font-weight="bold">{text}</text>')
    return "".join(out)


def svg(*parts):
    return HEAD_SVG + "".join(parts) + TAIL


FIGURES: dict[str, str] = {}





# 7. EpiPen: standing figure, auto-injector pressed into the outer mid-thigh.
FIGURES["epipen_thigh"] = svg(
    ground(258),
    # person (gray), standing
    head(150, 50, 22, GRAY),
    limb(150, 72, 150, 158, GRAY, 18),
    limb(150, 98, 118, 150, GRAY, 10),
    limb(150, 98, 184, 148, GRAY, 10),
    limb(150, 158, 132, 252, GRAY, 15),          # near leg
    limb(150, 158, 170, 252, GRAY, 15),          # far leg
    # helper's hand (dark) pressing the injector horizontally into the thigh
    f'<rect x="228" y="192" width="104" height="30" rx="12" fill="{DARK}"/>',
    limb(228, 207, 196, 207, DARK, 9),           # injector tip toward thigh
    dot(344, 207, 14, DARK),                     # gripping fist
    # red cue: target zone on the OUTER mid-thigh + press direction above the pen
    ring(172, 207, 21),
    arrow(336, 166, 236, 166),
    caption("PRESS HARD INTO OUTER MID-THIGH", "HOLD FOR 3 SECONDS - CLOTHES ARE OK"),
)


# 9. Burn cooling: tap, blue water onto the burn, 20-minute clock.
FIGURES["burn_cooling"] = svg(
    # tap (dark)
    polyline([(140, 22), (140, 62), (206, 62), (206, 96)], DARK, 12),
    # water (blue)
    limb(196, 108, 192, 176, BLUE, 6),
    limb(206, 108, 205, 180, BLUE, 6),
    limb(216, 108, 218, 176, BLUE, 6),
    # arm (gray) with red burn patch where the water lands
    limb(96, 198, 330, 206, GRAY, 30),
    dot(346, 207, 15, GRAY),
    f'<ellipse cx="206" cy="193" rx="22" ry="10" fill="{RED}" opacity="0.85"/>',
    # clock (dark ring, red hands) + label
    ring(408, 74, 34, DARK, 6),
    limb(408, 74, 408, 50, RED, 5),
    limb(408, 74, 424, 82, RED, 5),
    small_label(408, 138, "20 MIN", RED, 21),
    caption("COOL RUNNING WATER - 20 MINUTES", "NO ICE - NO BUTTER - NO CREAMS"),
)

# 10. Nosebleed: sit, lean forward, pinch the soft part; red drop falls clear.
FIGURES["nosebleed"] = svg(
    ground(258),
    # person (gray) seated, leaning forward
    head(180, 92, 25, GRAY),
    polyline([(198, 110), (252, 196)], GRAY, 18),          # leaning trunk
    polyline([(252, 196), (330, 200), (334, 252)], GRAY, 14),  # thigh + shin
    limb(244, 158, 192, 112, GRAY, 10),          # arm up to the nose
    dot(170, 106, 8, DARK),                      # pinching hand on the soft part
    limb(168, 104, 184, 112, DARK, 8),           # pinching fingers
    # red cues: lean-forward arc + one falling drop
    arc_arrow(96, 34, 130, 34, 152, 64),
    f'<path d="M158 138 q7 12 0 18 q-8 -6 0 -18" fill="{RED}"/>',
    caption("PINCH THE SOFT PART - 10-15 MINUTES", "LEAN FORWARD - NEVER TILT BACK"),
)

# 11. FAST stroke: three sign panels, then the red rule.
FIGURES["fast_stroke"] = svg(
    # panel 1: face with one drooping mouth corner
    ring(90, 100, 36, DARK, 6),
    dot(77, 88, 4.5), dot(104, 88, 4.5),
    f'<path d="M72 116 Q88 118 98 124 Q106 130 110 140" fill="none" '
    f'stroke="{DARK}" stroke-width="6" stroke-linecap="round"/>',
    arrow(118, 112, 113, 138, RED, 5, 12),
    small_label(90, 172, "FACE DROOPS"),
    # panel 2: mini person, one raised arm drifting down
    head(240, 52, 13),
    limb(240, 65, 240, 130, DARK, 10),
    limb(240, 82, 196, 58, DARK, 8),             # steady arm
    limb(240, 82, 282, 96, DARK, 8),             # drifting arm
    arrow(287, 66, 290, 92, RED, 5, 12),
    small_label(240, 172, "ARM DRIFTS"),
    # panel 3: speech bubble with a red garbled line
    f'<path d="M352 62 h84 v52 h-46 l-16 20 v-20 h-22 z" fill="none" '
    f'stroke="{DARK}" stroke-width="5" stroke-linejoin="round"/>',
    f'<path d="M366 88 q9 11 18 0 q9 -11 18 0 q9 11 18 0" fill="none" '
    f'stroke="{RED}" stroke-width="5" stroke-linecap="round"/>',
    small_label(394, 172, "SLURRED SPEECH"),
    # the rule
    f'<text x="240" y="232" font-size="25" font-weight="bold" fill="{RED}" '
    f'text-anchor="middle">ANY ONE SIGN = STROKE</text>',
    caption("CALL EMERGENCY SERVICES NOW", "NOTE THE TIME IT STARTED", DARK, DARK),
)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    total = 0
    for name, content in FIGURES.items():
        path = os.path.join(OUT, f"{name}.svg")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content + "\n")
        size = os.path.getsize(path)
        total += size
        print(f"{name}.svg  {size} bytes")
    print(f"total {total / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
