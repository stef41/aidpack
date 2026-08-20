"""Response assembly. All user-visible strings funnel through here so the
benchmark can sweep the complete output space."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    ems_number: str = "911, or 112"
    assistant_name: str = "AidPack"


@dataclass
class Response:
    text: str
    protocol_id: str | None = None
    node_id: str | None = None
    started_protocol: bool = False
    escalated: bool = False
    ems_banner: bool = False
    kind: str = "guidance"          # guidance|question|monitor|clarify|scope|idle
    figure: str | None = None       # bundled figure id (firstaid/figures/<id>.svg)
    lint_violations: list[str] = field(default_factory=list)


def ems_banner(cfg: Config) -> str:
    return (f"CALL {cfg.ems_number} NOW — or point at someone and tell them to call. "
            f"Put the phone on speaker and keep me with you.")


def step_suffix() -> str:
    return "Say \"next\" when done — or tell me at once if anything changes."


def monitor_suffix() -> str:
    return "I'm staying right here with you."


_WARNING_STARTERS = ("do not", "never", "don't", "no ", "nunca", "jamas",
                     "ne ", "n'", "jamais", "pas de")


def format_donots(donots: list[str]) -> str:
    out = []
    for d in donots:
        low = d.lower()
        if low.startswith(_WARNING_STARTERS):
            out.append(d)
        else:
            out.append(f"Do NOT: {d[0].lower() + d[1:]}")
    return "\n".join(out)


def render_turn(parts: list[str]) -> str:
    return "\n".join(p for p in parts if p).strip()


IDLE_GREETING = (
    "I'm your first-aid assistant. I work completely offline. "
    "Tell me what's happening — for example: \"my dad collapsed\", \"bad burn on her arm\", "
    "\"child is choking\" — and I'll guide you step by step. "
    "In any life-threatening emergency, calling your local emergency number always comes first."
)

IDLE_THANKS = "You did well. I'm here whenever you need me — just tell me what's happening."

STANDBY = ("Okay, standing by. If anything changes or a new emergency starts, "
           "just tell me what's happening.")

HELP_ARRIVED = ("Good — let the professionals take over now. Tell them what happened, "
                "what you did, and any times you noted. You did really well.")

REPROMPT_YN = "I just need a quick answer — yes, no, or not sure: "

AGE_QUESTION = "Quick check so I give the right technique: is this an adult, a child, or a baby under one?"

CLARIFY_TEMPLATE = "I want to get this right. It sounds like {guess} — is that correct? Say yes, or describe it differently."

NOT_UNDERSTOOD = ("Tell me what's happening in a few words — like \"deep cut on his arm\", "
                  "\"she fainted\", or \"chest pain\". If someone is in immediate danger, "
                  "call your local emergency number now.")

DISTRESS_ACK = ("You're doing everything that matters right now. Take one breath — "
                "I'm right here with you.")

DISTRESS_SAFE_BRANCH = ("That's okay — we'll assume the worst to stay safe and keep moving.")
