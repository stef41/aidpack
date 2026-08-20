"""Interactive CLI: chat with the assistant; optional voice-shaped output.

Usage:
  python -m firstaid.app.cli [--ems "911"] [--voice] [--log transcript.jsonl]
                             [--look photo.jpg clip.mp4 ...]

--look points the assistant at real camera media (photos/videos): frames are
captioned by the on-device VLM, mapped to protocol suggestions, and the chat
continues from there.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from ..adapters.speech import shape_for_speech
from ..engine import Config, Session
from ..engine.renderer import IDLE_GREETING
from ..kb import load_kb


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="firstaid", description="Offline first-aid assistant")
    ap.add_argument("--ems", default="911, or 112", help="local emergency number(s)")
    ap.add_argument("--voice", action="store_true", help="print voice-shaped output")
    ap.add_argument("--log", default=None, help="append transcript JSONL to this path")
    ap.add_argument("--look", nargs="+", metavar="MEDIA", default=None,
                    help="analyze photo/video file(s) before starting the chat")
    args = ap.parse_args(argv)

    session = Session(load_kb(), config=Config(ems_number=args.ems))
    log_f = open(args.log, "a", encoding="utf-8") if args.log else None

    print(f"\n[AidPack — offline first-aid assistant]\n")
    if args.look:
        from .look import describe_perception, perceive
        t0 = time.perf_counter()
        p = perceive(session, args.look)
        dt = time.perf_counter() - t0
        print(describe_perception(p) + f"  ({dt:.1f}s)")
        if p.response is not None:
            out = shape_for_speech(p.response.text) if args.voice else p.response.text
            print(f"\naid> {out}\n")
            if log_f:
                log_f.write(json.dumps({"ts": time.time(), "camera": args.look,
                                        "captions": p.captions,
                                        "assistant": p.response.text}) + "\n")
        elif not p.error:
            print("\naid> I couldn't see a clear emergency in that. "
                  "Tell me what's happening and I'll guide you.\n")
    else:
        print(IDLE_GREETING + "\n")
    try:
        while True:
            try:
                user = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user:
                continue
            if user.lower() in ("/quit", "/exit"):
                break
            t0 = time.perf_counter()
            resp = session.handle(user)
            dt_ms = (time.perf_counter() - t0) * 1000
            out = shape_for_speech(resp.text) if args.voice else resp.text
            fig = f"\n      [figure: firstaid/figures/{resp.figure}.svg]" if resp.figure else ""
            print(f"\naid> {out}{fig}\n      [{resp.protocol_id or '-'}/{resp.node_id or '-'} {dt_ms:.1f}ms]\n")
            if log_f:
                log_f.write(json.dumps({
                    "ts": time.time(), "user": user, "assistant": resp.text,
                    "protocol": resp.protocol_id, "node": resp.node_id,
                    "latency_ms": round(dt_ms, 2)}) + "\n")
                log_f.flush()
    finally:
        if log_f:
            log_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
