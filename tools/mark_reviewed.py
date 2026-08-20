#!/usr/bin/env python3
"""Mark the KB as clinically reviewed: records guideline snapshot, dates, and a
content hash. G20 fails if KB content changes without re-running this (after
actual review!) or if the review goes stale past its due date."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_DIR = os.path.join(ROOT, "firstaid", "kb")
OUT = os.path.join(ROOT, "deploy", "kb_review.json")

REVIEW_VALID_DAYS = 400  # annual cycle with grace


def kb_content_hash() -> str:
    h = hashlib.sha256()
    for fname in sorted(os.listdir(KB_DIR)):
        if fname.endswith(".json"):
            with open(os.path.join(KB_DIR, fname), "rb") as f:
                h.update(fname.encode())
                h.update(f.read())
    return h.hexdigest()


def main() -> int:
    today = date.today()
    record = {
        "last_reviewed": today.isoformat(),
        "review_due": (today + timedelta(days=REVIEW_VALID_DAYS)).isoformat(),
        "kb_sha256": kb_content_hash(),
        "guidelines_snapshot": [
            "AHA 2025 Guidelines for CPR & ECC (published 2025-10-22; core lay-rescuer numbers verified unchanged)",
            "AHA/ARC 2024 First Aid Focused Update",
            "ERC 2021 Guidelines (2025 edition monitored)",
            "IFRC International First Aid Guidelines 2020",
            "WHO rabies fact sheet (fetched 2026-08-20)",
            "NCPC Button Battery Triage Guideline (fetched 2026-08-20)",
            "Wilderness Medical Society practice guidelines",
            "Stop the Bleed (ACS)",
        ],
        "reviewer_note": sys.argv[1] if len(sys.argv) > 1 else "",
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
        f.write("\n")
    print(f"KB review recorded: due {record['review_due']}, hash {record['kb_sha256'][:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
