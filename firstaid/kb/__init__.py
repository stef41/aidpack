"""Knowledge-base loader and structural validator."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

KB_DIR = os.path.dirname(os.path.abspath(__file__))

VALID_SEVERITIES = {"life_threatening", "urgent", "moderate", "minor"}
VALID_EMS = {"immediate", "conditional", "self_care"}
VALID_NODE_TYPES = {"question", "step", "decision", "monitor", "escalate", "handoff"}
VALID_AGE_GROUPS = {"adult", "child", "infant", "any"}


@dataclass
class Protocol:
    id: str
    family: str
    name: str
    age_group: str
    severity: str
    ems: str
    sources: list[str]
    keywords: list[tuple[str, float]]
    exemplars: list[str]
    visual_findings: list[str]
    entry: str
    nodes: dict[str, dict]


@dataclass
class KnowledgeBase:
    protocols: dict[str, Protocol] = field(default_factory=dict)
    families: dict[str, list[str]] = field(default_factory=dict)

    def resolve(self, ref: str, facts: dict | None = None) -> Protocol | None:
        """Resolve a protocol id or family reference to a concrete protocol.

        Family refs with age variants pick the variant matching facts['age_group'];
        unknown age falls back to the adult variant (safest default for technique
        selection is handled by explicit age questions in the engine).
        """
        if ref in self.protocols:
            return self.protocols[ref]
        ids = self.families.get(ref)
        if not ids:
            return None
        if len(ids) == 1:
            return self.protocols[ids[0]]
        age = (facts or {}).get("age_group")
        for pid in ids:
            if self.protocols[pid].age_group == age:
                return self.protocols[pid]
        for pid in ids:
            if self.protocols[pid].age_group == "adult":
                return self.protocols[pid]
        return self.protocols[ids[0]]

    def family_has_variants(self, family: str) -> bool:
        return len(self.families.get(family, [])) > 1


def load_kb(kb_dir: str = KB_DIR) -> KnowledgeBase:
    kb = KnowledgeBase()
    for fname in sorted(os.listdir(kb_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(kb_dir, fname), encoding="utf-8") as f:
            data = json.load(f)
        for raw in data["protocols"]:
            p = Protocol(
                id=raw["id"],
                family=raw["family"],
                name=raw["name"],
                age_group=raw["age_group"],
                severity=raw["severity"],
                ems=raw["ems"],
                sources=raw.get("sources", []),
                keywords=[(k, float(w)) for k, w in raw.get("keywords", [])],
                exemplars=raw.get("exemplars", []),
                visual_findings=raw.get("visual_findings", []),
                entry=raw["entry"],
                nodes=raw["nodes"],
            )
            if p.id in kb.protocols:
                raise ValueError(f"duplicate protocol id: {p.id}")
            kb.protocols[p.id] = p
            kb.families.setdefault(p.family, []).append(p.id)
    errors = validate_kb(kb)
    if errors:
        raise ValueError("KB validation failed:\n" + "\n".join(errors))
    return kb


def validate_kb(kb: KnowledgeBase) -> list[str]:
    errors: list[str] = []
    for p in kb.protocols.values():
        loc = f"[{p.id}]"
        if p.severity not in VALID_SEVERITIES:
            errors.append(f"{loc} bad severity {p.severity!r}")
        if p.ems not in VALID_EMS:
            errors.append(f"{loc} bad ems {p.ems!r}")
        if p.age_group not in VALID_AGE_GROUPS:
            errors.append(f"{loc} bad age_group {p.age_group!r}")
        if p.severity == "life_threatening" and p.ems == "self_care":
            errors.append(f"{loc} life_threatening protocol cannot be self_care")
        if p.entry not in p.nodes:
            errors.append(f"{loc} entry {p.entry!r} not in nodes")
            continue
        for nid, node in p.nodes.items():
            nloc = f"{loc}.{nid}"
            ntype = node.get("type")
            if ntype not in VALID_NODE_TYPES:
                errors.append(f"{nloc} bad type {ntype!r}")
                continue
            if ntype == "question":
                if not node.get("prompt"):
                    errors.append(f"{nloc} question missing prompt")
                for branch in ("yes", "no"):
                    tgt = node.get(branch)
                    if not tgt or tgt not in p.nodes:
                        errors.append(f"{nloc} branch {branch!r} missing/bad target {tgt!r}")
                unsure = node.get("unsure")
                if unsure and unsure not in p.nodes:
                    errors.append(f"{nloc} unsure target {unsure!r} not in nodes")
            elif ntype == "step":
                if not node.get("text"):
                    errors.append(f"{nloc} step missing text")
                tgt = node.get("next")
                if not tgt or tgt not in p.nodes:
                    errors.append(f"{nloc} step missing/bad next {tgt!r}")
            elif ntype == "decision":
                if not node.get("fact"):
                    errors.append(f"{nloc} decision missing fact")
                cases = node.get("cases", {})
                for val, tgt in cases.items():
                    if tgt not in p.nodes:
                        errors.append(f"{nloc} case {val!r} bad target {tgt!r}")
                default = node.get("default")
                if not default or default not in p.nodes:
                    errors.append(f"{nloc} decision missing/bad default {default!r}")
            elif ntype == "monitor":
                if not node.get("text"):
                    errors.append(f"{nloc} monitor missing text")
            elif ntype == "escalate":
                ref = node.get("protocol")
                if not ref or (ref not in kb.protocols and ref not in kb.families):
                    errors.append(f"{nloc} escalate bad target {ref!r}")
            elif ntype == "handoff":
                if not node.get("text"):
                    errors.append(f"{nloc} handoff missing text")
        errors.extend(_check_reachability(p))
    return errors


def _check_reachability(p: Protocol) -> list[str]:
    seen: set[str] = set()
    stack = [p.entry]
    while stack:
        nid = stack.pop()
        if nid in seen or nid not in p.nodes:
            continue
        seen.add(nid)
        node = p.nodes[nid]
        for key in ("yes", "no", "unsure", "next", "default"):
            tgt = node.get(key)
            if isinstance(tgt, str):
                stack.append(tgt)
        for tgt in node.get("cases", {}).values():
            stack.append(tgt)
    unreachable = set(p.nodes) - seen
    return [f"[{p.id}] unreachable nodes: {sorted(unreachable)}" for _ in range(1) if unreachable]
