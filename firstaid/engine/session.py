"""Session engine: orchestrates NLU, safety, and protocol-graph walking.

One `Session` per conversation. `handle(text)` is the single entry point;
it returns a `Response` whose text has passed the forbidden-advice linter.
"""
from __future__ import annotations

from ..i18n import detect_language, load_packs
from ..kb import KnowledgeBase, Protocol
from ..nlu.dialog_acts import detect_dialog_act
from ..nlu.entities import extract_age_group, extract_entities
from ..nlu.intents import IntentClassifier
from ..safety.forbidden import (SCOPE_RESPONSES, folk_remedy_counter,
                                lint_advice, scope_guard)
from ..safety.redflags import DEFERRALS, covered_families, scan_red_flags
from ..text import normalize
from . import renderer as R

_TERMINAL_KINDS = {"monitor", "handoff"}
_INTERACTIVE = {"question", "step", "monitor", "handoff"}


class Session:
    def __init__(self, kb: KnowledgeBase, classifier: IntentClassifier | None = None,
                 config: R.Config | None = None):
        self.kb = kb
        self.classifier = classifier or IntentClassifier(kb)
        self.cfg = config or R.Config()
        self.facts: dict[str, str] = {}
        self.active: Protocol | None = None
        self.node_id: str | None = None
        self.step_no: int = 0
        self.awaiting: str | None = None          # "answer"|"step"|"age"|"clarify"|None
        self.pending_family: str | None = None    # for age question / clarify
        self.retries: int = 0
        self.greeted: bool = False
        self.lang: str = "en"
        self._packs = load_packs()
        self._turn_entities: dict[str, str] = {}
        self._fragments: list[str] = []

    # ------------------------------------------------------------------ public

    MAX_INPUT_CHARS = 4000  # ~40x a long spoken sentence; emergencies are short

    def handle(self, text: str) -> R.Response:
        if len(text) > self.MAX_INPUT_CHARS:
            # keep head and tail — buried emergencies usually trail a ramble
            half = self.MAX_INPUT_CHARS // 2
            text = text[:half] + " " + text[-half:]
        self.lang = detect_language(normalize(text), self.lang)
        resp = self._handle_inner(text)
        counter = folk_remedy_counter(normalize(text))
        if counter is not None:
            key, en_text = counter
            local = self._counter_text(key, en_text)
            if local.split(":", 1)[-1][:30] not in resp.text:
                resp.text = local + "\n" + resp.text
        return resp

    # ------------------------------------------------------------- i18n helpers

    def _t(self, key: str, default: str) -> str:
        pack = self._packs.get(self.lang)
        if pack is None:
            return default
        return pack.strings.get(key, default)

    def _counter_text(self, key: str, default: str) -> str:
        pack = self._packs.get(self.lang)
        if pack is None:
            return default
        return pack.counters.get(key, default)

    def _proto_name(self, proto) -> str:
        pack = self._packs.get(self.lang)
        if pack is not None:
            name = pack.proto_name(proto.id)
            if name:
                return name
        return proto.name.lower()

    def _scope_text(self, key: str) -> str:
        pack = self._packs.get(self.lang)
        if pack is not None and key in pack.scope:
            return pack.scope[key]
        return SCOPE_RESPONSES[key]

    def _node_field(self, node: dict, fld: str):
        pack = self._packs.get(self.lang)
        if pack is not None and self.active is not None and self.node_id:
            override = pack.node_field(self.active.id, self.node_id, fld)
            if override is not None:
                return override
        return node.get(fld)

    def _ems_banner(self) -> str:
        template = self._t("ems_banner", "")
        if template:
            num = self.cfg.ems_number
            if num == "911, or 112":  # unconfigured default -> localized default
                num = self._t("ems_number", num)
            return template.replace("{ems}", num)
        return R.ems_banner(self.cfg)

    def _protocol_translated(self, pid: str) -> bool:
        pack = self._packs.get(self.lang)
        return pack is None or pid in pack.protocols

    def _handle_inner(self, text: str) -> R.Response:
        norm = normalize(text)
        new_entities = extract_entities(text)
        self.facts.update(new_entities)
        # Inference: a person who is not breathing is unresponsive.
        if new_entities.get("breathing") in ("not_breathing", "abnormal") \
                and "consciousness" not in new_entities:
            self.facts["consciousness"] = "unresponsive"
            new_entities = dict(new_entities, consciousness="unresponsive")
        self._turn_entities = new_entities

        # 1. self-harm & scope guard (self-harm always wins)
        scope = scope_guard(norm)
        if scope == "self_harm":
            return self._finalize(R.Response(self._scope_text(scope), kind="scope"))

        # 2. red flags preempt everything
        active_family = self.active.family if self.active else None
        rf = scan_red_flags(norm, active_family)
        if rf is not None:
            # With no active protocol this is a fresh emergency, not a
            # deterioration — stale victim facts must not steer it.
            if self.active is None:
                self._clear_victim_facts()
            target_family = rf.target_family
            # Generic flags defer to a confident specific intent that itself
            # covers the signal (heat stroke for "unconscious", febrile for
            # "seizure", ...).
            deferral = DEFERRALS.get(rf.target_family)
            if deferral:
                result = self.classifier.classify(text, self.lang)
                if result.kind == "intent" and result.score >= 0.28 \
                        and result.family in deferral:
                    target_family = result.family
            proto = self.kb.resolve(target_family, self.facts)
            if proto is not None and (self.active is None or proto.id != self.active.id):
                notice = self._t("priority_notice", "This takes priority.") if self.active else ""
                # Choking technique differs radically by age — ask if unknown.
                # Cardiac arrest never waits: default to the adult sequence.
                if target_family != "cardiac_arrest" \
                        and self.kb.family_has_variants(target_family) \
                        and "age_group" not in self.facts:
                    self.awaiting = "age"
                    self.pending_family = target_family
                    prefix = (notice + " ") if notice else ""
                    return self._finalize(R.Response(prefix + self._t("age_question", R.AGE_QUESTION), kind="question"))
                return self._start_protocol(proto, notice=notice)
        elif self.active is not None and self.active.family == "choking":
            # Covered arrest signals inside choking: pivot through the graph's
            # own escalate node (keeps the mouth-check instruction).
            if self._turn_entities.get("consciousness") == "unresponsive" \
                    or self._turn_entities.get("breathing") in ("not_breathing", "abnormal"):
                jumped = self._jump_to_arrest()
                if jumped is not None:
                    return jumped
        elif self.active is not None and self.active.family == "cardiac_arrest" and (
                self._turn_entities.get("breathing") == "breathing"
                or self._turn_entities.get("consciousness") == "responsive"):
            # Signs of life during CPR: re-run the protocol's own entry checks
            # (decision nodes re-read the updated facts and route to recovery).
            return self._advance(self.active.entry)

        # 3. non-emergency scope redirects (only when idle; mid-protocol we answer + re-prompt)
        if scope is not None:
            resp_text = self._scope_text(scope)
            if self.active and self.node_id:
                resp_text += "\n" + self._render_current(reprompt=True).text
            return self._finalize(R.Response(resp_text, kind="scope"))

        # 4. structured dialog handling
        act = detect_dialog_act(text, awaiting_answer=self.awaiting in ("answer", "clarify", "step"))

        if act == "help_arrived":
            self._reset()
            return self._finalize(R.Response(self._t("help_arrived", R.HELP_ARRIVED), kind="idle"))
        if act == "stop":
            self._reset()
            return self._finalize(R.Response(self._t("standby", R.STANDBY), kind="idle"))
        if act == "repeat" and self.active and self.node_id:
            return self._render_current()
        if act == "confirm_continue" and self.active and self.node_id:
            resp = self._render_current()
            resp.text = self._t("keep_going", "Yes — keep going exactly like that.") + "\n" + resp.text
            return resp
        if act == "distress":
            return self._handle_distress()

        if self.awaiting == "age":
            return self._handle_age_answer(text, act)
        if self.awaiting == "clarify":
            return self._handle_clarify_answer(text, act)
        if self.awaiting == "answer" and self.active:
            return self._handle_question_answer(text, act, new_entities)
        if self.awaiting == "step" and self.active:
            return self._handle_step_progress(text, act)

        # 5. monitor state or idle: acknowledgments / new intents
        if act in ("thanks",):
            return self._finalize(R.Response(self._t("idle_thanks", R.IDLE_THANKS), kind="idle"))
        if act in ("greeting",):
            return self._finalize(R.Response(self._t("idle_greeting", R.IDLE_GREETING), kind="idle"))
        if act in ("done", "yes", "no", "unsure") and self.active:
            node = self._node()
            if node and node.get("type") == "monitor":
                return self._finalize(R.Response(
                    "Okay. Keep doing what we set up. " + node["text"],
                    protocol_id=self.active.id, node_id=self.node_id, kind="monitor"))

        return self._handle_new_intent(text)

    def handle_visual(self, findings) -> R.Response | None:
        """Consume vision findings (advisory). Returns a Response when the
        camera suggests starting a protocol; None when inconclusive or when
        an active life-threat protocol should not be disturbed."""
        from ..adapters.vision import FindingMapper
        if self.active and self.active.severity == "life_threatening":
            return None
        if not hasattr(self, "_finding_mapper"):
            self._finding_mapper = FindingMapper(self.kb)
        suggestion = self._finding_mapper.suggest(findings)
        if suggestion is None:
            return None
        family, _score = suggestion
        if self.active and self.active.family == family:
            return None
        proto = self.kb.resolve(family, self.facts)
        if proto is None:
            return None
        return self._start_protocol(
            proto, notice=self._t("camera_looks_like",
                                  "From the camera, this looks like: {name}.")
            .replace("{name}", self._proto_name(proto)))

    def handle_camera_caption(self, caption: str) -> R.Response | None:
        """Fallback perception route: classify the VLM's scene description with
        the same NLU used for user text. Advisory, confirmation-first."""
        if self.active and self.active.severity == "life_threatening":
            return None
        result = self.classifier.classify(caption)
        if result.kind != "intent" or result.score < 0.35:
            return None
        if self.active and result.family == self.active.family:
            return None
        proto = self.kb.resolve(result.family, self.facts)
        if proto is None or proto.family in ("general_help",):
            return None
        return self._start_protocol(
            proto, notice=self._t("camera_could_be",
                                  "From the camera, this could be: {name}. Tell me if that's wrong.")
            .replace("{name}", self._proto_name(proto)))

    def _handle_distress(self) -> R.Response:
        """Pure emotion: reassure and refocus. Repeated distress at a safety
        question takes the safe branch — dispatchers act on worst case rather
        than loop."""
        ack = self._t("distress_ack", R.DISTRESS_ACK)
        node = self._node()
        if self.active and node:
            if node.get("type") == "question":
                self.retries += 1
                if self.retries >= 2:
                    target = node.get("unsure") or node["no"]
                    resp = self._advance(target)
                    resp.text = self._t("distress_safe_branch", R.DISTRESS_SAFE_BRANCH) + "\n" + resp.text
                    return resp
            resp = self._render_current()
            resp.text = ack + "\n" + resp.text
            return resp
        # idle: comfort, then run the triage questions
        resp = self._start_family("general_help", fresh=False)
        resp.text = ack + "\n" + resp.text
        return resp

    def _try_fragments(self, text: str) -> R.Response | None:
        """Panic bursts arrive as weak fragments ("hes on the floor", "wont get
        up"). Accumulate them and classify the joined context."""
        self._fragments.append(text)
        if len(self._fragments) < 2:
            return None
        joined = " ".join(self._fragments[-4:])
        result = self.classifier.classify(joined)
        if result.kind == "intent" and result.family != "general_help":
            self._turn_entities = extract_entities(joined)
            self.facts.update(self._turn_entities)
            self._fragments.clear()
            return self._start_family(result.family)
        return None

    # ------------------------------------------------------- intent & starting

    def _handle_new_intent(self, text: str) -> R.Response:
        result = self.classifier.classify(text, self.lang)
        if result.kind == "intent":
            self._fragments.clear()
            return self._start_family(result.family, fresh=self.active is None)
        if result.kind == "clarify":
            proto = self.kb.resolve(result.family, self.facts)
            if proto is not None:
                self.awaiting = "clarify"
                self.pending_family = result.family
                guess = self._proto_name(proto)
                template = self._t("clarify_template", R.CLARIFY_TEMPLATE)
                return self._finalize(R.Response(
                    template.replace("{guess}", guess), kind="clarify"))
        if self.active:
            return self._render_current(reprompt=True)
        frag = self._try_fragments(text)
        if frag is not None:
            return frag
        return self._finalize(R.Response(self._t("not_understood", R.NOT_UNDERSTOOD), kind="idle"))

    def _start_family(self, family: str, fresh: bool = False) -> R.Response:
        if fresh:
            self._clear_victim_facts()
        if self.kb.family_has_variants(family) and "age_group" not in self.facts:
            self.awaiting = "age"
            self.pending_family = family
            return self._finalize(R.Response(self._t("age_question", R.AGE_QUESTION), kind="question"))
        proto = self.kb.resolve(family, self.facts)
        if proto is None:
            return self._finalize(R.Response(self._t("not_understood", R.NOT_UNDERSTOOD), kind="idle"))
        return self._start_protocol(proto)

    def _clear_victim_facts(self) -> None:
        """A new complaint may be a new victim: drop victim-specific facts not
        re-stated this turn. Escalation/red-flag paths (same-victim
        deterioration) never call this."""
        for k in ("consciousness", "breathing", "age_group"):
            if k not in self._turn_entities:
                self.facts.pop(k, None)

    def _start_protocol(self, proto: Protocol, notice: str = "") -> R.Response:
        self.active = proto
        self.node_id = proto.entry
        self.step_no = 0
        self.retries = 0
        self.awaiting = None
        parts = [notice]
        if self.lang != "en" and not self._protocol_translated(proto.id):
            parts.append(self._t("fallback_notice", ""))
        if proto.ems == "immediate":
            parts.append(self._ems_banner())
        walked = self._walk_to_interactive(parts)
        resp = self._render_node(walked_parts=parts)
        resp.started_protocol = True
        resp.ems_banner = proto.ems == "immediate"
        return resp

    # ----------------------------------------------------------- graph walking

    def _node(self) -> dict | None:
        if self.active and self.node_id:
            return self.active.nodes.get(self.node_id)
        return None

    def _walk_to_interactive(self, parts: list[str]) -> None:
        """Advance through decision/escalate nodes until an interactive node.
        Questions already answered by THIS turn's extracted entities are
        auto-answered (fresh evidence only — stale facts never skip questions)."""
        for _ in range(30):  # cycle guard
            node = self._node()
            if node is None:
                return
            ntype = node["type"]
            if ntype == "decision":
                val = self.facts.get(node["fact"])
                target = node.get("cases", {}).get(val, node["default"]) if val else node["default"]
                self.node_id = target
                continue
            if ntype == "question":
                auto = self._entity_answer(node, self._turn_entities)
                if auto is not None:
                    self._record_fact_from_answer(node, auto)
                    self.node_id = node[auto] if auto in ("yes", "no") else (node.get("unsure") or node["no"])
                    continue
                return
            if ntype == "escalate":
                if node.get("text"):
                    parts.append(self._node_field(node, "text") or node["text"])
                target = self.kb.resolve(node["protocol"], self.facts)
                if target is None:
                    return
                prev_ems = self.active.ems if self.active else "conditional"
                self.active = target
                self.node_id = target.entry
                self.step_no = 0
                if self.lang != "en" and not self._protocol_translated(target.id):
                    parts.append(self._t("fallback_notice", ""))
                if target.ems == "immediate" and prev_ems != "immediate":
                    parts.append(self._ems_banner())
                continue
            return

    def _advance(self, target: str) -> R.Response:
        self.node_id = target
        self.retries = 0
        parts: list[str] = []
        self._walk_to_interactive(parts)
        return self._render_node(walked_parts=parts)

    def _render_node(self, walked_parts: list[str] | None = None) -> R.Response:
        parts = walked_parts if walked_parts is not None else []
        node = self._node()
        if node is None or self.active is None:
            self._reset()
            return self._finalize(R.Response(R.NOT_UNDERSTOOD, kind="idle"))
        ntype = node["type"]
        kind = "guidance"
        if ntype == "question":
            parts.append(self._node_field(node, "prompt") or node["prompt"])
            self.awaiting = "answer"
            kind = "question"
        elif ntype == "step":
            self.step_no += 1
            step_word = self._t("step_word", "Step")
            parts.append(f"{step_word} {self.step_no}: {self._node_field(node, 'text') or node['text']}")
            donots = self._node_field(node, "donot") or node.get("donot")
            if donots:
                parts.append(R.format_donots(donots))
            parts.append(self._t("step_suffix", R.step_suffix()))
            self.awaiting = "step"
        elif ntype == "monitor":
            parts.append(self._node_field(node, "text") or node["text"])
            parts.append(self._t("monitor_suffix", R.monitor_suffix()))
            self.awaiting = None
            kind = "monitor"
        elif ntype == "handoff":
            parts.append(self._node_field(node, "text") or node["text"])
            proto = self.active
            self.active = None
            self.node_id = None
            self.awaiting = None
            resp = self._finalize(R.Response(R.render_turn(parts), protocol_id=proto.id, kind="question"))
            return resp
        return self._finalize(R.Response(
            R.render_turn(parts), protocol_id=self.active.id,
            node_id=self.node_id, kind=kind, figure=node.get("figure")))

    def _render_current(self, reprompt: bool = False) -> R.Response:
        node = self._node()
        if node is None or self.active is None:
            return self._finalize(R.Response(R.NOT_UNDERSTOOD, kind="idle"))
        ntype = node["type"]
        if ntype == "question":
            prefix = self._t("reprompt_yn", R.REPROMPT_YN) if reprompt else ""
            return self._finalize(R.Response(
                prefix + (self._node_field(node, "prompt") or node["prompt"]),
                protocol_id=self.active.id,
                node_id=self.node_id, kind="question", figure=node.get("figure")))
        if ntype == "step":
            step_word = self._t("step_word", "Step")
            text = f"{step_word} {self.step_no}: {self._node_field(node, 'text') or node['text']}"
            donots = self._node_field(node, "donot") or node.get("donot")
            if donots:
                text += "\n" + R.format_donots(donots)
            text += "\n" + self._t("step_suffix", R.step_suffix())
            return self._finalize(R.Response(
                text, protocol_id=self.active.id, node_id=self.node_id,
                figure=node.get("figure")))
        return self._finalize(R.Response(
            (self._node_field(node, "text") or node.get("text", "")) + "\n" +
            self._t("monitor_suffix", R.monitor_suffix()),
            protocol_id=self.active.id, node_id=self.node_id, kind="monitor"))

    # -------------------------------------------------------------- answers

    def _entity_answer(self, node: dict, entities: dict[str, str]) -> str | None:
        """Answer fact-questions directly from entities extracted this turn."""
        prompt = node.get("prompt", "").lower()
        if "breath" in prompt and "rescue breaths" not in prompt and "breathing" in entities:
            b = entities["breathing"]
            if "any breathing" in prompt:
                # "any breathing at all?" — slow/abnormal breathing counts as yes
                return "no" if b == "not_breathing" else "yes"
            return "no" if b in ("not_breathing", "abnormal") else "yes"
        if ("respond" in prompt or "awake" in prompt or "response" in prompt) \
                and "consciousness" in entities:
            return "no" if entities["consciousness"] == "unresponsive" else "yes"
        return None

    def _record_fact_from_answer(self, node: dict, answer: str) -> None:
        """Symmetric to _entity_answer: a yes/no answer to an *unambiguous*
        fact-question becomes a session fact for later decision nodes.
        Compound prompts (awake AND breathing) record nothing."""
        if answer not in ("yes", "no"):
            return
        prompt = node.get("prompt", "").lower()
        asks_breathing = "breathing normally" in prompt or "normal breathing" in prompt
        asks_conscious = "respond" in prompt or "awake" in prompt
        if asks_breathing and not asks_conscious:
            self.facts["breathing"] = "breathing" if answer == "yes" else "not_breathing"
        elif asks_conscious and not asks_breathing and "rescue breaths" not in prompt:
            self.facts["consciousness"] = "responsive" if answer == "yes" else "unresponsive"

    def _jump_to_arrest(self) -> R.Response | None:
        """Jump to the active protocol's own CPR-escalation node, if any."""
        if self.active is None:
            return None
        for nid, node in self.active.nodes.items():
            if node.get("type") == "escalate" and node.get("protocol") in (
                    "cardiac_arrest", "cpr_adult", "cpr_child", "cpr_infant"):
                if nid == self.node_id:
                    return None
                return self._advance(nid)
        return None

    def _may_switch(self, new_family: str, score: float) -> bool:
        """Mid-protocol switches must never downgrade severity: a life-threat
        protocol can only hand over to another life-threat family."""
        if self.active is None:
            return score >= 0.35
        if score < 0.45 or new_family == self.active.family:
            return False
        if self.active.severity == "life_threatening":
            target = self.kb.resolve(new_family, self.facts)
            return target is not None and target.severity == "life_threatening"
        return True

    def _handle_question_answer(self, text: str, act: str | None,
                                new_entities: dict[str, str]) -> R.Response:
        node = self._node()
        if node is None:
            return self._handle_new_intent(text)
        # Content beats particles: "no wait she IS breathing normally" must
        # take the breathing branch, not the leading-"no" branch.
        answer = self._entity_answer(node, new_entities)
        if answer is None and act in ("yes", "no", "unsure"):
            answer = act
        if answer is None and act in ("ack", "done"):
            # acknowledgment is not an answer — re-ask the safety question
            return self._render_current(reprompt=True)
        if answer is None:
            # contentful reply: maybe a brand-new emergency
            result = self.classifier.classify(text, self.lang)
            if result.kind == "intent" and self._may_switch(result.family, result.score):
                return self._start_family(result.family)
            # triage questions absorb panic fragments — try joined context
            if self.active and self.active.family == "general_help":
                frag = self._try_fragments(text)
                if frag is not None:
                    return frag
            self.retries += 1
            if self.retries >= 3:
                # safest branch: treat as unsure/no
                target = node.get("unsure") or node["no"]
                return self._advance(target)
            return self._render_current(reprompt=True)
        self._record_fact_from_answer(node, answer)
        if answer == "unsure":
            target = node.get("unsure") or node["no"]
        else:
            target = node[answer]
        return self._advance(target)

    def _handle_step_progress(self, text: str, act: str | None) -> R.Response:
        node = self._node()
        if node is None:
            return self._handle_new_intent(text)
        if act in ("done", "yes", "ack"):
            return self._advance(node["next"])
        if act in ("no", "unsure"):
            nxt = self.active.nodes.get(node["next"]) if self.active else None
            if nxt and nxt.get("type") == "question":
                # A "no" at a step usually answers the follow-up question
                # ("no, it's not out") — jump there and take its safe branch.
                self.node_id = node["next"]
                target = nxt.get("unsure") or nxt["no"] if act == "unsure" else nxt["no"]
                self._record_fact_from_answer(nxt, "no" if act == "no" else "no")
                return self._advance(target)
            return self._finalize(R.Response(
                "That's okay — take it one piece at a time. " + node["text"] +
                "\n" + R.step_suffix(),
                protocol_id=self.active.id if self.active else None,
                node_id=self.node_id))
        # contentful: new emergency or status update
        result = self.classifier.classify(text, self.lang)
        if result.kind == "intent" and self._may_switch(result.family, result.score):
            return self._start_family(result.family)
        return self._render_current()

    def _handle_age_answer(self, text: str, act: str | None) -> R.Response:
        age = extract_age_group(text)
        family = self.pending_family
        if age is None:
            self.retries += 1
            if self.retries >= 2:
                age = "adult"
            else:
                return self._finalize(R.Response(self._t("age_question", R.AGE_QUESTION), kind="question"))
        self.facts["age_group"] = age
        self.awaiting = None
        self.pending_family = None
        self.retries = 0
        if family:
            proto = self.kb.resolve(family, self.facts)
            if proto is not None:
                return self._start_protocol(proto)
        return self._finalize(R.Response(R.NOT_UNDERSTOOD, kind="idle"))

    def _handle_clarify_answer(self, text: str, act: str | None) -> R.Response:
        family = self.pending_family
        self.awaiting = None
        self.pending_family = None
        if act in ("yes", "done") and family:
            return self._start_family(family)
        # user re-describes; classify fresh, fall back to triage
        result = self.classifier.classify(text, self.lang)
        if result.kind == "intent":
            return self._start_family(result.family)
        return self._start_family("general_help")

    # ---------------------------------------------------------------- helpers

    def _reset(self) -> None:
        self.active = None
        self.node_id = None
        self.awaiting = None
        self.pending_family = None
        self.step_no = 0
        self.retries = 0
        self.facts.clear()

    def _finalize(self, resp: R.Response) -> R.Response:
        violations = lint_advice(resp.text)
        if violations:
            # Defense in depth: benchmarks guarantee this never fires in the shipped
            # KB; if it ever does at runtime, strip the offending lines.
            kept = [ln for ln in resp.text.split("\n") if not lint_advice(ln)]
            resp.text = "\n".join(kept)
            resp.lint_violations = violations
        return resp
