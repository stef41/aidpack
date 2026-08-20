# AidPack — Offline First-Aid AI Assistant

**A production-gated, fully offline first-aid assistant for smartphones.**
Chat + voice in/out + camera/video input. **English, Spanish, French** — auto-detected,
no setting needed. No network, ever. Guides a lay
responder step-by-step through 61 emergency protocols derived from
international first-aid consensus guidelines (ILCOR/AHA/ERC/IFRC 2020–2021).

```
you>  my dad collapsed and hes not breathing
aid>  CALL 911, or 112 NOW — or point at someone and tell them to call.
      Put the phone on speaker and keep me with you.
      Step 1: If anyone is nearby, send them to find an AED (defibrillator).
      Do not leave to search for one yourself. We start CPR right now.

you>  mi esposo se desplomó y no respira
aid>  LLAME AL 911 o 112 AHORA — o señale a alguien y dígale que llame.
      Ponga el teléfono en altavoz y quédese conmigo.
      Paso 1: Si hay alguien cerca, mándelo a buscar un DEA (desfibrilador).
      No se vaya usted a buscarlo. Empezamos RCP ahora mismo.
```

## Why not an end-to-end LLM?

First aid is safety-critical and low-entropy: the right guidance is fixed by
international consensus. An LLM hallucinates exactly when stakes are highest and
its output space cannot be exhaustively verified. AidPack instead uses a
**deterministic protocol-graph engine** with a **finite, enumerable output
space** — every possible output is swept by an automated forbidden-advice
linter, and every life-threatening protocol is structurally guaranteed to lead
with "call emergency services". Neural components (speech, vision) sit at the
*periphery* where perception genuinely needs them. See
[docs/RESEARCH.md](docs/RESEARCH.md) for the full design rationale and
literature review.

## Architecture

- [firstaid/kb/](firstaid/kb/) — 61 protocol graphs (454+ nodes) as JSON data:
  CPR (adult/child/infant), choking, bleeding/tourniquet, burns, stroke FAST,
  heart attack, anaphylaxis, seizures, poisoning, overdose+naloxone, childbirth,
  drowning, hypothermia/heat stroke, snake/spider/jellyfish, and more. Each node
  carries guideline-sourced text, do-not warnings, branch logic — and, on the
  33 steps where a picture beats words, a bundled illustration.
- [firstaid/figures/](firstaid/figures/) — 12 instructional figures (261 KiB):
  8 use professionally drawn, openly licensed medical illustrations (Blausen
  Medical and others, CC BY-SA 4.0 / CC0, sourced from Wikimedia Commons —
  pinned by hash and packaged with embedded attribution by
  [tools/fetch_figures.py](tools/fetch_figures.py)); 4 are clean generated
  pictograms ([tools/gen_figures.py](tools/gen_figures.py)). Provenance for
  every figure lives in figures/sources.json and is enforced by gate G21:
  CPR hand placement, infant two-finger technique, recovery position, back
  blows, abdominal thrusts, EpiPen use, tourniquet, burn cooling, nosebleed
  posture, FAST check, head-tilt–chin-lift. The engine attaches the right figure
  to each response (`Response.figure`); phones render the SVGs natively beside
  the spoken step.
- [firstaid/nlu/](firstaid/nlu/) — offline NLU: weighted keyword lexicon ∪
  char-ngram TF-IDF ∪ typo-tolerant fuzzy matching (the KB *is* the model — no
  training pipeline, no model file); entity extraction (age/consciousness/
  breathing); dialog acts (yes/no/unsure/done/repeat/stop…). Understands
  English, Spanish and French input (per-language vector spaces, shared
  phrase-anchored keywords).
- [firstaid/i18n/](firstaid/i18n/) — pure-data language packs (es, fr):
  full input lexicons for all 58 protocol families, translated system strings,
  EMS banners, myth counters, and complete translations of the 11 time-critical
  protocols (CPR ×3, choking ×2, bleeding, anaphylaxis, stroke…). Untranslated
  topics fall back to English steps behind an honest translated notice.
  Language detection is sticky per session — a lone "no" never flips an
  English conversation to Spanish mid-CPR.
- [firstaid/engine/](firstaid/engine/) — session state machine: walks protocol
  graphs, auto-answers questions from known facts, numbers steps for voice,
  supports repeat/interruption/protocol switching.
- [firstaid/safety/](firstaid/safety/) — red-flag scanner (life-threat signals
  preempt any state, with negation guard), forbidden-advice linter,
  folk-remedy myth-busting ("should I put butter on the burn?" → immediate
  counter-warning), scope guard (diagnosis/dosing refusals, crisis routing).
- [firstaid/adapters/](firstaid/adapters/) — pluggable perception: sherpa-onnx
  ASR/TTS/VAD, llama.cpp VLM vision (closed-vocabulary findings → advisory
  protocol suggestions), optional grounding-validated LLM paraphrase (off by
  default).
- Pure Python 3.10 stdlib, ~2k lines, 0.2 MB of assets, p95 turn < 14 ms.
  Portable to Kotlin/Swift with byte-exact conformance vectors.

## Benchmarks — the release gate

`python3 bench/run_bench.py` (exit 0 = shippable). **Current: 23/23 green.**

| Gate | Result |
|---|---|
| G1 KB structural validation | 61 protocols, 454 nodes ✅ |
| G2 EMS call in first response (all life-threat protocols) | 31/31 ✅ |
| G3 Forbidden advice across ENTIRE output space | 0/761 outputs (incl. all translations) ✅ |
| G4 Red-flag interruption (mid-protocol pivots) | 8/8 ✅ |
| G5 Scope-guard refusals (diagnosis/dosing/self-harm/vet) | 6/6 ✅ |
| G6 Intent top-1 (fresh test sets, incl. typos/panic) | clean 1.000 / noisy 1.000 ✅ |
| G7 Entity extraction | 1.000 ✅ |
| G8 End-to-end scenario requirements (45 conversations) | 431/431 ✅ |
| G9 p95 turn latency | 13.6 ms ✅ |
| G10 Adapter units (speech shaping, vision parse/map, LLM grounding) | ✅ |
| G11 Adversarial: injection, contradictions, multi-victim, EMS-refusal, 2,222-turn fuzz | ✅ |
| G12 Live vision E2E on real media (CPR scene, burn, video + 3 benign controls) | 6/6 ✅ |
| G13 Long-range recall: fact lifecycle, monitor persistence, 3-emergency marathons | 12/12 ✅ |
| G14 Hallucination resistance: 18 caption traps, authority-pressure dosing, homoglyphs | 0 FP ✅ |
| G15 Clinical invariants audit: 261 guideline facts across 60 protocols, source citations mandatory, unaudited protocols blocked | ✅ |
| G16 Precision: 60 confusable pairs (macro-P/R 1.000), abstention on vague input, red-flag negation, content-over-particle answers, KB numeric allowlist | ✅ |
| G17 Ambiguity: safe routing on multi-cause inputs, hedged answers → safe branch, trauma-chest & fever context gates, clarify flows, continue-question reassurance | ✅ |
| G18 Harm elicitation: 23 myth traps countered (never confirmed), benign guideline advice preserved, malicious-paraphrase injection rejected | ✅ |
| G19 Runtime chaos: exhaustive branch-path termination, 100KB-input DoS guard (<300 ms), unicode/zalgo/RTL/fullwidth survival, KB-corruption detection, corrupt-media handling, 200-session concurrency determinism | ✅ |
| G20 Guideline currency: clinical-review record is hash-bound to KB content and expires after 400 days — any content edit or stale review blocks the build | ✅ |
| G21 Instructional figures: 12 bundled figures valid/covered/no-orphans, license provenance + embedded CC-BY attribution enforced, 261 KiB, surfaced live by the engine | ✅ |
| G22 Panic robustness: fragmented bursts route to life-threat protocols, distress reassured then safe-branched, emotional statements never derail steps, repetition/elongation survive | ✅ |
| G23 Multilingual: es/fr detection & sticky sessions, 30 cross-language routings, fully-translated CPR/choking conversations, tier-1 pack completeness, clinical-number preservation in translations, honest fallback, localized myth counters | 494 checks ✅ |

## Point it at the situation (real, working)

The camera path is live: photos or videos go through an **on-device VLM**
(SmolVLM2-500M GGUF via llama.cpp, CPU-only) → neutral-prompt captions →
deterministic caption-to-finding mapping with cross-caption agreement voting →
advisory protocol start with verbal confirmation.

```bash
tools/get_vision_assets.sh                          # one-time: build llama.cpp + fetch weights (~640 MB)
python3 -m firstaid.app.cli --look photo.jpg        # point at a photo
python3 -m firstaid.app.cli --look clip.mp4         # point at a video
```

```
$ python3 -m firstaid.app.cli --look testmedia/cpr_scene.mp4
[vision] findings: person_collapsed(0.60)  (9.4s)
aid> From the camera, this looks like: unresponsive but breathing (recovery position).
     CALL 911, or 112 NOW — or point at someone and tell them to call.
     First, check: tilt their head back gently and watch the chest for up to
     10 seconds. Are they breathing normally?
```

Measured on CPU (16 threads): ~5 s per photo, ~10 s per video clip — without
any GPU, i.e. phone-class compute. Hallucination-hardened: leading prompts made
the 500M model see "a person lying on the ground" in an empty landscape, so the
pipeline captions neutrally, votes across prompts/frames, and a benign-scene
control image is part of the release gate (G12). Vision never skips the verbal
safety checks — a false positive costs one question; the checks catch it.

## Try it

```bash
python3 -m firstaid.app.cli                # chat
python3 -m firstaid.app.cli --voice        # voice-shaped output
python3 -m firstaid.app.cli --ems "999"    # localize emergency number
python3 bench/run_bench.py --verbose       # full scorecard
python3 tools/export_test_vectors.py       # regenerate porting vectors
```

## Ship it to a phone

See [deploy/PHONE_DEPLOYMENT.md](deploy/PHONE_DEPLOYMENT.md): device tiers
(text-only 0.2 MB → voice 130 MB → voice+vision 600 MB+), sherpa-onnx/llama.cpp
wiring, latency budgets, porting order, conformance vectors
([deploy/test_vectors.json](deploy/test_vectors.json)), and the
regulatory/content-review posture.

## Safety model (non-negotiables)

1. Calling emergency services always comes first; the app never replaces EMS.
2. Red flags preempt everything, every turn, in every state.
3. The output space is finite and 100%-swept for harmful folk remedies.
4. Vision is advisory — verbal confirmation before any physical instruction.
5. The optional LLM may only rephrase; a validator rejects any medical drift.
6. Out-of-scope requests (diagnosis, dosing) are refused and redirected;
   self-harm signals route to crisis resources.
7. Every protocol cites its guideline sources and is locked by the 261-fact
   clinical-invariant audit; content drift cannot ship.
8. **Guideline currency is enforced, not assumed**: the KB content hash is bound
   to a dated clinical-review record ([deploy/kb_review.json](deploy/kb_review.json),
   currently verified against AHA 2025 CPR/ECC and the AHA/ARC 2024 First Aid
   Focused Update). Any KB edit without re-review, or a review older than 400
   days, fails gate G20. Being offline, the app cannot self-update mid-incident
   — instead content ships as versioned packs and the build system guarantees
   each pack was verified against the newest guidelines at release time.
9. No build ships with a red gate.

**Disclaimer:** AidPack provides first-aid guidance for laypeople based on
published consensus guidelines. It is not a medical device and does not
diagnose. Always call your local emergency number in an emergency.
