# Research & Design Rationale

**Project:** AidPack — fully offline first-aid AI assistant for smartphones.
**Date:** 2026-08-20. Sources fetched live during the research phase are cited inline.

## 1. Problem statement

Build the most useful *self-contained* first-aid assistant that runs on a phone with
**no connectivity**: chat + voice in/out + video/camera input, guiding a lay responder
through emergencies until professional help takes over.

## 2. Key design finding: deterministic core beats end-to-end LLM

First aid is a **safety-critical, low-entropy domain**: the correct guidance for
"adult choking, can't speak" is fixed by international consensus (ILCOR/IFRC/AHA/ERC
guidelines) and does not benefit from generative variation. An end-to-end LLM:

- hallucinates under distribution shift (panicked, garbled, multilingual input);
- cannot be exhaustively verified (unbounded output space);
- costs 0.5–4 GB and 100–2000 ms/turn on phones;
- degrades exactly when stakes are highest.

A **protocol-graph engine** over a curated, guideline-derived knowledge base:

- has a **finite, enumerable output space** → we benchmark *every reachable output*
  against a forbidden-advice scanner and EMS-coverage gate (impossible with an LLM);
- responds in <50 ms on-CPU with ~1 MB of assets;
- is portable to Kotlin/Swift byte-for-byte via exported test vectors.

Therefore: **deterministic core, neural periphery.** Neural components (ASR, TTS,
vision) are pluggable adapters where perception genuinely needs them; an optional
on-device LLM may *paraphrase* protocol text but never author medical content.

## 3. On-device component survey (verified 2026-08-20)

| Layer | Choice | Evidence |
|---|---|---|
| ASR (speech→text) | **sherpa-onnx** streaming Zipformer / Whisper-tiny int8 | github.com/k2-fsa/sherpa-onnx: Apache-2.0, ASR+TTS+VAD+KWS on Android/iOS/arm64, 12 language APIs, prebuilt APKs; used by production apps (BreezeApp, MentraOS) |
| TTS (text→speech) | **sherpa-onnx + Piper/VITS voices** (~20–60 MB) | Piper (MIT, now OHF-Voice/piper1-gpl) voices run realtime on Raspberry Pi-class CPUs; sherpa-onnx ships them for Android/iOS |
| VAD / wake | silero-vad via sherpa-onnx | same runtime, ~2 MB |
| LLM/VLM runtime | **llama.cpp** (GGUF, 1.5–8-bit quant) | github.com/ggml-org/llama.cpp: MIT, ARM NEON first-class, Android/iOS builds, multimodal (mtmd) support |
| Vision model | **SmolVLM2-500M/2.2B** (Apache-2.0) or **MedGemma 1.5 4B** (multimodal, medical-tuned) | huggingface.co/blog/smolvlm: 2B VLM SOTA memory footprint, 81 tokens/384² patch, video via frame sampling (CinePile 27.14%); developers.google.com/health-ai-developer-foundations/medgemma: 4B multimodal, medical image+text, explicitly *not clinical-grade* → we use it only to produce *findings*, never advice |
| Medical guidance | **curated protocol KB** (this repo) | IFRC International First Aid, Resuscitation and Education Guidelines 2020; AHA/ILCOR 2020 CPR & ECC; ERC 2021; Stop-the-Bleed; consensus points encoded per-protocol (see kb/ `sources` fields) |

## 4. Prior art

- **BreezeApp (MediaTek Research)** — offline STT/TTS/chatbot/VQA app; proves the full
  offline multimodal stack ships on consumer phones today. No medical safety layer.
- **MedGemma/HAI-DEF** — strong medical perception, but Google explicitly requires
  downstream validation; unsuitable as sole guidance source.
- **Red Cross / St John first-aid apps** — offline *static content*, no NLU, no voice
  dialog, no vision. Validates offline-content demand; we add the interactive layer.
- Literature consensus (dispatcher-assisted CPR studies; T-CPR standards): rigid
  scripted instruction delivered stepwise *outperforms* free-form advice in emergencies —
  supporting the protocol-graph design (Kurz et al., Circulation 2020 T-CPR statement).

## 5. Architecture (result)

```
mic ──► VAD ──► ASR ─┐                                   ┌─► TTS ──► speaker
                     ├─► Normalizer ─► Red-flag scanner ─┤
keyboard ────────────┘        │             │            └─► screen (chat)
camera/video ► frame sampler ► VLM adapter ─► findings mapper
                              │             ▼
                              │      ┌─────────────────────────┐
                              └────► │  NLU: intent + entities │
                                     │  (lexicon ∪ char-ngram  │
                                     │   TF-IDF ∪ fuzzy match) │
                                     └───────────┬─────────────┘
                                                 ▼
                                     ┌─────────────────────────┐
                                     │ Protocol engine (graph  │
                                     │ walker, session state)  │
                                     └───────────┬─────────────┘
                                                 ▼
                                     ┌─────────────────────────┐
                                     │ Safety layer: EMS gate, │
                                     │ forbidden-advice lint,  │
                                     │ scope guard             │
                                     └───────────┬─────────────┘
                                                 ▼
                              optional constrained LLM paraphrase (off by default)
```

All arrows are on-device. Total mandatory assets ≈ **1 MB** (KB+NLU) + speech models
(~100 MB) + optional vision (~400 MB–2.6 GB). Runs down to 2 GB-RAM phones in
text-only mode; full multimodal targets ≥6 GB-RAM devices.

## 6. Safety model

1. **Triage-first:** red-flag scanner runs on *every* input in *every* state; a
   life-threat signal (e.g. "he stopped breathing" mid-burn-protocol) preempts
   the current protocol within the same turn.
2. **EMS gate:** every life-threatening protocol *must* instruct calling emergency
   services in its first response — enforced by benchmark gate, 100% required.
3. **Forbidden-advice lint:** the finite output space is swept for known-harmful
   folk remedies (butter on burns, head-back nosebleed, venom suction, gastric
   emptying, infant abdominal thrusts, …). Zero tolerance.
4. **Scope guard:** diagnosis/prescription/dosing questions are refused and
   redirected; self-harm signals route to crisis resources.
5. **Perception is advisory:** vision findings only *suggest* protocols; verbal
   confirmation is required before acting on them; uncertainty → ask.
6. The optional LLM layer may only rephrase engine output; a grounding validator
   rejects any paraphrase that adds/changes medical content tokens.

## 7. Quality gates (benchmarked, see bench/)

| Gate | Threshold | Type |
|---|---|---|
| EMS coverage on life threats | 100% | HARD |
| Forbidden advice in reachable outputs | 0 | HARD |
| Red-flag interruption success | 100% | HARD |
| Scope-guard refusals | 100% | HARD |
| Intent top-1, clean | ≥ 0.90 | HARD |
| Intent top-1, noisy (typos/panic) | ≥ 0.80 | HARD |
| Entity extraction F1 | ≥ 0.85 | HARD |
| Scenario requirement coverage | ≥ 0.95 | HARD |
| p95 turn latency (reference impl) | < 50 ms | HARD |
| Core asset size | < 5 MB | SOFT |

## 8. Non-goals

- Diagnosis, medication dosing beyond guideline-listed bystander assists
  (aspirin in adult ACS, epinephrine auto-injector, naloxone, reliever inhaler).
- Replacing EMS. The assistant's first reflex is always to summon professionals.
- Cloud anything.
