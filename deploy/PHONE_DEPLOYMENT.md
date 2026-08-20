# Phone Deployment Blueprint

Turns the reference implementation into a shipping Android/iOS app.
Everything below runs **fully offline**; nothing ever leaves the device.

## 1. Component manifest

| Component | Asset | Size | Runtime | Required? |
|---|---|---|---|---|
| Protocol KB + NLU lexicon | `firstaid/kb/*.json` (ship as-is) | **0.2 MB** | ported engine (Kotlin/Swift) | YES |
| Instructional figures | `firstaid/figures/*.svg` (12 line drawings, render natively) | 13 KB | AndroidSVG / SVGKit or precompiled vector drawables | YES |
| Engine | port of `firstaid/` (~2k lines, stdlib-only) | ~0 | native | YES |
| VAD | silero-vad v5 (sherpa-onnx) | 2 MB | sherpa-onnx | voice |
| ASR (streaming) | `sherpa-onnx-streaming-zipformer-en-2023-06-26` int8 | ~70 MB | sherpa-onnx | voice |
| ASR (alt, small) | whisper-tiny.en int8 | ~40 MB | sherpa-onnx | voice |
| TTS | Piper `en_US-lessac-medium` (VITS, via sherpa-onnx) | ~60 MB | sherpa-onnx | voice |
| TTS (alt, small) | Piper `en_US-lessac-low` | ~25 MB | sherpa-onnx | voice |
| Vision (tier B) | SmolVLM2-500M-Instruct Q4_K_M GGUF + mmproj | ~450 MB | llama.cpp (mtmd) | camera |
| Vision (tier A) | SmolVLM2-2.2B Q4_K_M / MedGemma 1.5 4B Q4 GGUF | 1.4–2.6 GB | llama.cpp | camera |
| LLM polish (optional, OFF) | Qwen2.5-1.5B-Instruct Q4_K_M | ~1 GB | llama.cpp | no |

**Device tiers**
- **Text-only** (any phone, ≥1 GB RAM): engine + KB. 0.2 MB. Instant.
- **Voice** (≥2 GB RAM): + VAD + ASR + TTS ≈ 130 MB assets, CPU realtime.
- **Voice+Vision** (≥6 GB RAM): + SmolVLM2-500M ≈ 600 MB total.
- **Max** (≥8 GB RAM, flagship): 2.2B/4B vision + optional LLM polish.

## 2. Runtime wiring

```
Android (Kotlin)                          iOS (Swift)
────────────────                          ───────────
sherpa-onnx AAR (maven: com.k2fsa)        sherpa-onnx.xcframework (SPM)
llama.cpp: llama-android (JNI)            llama.cpp xcframework (build-xcframework.sh)
Engine: Kotlin port (see §4)              Engine: Swift port (see §4)
```

**Voice loop** (both platforms):
1. Mic → 16 kHz PCM ring buffer → silero-VAD segments speech.
2. Segment → streaming Zipformer ASR → partial + final transcripts.
3. Final transcript → `Session.handle()` → response.
4. Response → `shape_for_speech()` port → Piper TTS → audio out.
   Synthesize sentence-by-sentence (`split_sentences`) so first audio starts <300 ms.
5. Barge-in: if VAD fires during TTS playback, pause TTS, process the new utterance
   (red flags may preempt mid-sentence — this is intentional and required).

**Camera/video loop** (implemented and gate-tested in this repo — see
`firstaid/app/look.py`, `firstaid/adapters/vlm_llamacpp.py`, bench gate G12):
1. CameraX / AVFoundation captures stills every 1 s (or the user records a clip;
   `sample_video_frames` mirrors the ffmpeg sampling: ≤6 even frames, 768 px).
2. Frames → VLM with **neutral caption prompts** (`CAPTION_PROMPTS`). Do NOT use
   leading prompts ("mention anyone lying down") — measured on SmolVLM2-500M,
   they induce hallucinated casualties in benign scenes. Videos caption
   first/middle/last frames separately; batching >2 frames dilutes captions.
3. Captions → `caption_findings()` (deterministic lexicon + cross-caption
   agreement voting) → `Session.handle_visual()`; if inconclusive,
   `Session.handle_camera_caption()` routes the caption through the same NLU
   used for user text.
4. Vision only *suggests*; the protocol's entry questions verbally confirm before
   any physical instruction. Never auto-act on vision alone. A benign-scene
   control image must stay negative in CI (gate G12).

Measured CPU-only reference numbers (SmolVLM2-500M Q8_0, 16 threads):
~2.4 s per caption pass, ~5 s per photo (2 passes), ~10 s per video (4 passes).
On phones use Q4_K_M + 384 px frames and expect 2–3× these times on 8 big
cores; run perception opportunistically — it never blocks the dialog.

**Emergency call button**: every screen shows a persistent "CALL <local EMS>"
button using the OS dialer (works without our app's involvement). Configure the
number from device region (`911`/`112`/`999`/`000`) at first launch, offline,
from a bundled region→number table.

## 3. Latency & battery budgets (measured targets)

| Stage | Budget | Notes |
|---|---|---|
| Engine turn | <5 ms | reference impl p95 = 13.6 ms in Python; native is faster |
| ASR final | <400 ms after end of speech | streaming zipformer int8 on 4 big cores |
| TTS first audio | <300 ms | sentence-chunked synthesis |
| VLM frame batch | <4 s (500M Q4) | run opportunistically, never blocks dialog |
| Idle battery | ~0 | everything event-driven; VAD is the only always-on stage |

## 4. Porting the engine (Kotlin/Swift)

The engine is deliberately stdlib-only, ~2k lines. Port order:
1. `text.py` — normalize/ngrams/edit-distance (pure functions).
2. `kb/__init__.py` — JSON load + validation (ship the same JSON files).
3. `nlu/` — entities (regex), dialog acts (lists), intents (lexicon+TF-IDF).
4. `safety/` — red flags, forbidden linter, scope guard, folk-remedy counters.
5. `engine/` — renderer + session state machine.

Then run the **conformance suite**: `deploy/test_vectors.json` (generated by
`tools/export_test_vectors.py`) contains input→output pairs for every layer plus
full conversation transcripts. A port must reproduce them byte-exactly
(strings) / exactly (labels, node ids). Wire it into the mobile CI.

Regex note: patterns use only `(?:…)`, alternation, `\b`, `{m,n}` lookahead —
supported by ICU (Android) and NSRegularExpression (iOS) unchanged.

## 5. App-store / clinical-safety posture

- Position as **first-aid guidance/education**, not diagnosis (see scope guard).
  This aligns with FDA "general wellness"/CDS discretion policies and similar
  EU MDR class-I-adjacent guidance for first-aid instruction apps — obtain
  local regulatory review before launch regardless.
- Content provenance: every protocol carries `sources` (ILCOR/AHA/IFRC/ERC
  2020–2021 consensus, WHO, NCPC) and is covered by the G15 clinical-invariant
  audit — 261 guideline facts (rates, depths, durations, doses, warn-only terms)
  asserted on every run; any new number or unaudited protocol fails the build
  until clinically reviewed. Verified against primary sources during
  development: WHO rabies (15-minute wound washing), NCPC button-battery triage
  (honey protocol), AHA 2020 opioid response (rescue breathing). Re-review the
  KB against guideline updates **annually** (ILCOR publishes CoSTR updates every
  ~5 years, interim statements yearly).
- The bench suite (`bench/run_bench.py`) is the release gate: **no build ships
  unless all 11 gates are green.** Wire into CI; the KB is data, so content
  changes re-run the full safety sweep automatically.
- Log nothing off-device. Transcripts stay local, user-erasable.

## 6. Model download & packaging strategy

Ship the APK/IPA with tier-0 (engine+KB, 0.2 MB) always working. Speech and
vision models are optional expansion packs downloaded **once** on Wi-Fi (with
checksums) and stored locally — after that the app never needs a network again.
For fully-offline distribution (disaster kits), preload the "Voice" tier on
sideloaded builds (≈130 MB APK).

**Content currency channel:** the KB is pure data, so guideline updates ship as
versioned content packs (a few hundred KB) independent of app releases. Each
pack must pass the full 23-gate suite, including G20: the pack's content hash
must match a dated clinical-review record (tools/mark_reviewed.py), reviews
expire after 400 days, and any unreviewed edit fails the build. Devices apply
packs opportunistically when connectivity exists; the bundled pack always keeps
working offline. Watch for: annual ILCOR CoSTR interim statements, the next
IFRC International First Aid Guidelines edition, and AHA focused updates.

## 7. Localization (shipped: en/es/fr)

Spanish and French are live as pure-data language packs
(`firstaid/i18n/<lang>/`), gated by G23 (494 checks):
- **Input**: full es/fr lexicons for all 58 protocol families merged into the
  classifier (per-language TF-IDF vector spaces so translations never shift
  English geometry); es/fr entities, dialog acts, red flags, myth patterns.
- **Output**: translated system strings, EMS banners (`LLAME AL…`/`APPELEZ
  LE…`), myth counters, and the 11 time-critical protocols fully translated
  (CPR ×3, choking ×2, severe bleeding, anaphylaxis, stroke, unresponsive-
  breathing, breathing difficulty, general help). Everything else falls back to
  English steps behind a translated notice — honest, never silently wrong.
- **Detection**: marker-based, accent-stripped, sticky per session (switching
  requires strong evidence; a lone shared word like "no" never flips language
  mid-protocol).
- **Voice models per locale** (sherpa-onnx):
  - es ASR `sherpa-onnx-streaming-zipformer-es` / whisper-tiny multilingual;
    TTS Piper `es_ES-sharvard-medium` (~65 MB).
  - fr ASR `sherpa-onnx-streaming-zipformer-fr-2023-04-14`;
    TTS Piper `fr_FR-siwis-medium` (~65 MB).
  - Route `Session.lang` to the TTS voice at the app layer.
- Adding a language = new `i18n/<lang>/` pack (lexicon.json + pack.json),
  extend `_MARKERS`/`_STRONG`, translate tier-1 protocols, then pass G23 and
  the full 23-gate suite. No engine changes.
