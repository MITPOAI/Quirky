# Quirky Layers — the next-generation architecture

**A deep-dive of the open-source landscape (verified July 2026) plus a concrete plan to turn
Quirky from a one-shot pipeline into a full retouch / de-AI / re-map / clean studio — with
every capability available in a zero-weight classical form AND an optional, better, AI form.**

This doc is the output of researching ~35 open-source projects across image restoration,
segmentation, detection, provenance, classical signal-processing, audio, text and video —
with licenses verified against the actual repo/model-card, not assumed from memory. It ends
with a proposed architecture ("Quirky Layers") that ties everything into one coherent app,
a module layout consistent with the existing codebase, and a phased build order.

---

## 1. The architecture: "Quirky Layers"

Today's `ImageHumanizer.humanize()` is one flat function: measure a few things, apply a
fixed sequence of corrections, write the output. The `diagnose` layer (built last session)
already cracked this open into named, toggleable "fix cards" — that's the seed of something
bigger. The proposal is to formalize it into a **non-destructive layer stack**, the same
mental model as Photoshop adjustment layers, run through five stages:

```
 FINGERPRINT      SEGMENT           LAYERS                    RE-MAP        CLEAN
 (exists)         (new)             (extends fix-cards)       (new)         (new)
 ┌──────────┐    ┌──────────┐    ┌─────────────────────┐    ┌─────────┐   ┌──────────┐
 │ which     │    │ precise   │    │ ordered, toggleable  │    │ re-run   │   │ strip /   │
 │ generator │───▶│ mask:     │───▶│ corrections; EACH one │───▶│ diagnose │──▶│ read C2PA │
 │ made this │    │ subject / │    │ has a classical impl  │    │ + delta  │   │ + EXIF;   │
 │           │    │ face / bg │    │ (Tier 0) AND an       │    │ heatmap; │   │ optional  │
 │           │    │           │    │ optional AI impl      │    │ loop or  │   │ sign as   │
 │           │    │           │    │ (Tier 1)              │    │ stop     │   │ "edited"  │
 └──────────┘    └──────────┘    └─────────────────────┘    └─────────┘   └──────────┘
```

Every stage after Fingerprint is genuinely new or a formalization of something partial:

- **Segment** — today's masking is a skin-color filter blended with classical saliency
  (`image/pipeline.py`). Good enough for a single portrait, not for "select the sky" or
  "select this jacket." A real segmentation model turns Quirky from portrait-only into
  general-purpose.
- **Layers** — the diagnose fix-cards already ship as accept/reject toggles. The extension
  is giving each one a *second* implementation: e.g. the "plastic skin" fix currently only
  has the classical bilateral+grain path; with the `[dl]` registry it can offer GFPGAN or
  RestoreFormer++ as a one-click "use AI for this layer" upgrade, same UI, same fix-card,
  just a better engine behind it when the user opts in.
- **Re-map** — `humanize_locked()` already loops intensity against a target; formalizing
  this as an explicit re-diagnose-and-diff step after *any* layer or stack of layers is what
  "map it again" means literally — show the user the new heatmap next to the old one.
- **Clean** — `prompt_leak_score` already *detects* generator metadata leaks; there's no
  code that actually strips them yet. This closes that loop, and adds the option to go the
  other way: sign an honest "edited with Quirky" C2PA credential instead of hiding the edit
  (this is the direction `docs/RESPONSIBLE_USE.md` already points).

The Tier 0 / Tier 1 split is exactly the pattern `pyproject.toml` and `quirky/plugins/dl.py`
already establish (`[dl]` extra, `MODEL_REGISTRY`, commercial-safe licenses only). Nothing
below breaks that pattern — it extends the same registry.

---

## 2. Tier 0 — classical, zero-weight additions (core, always on)

Pure CPU signal processing. No downloads, no license risk, ships in the base install.

| Library | License | Adds to Quirky | Plugs into |
|---|---|---|---|
| **colour-science** | BSD-3-Clause | Real colorimetry: chromatic-adaptation white balance, CAM tone mapping — replaces the current gray-world approximation with proper color science | `image/pipeline.py` white-balance stage |
| **OpenColorIO** (PyOpenColorIO) | BSD-3-Clause | Film/ACES color-managed grading, real LUT/CDL transforms | new `image/grade.py` |
| **Little CMS 2** (via `Pillow.ImageCms`) | MIT/HPND | ICC-profile-aware color grading | `image/grade.py` |
| **rawpy** (+ LibRaw as a dependency, LGPL-2.1/CDDL — fine unmodified, don't vendor source) | MIT wrapper | Decode actual camera RAW files — real sensor data instead of approximating one | new `image/raw.py` |
| **colour-demosaicing** | BSD-3-Clause | Bilinear/Malvar/Menon CFA demosaic algorithms — a much better Bayer round-trip than the current hand-rolled one in `bayer_demosaic_roundtrip()` | `image/pipeline.py` |
| **filmgrainer** | MIT | Turnkey, tunable film-grain synthesis to complement the existing Poisson-Gaussian grain | `image/pipeline.py` |
| **hitherdither** | MIT | Halftone/dither patterns — useful for a "print scan" humanize preset | new preset |
| **OpenCV `fastNlMeansDenoising`** + **scikit-image `restoration`** (wavelet, TV-Chambolle) | Apache-2.0 / BSD-3 | Real denoise for the "clean up a noisy/over-sharpened AI image" direction (the inverse of humanize — useful for retouch mode) | new `image/denoise.py` |
| **piexif** + **Pillow** re-encode | MIT / HPND | Actually strip EXIF/PNG metadata (the "Clean" stage) — pairs with the existing `prompt_leak_score` detector, which currently only detects, never acts | new `clean/metadata.py` |
| **dlib** | Boost Software License | 68-point face landmarks as a lighter-weight alternative/companion to MediaPipe for retouch targeting | `image/transforms.py` |
| **pyworld / WORLD vocoder** | MIT wrapper + Modified-BSD core | Proper F0/jitter/shimmer extraction for audio — replaces the current envelope-based approximation in `detector/formulas.py` `compute_audio_plastic_score`, and is the commercial-safe alternative to GPL'd `praat-parselmouth` | `audio/pipeline.py` |

**Avoid bundling (copyleft/non-commercial landmines found in this category):**
`bm3d`/`bm4d` (Tampere non-commercial-only license — do not ship, even though it's the
"best" denoiser by reputation), `pyexiv2`/`exiv2` (GPL-2/3 — use piexif+Pillow instead, or
shell out to a separate ExifTool process under its Artistic License if deeper IPTC/XMP is
ever needed), `G'MIC`/`gmic-py` (CeCILL-2.1, GPL-equivalent — invoke the standalone `gmic`
CLI as an optional external tool if wanted, never link it in), Alasdair Newson's reference
film-grain-rendering code (GPL-3 — the *algorithm* is a published IPOL paper, safe to
reimplement from the paper, just don't copy the GPL code).

---

## 3. Tier 1 — optional AI, commercial-safe weights (extends `[dl]`)

Same pattern as today: `pip install quirky[dl]`, ONNX Runtime, CPU by default, weights
cached on first use, registered only if Apache/MIT/BSD.

| Tool | Code | Weights | CPU/ONNX | Adds | Registry task |
|---|---|---|---|---|---|
| **MobileSAM** | Apache-2.0 ✅ *(verified)* | Apache-2.0, distilled from SAM | Yes — built for CPU/edge, ~40MB | Click-to-mask precise segmentation — powers the new Segment stage | `task: "segment"` |
| **rembg** (IS-Net/U²-Net backends) | MIT | Apache-2.0 (verify per bundled checkpoint) | Yes, ONNXRuntime CPU-first | One-click subject cutout, no click needed | `task: "segment_auto"` |
| **DRCT** or **SwinIR** | MIT / Apache-2.0 | same | ONNX inference script / community exports | Better upscaler than the current Real-ESRGAN-only path for compressed/degraded inputs | extends `task: "upscale"` |
| **ZITS** | Apache-2.0 | same | unverified, GPU-typical | Structure-aware inpaint alternative to LaMa for large holes | extends `task: "inpaint"` |
| **RestoreFormer++** | Apache-2.0 ✅ *(verified)* | same | unverified | Alternative to GFPGAN for face restore — ⚠️ **last release Sep 2023, only 58 commits** — verify it still runs against current onnxruntime before shipping, don't treat as actively maintained | extends `task: "face_restore"` |
| **DeepFilterNet** | MIT/Apache-2.0 dual | same | Yes — real ONNX export + Intel OpenVINO port | Actual speech denoise — new capability, not currently in Quirky at all | new `task: "audio_denoise"` |
| **roberta-base-openai-detector** | MIT ✅ *(verified)* | same | Yes — 125M params, ONNX-exportable | One more *weak signal* for the text side of the honesty oracle (see §5 — do not treat as authoritative, the model card itself says so) | new `task: "text_detect_signal"` |
| **RIFE** — official `hzwer/Practical-RIFE` or `hzwer/ECCV2022-RIFE` repos ONLY | MIT | MIT | PyTorch native + community ncnn/VapourSynth ports | Real frame interpolation → genuine handheld-motion smoothing for the video pipeline, which today only does frame-wise texture, not motion | new `video/interpolate.py` |
| **c2pa-python** / **c2patool** | Apache-2.0 ✅ *(verified, Adobe copyright)* | n/a | N/A (metadata/signing) | Read, strip, or optionally sign C2PA Content Credentials — the "Clean" stage's provenance half | new `clean/provenance.py` |

**Avoid (confirmed landmines):**

| Tool | Why it's blocked |
|---|---|
| **CodeFormer** | S-Lab License 1.0 — non-commercial. Quirky already correctly avoids this; confirmed again this session. |
| **SUPIR** | Custom "SUPIR Software License" — explicitly non-commercial without a paid agreement, plus it drags in multi-GB SDXL+LLaVA-13B. |
| **GPEN** | No LICENSE file anywhere in the repo — no license = all rights reserved by default. Treat as unsafe, not just "research-only." |
| **IOPaint** (ex lama-cleaner) | Apache-2.0 code, but **the repo was archived Aug 13, 2025** — no longer maintained. Fine as a reference for its model catalog/UX, but don't depend on it live; fork/vendor a frozen snapshot if anything is reused, and note it also optionally bundles Stable-Diffusion-based erase models that carry separate, non-commercial terms. |
| **FastSAM** | AGPL-3.0 (via Ultralytics/YOLOv8) — copyleft, incompatible with Apache-2.0 distribution. |
| **so-vits-svc** | AGPL-3.0 — same problem. |
| **Coqui XTTS-v2 weights** | Coqui Public Model License — non-commercial, and Coqui Inc. is defunct so there's no commercial license to purchase even if you wanted one. (Piper TTS, MIT, is the safe alternative if TTS synthesis is ever needed.) |
| **DIRE** | Needs a full diffusion-model reconstruction pass to score one image — multi-GB, GPU-only, not viable for a CPU-first local tool. |
| **Ghostbuster** | Requires calling the OpenAI API — disqualifies it outright for an offline/local tool. |
| **language_tool_python** | GPL-3.0 wrapper (the underlying LanguageTool engine itself is LGPL) — if grammar-checking is ever wanted, run the LanguageTool server directly and call its HTTP API instead of importing this GPL wrapper. |

---

## 4. What this actually unlocks (mapped to what you asked for)

**"Retouch, touch up"** — Segment (precise masks beyond faces) + the AI half of the Layers
stage (RestoreFormer++/GFPGAN, DRCT/Real-ESRGAN, ZITS/LaMa) turns Quirky from
portrait-humanizer into a general local retouch tool: select any region, remove any object,
restore any face, upscale anything.

**"Make it less AI"** — unchanged core promise, but now every Layer has a *better* optional
engine (neural face-restore instead of bilateral-filter approximation) while the free path
still works with zero downloads.

**"Some layer, add new thing"** — that's the literal Layers architecture: each fix is an
independent, reorderable, toggleable operation, classical-or-AI per layer, instead of one
fixed function.

**"Map it again"** — the Re-map stage: after any edit, re-run `composite_slop_map` +
`diagnose_image` and show the before/after delta heatmap, feeding back into
`humanize_locked()`'s existing minimal-edit loop.

**"Clean"** — the Clean stage actually acts on what `prompt_leak_score` already detects:
strip EXIF/PNG/C2PA generator fingerprints — or, the more honest option, *sign* a real "this
was edited with Quirky" C2PA credential instead of scrubbing evidence of AI origin. Worth
deciding deliberately, since it's a values choice, not just an engineering one — flagging it
rather than silently picking a default.

**"Without AI or using AI better"** — the Tier 0 / Tier 1 split, exactly as it works today
for `upscale`/`repaint`/`voice-clone`, just extended to cover the new capabilities too.

---

## 5. Honesty note: the detector oracle should get *wider*, not more "authoritative"

None of the open AI-image detectors researched (organika/sdxl-detector, umm-maybe,
UniversalFakeDetect, DIRE, NPR/AIDE) are reliable general-purpose classifiers — they overfit
to the generator families in their training set and degrade toward chance on newer
generators or after simple recompression. Same story on the text side:
`roberta-base-openai-detector`'s own model card says outright it's "not high enough accuracy
for standalone detection" and should not be used to make serious allegations. SynthID is not
a self-hostable detector at all — it's a Google-side watermark/verification service, useful
only as a corroborating signal for Google-generated content specifically.

The right move for `diagnose/oracle.py` is *not* to swap in a "real" neural detector as
ground truth. It's to extend `EnsembleHeuristicOracle` into a genuine multi-signal ensemble
— classical statistics (already built) + optional roberta-signal for text + optional
image-detector signal — and keep reporting it the way `audit()` already frames it: a
transfer measurement against an independent signal, never a pass/fail verdict. This is
already the right instinct in the existing code; the research just confirms it and gives two
more weak-but-independent signals to add to the ensemble.

---

## 6. Proposed module layout

Consistent with the existing package structure (`quirky/detector`, `quirky/image`,
`quirky/diagnose`, `quirky/plugins/dl.py`):

```
quirky/
├── segment/              # NEW
│   ├── engine.py          # SAM/MobileSAM/rembg wrappers behind one interface
│   └── masks.py           # mask utilities: subject / face / sky / "brushed region"
├── clean/                 # NEW
│   ├── metadata.py         # piexif + Pillow re-encode: strip EXIF/PNG generator chunks
│   └── provenance.py       # c2pa-python: read / strip / optionally sign C2PA
├── image/
│   ├── grade.py            # NEW: colour-science + OpenColorIO real color grading
│   ├── raw.py              # NEW: rawpy RAW decode + colour-demosaicing
│   └── denoise.py          # NEW: classical denoise (inverse-direction retouch)
├── audio/
│   └── denoise.py          # NEW: DeepFilterNet wrapper (Tier 1)
├── video/
│   └── interpolate.py      # NEW: RIFE frame interpolation (Tier 1)
├── diagnose/
│   ├── layers.py           # NEW: formalizes fix-cards into an ordered, toggleable stack
│   ├── remap.py            # NEW: before/after delta heatmap + re-diagnose loop
│   └── oracle.py           # EXTEND: add roberta text-signal + image-detector signal to the ensemble
└── plugins/
    └── dl.py                # EXTEND MODEL_REGISTRY with: mobilesam, rembg/isnet, drct,
                              #   zits, restoreformerpp, deepfilternet, roberta_detector, rife
```

Every new module follows the same rule already in place: Tier 0 modules (`clean/metadata`,
`image/grade`, `image/raw`, `image/denoise`) have zero dependency on `[dl]` and work in the
base install; Tier 1 modules (`segment/engine` for the AI backends, `audio/denoise`,
`video/interpolate`, and the neural half of `diagnose/layers`) route through
`quirky.plugins.dl.require_dl()` exactly like `upscale()`/`repaint()` do today.

---

## 7. Suggested build order

Roughly in order of "highest leverage for lowest new-dependency weight":

1. **Clean (metadata)** — piexif + Pillow, zero new heavy deps, directly closes the gap
   where `prompt_leak_score` detects but nothing acts. Small, fast, high trust value.
2. **Re-map** — mostly wiring existing `composite_slop_map` + `diagnose_image` +
   `humanize_locked` together with a delta view; no new dependencies at all.
3. **Layers (formalize fix-cards)** — restructure the existing accept/reject cards into a
   reorderable stack; still zero new dependencies, mostly an API/UI change.
4. **Segment (MobileSAM + rembg)** — ~40MB MobileSAM is cheap; this unlocks general (not
   just portrait) retouch and every Tier-1 image layer benefits from precise masks.
5. **Layers, AI half (RestoreFormer++, DRCT, ZITS)** — bigger downloads, register in
   `MODEL_REGISTRY` following the existing pattern exactly.
6. **Provenance (c2pa-python)** — small dependency, but needs a deliberate decision on
   strip-vs-sign before shipping (see §4).
7. **Audio denoise (DeepFilterNet) / Video interpolation (RIFE)** — new modalities of
   capability, larger scope, good candidates for their own dedicated pass.

Text-detector-as-oracle-signal and RAW/color-science are lower urgency — nice depth, not
blocking anything else.

---

*Sources checked live this session (license/maintenance verified against the actual repo,
LICENSE file, or model card rather than assumed): MobileSAM, c2pa-python, SAM/SAM2,
so-vits-svc, praat-parselmouth, RIFE (hzwer repos), Real-ESRGAN, colour-science,
rawpy/LibRaw, bm3d, pyexiv2/exiv2, G'MIC, CodeFormer, IOPaint, SwinIR, HAT, DRCT, SUPIR,
RestoreFormer++, GPEN, roberta-base-openai-detector, DeepFilterNet, textstat,
language_tool_python. A handful of secondary items (MAT, DiffBIR, exact rembg per-checkpoint
provenance) are marked unverified in the detailed research and should get a final license
check before anything ships commercially.*
