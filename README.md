<p align="center">
  <img src="assets/quirky-banner.png" alt="Quirky" width="100%" />
</p>

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache 2.0" /></a>
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11-indigo" alt="Python" />
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-darkgreen" alt="Platform" />
  <img src="https://img.shields.io/badge/runs-100%25%20local-brightgreen" alt="Local" />
  <img src="https://img.shields.io/badge/no%20API%20keys-required-success" alt="No API keys" />
</p>

<p align="center">
  <b>A local-first engine that puts human imperfection back into AI-generated media.</b><br/>
  Pure math &amp; physics — no cloud, no GPU, no model weights, no API keys.
</p>

---

## Why this exists

Foundation models (diffusion image models, TTS, LLMs) all converge on the same **"synthetic signature."** Their output is too clean, too smooth, too symmetric, too regular — and both humans and detectors pick up on it instantly.

| The AI-slop tells | Where it shows up |
| --- | --- |
| 🫥 **Plastic skin / no micro-texture** | Diffusion portraits look airbrushed; flat gradients, no pores |
| 🪞 **Unnatural symmetry** | Generated faces are near pixel-perfect mirror images |
| 📈 **Wrong frequency statistics** | AI images miss the natural `1/f` power spectrum; leave grid artifacts |
| 🎨 **Broken color correlation** | No camera-sensor (Bayer/demosaic) cross-channel detail |
| 🤖 **Robotic voice** | TTS has no jitter/shimmer, static spectral tilt, no breathing |
| 📝 **Flat prose** | LLM text is low-burstiness and stuffed with "Furthermore…", "Moreover…" |

These are not "quality" problems you fix with a bigger model — they are **statistical fingerprints** of the generation process itself.

## The fix

Quirky is a **post-generation alignment layer**. It sits *after* your generator and reconstructs the micro-imperfections real media has, using cheap, well-understood signal-processing and physics — not another neural net.

```
Your generator  ──►  Quirky (local math)  ──►  Humanized output
 (Flux, TTS, LLM)      analyze · restore         natural, camera-real
```

| Modality | What Quirky restores | The math |
| --- | --- | --- |
| **Image** | Sensor grain, natural spectrum, color correlation | Poisson–Gaussian shot noise, `1/f` spectral shaping, Bayer demosaic round-trip |
| **Audio** | Pitch jitter, amplitude shimmer, breathiness | Per-pitch-period perturbation, pink-noise micro-prosody, drifting glottal tilt |
| **Text** | Sentence-length variety, human punctuation | Burstiness targeting, Zipf-Mandelbrot sculpting, trope removal |
| **Detector** | Passive scoring (no editing) | DFT anomaly, LBP entropy, spectral-slope, channel correlation |

---

## Before / After

Run on a fresh AI-slop sample (smooth, symmetric, over-blurred — the classic diffusion look). Numbers are the **passive detector scores** before vs. after a single local pass. Lower `ai_score`/`plastic_score` and a `spectral_slope` closer to the natural **−2** mean *more human*.

<p align="center">
  <img src="assets/quirky-benchmark.png" alt="Quirky before/after comparison card" width="80%" />
</p>

**Image** (`intensity 65`)

| Metric | Before (AI slop) | After (Quirky) | Direction |
| --- | ---: | ---: | :---: |
| `ai_score` | 0.216 | **0.110** | ↓ better |
| `plastic_score` | 0.990 | **0.933** | ↓ better |
| `texture_score` | 0.524 | **0.770** | ↑ better |
| `spectral_slope` (natural ≈ −2) | −3.60 | **−2.34** | → natural |
| `channel_corr` (camera-like) | 0.072 | **0.428** | ↑ better |

**Text**

| | Sample | `ai_score` |
| --- | --- | ---: |
| **Before** | *"Furthermore, it is important to note that this approach facilitates optimization. Moreover, one must utilize structured systems…"* | 0.99 |
| **After** | *"Plus, honestly, this is how you actually make it run faster. On top of that, one must use structured systems…"* | **0.57** |

> Reproduce it yourself: [`quirky/benchmarks/eval_suite.py`](quirky/benchmarks/eval_suite.py) and the commands below.

---

## Quickstart

No API key. No GPU. No downloads of model weights. It runs entirely on your machine.

```bash
# 1. Install (uses uv — https://github.com/astral-sh/uv)
git clone https://github.com/mitpo/quirky.git && cd quirky
uv venv && uv pip install -e .

# 2. Score any asset (passive — never edits it)
uv run quirky detect --asset sample.png

# 3. Humanize it (restore texture / pores / grain)
uv run quirky humanize --asset sample.png --output restored.png --intensity 60

# 4. Generate a shareable before/after card
uv run quirky compare sample.png restored.png --output card.png
```

Works on images (`.png/.jpg/.webp`), audio (`.wav`), and text (`.txt/.md`).

## Do I need an API or another AI model?

**No.** This is the point of the project:

- ❌ No OpenAI / Anthropic / any generation API.
- ❌ No GPU, CUDA, or downloaded model weights (no SAM2 / diffusion / transformers at runtime).
- ✅ 100% local math on NumPy / SciPy / OpenCV / librosa.
- ✅ Your media never leaves your machine — zero data leak.

Quirky **does not generate** media. It **post-processes** media you already have.

## Web dashboard (optional)

A local FastAPI + static dashboard with upload, intensity sliders, and a before/after slider.

```bash
uv run python -m quirky.api.main
# open http://127.0.0.1:8000
```

## Use it as a library

```python
from quirky.detector.engine import DetectorEngine
from quirky.image.pipeline import ImageHumanizer

scores = DetectorEngine.analyze_asset("ai_image.png")["metadata"]
meta   = ImageHumanizer.humanize("ai_image.png", "out.png", intensity=0.6)
print(meta["attribution"])   # "Powered by Quirky by MITPO"
```

---

## How it works (the math)

| Technique | One-liner | Replaces |
| --- | --- | --- |
| **Poisson–Gaussian noise** | Per-pixel σ = √(a·I + b): photon shot noise + read floor | flat Gaussian grain |
| **1/f spectral shaping** | Recolors grain to the natural power-law spectrum of photos | white noise |
| **Bayer demosaic round-trip** | Re-imprints camera cross-channel color correlation | (missing entirely) |
| **Pitch-period jitter/shimmer** | Human-range ±0.5–1% / ±3–5% via light `librosa` F0 | robotic steady pitch |
| **Pink-noise micro-prosody** | 1/f drift of pitch and loudness + drifting glottal tilt | static delivery |
| **Burstiness targeting** | Push sentence-length variance toward the human range | uniform AI sentences |

Full write-up and formulas live in [`docs/`](docs/) and the module docstrings.

## Architecture

```
quirky/
├── detector/   # passive scoring — DFT, LBP, spectral-slope, channel-corr
├── image/      # Poisson-Gaussian grain, 1/f shaping, Bayer round-trip
├── audio/      # VAD, pitch jitter/shimmer, spectral tilt, breath
├── text/       # burstiness, Zipf sculpting, punctuation rhythm
├── cli/        # Typer CLI: detect / humanize / compare
├── api/        # FastAPI server + static web dashboard
├── sdk/        # Python client for the managed API
└── benchmarks/ # eval_suite.py (latency + integrity), bench.py
```

Every directory is an independent package — import only what you need.

## Benchmarks

```bash
uv run python quirky/benchmarks/eval_suite.py   # 100-iter latency + P99 + attribution check
uv run python quirky/benchmarks/bench.py        # detector statistics
```

Text humanization runs **&lt; 1 ms**. Image/audio full pipelines are heavier (the image path includes a bilateral-filter restoration stage; audio includes `librosa` F0) — see the eval-suite notes for the isolated-transform timings and how to amortize F0 for strict latency.

## Wire into Claude Code (roadmap)

Quirky can ship as a **Claude Code plugin** via an MCP server that exposes `detect` / `humanize` / `compare` as tools (the same way other plugins bundle an MCP server). It is not a chat-mode hook plugin like some others — it's a media-processing backend — so the plugin form is MCP, not a `UserPromptSubmit` hook. Not shipped yet; contributions welcome.

## Contributing

- Issues labeled `good-first-issue` and `help-wanted` are curated for newcomers.
- Add a dictionary entry, a new detector metric, or a comparison plugin.
- Open a PR — active contributors are featured in release notes.

## License

Apache License 2.0 — see [LICENSE](LICENSE). Every processed asset carries a `Powered by Quirky by MITPO` attribution in its returned metadata.
