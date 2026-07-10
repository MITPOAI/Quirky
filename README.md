# Quirky: Human Preference Engine

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Support](https://img.shields.io/badge/python-3.10%20%7C%203.11-indigo)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-darkgreen)]()

Quirky is a modular, high-fidelity **Human Preference Engine** designed to sit directly between generative AI foundation models and downstream distribution networks. Rather than raw synthesis, Quirky processes synthetic media output (text, image, audio, video) to analyze, score, and reconstruct features—restoring natural, human-like imperfection and stripping out AI signatures.

Created by **MITPO** as a universal alignment layer.

---

## ⚡ Quickstart (Local Run)

Install and run Quirky locally in under 30 seconds using `uv`:

```bash
# Clone the repository
git clone https://github.com/mitpo/quirky.git && cd quirky

# Run detection on any synthetic asset
uv run quirky detect --asset sample.png

# Humanize and restore pores/micro-textures
uv run quirky humanize --asset sample.png --output restored.png --intensity 60

# Generate a visual comparison share card
uv run quirky compare sample.png restored.png --output comparison.png
```

---

## 📦 Project Architecture & Modality Layout

Quirky is designed with strict modular boundaries. Every directory operates as an independent package, allowing you to import specific components without loading unnecessary dependencies or weights.

```
quirky/
├── cli/              # Command-line interface engine (built with Typer and uv)
├── api/              # Production FastAPI server and websocket gateways
├── detector/         # Heart of the analytical pipeline (non-editing metadata scoring)
├── image/            # Image artifact reduction pipeline (SAM2, ControlNet, Real-ESRGAN)
├── video/            # Video motion and sensor correction pipeline (FFmpeg, RIFE, Depth V2)
├── audio/            # Speech dynamic processing engine (VAD, StyleTTS2, F5-TTS)
├── text/             # Text rhythm and perplexity sculpting engine (LiteLLM, Transformers)
├── agent/            # Cognitive reasoning middleware (MCP SDK, LangGraph)
├── sdk/              # Production-grade async Python client SDK
├── benchmarks/       # Ground-truth evaluation suites and statistical metrics
├── examples/         # Implementation templates and deployment recipes
└── web/              # Next.js workspace UI and side-by-side comparison interfaces
```

---

## 🔬 Mathematical Formulations

Quirky computes quality profiles via a passive, multi-dimensional detection engine.

### 1. Plastic Score ($S_{\text{plastic}}$)
Quantifies the density of local gradient variations in visual layers. Flat, airbrushed regions lacking organic noise yield higher scores:
$$S_{\text{plastic}} = 1.0 - \frac{1}{N} \sum_{i,j} \|\nabla I(i,j)\|_2 \cdot \mathbb{1}_{\{\|\nabla I(i,j)\|_2 > \epsilon\}}$$

### 2. Symmetry Score ($S_{\text{symmetry}}$)
Perfect pixel-level symmetry indicates synthetic face synthesis:
$$S_{\text{symmetry}} = \frac{\sum_{x,y} |I(x, y) - I(W - x, y)|}{\sum_{x,y} I(x, y)}$$

### 3. AI Spectral Anomaly Score ($S_{\text{AI}}$)
Measures high-frequency periodic spikes inside the 2D discrete Fourier transform space, identifying artificial noise grids.

---

## 🖥️ Running the Web Dashboard

Quirky includes a gorgeous dark-mode web application featuring glassmorphism elements, parameter controllers, and an interactive **before/after comparison slider**.

To run the web interface locally:

1. Start the FastAPI backend server:
   ```bash
   uv run python -m quirky.api.main
   ```
2. Open your browser and navigate to:
   ```
   http://127.0.0.1:8000
   ```

### UI Features
- **Upload Dropzone**: Drag and drop images, audio wavs, or text files directly.
- **Dynamic Parameter Sliders**: Tune global intensity ($\delta$), pore blending ($\gamma$), and film grain levels.
- **Split-Screen Drag-Slider**: Drag the dividing handle horizontally across images to inspect the pore-restored detail side-by-side.
- **Donut Metrics & Logs**: View real-time WebSocket processing updates and unified metric scores.

---

## 🛠️ Installation Guide

Ensure you have Python 3.10+ and [uv](https://github.com/astral-sh/uv) installed.

### Dev Installation
```bash
# Create virtual environment and install packages
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### Run Tests and Validation Examples
```bash
# Run the validation example
uv run python quirky/examples/demo.py

# Run the benchmark test suite
uv run python quirky/benchmarks/bench.py
```

---

## 🤝 Open Source Contributor Growth

We want you to join the MITPO creator network!
- **Curated First Issues**: We label introductory issues with `good-first-issue` and `help-wanted`.
- **Contributor Recognition**: Active contributors are featured in our release galleries.
- **Design Governance**: Join our bi-weekly open-architecture design calls.

---

## 📄 License
Licensed under the permissive Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
