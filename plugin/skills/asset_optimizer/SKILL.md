---
name: asset_optimizer
description: >
  Detects an image, audio, or video file by its magic bytes and runs the
  matching signal-processing pipeline to restore human sensor imperfections.
  Trigger: synthetic media (Flux, TTS, generated video) that looks too clean.
---

# Asset Optimizer Skill

Generated media is too perfect: no sensor grain, flat lighting, static pitch, locked framing. This skill puts the physical imperfections back.

## When to use

- An image came out of a diffusion model and reads plastic.
- TTS audio holds a dead, unwavering pitch.
- Generated video sits on a tripod smooth path no hand ever makes.

## Format detection (magic bytes)

The first 16 bytes decide the modality. The file extension is only a fallback when no signature matches.

```text
PNG        89 50 4E 47 0D 0A 1A 0A          -> .png
JPEG       FF D8 FF                          -> .jpg
BMP        42 4D  ("BM")                     -> .bmp
RIFF/WEBP  52 49 46 46 .. .. .. .. 57 45 42 50 -> .webp
RIFF/WAVE  52 49 46 46 .. .. .. .. 57 41 56 45 -> .wav
MP4        .. .. .. .. 66 74 79 70  ("ftyp")  -> .mp4
```

## Routing

```text
.png .jpg .jpeg .webp .bmp  -> ImageHumanizer
.wav                        -> AudioHumanizer
.mp4 .avi .mov              -> VideoHumanizer
anything else               -> TextHumanizer (fallback)
```

## Image pipeline

- **Bayer demosaic round trip.** Simulate a color-filter-array capture and bilinear demosaic to reinstate inter channel color correlation.
- **Poisson-Gaussian sensor noise.** Heteroscedastic photon-transfer grain, spectrally shaped toward pink (`beta` near `1.0`) so it reads like film, not flat white noise.
- **CLAHE lighting.** Local contrast on the LAB `L` channel breaks flat, variance-dead lighting.
- **Retinex relighting.** Single-scale Retinex (log `I` minus log Gaussian `I`) plus a soft vignette, face targeted when a face is detected.
- **Gray-world white balance.** A small cast correction toward neutral.

## Audio pipeline

- **Voice activity partition.** Split speech from silence first.
- **Pitch-period jitter.** Per period timing drift at roughly `0.008 * intensity`, driven by pink noise.
- **Shimmer.** Per period amplitude drift at roughly `0.04 * intensity`, also pink.
- **Phrase-final lengthening.** Stretch the last stretch of a phrase, the way people slow down at the end.
- **Glottal tilt drift and breath.** A slowly drifting spectral tilt plus aspiration and inserted breaths.

Every micro variation follows a 1/f pink spectrum, which is how real vocal drift behaves, not white noise.

## Video pipeline

- **Fractal camera drift.** A smoothed handheld trajectory over `dx`, `dy`, and rotation.
- **Rolling-shutter correction.** Sub-frame correction, blended selectively inside a tracked region.

## Parameters and output

- `intensity` runs `0.0` to `1.0` (default `0.5`); the CLI takes `0` to `100`.
- Output defaults to `{stem}.humanized{ext}` when no path is given.

## Before and after

Measure it, don't eyeball it. Run the detector on the source and the result and compare the metrics.

```text
quirky detect --asset input.png
quirky humanize --asset input.png --output out.png --intensity 50
quirky detect --asset out.png
```
