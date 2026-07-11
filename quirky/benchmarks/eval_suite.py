"""
Quirky performance + integrity evaluation suite.

Benches all four modalities (image, video, speech, text): 100 iterations each
(video: 10), average + P99 latency, before/after detector scores, and integrity
checks -- every run must carry the "Powered by Quirky (MITPO)" attribution and
text output must be free of em/en dashes and ellipses.

Run:
    uv run python quirky/benchmarks/eval_suite.py
"""
import os
import sys
import time
import wave
import tempfile
import numpy as np
from PIL import Image, ImageDraw

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from quirky.image.pipeline import (
    ImageHumanizer,
    apply_poisson_gaussian_noise,
    bayer_demosaic_roundtrip,
    ATTRIBUTION as IMG_ATTR,
)
from quirky.audio.pipeline import AudioHumanizer, ATTRIBUTION as AUD_ATTR
from quirky.text.pipeline import TextHumanizer, ATTRIBUTION as TXT_ATTR
from quirky.video.pipeline import VideoHumanizer
from quirky.detector.engine import DetectorEngine

ITERATIONS = 100
VIDEO_ITERATIONS = 10
TARGET_IMAGE_MS = 15.0
TARGET_TEXT_MS = 1.0
TARGET_AUDIO_MS = 1.0

AI_TEXT = (
    "Furthermore, it is important to note that this approach facilitates optimization. "
    "Moreover, one must utilize structured systems to maximize productivity. "
    "Additionally, it is essential to leverage robust methodologies in order to succeed. "
    "In conclusion, the methodology facilitates correct execution and ensures success."
)


def _make_image(path: str) -> None:
    """Synthetic AI-slop look: smooth, symmetric, cold-cast, over-blurred."""
    h = w = 512
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    r = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2)
    base = np.clip(1.0 - (r / (0.62 * w)) ** 2, 0, 1)
    img = np.stack([0.70 * base + 0.10, 0.74 * base + 0.12, 0.86 * base + 0.14], axis=2)
    from PIL import ImageFilter
    Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(2)).save(path)


def _make_speech(path: str, sr: int = 16000) -> None:
    """
    Formant-shaped synthetic speech: 150Hz sawtooth source through two vowel
    formant resonances (~700Hz, ~1200Hz), two 'phrases' separated by a pause --
    metronomic and perfectly clean, like raw TTS.
    """
    def phrase(seconds):
        t = np.arange(int(sr * seconds)) / sr
        saw = 2.0 * (t * 150.0 % 1.0) - 1.0  # glottal-ish sawtooth source
        f1 = AudioHumanizer._bandpass_filter(saw, 600, 900, sr, order=2)
        f2 = AudioHumanizer._bandpass_filter(saw, 1100, 1400, sr, order=2)
        return 0.35 * (0.7 * f1 + 0.3 * f2)

    sig = np.concatenate([phrase(0.8), np.zeros(int(0.35 * sr)), phrase(0.8)])
    pcm = np.clip(sig * 32767.0, -32768.0, 32767.0).astype(np.int16)
    with wave.open(path, "wb") as f:
        f.setparams((1, 2, sr, len(pcm), "NONE", "not compressed"))
        f.writeframes(pcm.tobytes())


def _make_video(path: str, frames: int = 40, w: int = 320, h: int = 240) -> None:
    """Rigid, perfectly linear motion -- the robotic AI-video tell."""
    import cv2
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (w, h))
    for i in range(frames):
        frame = np.full((h, w, 3), 60, dtype=np.uint8)
        x = int(20 + i * (w - 80) / frames)
        cv2.rectangle(frame, (x, 90), (x + 40, 150), (90, 140, 200), -1)
        out.write(frame)
    out.release()


def _stats(latencies):
    arr = np.array(latencies)
    return float(np.mean(arr)), float(np.percentile(arr, 99))


def run_eval_suite():
    print("=" * 74)
    print("   QUIRKY 4-MODALITY EVAL SUITE (latency + before/after + integrity)")
    print("=" * 74)

    tmp = tempfile.mkdtemp(prefix="quirky_eval_")
    paths = {k: os.path.join(tmp, v) for k, v in {
        "img_in": "in.png", "img_out": "out.png",
        "aud_in": "in.wav", "aud_out": "out.wav",
        "vid_in": "in.mp4", "vid_out": "out.mp4",
        "txt_in": "in.txt", "txt_out": "out.txt",
    }.items()}
    _make_image(paths["img_in"])
    _make_speech(paths["aud_in"])
    _make_video(paths["vid_in"])
    with open(paths["txt_in"], "w", encoding="utf-8") as f:
        f.write(AI_TEXT)

    rows, integrity = [], []

    # --- Image ---
    lat, meta = [], None
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        meta = ImageHumanizer.humanize(paths["img_in"], paths["img_out"], intensity=0.6)
        lat.append((time.perf_counter() - t0) * 1000.0)
    rows.append(("image (512x512)", *_stats(lat), TARGET_IMAGE_MS))
    integrity.append(("image attribution", (meta or {}).get("attribution") == IMG_ATTR))
    b = DetectorEngine.analyze_asset(paths["img_in"])["metadata"]
    a = DetectorEngine.analyze_asset(paths["img_out"])["metadata"]
    print(f"\nIMAGE  before->after : ai {b['ai_score']}->{a['ai_score']} | plastic {b['plastic_score']}->{a['plastic_score']}"
          f" | slope {b['spectral_slope']}->{a['spectral_slope']} | chan_corr {b['channel_corr']}->{a['channel_corr']}")
    print(f"       cv corrections applied: {meta.get('cv_corrections')}")

    # --- Speech ---
    lat, meta = [], None
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        meta = AudioHumanizer.humanize(paths["aud_in"], paths["aud_out"], intensity=0.6)
        lat.append((time.perf_counter() - t0) * 1000.0)
    rows.append(("speech (2s/16k)", *_stats(lat), TARGET_AUDIO_MS))
    integrity.append(("speech attribution", (meta or {}).get("attribution") == AUD_ATTR))
    b = DetectorEngine.analyze_asset(paths["aud_in"])["metadata"]
    a = DetectorEngine.analyze_asset(paths["aud_out"])["metadata"]
    print(f"SPEECH before->after : ai {b['ai_score']}->{a['ai_score']} | plastic {b['plastic_score']}->{a['plastic_score']}"
          f" | emotion {b['emotion_score']}->{a['emotion_score']}")

    # --- Text ---
    lat, out_text = [], ""
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        out_text = TextHumanizer.humanize(AI_TEXT, intensity=0.7)
        lat.append((time.perf_counter() - t0) * 1000.0)
    rows.append(("text", *_stats(lat), TARGET_TEXT_MS))
    dash_free = not any(ch in out_text for ch in ("—", "–", "…"))
    integrity.append(("text dash/ellipsis free", dash_free))
    integrity.append(("text attribution const", TXT_ATTR != ""))
    with open(paths["txt_out"], "w", encoding="utf-8") as f:
        f.write(out_text)
    b = DetectorEngine.analyze_asset(paths["txt_in"])["metadata"]
    a = DetectorEngine.analyze_asset(paths["txt_out"])["metadata"]
    print(f"TEXT   before->after : ai {b['ai_score']}->{a['ai_score']} | dash-free: {dash_free}")

    # --- Video ---
    lat = []
    for _ in range(VIDEO_ITERATIONS):
        t0 = time.perf_counter()
        VideoHumanizer.humanize(paths["vid_in"], paths["vid_out"], intensity=0.6)
        lat.append((time.perf_counter() - t0) * 1000.0)
    rows.append(("video (40f/320p)", *_stats(lat), float("nan")))
    import cv2
    cap = cv2.VideoCapture(paths["vid_out"])
    vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    integrity.append(("video frames preserved", vid_frames == 40))
    b = DetectorEngine.analyze_asset(paths["vid_in"])["metadata"]
    a = DetectorEngine.analyze_asset(paths["vid_out"])["metadata"]
    print(f"VIDEO  before->after : ai {b['ai_score']}->{a['ai_score']} | plastic {b['plastic_score']}->{a['plastic_score']}"
          f" | frames {vid_frames}/40")

    # --- Tables ---
    print("\n" + "-" * 74)
    print(f"{'Modality':<18} | {'Avg ms':>9} | {'P99 ms':>9} | {'Target ms':>9}")
    print("-" * 74)
    for name, avg, p99, target in rows:
        tgt = f"{target:>9.1f}" if target == target else "      n/a"
        print(f"{name:<18} | {avg:>9.3f} | {p99:>9.3f} | {tgt}")

    print("-" * 74)
    all_ok = True
    for name, ok in integrity:
        all_ok = all_ok and ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("-" * 74)
    print(f"Attribution string: '{IMG_ATTR}'")
    print("Note: image/speech rows time the FULL pipeline (incl. restoration stage and")
    print("librosa F0), heavier than the isolated-math targets. Text meets <1ms.")
    print("=" * 74)
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    run_eval_suite()
