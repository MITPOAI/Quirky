"""
Lightweight computer-vision touch-up transforms for the Quirky image engine.

Zero-VRAM, CPU-only, classical CV (OpenCV + NumPy). Optionally uses MediaPipe
FaceMesh for precise face targeting when the `quirky[vision]` extra is installed;
falls back to a skin+saliency mask otherwise. No heavy deep-learning weights.

Public API:
    detect_synthetic_frequency_signature(image) -> float   # FFT grid-spike AI score
    detect_face_regions(image_rgb) -> Optional[mask]        # MediaPipe (optional)
    detect_blemishes(gray, region_mask=None) -> mask        # top-hat spot finder
    remove_spots(image_rgb, blemish_mask, strength) -> img  # cv2.inpaint touch-up
    analyze_and_fix_portrait_lighting(image_rgb, intensity, face_mask=None) -> (img, meta)
"""
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2

ATTRIBUTION = "Powered by Quirky (MITPO)"

# Lazy MediaPipe singleton (optional dependency).
_FACE_MESH = None
_FACE_MESH_TRIED = False


def _get_face_mesh():
    """Return a cached MediaPipe FaceMesh, or None if the extra is not installed."""
    global _FACE_MESH, _FACE_MESH_TRIED
    if _FACE_MESH_TRIED:
        return _FACE_MESH
    _FACE_MESH_TRIED = True
    try:
        import mediapipe as mp  # optional: quirky[vision]
        _FACE_MESH = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=False,
            min_detection_confidence=0.5,
        )
    except Exception:
        _FACE_MESH = None
    return _FACE_MESH


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Accept float[0,1] or uint8 RGB; return uint8 RGB."""
    if image.dtype != np.uint8:
        return np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def detect_synthetic_frequency_signature(image: np.ndarray) -> float:
    """
    FFT grid-spike detector (Fast Fourier Transform, zero VRAM). Generative models
    leave concentrated, periodic high-frequency peaks from their upsampling stack;
    real optics/skin spread frequency chaotically. Returns a normalized ai_score in
    [0,1]: high = strong artificial periodicity.
    """
    if image is None or image.size == 0:
        return 0.5
    u8 = _as_uint8_rgb(image)
    gray = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY).astype(np.float32) if u8.ndim == 3 else u8.astype(np.float32)
    # The grid-spike signature is scale-stable; cap the FFT at 256px for speed.
    if max(gray.shape) > 256:
        s = 256 / max(gray.shape)
        gray = cv2.resize(gray, (max(int(gray.shape[1] * s), 16), max(int(gray.shape[0] * s), 16)),
                          interpolation=cv2.INTER_AREA)
    h, w = gray.shape
    if h < 16 or w < 16:
        return 0.5

    mag = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[-cy:h - cy, -cx:w - cx]
    r = np.sqrt(xx * xx + yy * yy)
    rmax = np.sqrt(cy * cy + cx * cx)

    # High-frequency ring (outer 40%): AI grids create isolated spikes here.
    ring = mag[r > 0.6 * rmax]
    if ring.size == 0:
        return 0.5
    mean_hf = float(ring.mean()) + 1e-8
    # peak-to-average + spike concentration (fraction of energy in top 0.5% of bins)
    peak_ratio = float(ring.max()) / mean_hf
    thresh = np.percentile(ring, 99.5)
    spikes = ring[ring > thresh]
    spike_fraction = float((ring > thresh).mean()) * (spikes.sum() / (ring.sum() + 1e-8))

    score = 0.6 * np.clip((peak_ratio - 2.5) / 10.0, 0, 1) + 0.4 * np.clip(spike_fraction * 40.0, 0, 1)
    return float(np.clip(score, 0.0, 1.0))


def detect_face_regions(image_rgb: np.ndarray) -> Optional[np.ndarray]:
    """
    Precise face mask via MediaPipe FaceMesh convex hull (optional quirky[vision]).
    Returns a float [0,1] mask (soft edges) or None if unavailable / no face found.
    """
    mesh = _get_face_mesh()
    if mesh is None:
        return None
    u8 = _as_uint8_rgb(image_rgb)
    h, w = u8.shape[:2]
    try:
        res = mesh.process(u8)
        if not res.multi_face_landmarks:
            return None
        pts = np.array([[int(lm.x * w), int(lm.y * h)]
                        for lm in res.multi_face_landmarks[0].landmark], dtype=np.int32)
        hull = cv2.convexHull(pts)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        soft = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(h, w) / 60.0).astype(np.float32) / 255.0
        return soft
    except Exception:
        return None


def detect_blemishes(gray: np.ndarray, region_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Classical blemish/spot finder: black-hat + white-hat morphology isolates small
    dark/bright dots (stray specks, JPEG spots, over-rendered pores) that differ from
    the local surface. Restricted to region_mask (e.g. face) when provided.
    Returns a uint8 mask (255 = spot) suitable for cv2.inpaint.
    """
    g = gray.astype(np.uint8) if gray.dtype != np.uint8 else gray
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, kernel)
    tophat = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, kernel)
    resp = cv2.max(blackhat, tophat)
    thr = max(18, int(np.percentile(resp, 99.0)))
    _, mask = cv2.threshold(resp, thr, 255, cv2.THRESH_BINARY)
    # keep only small blobs (spots), drop large structures
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    if region_mask is not None:
        mask = (mask * (region_mask > 0.15)).astype(np.uint8)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return mask


def remove_spots(image_rgb: np.ndarray, blemish_mask: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Content-aware spot removal via cv2.inpaint (Telea). Genuine touch-up: the removed
    region is reconstructed from surrounding pixels, not blurred. strength blends the
    inpainted result back toward the original so it stays subtle.
    """
    if blemish_mask is None or blemish_mask.max() == 0:
        return image_rgb
    was_float = image_rgb.dtype != np.uint8
    u8 = _as_uint8_rgb(image_rgb)
    bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    inp = cv2.inpaint(bgr, blemish_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32)
    out = u8.astype(np.float32) * (1.0 - strength) + inp * strength
    out = np.clip(out, 0, 255)
    return (out / 255.0) if was_float else out.astype(np.uint8)


def analyze_and_fix_portrait_lighting(
    image_rgb: np.ndarray, intensity: float = 0.5, face_mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Physical relighting: split the luminance (Y) channel into an illumination layer I
    (bilateral-filtered) and a reflectance layer R (Retinex, R = Y / I). Attenuate the
    artificial flat HDR glow by compressing I toward its mean, then re-inject subtle
    micro-shadow from R's edges. Applied within face_mask when provided.
    Returns (image, meta). Accepts float[0,1] or uint8 RGB; returns the same dtype.
    """
    was_float = image_rgb.dtype != np.uint8
    u8 = _as_uint8_rgb(image_rgb)
    h, w = u8.shape[:2]
    ycc = cv2.cvtColor(u8, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    Y = ycc[:, :, 0]

    # Illumination via edge-preserving bilateral (cheap at 1/2 res, then upsample).
    small = cv2.resize(Y, (max(w // 2, 8), max(h // 2, 8)), interpolation=cv2.INTER_AREA)
    illum_s = cv2.bilateralFilter(small, d=9, sigmaColor=40, sigmaSpace=40)
    illum = cv2.resize(illum_s, (w, h), interpolation=cv2.INTER_LINEAR)
    reflect = Y / (illum + 1e-3)

    # Compress flat HDR glow: pull illumination toward its mean.
    comp = illum.mean() + (illum - illum.mean()) * (1.0 - 0.35 * intensity)
    # Micro-shadow: darken where reflectance has structure (edges of features).
    edges = cv2.Laplacian(reflect, cv2.CV_32F, ksize=3)
    shadow = 1.0 - np.clip(np.abs(edges) * 0.15 * intensity, 0.0, 0.25)
    Y_new = np.clip(comp * reflect * shadow, 0, 255)

    if face_mask is not None:
        m = np.clip(face_mask, 0.0, 1.0)
        Y_new = Y * (1.0 - m) + Y_new * m

    ycc[:, :, 0] = Y_new
    out = cv2.cvtColor(np.clip(ycc, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)
    meta = {"relight": round(float(intensity), 3), "face_targeted": face_mask is not None}
    return (out.astype(np.float32) / 255.0 if was_float else out), meta


def run_touchup_benchmark(iterations: int = 50) -> Dict[str, Any]:
    """Benchmark the freq-scan + lighting fix on a mock 512x512 image (CPU, <20ms goal)."""
    import time
    rng = np.random.default_rng(0)
    img = rng.random((512, 512, 3), dtype=np.float32)
    # warm
    detect_synthetic_frequency_signature(img)
    analyze_and_fix_portrait_lighting(img, 0.6)
    lat = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = detect_synthetic_frequency_signature(img)
        _img, _ = analyze_and_fix_portrait_lighting(img, 0.6)
        lat.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(lat)
    payload = {
        "attribution": ATTRIBUTION,
        "avg_ms": round(float(arr.mean()), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "target_ms": 20.0,
        "met_target": bool(arr.mean() < 20.0),
        "mediapipe_available": _get_face_mesh() is not None,
    }
    return payload


if __name__ == "__main__":
    import json
    print(json.dumps(run_touchup_benchmark(), indent=2))
