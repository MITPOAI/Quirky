import numpy as np
from PIL import Image
import math
import os
import re

def compute_image_ai_score(img_gray: np.ndarray) -> float:
    """
    Measures high-frequency spectral distribution anomalies in the Discrete Fourier Transform (DFT) space.
    Generative models often leave periodic high-frequency patterns/grid artifacts on a flat canvas.
    """
    try:
        h, w = img_gray.shape
        # Compute 2D DFT
        dft = np.fft.fft2(img_gray)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = np.abs(dft_shift)
        
        # Define high-frequency mask (outer ring)
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        r = np.sqrt(x*x + y*y)
        
        # Threshold for high frequency (outer 40% of frequencies)
        max_r = np.sqrt(cy**2 + cx**2)
        high_freq_mask = r > (0.6 * max_r)
        
        high_freq_mags = magnitude_spectrum[high_freq_mask]
        mean_hf = np.mean(high_freq_mags) + 1e-8
        max_hf = np.max(high_freq_mags)
        
        # Peak-to-average ratio of high frequencies.
        # AI grid artifacts create sharp, isolated Fourier spikes (high ratio).
        # Random photographic grain distributes energy uniformly (low ratio).
        peak_to_average = max_hf / mean_hf
        
        # Normalize anomaly ratio: typical AI grid spikes yield ratios > 8.0
        grid_anomaly = float(np.clip((peak_to_average - 2.5) / 10.0, 0.05, 0.98))
        
        # Calculate local gradient flatness. Synthetic images are overly flat/smoothed.
        grad_y, grad_x = np.gradient(img_gray.astype(float) / 255.0)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        mean_grad = np.mean(grad_mag)
        
        # Flatness index: close to 1.0 for flat images, close to 0.0 for heavily textured images
        flatness = float(np.clip(1.0 - (mean_grad / 0.035), 0.0, 1.0))
        
        # Combined AI score: scales with grid anomaly and flatness
        # If we add uniform noise, grid_anomaly drops and flatness drops, decreasing the score.
        ai_prob = grid_anomaly * (0.3 + 0.7 * flatness)
        return float(np.clip(ai_prob, 0.05, 0.98))
    except Exception:
        return 0.5

def compute_image_plastic_score(img_gray: np.ndarray, epsilon: float = 0.015) -> float:
    """
    Quantifies the density of local gradient variations:
    S_plastic = 1.0 - (1/N) * sum_{i,j} (||grad I(i,j)||_2 * II(||grad I(i,j)||_2 > epsilon))
    Where flat, over-smoothed patches lacking micro-textures yield an elevated score.
    """
    try:
        h, w = img_gray.shape
        N = h * w
        
        # Calculate gradients (Sobel filters or basic finite differences)
        grad_y, grad_x = np.gradient(img_gray.astype(float) / 255.0)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Apply indicator function where gradient magnitude > epsilon
        active_gradients = grad_mag * (grad_mag > epsilon)
        sum_active = np.sum(active_gradients)
        
        # Calibrate: instead of dividing by N (which keeps the value tiny),
        # normalize against a typical camera skin/pore gradient density factor (e.g. 0.03)
        grad_density = sum_active / N
        norm_factor = 0.025
        
        plastic_score = 1.0 - (grad_density / norm_factor)
        return float(np.clip(plastic_score, 0.05, 0.99))
    except Exception:
        return 0.8

def compute_image_symmetry_score(img_gray: np.ndarray) -> float:
    """
    Measures bilateral facial and geometric reflection coefficients across vertical midline:
    S_symmetry = sum_{x,y} |I(x,y) - I(W-x, y)| / sum_{x,y} I(x,y)
    
    Perfect pixel-level symmetry yields close to 0 in this raw value.
    We return 1.0 - raw_value for the indicator score so that 1.0 = perfectly symmetric (machine-like).
    """
    try:
        I = img_gray.astype(float)
        # Flip image horizontally
        I_flipped = np.fliplr(I)
        
        numerator = np.sum(np.abs(I - I_flipped))
        denominator = np.sum(I) + 1e-8
        
        raw_symmetry = numerator / denominator
        # Convert to score where 1.0 is perfectly symmetric, 0.0 is completely asymmetric
        symmetry_score = 1.0 - min(max(raw_symmetry * 2.0, 0.0), 1.0)
        return float(symmetry_score)
    except Exception:
        return 0.5

def compute_image_lighting_score(img_gray: np.ndarray) -> float:
    """
    Evaluates lighting anomalies and impossible ambient shading.
    Computes local intensity variance compared to regional averages to check for HDR halos and flat lighting.
    """
    try:
        # A simple estimate: ratio of high-luminance pixels without matching shadows
        I = img_gray.astype(float) / 255.0
        # Calculate local mean (ambient estimation)
        from scipy.ndimage import uniform_filter
        local_mean = uniform_filter(I, size=15)
        local_var = uniform_filter((I - local_mean)**2, size=15)
        
        # AI images tend to have unnaturally balanced/flat local variance (low contrast in textures)
        # combined with excessive global contrast (HDR halos).
        flat_variance_ratio = np.mean(local_var < 0.01)
        return float(np.clip(flat_variance_ratio, 0.0, 1.0))
    except Exception:
        return 0.4

def compute_image_texture_score(img_gray: np.ndarray) -> float:
    """
    Computes local binary pattern (LBP) entropy.
    Low scores indicate flat surface shading or repetitive pixel structures typical of diffusion outputs.
    """
    try:
        h, w = img_gray.shape
        I = img_gray.astype(int)
        
        # Calculate basic 8-neighbor LBP
        lbp = np.zeros((h-2, w-2), dtype=uint8 if 'uint8' in globals() else np.uint8)
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                # Shift and compare
                neighbor = I[i:i+h-2, j:j+w-2]
                center = I[1:-1, 1:-1]
                lbp += ((neighbor >= center) * (2 ** (i*3 + j))).astype(np.uint8)
                
        # Calculate histogram entropy
        hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]
        entropy = -np.sum(prob * np.log2(prob))
        
        # Maximum possible entropy for 8-bit LBP is 8.0.
        # AI images often have lower entropy due to flat regions and repetitive micro-structures.
        normalized_texture_score = float(entropy / 8.0)
        return normalized_texture_score
    except Exception:
        return 0.5

def compute_image_repetition_score(img_gray: np.ndarray) -> float:
    """
    Measures structural duplication and copy-move patterns.
    Uses 1D/2D auto-correlation.
    """
    try:
        I = img_gray.astype(float) - np.mean(img_gray)
        # Auto-correlation using FFT
        f = np.fft.fft2(I)
        f_conj = np.conj(f)
        r = np.real(np.fft.ifft2(f * f_conj))
        
        # Normalize and find off-center peaks (repetition indicator)
        r = r / (r[0, 0] + 1e-8)
        h, w = r.shape
        r[0, 0] = 0.0  # Zero out center
        max_peak = np.max(np.abs(r))
        return float(np.clip(max_peak * 2.0, 0.0, 1.0))
    except Exception:
        return 0.3

# --- Text Detection Formulas ---

def compute_text_ai_score(text: str) -> float:
    """
    Measures syntactic uniformity and predictability.
    """
    if len(text.strip()) < 10:
        return 0.2
    
    # AI text has highly stable sentence length variance and uses specific trigger words.
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return 0.5
        
    sentence_lengths = [len(s.split()) for s in sentences]
    # Human text has high variance in sentence lengths (high burstiness)
    length_variance = np.var(sentence_lengths) if len(sentence_lengths) > 1 else 0.0
    
    # Lower variance -> more likely AI
    burstiness_score = 1.0 - min(length_variance / 100.0, 1.0)
    
    # Common LLM transition words / boilerplate check
    ai_clichés = ["furthermore", "moreover", "in conclusion", "it is important to note", "as an ai", "firstly", "lastly", "consequently"]
    cliché_count = sum(1 for w in ai_clichés if w in text.lower())
    cliché_score = min(cliché_count / 3.0, 1.0)
    
    # Combined score
    ai_score = 0.6 * burstiness_score + 0.4 * cliché_score
    return float(np.clip(ai_score, 0.0, 1.0))

def compute_text_repetition_score(text: str) -> float:
    """
    Tracks n-gram recurrence to check for repetitive structure loops.
    """
    words = [w.lower() for w in re.findall(r'\b\w+\b', text)]
    if len(words) < 5:
        return 0.0
        
    # Check 3-grams
    trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
    if not trigrams:
        return 0.0
    unique_trigrams = set(trigrams)
    repetition = 1.0 - (len(unique_trigrams) / len(trigrams))
    return float(repetition)

def compute_text_prompt_leak_score(text: str) -> float:
    """
    Analyzes text for typical system instruction formats or formatting leftovers.
    """
    patterns = [
        r"(as an AI language model|system instructions|ignore previous instructions)",
        r"(Output format:|JSON format:|User:|Assistant:)",
        r"(\\n|### Instruction|### Response)",
        r"(helpful, harmless, and honest)"
    ]
    score = 0.0
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            score += 0.25
    return float(np.clip(score, 0.0, 1.0))

# --- Audio Detection Formulas ---

def compute_audio_plastic_score(audio_data: np.ndarray, sr: int) -> float:
    """
    Measures lack of spectral tilt and fundamental frequency (f0) micro-instability.
    Synthetic voices are too perfect and lack micro-jitter/shimmer.
    """
    try:
        # Simple micro-instability check using signal envelope difference
        env = np.abs(audio_data)
        env_diff = np.diff(env)
        # Synthetic speech is clean and has low envelope variance during pauses/vowels
        jitter_estimate = np.std(env_diff) / (np.mean(env) + 1e-8)
        # If jitter is extremely low (flat waves/perfect pitch), plastic score is high
        plastic = 1.0 - min(jitter_estimate * 50.0, 1.0)
        return float(np.clip(plastic, 0.0, 1.0))
    except Exception:
        return 0.6

def compute_audio_emotion_score(audio_data: np.ndarray, sr: int) -> float:
    """
    Evaluates prosodic range variation (variation in fundamental frequency amplitude).
    """
    try:
        # Estimate dynamic range variance
        env = np.abs(audio_data)
        env_mean = np.mean(env)
        if env_mean == 0:
            return 0.0
        dyn_range = np.std(env) / env_mean
        # A higher dynamic range variance correlates to higher vocal expression
        emotion = min(dyn_range * 1.5, 1.0)
        return float(emotion)
    except Exception:
        return 0.3

# --- Natural-Statistics Image Formulas (physically-grounded realism metrics) ---

def compute_image_spectral_slope(img_gray: np.ndarray) -> float:
    """
    Azimuthally-averaged (radial) power-spectrum slope in log-log space.
    Natural photographs follow a power law P(f) ~ f^alpha with alpha ~ -2.
    Diffusion/GAN output deviates (excess or deficit high-frequency energy),
    so a slope pushed back toward -2 indicates more natural statistics.
    """
    try:
        I = img_gray.astype(float)
        step = max(1, max(I.shape) // 256)  # coarse metric: cap work at ~256px
        if step > 1:
            I = I[::step, ::step]
        I = I - I.mean()
        f = np.fft.fftshift(np.fft.fft2(I))
        ps = np.abs(f) ** 2

        h, w = ps.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[-cy:h - cy, -cx:w - cx]
        r = np.sqrt(x * x + y * y).astype(int)

        rmax = int(min(cy, cx))
        if rmax < 8:
            return -2.0
        tbin = np.bincount(r.ravel(), ps.ravel())
        nr = np.bincount(r.ravel())
        radial = tbin[1:rmax] / (nr[1:rmax] + 1e-8)
        freqs = np.arange(1, rmax)

        mask = radial > 0
        if mask.sum() < 8:
            return -2.0
        slope, _ = np.polyfit(np.log(freqs[mask]), np.log(radial[mask]), 1)
        return float(slope)
    except Exception:
        return -2.0

def compute_image_channel_correlation(img_rgb: np.ndarray) -> float:
    """
    Mean absolute inter-channel correlation of high-pass residuals.
    Real cameras (Bayer CFA + demosaic) leave strongly correlated fine detail
    across R, G, B; many synthetic images do not. Higher = more camera-like.
    """
    try:
        arr = img_rgb.astype(float)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return 0.0
        step = max(1, max(arr.shape[:2]) // 256)  # coarse metric: cap work at ~256px
        if step > 1:
            arr = arr[::step, ::step, :]
        from scipy.ndimage import uniform_filter
        hp = [arr[:, :, c] - uniform_filter(arr[:, :, c], size=3) for c in range(3)]
        pairs = [
            np.corrcoef(hp[0].ravel(), hp[1].ravel())[0, 1],
            np.corrcoef(hp[1].ravel(), hp[2].ravel())[0, 1],
            np.corrcoef(hp[0].ravel(), hp[2].ravel())[0, 1],
        ]
        vals = [v for v in pairs if np.isfinite(v)]
        return float(np.mean(np.abs(vals))) if vals else 0.0
    except Exception:
        return 0.0
