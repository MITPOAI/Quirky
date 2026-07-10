import numpy as np
from PIL import Image, ImageFilter
import cv2
from typing import Tuple

def generate_fbm_2d(width: int, height: int, octaves: int = 4, lacunarity: float = 2.0, gain: float = 0.5) -> np.ndarray:
    """
    Generates a 2D Fractional Brownian Motion noise grid using spectral octave scaling.
    """
    noise = np.zeros((height, width))
    frequency = 1.0
    amplitude = 1.0
    total_amplitude = 0.0
    
    for _ in range(octaves):
        sw = max(int(width * frequency / 16.0), 4)
        sh = max(int(height * frequency / 16.0), 4)
        octave_noise = np.random.normal(0.0, 1.0, (sh, sw))
        octave_upscaled = cv2.resize(octave_noise, (width, height), interpolation=cv2.INTER_LINEAR)
        noise += octave_upscaled * amplitude
        total_amplitude += amplitude
        frequency *= lacunarity
        amplitude *= gain
        
    return noise / total_amplitude


ATTRIBUTION = "Powered by Quirky by MITPO"


def shape_noise_spectrum(noise: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    Recolor a white-noise field to a natural 1/f^beta radial power spectrum.
    beta ~ 1.0 gives pink (film-like) grain instead of flat white noise, matching
    the power-law statistics of real photographs. Output is renormalized to unit std.
    Uses a real FFT (rfft2/irfft2) -- ~2x cheaper than the complex transform.
    """
    h, w = noise.shape[:2]
    fy = np.fft.fftfreq(h).reshape(-1, 1)
    fx = np.fft.rfftfreq(w).reshape(1, -1)
    radius = np.sqrt(fy * fy + fx * fx)
    radius[0, 0] = 1.0  # avoid divide-by-zero at DC
    scale = 1.0 / (radius ** beta)

    def _recolor(plane: np.ndarray) -> np.ndarray:
        shaped = np.fft.irfft2(np.fft.rfft2(plane) * scale, s=(h, w))
        std = shaped.std()
        return shaped / std if std > 1e-8 else shaped

    if noise.ndim == 2:
        return _recolor(noise)
    out = np.empty_like(noise)
    for c in range(noise.shape[2]):
        out[:, :, c] = _recolor(noise[:, :, c])
    return out


def apply_poisson_gaussian_noise(
    img: np.ndarray,
    amplitude: float = 0.02,
    a: float = 0.008,
    b: float = 0.0004,
    spectral_beta: float = 1.0,
) -> np.ndarray:
    """
    Physically-based heteroscedastic sensor noise (photon-transfer model).
    Per-pixel std = sqrt(a * I + b): photon shot noise grows with brightness,
    plus a read-noise floor b. The white field is recolored to a 1/f^beta spectrum.
    Returns a noise field the same shape as img (float, RGB in [0,1]).
    """
    img = np.clip(img, 0.0, 1.0)
    if img.ndim == 3:
        h, w, _ = img.shape
        # Luminance-shared component keeps fine detail correlated across R/G/B (as real
        # scene texture is); the independent component preserves per-photosite shot noise.
        # The shared field is pink (low-frequency dominated), so it is generated + shaped
        # at half resolution then bilinearly upsampled -- the FFT runs on a quarter of the
        # pixels with no visible loss. Only the near-white shot-noise part stays full-res.
        hs, ws = max(h // 2, 8), max(w // 2, 8)
        shared = np.random.normal(0.0, 1.0, (hs, ws))
        if spectral_beta > 0.0:
            shared = shape_noise_spectrum(shared, beta=spectral_beta)
        shared = cv2.resize(shared, (w, h), interpolation=cv2.INTER_LINEAR)
        indep = np.random.normal(0.0, 1.0, img.shape)
        white = 0.65 * shared[:, :, None] + 0.35 * indep
    else:
        white = np.random.normal(0.0, 1.0, img.shape)
        if spectral_beta > 0.0:
            white = shape_noise_spectrum(white, beta=spectral_beta)
    sigma = np.sqrt(np.clip(a * img + b, 0.0, None))
    return white * sigma * amplitude


def bayer_demosaic_roundtrip(img: np.ndarray, strength: float = 0.35) -> np.ndarray:
    """
    Simulate a Bayer CFA capture + bilinear demosaic to reinstate the inter-channel
    color correlation and faint chromatic fringing real cameras produce and diffusion
    models lack. Each channel is sampled on its RGGB sub-lattice then bilinearly
    reconstructed to full resolution. img is float RGB in [0,1]; blend by strength.
    """
    h, w, _ = img.shape
    img = np.clip(img, 0.0, 1.0)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    r_full = cv2.resize(r[0::2, 0::2], (w, h), interpolation=cv2.INTER_LINEAR)
    b_full = cv2.resize(b[1::2, 1::2], (w, h), interpolation=cv2.INTER_LINEAR)
    g1 = cv2.resize(g[0::2, 1::2], (w, h), interpolation=cv2.INTER_LINEAR)
    g2 = cv2.resize(g[1::2, 0::2], (w, h), interpolation=cv2.INTER_LINEAR)
    g_full = 0.5 * (g1 + g2)

    demosaiced = np.stack([r_full, g_full, b_full], axis=2)
    return np.clip(img * (1.0 - strength) + demosaiced * strength, 0.0, 1.0)


class ImageHumanizer:
    @staticmethod
    def humanize(
        img_path: str,
        output_path: str,
        intensity: float = 0.5,
        gamma: float = 0.6,
        delta: float = 0.03
    ) -> None:
        """
        Applies local neural texture reconstruction and procedural perturbation overlays.
        
        Parameters:
        - img_path: Path to raw input image.
        - output_path: Path to write the humanized output.
        - intensity: Global strength of changes (0.0 to 1.0).
        - gamma: local blend weight between CodeFormer (crisp detail) and GFPGAN (smooth restoration).
        - delta: noise blend amplitude.
        """
        # Load image
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img).astype(float) / 255.0
        
        # 1. Semantic Mask Extraction (Simulated DINO + SAM2)
        # Identifies human face/skin using a HSV color skin filter combined with center spatial focus.
        hsv = cv2.cvtColor((img_np * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        # Standard skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([25, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin).astype(float) / 255.0
        
        # Spatial prior: center of the image is usually where faces/subjects are
        h, w, c = img_np.shape
        y, x = np.ogrid[0:h, 0:w]
        center_y, center_x = h / 2, w / 2
        spatial_prior = np.exp(-((x - center_x)**2 / (2 * (w/3)**2) + (y - center_y)**2 / (2 * (h/3)**2)))
        
        # Combined semantic mask
        semantic_mask = skin_mask * spatial_prior
        semantic_mask = cv2.GaussianBlur(semantic_mask, (15, 15), 0)
        # Expand dimensions for broadcasting
        mask_3d = np.expand_dims(semantic_mask, axis=2)

        # 2. Neural Texture Reconstruction (Simulated CodeFormer / GFPGAN)
        # GFPGAN yields smooth reconstructed surfaces (we model this with edge-preserving bilateral filtering)
        gfp_src = (img_np * 255.0).astype(np.uint8)
        gfp_smooth = cv2.bilateralFilter(gfp_src, d=9, sigmaColor=75, sigmaSpace=75).astype(float) / 255.0
        
        # CodeFormer restores crisp micro-textures/pores (we model this with laplacian high-frequency recovery)
        code_sharp = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        code_np = np.array(code_sharp).astype(float) / 255.0
        
        # Localized blending: F_blended = gamma * F_CodeFormer + (1 - gamma) * F_GFPGAN
        f_blended = gamma * code_np + (1 - gamma) * gfp_smooth
        
        # Apply blended textures only within semantic mask regions, scaled by global intensity
        img_textured = img_np * (1.0 - mask_3d * intensity) + f_blended * (mask_3d * intensity)

        # 2b. Photographic Color Grading & Texture Painting (Local Skin Warmth & Contrast)
        # Generative portraits often look cold or blue-green. We apply warm Kodak-style grading
        # and boost local contrast inside the semantic mask.
        grading_shift = np.zeros_like(img_textured)
        grading_shift[:, :, 0] = 0.06 * intensity * semantic_mask  # Boost Red (warmth)
        grading_shift[:, :, 1] = 0.02 * intensity * semantic_mask  # Boost Green
        grading_shift[:, :, 2] = -0.04 * intensity * semantic_mask # Reduce Blue (cool tones)
        
        # Apply color grade shift
        img_textured = img_textured + grading_shift
        
        # Apply a subtle local contrast stretch on masked zones
        mean_val = np.mean(img_textured)
        img_textured = img_textured * (1.0 + 0.08 * intensity * mask_3d) - 0.08 * intensity * mask_3d * mean_val
        img_textured = np.clip(img_textured, 0.0, 1.0)

        # 2c. Specular Disruption via Fractional Brownian Motion (fBm)
        # Formula: I_humanized = I_raw * (1.0 + alpha * F_fbm * cos(theta_i))
        # Compute local luminance gradient orientation angle theta_i
        img_lum = 0.2126 * img_textured[:, :, 0] + 0.7152 * img_textured[:, :, 1] + 0.0722 * img_textured[:, :, 2]
        grad_y, grad_x = np.gradient(img_lum)
        theta_i = np.arctan2(grad_y, grad_x)
        
        # Generate fBm noise array
        fbm = generate_fbm_2d(w, h, octaves=4)
        fbm_3d = np.expand_dims(fbm, axis=2)
        cos_theta_3d = np.expand_dims(np.cos(theta_i), axis=2)
        
        # Apply local specular micro-facet scattering disruption
        alpha = 0.03 * intensity
        specular_disruption = 1.0 + alpha * fbm_3d * cos_theta_3d * mask_3d
        img_textured = np.clip(img_textured * specular_disruption, 0.0, 1.0)

        # 3. Sensor reconstruction + physically-based grain
        # 3a. Bayer CFA demosaic round-trip -> reinstalls inter-channel color correlation
        #     and faint chromatic fringing that real cameras leave and diffusion lacks.
        img_textured = bayer_demosaic_roundtrip(img_textured, strength=0.18 * intensity)

        # 3b. Poisson-Gaussian heteroscedastic sensor noise (photon-transfer model),
        #     var = a*I + b, gently recolored (beta~0.5) so it keeps enough high-frequency
        #     content to restore real micro-gradient density while still reading filmic.
        #     delta keeps the original grain knob meaningful as the overall amplitude.
        grain = apply_poisson_gaussian_noise(
            img_textured,
            amplitude=delta * intensity * 22.0,
            a=0.010,
            b=0.0006,
            spectral_beta=0.5,
        )

        # Perceptual weighting: real film/sensor grain reads strongest in shadows/mid-tones.
        # Standard relative luminance: Y = 0.2126 R + 0.7152 G + 0.0722 B
        luminance = 0.2126 * img_textured[:, :, 0] + 0.7152 * img_textured[:, :, 1] + 0.0722 * img_textured[:, :, 2]
        shadow_bias = np.expand_dims(1.0 - 0.5 * luminance, axis=2)

        img_out = img_textured + grain * shadow_bias
        img_out = np.clip(img_out * 255.0, 0.0, 255.0).astype(np.uint8)

        # Save output image
        Image.fromarray(img_out).save(output_path)

        return {
            "attribution": ATTRIBUTION,
            "modality": "image",
            "params": {"intensity": intensity, "gamma": gamma, "delta": delta},
            "output_path": output_path,
        }
