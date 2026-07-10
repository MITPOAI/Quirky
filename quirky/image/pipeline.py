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

        # 3. Procedural Perturbation Overlay (mimic analog camera sensor grain)
        # Formula: I_out(x,y) = I_in(x,y) + delta * N(x,y) * (1 - L(x,y))
        # Compute local luminance L(x,y)
        # Standard relative luminance coefficients: Y = 0.2126 R + 0.7152 G + 0.0722 B
        luminance = 0.2126 * img_textured[:, :, 0] + 0.7152 * img_textured[:, :, 1] + 0.0722 * img_textured[:, :, 2]
        luminance_3d = np.expand_dims(luminance, axis=2)
        
        # Generate multi-scale noise N(x, y)
        noise_fine = np.random.normal(0, 0.5, img_textured.shape)
        noise_coarse = np.random.normal(0, 0.5, (h // 2, w // 2, c))
        noise_coarse_upscaled = cv2.resize(noise_coarse, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Multi-scale procedural noise
        noise_pattern = 0.7 * noise_fine + 0.3 * noise_coarse_upscaled
        
        # Apply grain heavily in mid-tones and shadows: delta * N * (1 - L)
        # We scale this by a factor of 3.0 to make the film grain crisp and visually apparent
        grain = delta * intensity * noise_pattern * (1.0 - luminance_3d) * 3.0
        
        # Add grain to image and clip to valid bounds
        img_out = img_textured + grain
        img_out = np.clip(img_out * 255.0, 0.0, 255.0).astype(np.uint8)
        
        # Save output image
        Image.fromarray(img_out).save(output_path)
