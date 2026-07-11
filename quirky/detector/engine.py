import os
import uuid
import wave
import numpy as np
from PIL import Image
from typing import Dict, Any

from quirky.detector.formulas import (
    compute_image_ai_score,
    compute_image_plastic_score,
    compute_image_symmetry_score,
    compute_image_lighting_score,
    compute_image_texture_score,
    compute_image_repetition_score,
    compute_text_ai_score,
    compute_text_repetition_score,
    compute_text_prompt_leak_score,
    compute_audio_plastic_score,
    compute_audio_emotion_score,
    compute_image_spectral_slope,
    compute_image_channel_correlation
)
from quirky.detector.calibrate import calibrated_text_score

class DetectorEngine:
    @staticmethod
    def analyze_asset(filepath: str) -> Dict[str, Any]:
        """
        Conducts passive multi-dimensional statistical, geometric,
        and linguistic analysis on the input asset.
        Returns a unified JSON schema.
        """
        asset_id = str(uuid.uuid4())
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Asset file not found at {filepath}")
            
        ext = os.path.splitext(filepath)[1].lower()
        
        # Default empty metadata dictionary
        scores = {
            "ai_score": 0.0,
            "ai_probability": 0.0,
            "score_source": "heuristic",
            "plastic_score": 0.0,
            "emotion_score": 0.0,
            "symmetry_score": 0.0,
            "lighting_score": 0.0,
            "texture_score": 0.0,
            "repetition_score": 0.0,
            "prompt_leak_score": 0.0
        }
        
        # Modality checks
        if ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
            scores = DetectorEngine._analyze_image(filepath)
        elif ext in [".txt", ".md", ".json", ".html"]:
            scores = DetectorEngine._analyze_text(filepath)
        elif ext in [".wav", ".mp3", ".ogg", ".flac"]:
            scores = DetectorEngine._analyze_audio(filepath)
        elif ext in [".mp4", ".avi", ".mkv", ".mov"]:
            scores = DetectorEngine._analyze_video(filepath)
        else:
            # Fallback (try loading as text)
            try:
                scores = DetectorEngine._analyze_text(filepath)
            except Exception:
                pass
                
        return {
            "asset_id": asset_id,
            "filepath": filepath,
            "file_type": ext,
            "metadata": scores
        }
        
    @staticmethod
    def _analyze_image(filepath: str) -> Dict[str, float]:
        try:
            with Image.open(filepath) as img:
                img_gray = np.array(img.convert("L"))
                img_rgb = np.array(img.convert("RGB"))

            ai = compute_image_ai_score(img_gray)
            plastic = compute_image_plastic_score(img_gray)
            symmetry = compute_image_symmetry_score(img_gray)
            lighting = compute_image_lighting_score(img_gray)
            texture = compute_image_texture_score(img_gray)
            repetition = compute_image_repetition_score(img_gray)
            spectral_slope = compute_image_spectral_slope(img_gray)
            channel_corr = compute_image_channel_correlation(img_rgb)
            
            # Simple prompt leak check for images: read exif / png info chunks
            leak = 0.0
            with Image.open(filepath) as img:
                for k, v in img.info.items():
                    if any(t in str(k).lower() or t in str(v).lower() for t in ["parameters", "prompt", "negative prompt", "steps", "sampler"]):
                        leak = 0.95
                        break
            
            # Emotion score on images: mock facial landmark emotion (based on intensity dynamics)
            # In a real pipeline, this would route to a facial emotion classifier.
            emotion = float(np.clip(1.0 - plastic, 0.0, 1.0))
            
            return {
                "ai_score": round(ai, 3),
                "plastic_score": round(plastic, 3),
                "emotion_score": round(emotion, 3),
                "symmetry_score": round(symmetry, 3),
                "lighting_score": round(lighting, 3),
                "texture_score": round(texture, 3),
                "repetition_score": round(repetition, 3),
                "prompt_leak_score": round(leak, 3),
                "spectral_slope": round(spectral_slope, 3),
                "channel_corr": round(channel_corr, 3)
            }
        except Exception:
            return {
                "ai_score": 0.5, "plastic_score": 0.5, "emotion_score": 0.5, "symmetry_score": 0.5,
                "lighting_score": 0.5, "texture_score": 0.5, "repetition_score": 0.5, "prompt_leak_score": 0.0,
                "spectral_slope": -2.0, "channel_corr": 0.0
            }

    @staticmethod
    def _analyze_text(filepath: str) -> Dict[str, float]:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                
            ai = compute_text_ai_score(content)
            prob, source = calibrated_text_score(content)
            repetition = compute_text_repetition_score(content)
            leak = compute_text_prompt_leak_score(content)
            
            # Text plastic score: machine-like uniformity (e.g. low perplexity variance, lack of punctuation burstiness)
            sentences = [s.strip() for s in content.split(".") if s.strip()]
            if len(sentences) > 1:
                lens = [len(s.split()) for s in sentences]
                std_lens = np.std(lens)
                # Low standard deviation of sentence lengths indicates rigid/monotonous AI writing
                plastic = float(np.clip(1.0 - (std_lens / 15.0), 0.0, 1.0))
            else:
                plastic = 0.5
                
            # Emotion score: semantic variance (approximated by sentiment word check)
            emotion_words = ["love", "hate", "happy", "sad", "angry", "fear", "excited", "wow", "amazing", "horrible"]
            words_lower = content.lower().split()
            emotion_count = sum(1 for w in words_lower if w in emotion_words)
            emotion = float(np.clip(emotion_count / (len(words_lower) * 0.05 + 1e-8), 0.0, 1.0))
            
            return {
                "ai_score": round(ai, 3),
                "ai_probability": round(prob, 3),
                "score_source": source,
                "plastic_score": round(plastic, 3),
                "emotion_score": round(emotion, 3),
                "symmetry_score": 0.0,
                "lighting_score": 0.0,
                "texture_score": round(1.0 - plastic, 3),
                "repetition_score": round(repetition, 3),
                "prompt_leak_score": round(leak, 3)
            }
        except Exception:
            return {
                "ai_score": 0.5,
                "ai_probability": 0.5,
                "score_source": "heuristic",
                "plastic_score": 0.5, "emotion_score": 0.5, "symmetry_score": 0.0,
                "lighting_score": 0.0, "texture_score": 0.5, "repetition_score": 0.5, "prompt_leak_score": 0.0
            }

    @staticmethod
    def _analyze_audio(filepath: str) -> Dict[str, float]:
        try:
            # Read wave file using standard wave module
            with wave.open(filepath, 'rb') as wav:
                params = wav.getparams()
                nchannels, sampwidth, framerate, nframes = params[:4]
                str_data = wav.readframes(nframes)
                audio_data = np.frombuffer(str_data, dtype=np.int16)
                
            plastic = compute_audio_plastic_score(audio_data, framerate)
            emotion = compute_audio_emotion_score(audio_data, framerate)
            
            # Simple AI audio signature: absence of sub-100Hz breathing frequencies
            # combined with flat spectral harmonics
            ai = float(np.clip(plastic * 1.2 - emotion * 0.4, 0.0, 1.0))
            
            return {
                "ai_score": round(ai, 3),
                "plastic_score": round(plastic, 3),
                "emotion_score": round(emotion, 3),
                "symmetry_score": 0.0,
                "lighting_score": 0.0,
                "texture_score": 0.0,
                "repetition_score": round(plastic * 0.3, 3),
                "prompt_leak_score": 0.0
            }
        except Exception:
            # Fallback if it's compressed (like mp3) or corrupts
            return {
                "ai_score": 0.5, "plastic_score": 0.5, "emotion_score": 0.5, "symmetry_score": 0.0,
                "lighting_score": 0.0, "texture_score": 0.0, "repetition_score": 0.3, "prompt_leak_score": 0.0
            }

    @staticmethod
    def _analyze_video(filepath: str) -> Dict[str, float]:
        # Video is treated as a sequence of frames for detection
        # We sample 3 frames, run image detection on them, and average the results
        # In a real system, we'd use cv2.VideoCapture
        try:
            import cv2
            cap = cv2.VideoCapture(filepath)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames = []
            
            # Sample 3 frames (beginning, middle, end)
            for idx in [frame_count // 4, frame_count // 2, (3 * frame_count) // 4]:
                if idx < 0:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frames.append(gray)
            cap.release()
            
            if not frames:
                return {"ai_score": 0.5, "plastic_score": 0.5, "emotion_score": 0.5, "symmetry_score": 0.5, "lighting_score": 0.5, "texture_score": 0.5, "repetition_score": 0.5, "prompt_leak_score": 0.0}
                
            ais = [compute_image_ai_score(f) for f in frames]
            plastics = [compute_image_plastic_score(f) for f in frames]
            symmetries = [compute_image_symmetry_score(f) for f in frames]
            lightings = [compute_image_lighting_score(f) for f in frames]
            textures = [compute_image_texture_score(f) for f in frames]
            repetitions = [compute_image_repetition_score(f) for f in frames]
            
            return {
                "ai_score": round(float(np.mean(ais)), 3),
                "plastic_score": round(float(np.mean(plastics)), 3),
                "emotion_score": round(float(1.0 - np.mean(plastics)), 3),
                "symmetry_score": round(float(np.mean(symmetries)), 3),
                "lighting_score": round(float(np.mean(lightings)), 3),
                "texture_score": round(float(np.mean(textures)), 3),
                "repetition_score": round(float(np.mean(repetitions)), 3),
                "prompt_leak_score": 0.0
            }
        except Exception:
            return {
                "ai_score": 0.65, "plastic_score": 0.72, "emotion_score": 0.28, "symmetry_score": 0.5,
                "lighting_score": 0.6, "texture_score": 0.4, "repetition_score": 0.35, "prompt_leak_score": 0.0
            }
