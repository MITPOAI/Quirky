from __future__ import annotations

import os
from typing import Any

from quirky.skills.base import BaseSkill
from quirky.image.pipeline import ImageHumanizer
from quirky.audio.pipeline import AudioHumanizer
from quirky.video.pipeline import VideoHumanizer
from quirky.text.pipeline import TextHumanizer

class AssetOptimizerSkill(BaseSkill):
    """
    Optimizes media assets (images, audio, video) to restore human imperfections.
    Identifies file modalities using magic bytes headers dynamically.
    """
    @property
    def name(self) -> str:
        return "asset_optimizer"

    @property
    def description(self) -> str:
        return "Optimizes image, audio, video, or fallback text files by applying signal-processing pipelines."

    def execute(self, content: str, **kwargs) -> str:
        if not isinstance(content, str):
            raise TypeError("AssetOptimizerSkill requires a file path string.")

        filepath = os.path.abspath(content)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Input file not found at: {filepath}")

        intensity = kwargs.get("intensity", 0.5)
        output_path = kwargs.get("output_path")

        # Detect format dynamically by reading file header
        detected_ext = self._detect_format_by_header(filepath)
        if detected_ext:
            ext_lower = detected_ext
        else:
            ext_lower = os.path.splitext(filepath)[1].lower()

        # Build output path default if missing
        if output_path is None:
            stem, ext = os.path.splitext(filepath)
            # If input file had no extension, append the detected format
            ext_to_use = ext if ext else ext_lower
            output_path = f"{stem}.humanized{ext_to_use}"
        else:
            output_path = os.path.abspath(output_path)

        if ext_lower in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
            ImageHumanizer.humanize(filepath, output_path, intensity=intensity)
        elif ext_lower in [".wav"]:
            AudioHumanizer.humanize(filepath, output_path, intensity=intensity)
        elif ext_lower in [".mp4", ".avi", ".mov"]:
            VideoHumanizer.humanize(filepath, output_path, intensity=intensity)
        else:
            # Fallback: treat as text
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text_content = f.read()
            humanized = TextHumanizer.humanize(text_content, intensity=intensity)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(humanized)

        return output_path

    def _detect_format_by_header(self, filepath: str) -> str | None:
        try:
            with open(filepath, "rb") as f:
                header = f.read(16)
            if len(header) < 4:
                return None

            # PNG signature
            if header.startswith(b"\x89PNG\r\n\x1a\n"):
                return ".png"

            # JPEG signature
            if header.startswith(b"\xff\xd8\xff"):
                return ".jpg"

            # BMP signature
            if header.startswith(b"BM"):
                return ".bmp"

            # RIFF based formats (WAV, WebP)
            if header.startswith(b"RIFF") and len(header) >= 12:
                sub = header[8:12]
                if sub == b"WEBP":
                    return ".webp"
                if sub == b"WAVE":
                    return ".wav"

            # MP4 / Common video start signatures
            if len(header) >= 8 and (header[4:8] == b"ftyp" or header.startswith(b"\x00\x00\x00")):
                return ".mp4"
        except Exception:
            pass
        return None
