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

        stem, ext = os.path.splitext(filepath)
        ext_lower = ext.lower()

        if output_path is None:
            output_path = f"{stem}.humanized{ext}"
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
