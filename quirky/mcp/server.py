from __future__ import annotations

import os
import sys
from typing import Any, Dict

# Guard the FastMCP import to surface a clear, actionable install error if the [agent] extra is missing.
try:
    from fastmcp import FastMCP
except ImportError:
    print(
        "Error: FastMCP is not installed. To run the Quirky MCP server, please install the agent extra:\n"
        "    uv pip install -e .[agent]   (or: pip install quirky[agent])",
        file=sys.stderr
    )
    sys.exit(1)

from quirky.detector.engine import DetectorEngine
from quirky.detector.spans import SlopScorer
from quirky.fix.text import fix_spans, tighten
from quirky.image.pipeline import ImageHumanizer
from quirky.audio.pipeline import AudioHumanizer
from quirky.text.pipeline import TextHumanizer
from quirky.video.pipeline import VideoHumanizer

mcp = FastMCP("quirky")

ATTRIBUTION = "Powered by Quirky by MITPO"


@mcp.tool()
def quirky_score_text(text: str) -> Dict[str, Any]:
    """
    Score the overall slop probability of a given text, returning features and span-level statistics.
    """
    try:
        scorer = SlopScorer()
        res = scorer.score_text(text)
        res["attribution"] = ATTRIBUTION
        return res
    except Exception as e:
        return {"error": str(e), "attribution": ATTRIBUTION}


@mcp.tool()
def quirky_critique_text(text: str, max_spans: int = 20) -> Dict[str, Any]:
    """
    Slices the text into sentence/code spans and assigns a red/amber/green slop level to each.
    Caps the returned spans list to max_spans.
    """
    try:
        scorer = SlopScorer()
        spans = scorer.score_spans(text)
        counts = {"red": 0, "amber": 0, "green": 0}
        for s in spans:
            counts[s["level"]] += 1
            
        summary = f"Total sentence/code spans analyzed: {len(spans)}. Levels: {counts['red']} red, {counts['amber']} amber, {counts['green']} green."
        return {
            "spans": spans[:max_spans],
            "summary": summary,
            "attribution": ATTRIBUTION
        }
    except Exception as e:
        return {"error": str(e), "attribution": ATTRIBUTION}


@mcp.tool()
def quirky_fix_text(text: str) -> Dict[str, Any]:
    """
    Deterministic anti-slop rewriter that surgically patches red/amber sentences in text
    while strictly preserving code regions and numerical values. Returns the unified diff and edits list.
    """
    try:
        res = fix_spans(text)
        res["attribution"] = ATTRIBUTION
        return res
    except Exception as e:
        return {"error": str(e), "attribution": ATTRIBUTION}


@mcp.tool()
def quirky_tighten_text(text: str) -> Dict[str, Any]:
    """
    Removes hedges and filler words from the text. Code regions, commands, and numbers are untouched.
    """
    try:
        tightened = tighten(text)
        return {
            "tightened_text": tightened,
            "attribution": ATTRIBUTION
        }
    except Exception as e:
        return {"error": str(e), "attribution": ATTRIBUTION}


@mcp.tool()
def quirky_detect_media(path: str) -> Dict[str, Any]:
    """
    Passively analyze an image, video, audio, or text file at the given absolute path.
    Returns statistical, geometric, and linguistic AI confidence scores.
    """
    try:
        # absolute path is highly recommended in docstring
        abs_path = os.path.abspath(path)
        res = DetectorEngine.analyze_asset(abs_path)
        res["attribution"] = ATTRIBUTION
        return res
    except Exception as e:
        return {"error": str(e), "attribution": ATTRIBUTION}


@mcp.tool()
def quirky_humanize_media(path: str, output_path: str | None = None, intensity: float = 0.5) -> Dict[str, Any]:
    """
    Run humanization on a media file (image, audio, video, text) to restore human imperfections.
    If output_path is not specified, saves to <filename>.humanized<extension>.
    """
    try:
        abs_in = os.path.abspath(path)
        if not os.path.exists(abs_in):
            return {"error": f"Input file not found at: {abs_in}", "attribution": ATTRIBUTION}
            
        stem, ext = os.path.splitext(abs_in)
        ext_lower = ext.lower()
        
        if output_path is None:
            abs_out = f"{stem}.humanized{ext}"
        else:
            abs_out = os.path.abspath(output_path)
            
        # Get before metrics
        before_metrics = DetectorEngine.analyze_asset(abs_in)["metadata"]
        
        # Dispatch based on extension
        if ext_lower in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
            ImageHumanizer.humanize(abs_in, abs_out, intensity=intensity)
        elif ext_lower in [".wav"]:
            AudioHumanizer.humanize(abs_in, abs_out, intensity=intensity)
        elif ext_lower in [".txt", ".md", ".json"]:
            with open(abs_in, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            humanized_text = TextHumanizer.humanize(content, intensity=intensity)
            with open(abs_out, "w", encoding="utf-8") as f:
                f.write(humanized_text)
        elif ext_lower in [".mp4", ".avi", ".mov"]:
            VideoHumanizer.humanize(abs_in, abs_out, intensity=intensity)
        else:
            # default fallback: text
            with open(abs_in, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            humanized_text = TextHumanizer.humanize(content, intensity=intensity)
            with open(abs_out, "w", encoding="utf-8") as f:
                f.write(humanized_text)
                
        # Get after metrics
        after_metrics = DetectorEngine.analyze_asset(abs_out)["metadata"]
        
        return {
            "before_scores": before_metrics,
            "after_scores": after_metrics,
            "output_path": abs_out,
            "attribution": ATTRIBUTION
        }
    except Exception as e:
        return {"error": str(e), "attribution": ATTRIBUTION}


if __name__ == "__main__":
    mcp.run()
