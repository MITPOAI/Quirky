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

ATTRIBUTION = "Powered by Quirky (MITPO)"


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


@mcp.tool()
def quirky_list_skills() -> Dict[str, Any]:
    """
    Lists the names and descriptions of all registered agent skills.
    """
    try:
        from quirky.agent.core import QuirkyAgent
        agent = QuirkyAgent()
        return {
            "skills": agent.list_skills(),
            "attribution": ATTRIBUTION
        }
    except Exception as e:
        return {"error": str(e), "attribution": ATTRIBUTION}


@mcp.tool()
def quirky_run_agent(
    input_data: str,
    is_file: bool = True,
    output_path: str | None = None,
    intensity: float = 0.5
) -> Dict[str, Any]:
    """
    Runs QuirkyAgent on the input (file path or direct text content) to sequentially
    apply skills, analyze scores, and verify safety guards.
    """
    try:
        from quirky.agent.core import QuirkyAgent
        agent = QuirkyAgent()
        res = agent.run(
            input_data=input_data,
            is_file=is_file,
            output_path=output_path,
            intensity=intensity
        )
        res["attribution"] = ATTRIBUTION
        return res
    except Exception as e:
        return {"error": str(e), "attribution": ATTRIBUTION}


@mcp.tool()
def quirky_diagnose_image(path: str, intensity: float = 0.6) -> Dict[str, Any]:
    """
    Prescriptive diagnosis of an image: a list of named defects (color cast, flat
    lighting, plastic skin, spectrum, etc.) each with severity, a plain-English detail,
    and the specific fix id it recommends. Feeds accept/reject "fix cards".
    """
    try:
        from quirky.diagnose import diagnose_image
        res = diagnose_image(os.path.abspath(path), intensity=intensity)
        res["attribution"] = ATTRIBUTION
        return res
    except Exception as e:
        return {"error": str(e), "attribution": ATTRIBUTION}


@mcp.tool()
def quirky_xray_image(path: str, output_path: str | None = None, alpha: float = 0.55) -> Dict[str, Any]:
    """
    Render a Slop X-ray heatmap overlay (PNG) that shows WHERE an image reads as
    synthetic. Writes to <name>.xray.png unless output_path is given.
    """
    try:
        from quirky.diagnose import maps
        abs_in = os.path.abspath(path)
        stem, _ = os.path.splitext(abs_in)
        abs_out = os.path.abspath(output_path) if output_path else f"{stem}.xray.png"
        gray, rgb = maps.load_gray_rgb(abs_in)
        comp, _ = maps.composite_slop_map(gray)
        maps.render_heatmap_overlay(rgb, comp, abs_out, alpha=alpha)
        return {
            "output_path": abs_out,
            "slop_mean": round(float(comp.mean()), 4),
            "slop_peak": round(float(comp.max()), 4),
            "hotspots": maps.region_scores(comp),
            "attribution": ATTRIBUTION,
        }
    except Exception as e:
        return {"error": str(e), "attribution": ATTRIBUTION}


@mcp.tool()
def quirky_fingerprint_image(path: str) -> Dict[str, Any]:
    """
    Heuristic guess of which generator likely produced an image (SD/SDXL VAE grid, GAN,
    over-smoothed render, generic diffusion, or real camera) plus the inverse fixes it
    calls for. A statistical prior, not a trained classifier.
    """
    try:
        from quirky.diagnose import fingerprint_image
        res = fingerprint_image(os.path.abspath(path))
        res["attribution"] = ATTRIBUTION
        return res
    except Exception as e:
        return {"error": str(e), "attribution": ATTRIBUTION}


@mcp.tool()
def quirky_humanize_locked(
    path: str,
    output_path: str | None = None,
    target_ai: float = 0.15,
    min_ssim: float = 0.86,
    fixes: str | None = None,
) -> Dict[str, Any]:
    """
    Minimal-edit humanize for images: climb intensity only until ai_score <= target_ai
    while SSIM to the original stays >= min_ssim. Returns the chosen intensity, whether
    the target was met, and the per-step trace. `fixes` is an optional comma-separated
    list of fix ids to restrict which corrections run.
    """
    try:
        from quirky.diagnose import humanize_locked
        abs_in = os.path.abspath(path)
        stem, ext = os.path.splitext(abs_in)
        abs_out = os.path.abspath(output_path) if output_path else f"{stem}.locked{ext}"
        enabled = set(f.strip() for f in fixes.split(",") if f.strip()) if fixes else None
        res = humanize_locked(abs_in, abs_out, target_ai=target_ai, min_ssim=min_ssim,
                              enabled_fixes=enabled)
        res["attribution"] = ATTRIBUTION
        return res
    except Exception as e:
        return {"error": str(e), "attribution": ATTRIBUTION}


@mcp.tool()
def quirky_audit_transfer(before: str, after: str, oracle: str = "auto") -> Dict[str, Any]:
    """
    Honesty check: score before/after against an EXTERNAL detector oracle (separate from
    Quirky's own ai_score) and report the true reduction. oracle: auto | ensemble | neural.
    """
    try:
        from quirky.diagnose import audit, get_oracle, EnsembleHeuristicOracle
        orc = EnsembleHeuristicOracle() if oracle == "ensemble" else get_oracle(oracle)
        res = audit(os.path.abspath(before), os.path.abspath(after), orc)
        res["attribution"] = ATTRIBUTION
        return res
    except Exception as e:
        return {"error": str(e), "attribution": ATTRIBUTION}


@mcp.tool()
def quirky_scan_metadata(path: str) -> Dict[str, Any]:
    """
    Read-only report of every embedded metadata field in an image (EXIF, PNG text
    chunks, etc), flagging entries that look like AI-generator leaks (prompt, seed,
    sampler, model hash...). Never modifies the file.
    """
    try:
        from quirky.clean import scan_metadata
        res = scan_metadata(os.path.abspath(path))
        res["attribution"] = ATTRIBUTION
        return res
    except Exception as e:
        return {"error": str(e), "attribution": ATTRIBUTION}


@mcp.tool()
def quirky_clean_metadata(path: str, output_path: str | None = None, attribute: bool = False) -> Dict[str, Any]:
    """
    Clean: scrub all embedded metadata via a clean re-encode (drops EXIF, PNG text
    chunks, generator-parameter leaks). If attribute=True, leaves one honest
    "Powered by Quirky (MITPO)" tag instead of a silent blank. Writes to
    <name>.clean<ext> unless output_path is given.
    """
    try:
        from quirky.clean import clean_metadata
        abs_in = os.path.abspath(path)
        stem, ext = os.path.splitext(abs_in)
        abs_out = os.path.abspath(output_path) if output_path else f"{stem}.clean{ext}"
        res = clean_metadata(abs_in, abs_out, attribute=attribute)
        res["attribution"] = ATTRIBUTION
        return res
    except Exception as e:
        return {"error": str(e), "attribution": ATTRIBUTION}


@mcp.tool()
def quirky_remap_image(before: str, after: str, intensity: float = 0.6) -> Dict[str, Any]:
    """
    Re-map: re-diagnose an edited image against a fresh diagnosis of the original.
    Returns resolved/remaining/newly-introduced defect ids, the slop-score delta, a
    red(worse)/green(better) delta heatmap, and a plain recommendation (clean,
    keep_going, diminishing_returns, over_cooked).
    """
    try:
        from quirky.diagnose import remap_image
        res = remap_image(os.path.abspath(before), os.path.abspath(after), intensity=intensity)
        res["attribution"] = ATTRIBUTION
        return res
    except Exception as e:
        return {"error": str(e), "attribution": ATTRIBUTION}


@mcp.tool()
def quirky_remap_loop(
    path: str,
    output_path: str | None = None,
    min_ssim: float = 0.80,
    max_rounds: int = 3,
) -> Dict[str, Any]:
    """
    Closed loop: diagnose -> humanize -> re-map, repeated until clean, diminishing
    returns, the fidelity floor to the ORIGINAL would be breached, or max_rounds is
    hit. Returns the full round-by-round trace. Writes to <name>.remapped<ext>
    unless output_path is given.
    """
    try:
        from quirky.diagnose import remap_loop
        abs_in = os.path.abspath(path)
        stem, ext = os.path.splitext(abs_in)
        abs_out = os.path.abspath(output_path) if output_path else f"{stem}.remapped{ext}"
        res = remap_loop(abs_in, abs_out, min_ssim=min_ssim, max_rounds=max_rounds)
        res["attribution"] = ATTRIBUTION
        return res
    except Exception as e:
        return {"error": str(e), "attribution": ATTRIBUTION}


if __name__ == "__main__":
    mcp.run()

