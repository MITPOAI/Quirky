import os
import json
import typer
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from quirky.detector.engine import DetectorEngine
from quirky.image.pipeline import ImageHumanizer
from quirky.audio.pipeline import AudioHumanizer
from quirky.text.pipeline import TextHumanizer
from quirky.video.pipeline import VideoHumanizer

app = typer.Typer(help="Quirky Human Preference Engine CLI")

@app.command()
def detect(
    asset: str = typer.Option(..., "--asset", "-a", help="Path to the synthetic asset to analyze")
):
    """
    Passively analyzes a synthetic media asset and outputs a unified metric JSON schema.
    """
    try:
        if not os.path.exists(asset):
            typer.secho(f"Error: Asset file '{asset}' does not exist.", fg=typer.colors.RED)
            raise typer.Exit(1)
            
        typer.secho(f"Analyzing {asset}...", fg=typer.colors.CYAN)
        report = DetectorEngine.analyze_asset(asset)
        
        # Print JSON output
        print(json.dumps(report, indent=2))
    except Exception as e:
        typer.secho(f"Execution Error: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(1)

@app.command()
def humanize(
    asset: str = typer.Option(..., "--asset", "-a", help="Path to the synthetic asset to optimize"),
    output: str = typer.Option(..., "--output", "-o", help="Path to write the humanized asset"),
    intensity: int = typer.Option(50, "--intensity", "-i", min=0, max=100, help="Humanization intensity (0-100)"),
    fixes: str = typer.Option(None, "--fixes", help="Comma-separated fix ids to apply (images only). "
                                                    "Default = all. e.g. white_balance,clahe_lighting,plastic_texture"),
    lock: bool = typer.Option(False, "--lock/--no-lock", help="Minimal-edit fidelity lock (images only): climb "
                                                              "intensity only until the target is met."),
    target: float = typer.Option(0.15, "--target", min=0.0, max=1.0, help="Target ai_score for --lock"),
    min_ssim: float = typer.Option(0.86, "--min-ssim", min=0.0, max=1.0, help="SSIM fidelity floor for --lock"),
):
    """
    Optimizes a synthetic asset to restore realistic human imperfections and details.
    """
    try:
        if not os.path.exists(asset):
            typer.secho(f"Error: Asset file '{asset}' does not exist.", fg=typer.colors.RED)
            raise typer.Exit(1)

        ext = os.path.splitext(asset)[1].lower()
        strength = intensity / 100.0
        enabled_fixes = set(f.strip() for f in fixes.split(",") if f.strip()) if fixes else None

        typer.secho(f"Humanizing {asset} with intensity {intensity}%...", fg=typer.colors.CYAN)

        if ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
            if lock:
                from quirky.diagnose import humanize_locked
                res = humanize_locked(asset, output, target_ai=target, min_ssim=min_ssim,
                                      enabled_fixes=enabled_fixes)
                typer.secho(
                    f"  lock: target_met={res['target_met']} intensity={res['chosen_intensity']} "
                    f"ai {res['original_ai_score']}->{res['final_ai_score']} ssim={res['final_ssim']}",
                    fg=typer.colors.YELLOW)
                typer.secho(f"  {res['reason']}", fg=typer.colors.WHITE)
            else:
                ImageHumanizer.humanize(asset, output, intensity=strength, enabled_fixes=enabled_fixes)
        elif ext in [".wav"]:
            AudioHumanizer.humanize(asset, output, intensity=strength)
        elif ext in [".txt", ".md", ".json"]:
            with open(asset, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            humanized_text = TextHumanizer.humanize(content, intensity=strength)
            with open(output, "w", encoding="utf-8") as f:
                f.write(humanized_text)
        elif ext in [".mp4", ".avi", ".mov"]:
            VideoHumanizer.humanize(asset, output, intensity=strength)
        else:
            typer.secho(f"Unsupported modality '{ext}'. Treating as text...", fg=typer.colors.YELLOW)
            with open(asset, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            humanized_text = TextHumanizer.humanize(content, intensity=strength)
            with open(output, "w", encoding="utf-8") as f:
                f.write(humanized_text)
                
        typer.secho(f"Success! Saved humanized output to {output}", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Execution Error: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(1)

@app.command()
def compare(
    before: str = typer.Argument(..., help="Path to raw/input asset"),
    after: str = typer.Argument(..., help="Path to modified/humanized asset"),
    output: str = typer.Option(..., "--output", "-o", help="Path to save the generated comparison share card (PNG)")
):
    """
    Generates a high-fidelity visual side-by-side comparison card for social media.
    """
    try:
        if not os.path.exists(before) or not os.path.exists(after):
            typer.secho("Error: Both before and after files must exist.", fg=typer.colors.RED)
            raise typer.Exit(1)
            
        # Get metrics
        before_report = DetectorEngine.analyze_asset(before)
        after_report = DetectorEngine.analyze_asset(after)
        
        before_scores = before_report["metadata"]
        after_scores = after_report["metadata"]
        
        ext = os.path.splitext(before)[1].lower()
        
        # Dimensions
        card_w, card_h = 1000, 750
        card = Image.new("RGB", (card_w, card_h), "#131316")
        draw = ImageDraw.Draw(card)
        
        # Draw Header
        draw.rectangle([(0, 0), (card_w, 80)], fill="#1a1a20")
        draw.text((30, 25), "QUIRKY COMPARE: PORTRAIT RESTORATION", fill="#ffffff", font=None)
        
        # Modality specific drawing
        if ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
            # Load images
            img_b = Image.open(before).convert("RGB")
            img_a = Image.open(after).convert("RGB")
            
            # Resize to fit side-by-side
            img_sz = 420
            img_b_res = img_b.resize((img_sz, img_sz), Image.Resampling.LANCZOS)
            img_a_res = img_a.resize((img_sz, img_sz), Image.Resampling.LANCZOS)
            
            # Paste with spacing
            card.paste(img_b_res, (50, 150))
            card.paste(img_a_res, (530, 150))
            
            # Draw Labels
            draw.text((50, 110), "BEFORE (Plastic Base)", fill="#8e8e9f")
            draw.text((530, 110), "AFTER (Pore Restored)", fill="#10b981")
            
        else:
            # Drawing text comparison
            draw.rectangle([(50, 150), (470, 500)], fill="#1d1d24", outline="#2e2e38")
            draw.rectangle([(530, 150), (950, 500)], fill="#1d1d24", outline="#2e2e38")
            
            draw.text((50, 110), "BEFORE (Synthetic Base)", fill="#8e8e9f")
            draw.text((530, 110), "AFTER (Humanized)", fill="#10b981")
            
            # Read snippet of texts
            with open(before, "r", encoding="utf-8", errors="ignore") as f:
                txt_b = f.read(250) + "..."
            with open(after, "r", encoding="utf-8", errors="ignore") as f:
                txt_a = f.read(250) + "..."
                
            def draw_wrapped_text(text, x, y, max_w):
                words = text.split()
                lines = []
                curr_line = ""
                for w in words:
                    if len(curr_line + " " + w) * 7 > max_w:
                        lines.append(curr_line)
                        curr_line = w
                    else:
                        curr_line += " " + w if curr_line else w
                if curr_line:
                    lines.append(curr_line)
                for line in lines[:14]:
                    draw.text((x, y), line, fill="#d1d1e0")
                    y += 22
                    
            draw_wrapped_text(txt_b, 70, 170, 380)
            draw_wrapped_text(txt_a, 550, 170, 380)
            
        # Draw Metrics Splitter
        draw.rectangle([(0, 580), (card_w, 680)], fill="#16161c")
        
        # Display AI & Plastic Scores
        b_ai = before_scores.get("ai_score", 0.0)
        b_pl = before_scores.get("plastic_score", 0.0)
        a_ai = after_scores.get("ai_score", 0.0)
        a_pl = after_scores.get("plastic_score", 0.0)
        
        draw.text((50, 600), f"AI SCORE: {b_ai}", fill="#ef4444")
        draw.text((50, 630), f"PLASTIC SCORE: {b_pl}", fill="#ef4444")
        
        draw.text((530, 600), f"AI SCORE: {a_ai}", fill="#10b981")
        draw.text((530, 630), f"PLASTIC SCORE: {a_pl}", fill="#10b981")
        
        # Draw Attribution Footer
        draw.rectangle([(0, 680), (card_w, card_h)], fill="#0a0a0c")
        draw.text((30, 700), "Powered by Quirky", fill="#8e8e9f")
        draw.text((30, 720), "by MITPO", fill="#4f46e5")
        
        card.save(output)
        typer.secho(f"Share card generated and saved to {output}", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Execution Error: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(1)

@app.command()
def xray(
    asset: str = typer.Option(..., "--asset", "-a", help="Image to analyze"),
    output: str = typer.Option("xray.png", "--output", "-o", help="Where to write the heatmap overlay PNG"),
    alpha: float = typer.Option(0.55, "--alpha", min=0.0, max=1.0, help="Heatmap blend strength"),
):
    """
    Slop X-ray: render an explainable heatmap showing WHERE an image reads as synthetic.
    """
    try:
        from quirky.diagnose import maps
        gray, rgb = maps.load_gray_rgb(asset)
        comp, _ = maps.composite_slop_map(gray)
        maps.render_heatmap_overlay(rgb, comp, output, alpha=alpha)
        typer.secho(f"Slop X-ray written to {output}", fg=typer.colors.GREEN)
        typer.secho(f"  slop mean={comp.mean():.3f}  peak={comp.max():.3f}", fg=typer.colors.YELLOW)
        typer.secho(f"  hotspots: {maps.region_scores(comp)}", fg=typer.colors.WHITE)
    except Exception as e:
        typer.secho(f"X-ray Error: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(1)


@app.command()
def diagnose(
    asset: str = typer.Option(..., "--asset", "-a", help="Image to diagnose"),
    intensity: int = typer.Option(60, "--intensity", "-i", min=0, max=100, help="Intensity used to size fix params"),
):
    """
    Prescriptive diagnosis: list the named defects and the fix each one recommends.
    """
    try:
        from quirky.diagnose import diagnose_image
        d = diagnose_image(asset, intensity=intensity / 100.0)
        print(json.dumps(d, indent=2))
    except Exception as e:
        typer.secho(f"Diagnose Error: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(1)


@app.command()
def fingerprint(
    asset: str = typer.Option(..., "--asset", "-a", help="Image to fingerprint"),
):
    """
    Guess which generator likely produced an image (heuristic) + the inverse fixes it calls for.
    """
    try:
        from quirky.diagnose import fingerprint_image
        print(json.dumps(fingerprint_image(asset), indent=2))
    except Exception as e:
        typer.secho(f"Fingerprint Error: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(1)


@app.command()
def audit(
    before: str = typer.Argument(..., help="Original (pre-humanize) asset"),
    after: str = typer.Argument(..., help="Humanized asset"),
    oracle: str = typer.Option("auto", "--oracle", help="Oracle: auto | ensemble | neural"),
):
    """
    Honesty check: score before/after against an EXTERNAL detector, not Quirky's own metric.
    """
    try:
        from quirky.diagnose import audit as run_audit, get_oracle, EnsembleHeuristicOracle
        orc = EnsembleHeuristicOracle() if oracle == "ensemble" else get_oracle(oracle)
        print(json.dumps(run_audit(before, after, orc), indent=2))
    except Exception as e:
        typer.secho(f"Audit Error: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(1)


@app.command()
def clean(
    asset: str = typer.Option(..., "--asset", "-a", help="Image to scrub metadata from"),
    output: str = typer.Option(None, "--output", "-o", help="Where to write the cleaned image (default: <name>.clean<ext>)"),
    attribute: bool = typer.Option(False, "--attribute/--no-attribute",
                                   help="Leave one honest 'Powered by Quirky' tag instead of a silent blank"),
    scan_only: bool = typer.Option(False, "--scan-only", help="Report embedded metadata; write nothing"),
):
    """
    Clean: scrub embedded metadata (EXIF / PNG-text / generator-parameter leaks) via a clean re-encode.
    """
    try:
        from quirky.clean import scan_metadata, clean_metadata
        if not os.path.exists(asset):
            typer.secho(f"Error: Asset file '{asset}' does not exist.", fg=typer.colors.RED)
            raise typer.Exit(1)

        if scan_only:
            report = scan_metadata(asset)
            print(json.dumps(report, indent=2))
            typer.secho(f"  {report['meaningful_count']} meaningful field(s), "
                        f"{report['leak_count']} look like generator leaks", fg=typer.colors.YELLOW)
            return

        stem, ext = os.path.splitext(asset)
        out = output or f"{stem}.clean{ext}"
        res = clean_metadata(asset, out, attribute=attribute)
        print(json.dumps(res, indent=2))
        color = typer.colors.GREEN if res["fully_clean"] else typer.colors.YELLOW
        typer.secho(f"Cleaned -> {out}  (fully_clean={res['fully_clean']})", fg=color)
    except Exception as e:
        typer.secho(f"Clean Error: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(1)


@app.command()
def remap(
    before: str = typer.Argument(..., help="Original (pre-edit) asset"),
    after: str = typer.Argument(..., help="Edited asset to re-diagnose"),
    intensity: int = typer.Option(60, "--intensity", "-i", min=0, max=100, help="Intensity used to size fix params"),
    output: str = typer.Option(None, "--output", "-o", help="Optional path to save the red/green delta heatmap PNG"),
):
    """
    Re-map: re-diagnose an edited image against the original -- resolved / remaining / new
    defects, plus a red(worse)/green(better) delta heatmap.
    """
    try:
        from quirky.diagnose import remap_image
        res = remap_image(before, after, intensity=intensity / 100.0)
        if output:
            import base64
            b64 = res["delta_heatmap"].split(",", 1)[1]
            with open(output, "wb") as f:
                f.write(base64.b64decode(b64))
            typer.secho(f"Delta heatmap written to {output}", fg=typer.colors.GREEN)
        printable = {k: v for k, v in res.items() if k != "delta_heatmap"}
        print(json.dumps(printable, indent=2))
        typer.secho(f"  recommendation: {res['recommendation']} -- {res['note']}", fg=typer.colors.YELLOW)
    except Exception as e:
        typer.secho(f"Remap Error: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(1)


@app.command(name="remap-loop")
def remap_loop_cmd(
    asset: str = typer.Option(..., "--asset", "-a", help="Image to iteratively diagnose -> humanize -> re-map"),
    output: str = typer.Option(..., "--output", "-o", help="Where to write the final result"),
    min_ssim: float = typer.Option(0.80, "--min-ssim", min=0.0, max=1.0, help="Fidelity floor vs. the original"),
    max_rounds: int = typer.Option(3, "--max-rounds", min=1, max=10),
    start_intensity: int = typer.Option(40, "--start-intensity", min=0, max=100),
    step: int = typer.Option(20, "--step", min=1, max=100),
):
    """
    Closed loop: diagnose -> humanize -> re-map, repeated until clean, diminishing
    returns, the fidelity floor would be breached, or max_rounds is hit.
    """
    try:
        from quirky.diagnose import remap_loop
        res = remap_loop(asset, output, min_ssim=min_ssim, max_rounds=max_rounds,
                         start_intensity=start_intensity / 100.0, step=step / 100.0)
        print(json.dumps(res, indent=2))
        typer.secho(f"Saved final result to {output} after {res['total_rounds']} round(s)", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Remap-loop Error: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(1)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address for the web dashboard"),
    port: int = typer.Option(8000, "--port", "-p", help="Port for the web dashboard")
):
    """
    Starts the local web dashboard (FastAPI + static UI) at http://HOST:PORT.
    """
    typer.secho(f"Starting Quirky web dashboard at http://{host}:{port} ...", fg=typer.colors.CYAN)
    from quirky.api.main import start_server
    start_server(host=host, port=port)


@app.command()
def init(
    path: str = typer.Option(".", "--path", "-p", help="Target repository directory to initialize"),
    force: bool = typer.Option(False, "--force", "-f", help="Force overwrite existing files even if they lack markers")
):
    """
    Scaffolds active anti-slop guidelines and MCP server configurations into the target repository.
    """
    try:
        from quirky.cli.scaffold import init_repo
        typer.secho(f"Initializing Quirky cross-tool rules in target repo '{path}'...", fg=typer.colors.CYAN)
        res = init_repo(path, force=force)
        for name, status in res.items():
            if status == "created":
                fg_color = typer.colors.GREEN
            elif status == "updated":
                fg_color = typer.colors.YELLOW
            else:
                fg_color = typer.colors.WHITE
            typer.secho(f"  [{status.upper()}] {name}", fg=fg_color)
        typer.secho("Success! Quirky cross-tool scaffolding completed.", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Scaffolding Error: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(1)



def _run_dl(fn_name: str, *args):
    """Run a quirky.plugins.dl function, surfacing a clean install hint if the extra is missing."""
    from quirky.plugins import dl
    try:
        return getattr(dl, fn_name)(*args)
    except dl.DLNotInstalled as e:
        typer.secho(str(e), fg=typer.colors.YELLOW)
        raise typer.Exit(1)


@app.command()
def upscale(
    asset: str = typer.Option(..., "--asset", "-a", help="Image to upscale"),
    output: str = typer.Option(..., "--output", "-o", help="Where to write the result")
):
    """Neural super-resolution (Real-ESRGAN). Requires the optional quirky[dl] extra."""
    import numpy as np
    from PIL import Image as _Image
    img = np.array(_Image.open(asset).convert("RGB"))
    out = _run_dl("upscale", img)
    _Image.fromarray(out).save(output)
    typer.secho(f"Upscaled -> {output}", fg=typer.colors.GREEN)


@app.command()
def repaint(
    asset: str = typer.Option(..., "--asset", "-a", help="Image to repaint"),
    mask: str = typer.Option(..., "--mask", "-m", help="Mask PNG (white = area to repaint)"),
    output: str = typer.Option(..., "--output", "-o", help="Where to write the result")
):
    """Neural inpaint / object+spot repaint (LaMa). Requires the optional quirky[dl] extra."""
    import numpy as np
    from PIL import Image as _Image
    img = np.array(_Image.open(asset).convert("RGB"))
    msk = np.array(_Image.open(mask).convert("L"))
    out = _run_dl("repaint", img, msk)
    _Image.fromarray(out).save(output)
    typer.secho(f"Repainted -> {output}", fg=typer.colors.GREEN)


@app.command(name="voice-clone")
def voice_clone(
    asset: str = typer.Option(..., "--asset", "-a", help="Source speech to re-timbre"),
    reference: str = typer.Option(..., "--reference", "-r", help="Reference clip of the target voice"),
    output: str = typer.Option(..., "--output", "-o", help="Where to write the cloned-voice audio")
):
    """Zero-shot voice conversion toward a reference voice. Requires the optional quirky[dl] extra."""
    _run_dl("clone_voice", asset, reference, output)
    typer.secho(f"Voice-converted -> {output}", fg=typer.colors.GREEN)


agent_app = typer.Typer(help="Manage and run Quirky agents and skills")

@agent_app.command(name="list-skills")
def agent_list_skills():
    """
    Lists all available agent skills.
    """
    from quirky.agent.core import QuirkyAgent
    agent = QuirkyAgent()
    skills = agent.list_skills()
    typer.secho("Registered Quirky Agent Skills:", fg=typer.colors.CYAN, bold=True)
    for s in skills:
        typer.secho(f"  * {s['name']}: {s['description']}", fg=typer.colors.GREEN)

@agent_app.command(name="run")
def agent_run(
    asset: str = typer.Option(..., "--asset", "-a", help="Path to the synthetic asset or text file to optimize"),
    output: str = typer.Option(None, "--output", "-o", help="Path to save the optimized result"),
    intensity: int = typer.Option(50, "--intensity", "-i", min=0, max=100, help="Optimization intensity (0-100)")
):
    """
    Runs QuirkyAgent on the specified asset.
    """
    try:
        from quirky.agent.core import QuirkyAgent
        if not os.path.exists(asset):
            typer.secho(f"Error: Asset file '{asset}' does not exist.", fg=typer.colors.RED)
            raise typer.Exit(1)

        typer.secho(f"Initializing QuirkyAgent for {asset}...", fg=typer.colors.CYAN)
        agent = QuirkyAgent()
        result = agent.run(
            input_data=asset,
            is_file=True,
            output_path=output,
            intensity=intensity / 100.0
        )
        
        # Display logs
        typer.secho("\nAgent Execution Audit Log:", fg=typer.colors.CYAN, bold=True)
        for log in result.get("logs", []):
            typer.secho(f"  {log}", fg=typer.colors.WHITE)
            
        if result["status"] == "success":
            typer.secho("\nSuccess!", fg=typer.colors.GREEN, bold=True)
            typer.secho(f"Original AI Score: {result['original_score']}", fg=typer.colors.YELLOW)
            typer.secho(f"Final AI Score:    {result['final_score']}", fg=typer.colors.GREEN)
            if "output_path" in result:
                typer.secho(f"Saved to:          {result['output_path']}", fg=typer.colors.GREEN)
        else:
            typer.secho(f"\nFailure: {result.get('error')}", fg=typer.colors.RED, bold=True)
            raise typer.Exit(1)
            
    except Exception as e:
        typer.secho(f"Execution Error: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(1)


def find_git_dir(start_path: str = ".") -> str | None:
    curr = os.path.abspath(start_path)
    while True:
        git_dir = os.path.join(curr, ".git")
        if os.path.isdir(git_dir):
            return git_dir
        parent = os.path.dirname(curr)
        if parent == curr:
            break
        curr = parent
    return None


@agent_app.command(name="install-hook")
def agent_install_hook():
    """
    Installs a Git pre-commit hook that automatically runs QuirkyAgent on staged files.
    """
    try:
        git_dir = find_git_dir()
        if not git_dir:
            typer.secho("Error: No .git directory found. Run this in a Git repo.", fg=typer.colors.RED)
            raise typer.Exit(1)
            
        hooks_dir = os.path.join(git_dir, "hooks")
        os.makedirs(hooks_dir, exist_ok=True)
        hook_path = os.path.join(hooks_dir, "pre-commit")
        
        script_content = """#!/bin/sh
# Quirky Anti-Slop Git Pre-Commit Hook
# Automatically cleans and humanizes staged text and source files before committing.

# Get list of staged files (Added, Copied, Modified)
staged_files=$(git diff --cached --name-only --diff-filter=ACM)

for file in $staged_files; do
  if [ -f "$file" ]; then
    ext="${file##*.}"
    ext_lower=$(echo "$ext" | tr '[:upper:]' '[:lower:]')
    
    if [ "$ext_lower" = "md" ] || [ "$ext_lower" = "txt" ] || [ "$ext_lower" = "py" ] || [ "$ext_lower" = "js" ] || [ "$ext_lower" = "ts" ]; then
      echo "Quirky: Cleaning slop from staged file: $file"
      quirky agent run --asset "$file" --output "$file" --intensity 50
      git add "$file"
    fi
  fi
done

exit 0
"""
        with open(hook_path, "w", encoding="utf-8") as f:
            f.write(script_content)
            
        try:
            os.chmod(hook_path, 0o755)
        except Exception:
            pass
            
        typer.secho(f"Success! Git pre-commit hook installed at {hook_path}", fg=typer.colors.GREEN)
        typer.secho("Staged files (.md, .txt, .py, .js, .ts) will now be automatically humanized on commit.", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Error installing hook: {str(e)}", fg=typer.colors.RED)
        raise typer.Exit(1)


app.add_typer(agent_app, name="agent")


if __name__ == "__main__":
    app()


