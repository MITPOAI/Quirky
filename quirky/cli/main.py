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
    intensity: int = typer.Option(50, "--intensity", "-i", min=0, max=100, help="Humanization intensity (0-100)")
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
        
        typer.secho(f"Humanizing {asset} with intensity {intensity}%...", fg=typer.colors.CYAN)
        
        if ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
            ImageHumanizer.humanize(asset, output, intensity=strength)
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

if __name__ == "__main__":
    app()
