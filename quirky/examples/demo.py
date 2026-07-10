import os
import sys
from PIL import Image, ImageDraw

# Add current directory to path to support running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from quirky.detector.engine import DetectorEngine
from quirky.image.pipeline import ImageHumanizer
from quirky.text.pipeline import TextHumanizer
from quirky.cli.main import compare as run_compare_cmd

def run_demo():
    print("=" * 70)
    print("       QUIRKY HUMAN PREFERENCE ENGINE - END-TO-END DEMO")
    print("=" * 70)

    # 1. Setup workspace assets
    demo_dir = "demo_assets"
    os.makedirs(demo_dir, exist_ok=True)
    
    img_in = os.path.join(demo_dir, "ai_image_raw.png")
    img_out = os.path.join(demo_dir, "ai_image_aligned.png")
    txt_in = os.path.join(demo_dir, "ai_text_raw.txt")
    txt_out = os.path.join(demo_dir, "ai_text_aligned.txt")
    card_out = os.path.join(demo_dir, "social_share_card.png")
    
    # Generate Synthetic-like portrait base (smooth gradient sphere representing plastic facial shading)
    print("[1/6] Creating mock synthetic image with airbrushed characteristics...")
    img = Image.new("RGB", (512, 512), "#111116")
    draw = ImageDraw.Draw(img)
    # Perfectly flat lighting gradient
    for r in range(200, 0, -2):
        color = int(100 + (200 - r) * 0.5)
        draw.ellipse([256 - r, 256 - r, 256 + r, 256 + r], fill=(color, color - 10, color + 20))
    img.save(img_in)
    
    # Generate standard AI text
    print("[2/6] Creating mock AI text using common clichés...")
    robotic_text = (
        "Furthermore, it is important to note that this approach facilitates optimization. "
        "Moreover, one must utilize structured systems to maximize productivity. "
        "In conclusion, the methodology facilitates correct execution."
    )
    with open(txt_in, "w", encoding="utf-8") as f:
        f.write(robotic_text)
        
    # 2. Run passive detection on raw assets
    print("\n[3/6] Running passive analytics on raw assets...")
    print("-" * 50)
    report_img_raw = DetectorEngine.analyze_asset(img_in)
    print(f"RAW IMAGE -> AI Score: {report_img_raw['metadata']['ai_score']}, Plastic Score: {report_img_raw['metadata']['plastic_score']}")
    
    report_txt_raw = DetectorEngine.analyze_asset(txt_in)
    print(f"RAW TEXT  -> AI Score: {report_txt_raw['metadata']['ai_score']}, Plastic Score: {report_txt_raw['metadata']['plastic_score']}")
    print("-" * 50)

    # 3. Apply humanization pipelines
    print("\n[4/6] Applying Quirky alignment humanizer...")
    ImageHumanizer.humanize(img_in, img_out, intensity=0.6, gamma=0.6, delta=0.04)
    
    with open(txt_in, "r", encoding="utf-8") as f:
        raw_text = f.read()
    humanized_text = TextHumanizer.humanize(raw_text, intensity=0.7)
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write(humanized_text)
        
    # 4. Measure optimized scores
    print("\n[5/6] Measuring aligned output scores...")
    print("-" * 50)
    report_img_aligned = DetectorEngine.analyze_asset(img_out)
    print(f"ALIGNED IMAGE -> AI Score: {report_img_aligned['metadata']['ai_score']}, Plastic Score: {report_img_aligned['metadata']['plastic_score']}")
    
    report_txt_aligned = DetectorEngine.analyze_asset(txt_out)
    print(f"ALIGNED TEXT  -> AI Score: {report_txt_aligned['metadata']['ai_score']}, Plastic Score: {report_txt_aligned['metadata']['plastic_score']}")
    print("-" * 50)
    
    # 5. Generate visual comparison share card
    print("\n[6/6] Rendering social share comparison card...")
    # Invoke compare logic
    try:
        run_compare_cmd(img_in, img_out, card_out)
        print(f"Successfully generated comparison card at: {card_out}")
    except Exception as e:
        print(f"Failed to generate card: {str(e)}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETED SUCCESSFULLY. Check the 'demo_assets/' folder.")
    print("=" * 70)

if __name__ == "__main__":
    run_demo()
