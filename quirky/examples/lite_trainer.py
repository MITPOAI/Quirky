import os
import sys
import numpy as np

# Add Quirky package to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from quirky.detector.engine import DetectorEngine
from quirky.image.pipeline import ImageHumanizer
from quirky.audio.pipeline import AudioHumanizer
from quirky.text.pipeline import TextHumanizer

def loss_function(scores: dict) -> float:
    """
    Computes loss based on how far the asset is from target human scores:
    Target AI Score: < 0.10
    Target Plastic Score: < 0.60
    """
    ai_score = scores.get("ai_score", 0.5)
    plastic_score = scores.get("plastic_score", 0.5)
    
    # Minimize both AI score and plastic score
    loss = (ai_score - 0.05)**2 + (plastic_score - 0.55)**2
    return float(loss)

def run_lite_training(asset_in: str, asset_out: str):
    """
    Runs a coordinate-descent 'lite training' loop to optimize humanization parameters.
    Supports Image, Audio, and Text modalities.
    """
    print("=" * 70)
    print("      QUIRKY PARAMETER TUNER - MULTI-MODAL SOLVER")
    print("=" * 70)
    
    if not os.path.exists(asset_in):
        print(f"Error: Target asset '{asset_in}' not found.")
        return
        
    ext = os.path.splitext(asset_in)[1].lower()
    
    # Initialize parameters based on modality
    if ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
        modality = "image"
        params = {
            "intensity": {"val": 0.3, "step": 0.15, "bounds": (0.1, 0.9)},
            "gamma": {"val": 0.5, "step": 0.15, "bounds": (0.2, 0.8)},
            "delta": {"val": 0.02, "step": 0.02, "bounds": (0.01, 0.08)}
        }
    elif ext in [".wav"]:
        modality = "audio"
        params = {
            "intensity": {"val": 0.3, "step": 0.15, "bounds": (0.1, 0.9)},
            "breath_freq": {"val": 0.5, "step": 0.15, "bounds": (0.2, 0.9)}
        }
    else:
        modality = "text"
        params = {
            "intensity": {"val": 0.3, "step": 0.15, "bounds": (0.1, 0.9)}
        }
        
    print(f"Detected Modality: {modality.upper()} ({ext})")
    
    best_loss = 999.0
    iterations = 4
    print(f"Starting solver loops. Target baseline: AI Score < 0.10, Plastic Score < 0.60\n")
    
    for epoch in range(1, iterations + 1):
        print(f"--- Epoch {epoch}/{iterations} ---")
        
        for p_name, p_data in params.items():
            best_local_val = p_data["val"]
            
            # Evaluate current, step up, and step down
            test_vals = [
                p_data["val"],
                min(p_data["val"] + p_data["step"], p_data["bounds"][1]),
                max(p_data["val"] - p_data["step"], p_data["bounds"][0])
            ]
            
            local_best_loss = best_loss
            
            for t_val in set(test_vals):
                # Temporary parameters dict
                temp_params = {k: v["val"] for k, v in params.items()}
                temp_params[p_name] = t_val
                
                # Execute humanizer step
                if modality == "image":
                    ImageHumanizer.humanize(
                        asset_in, 
                        asset_out, 
                        intensity=temp_params["intensity"], 
                        gamma=temp_params["gamma"], 
                        delta=temp_params["delta"]
                    )
                elif modality == "audio":
                    AudioHumanizer.humanize(
                        asset_in,
                        asset_out,
                        intensity=temp_params["intensity"],
                        breath_freq=temp_params["breath_freq"]
                    )
                elif modality == "text":
                    with open(asset_in, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    humanized_text = TextHumanizer.humanize(content, intensity=temp_params["intensity"])
                    with open(asset_out, "w", encoding="utf-8") as f:
                        f.write(humanized_text)
                
                # Analyze output scores
                report = DetectorEngine.analyze_asset(asset_out)
                loss = loss_function(report["metadata"])
                
                if loss < local_best_loss:
                    local_best_loss = loss
                    best_local_val = t_val
            
            # Update best parameter state
            p_data["val"] = best_local_val
            best_loss = local_best_loss
            
        tuning_str = ", ".join([f"{k}={v['val']:.2f}" for k, v in params.items()])
        print(f"Current Tuning: [{tuning_str}]")
        print(f"Current Target Loss: {best_loss:.5f}\n")
        
    print("=" * 70)
    print("      TRAINING COMPLETED - OPTIMIZATION CONVERGED")
    print("=" * 70)
    
    # Run final build with optimal weights
    opt_params = {k: v["val"] for k, v in params.items()}
    
    if modality == "image":
        ImageHumanizer.humanize(asset_in, asset_out, intensity=opt_params["intensity"], gamma=opt_params["gamma"], delta=opt_params["delta"])
    elif modality == "audio":
        AudioHumanizer.humanize(asset_in, asset_out, intensity=opt_params["intensity"], breath_freq=opt_params["breath_freq"])
    elif modality == "text":
        with open(asset_in, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        humanized_text = TextHumanizer.humanize(content, intensity=opt_params["intensity"])
        with open(asset_out, "w", encoding="utf-8") as f:
            f.write(humanized_text)
            
    final_report = DetectorEngine.analyze_asset(asset_out)
    
    print(f"Optimal Parameters Mapped:")
    for k, v in opt_params.items():
        print(f"  - {k.capitalize()}: {v:.3f}")
    print("-" * 70)
    print(f"Aligned Scores:")
    print(f"  - Final AI Score:        {final_report['metadata']['ai_score']}")
    print(f"  - Final Plastic Score:   {final_report['metadata']['plastic_score']}")
    print("=" * 70)

if __name__ == "__main__":
    # Optimize the demo AI image
    run_lite_training("demo_assets/ai_image_raw.png", "demo_assets/ai_image_aligned.png")
