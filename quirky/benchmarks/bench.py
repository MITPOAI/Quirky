import time
import os
import numpy as np
from PIL import Image

from quirky.detector.engine import DetectorEngine

def run_benchmarks(test_dir: str = "benchmark_assets"):
    """
    Runs passive analysis checks over a directory of files,
    measuring execution times and computing average scores.
    """
    print("=" * 60)
    print("   QUIRKY HUMAN PREFERENCE ENGINE - STATISTICAL BENCHMARKS")
    print("=" * 60)
    
    if not os.path.exists(test_dir):
        print(f"Creating sample assets in '{test_dir}' for validation...")
        os.makedirs(test_dir, exist_ok=True)
        
        # 1. Create a dummy synthetic image (perfect gradients, high symmetry)
        img = Image.new("RGB", (512, 512), "#888888")
        # Draw a perfectly symmetrical gradient circle
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.ellipse([128, 128, 384, 384], fill="#cccccc", outline="#ffffff")
        img.save(os.path.join(test_dir, "synthetic_face.png"))
        
        # 2. Create a dummy synthetic text (highly repetitive, robotic transitions)
        robotic_text = (
            "Furthermore, it is important to note that this approach facilitates optimization. "
            "Moreover, one must utilize structured systems to maximize productivity. "
            "In conclusion, the methodology facilitates correct execution."
        )
        with open(os.path.join(test_dir, "synthetic_text.txt"), "w") as f:
            f.write(robotic_text)
            
    files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
    
    if not files:
        print("No files to benchmark.")
        return
        
    latencies = []
    ai_scores = []
    plastic_scores = []
    
    print(f"Loaded {len(files)} benchmark assets. Running detector evaluation...")
    print("-" * 60)
    print(f"{'Filename':<30} | {'Type':<6} | {'AI Score':<8} | {'Latency (ms)':<12}")
    print("-" * 60)
    
    for fpath in files:
        t0 = time.perf_counter()
        report = DetectorEngine.analyze_asset(fpath)
        t1 = time.perf_counter()
        
        latency = (t1 - t0) * 1000.0
        latencies.append(latency)
        
        meta = report["metadata"]
        ai_scores.append(meta["ai_score"])
        plastic_scores.append(meta["plastic_score"])
        
        fname = os.path.basename(fpath)
        ftype = report["file_type"]
        print(f"{fname:<30} | {ftype:<6} | {meta['ai_score']:<8.3f} | {latency:<12.2f}")
        
    print("=" * 60)
    print("   SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total files analyzed:    {len(files)}")
    print(f"Mean Latency:            {np.mean(latencies):.2f} ms")
    print(f"Latency StdDev:          {np.std(latencies):.2f} ms")
    print(f"Mean AI Score:           {np.mean(ai_scores):.3f}")
    print(f"Mean Plasticity Score:   {np.mean(plastic_scores):.3f}")
    print("=" * 60)

if __name__ == "__main__":
    run_benchmarks()
