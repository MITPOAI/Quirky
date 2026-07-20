import os
import base64
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from quirky.detector.engine import DetectorEngine
from quirky.image.pipeline import ImageHumanizer
from quirky.audio.pipeline import AudioHumanizer
from quirky.text.pipeline import TextHumanizer

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

app = FastAPI(
    title="Quirky Human Preference Engine API",
    description="Managed Engine API for passive synthetic media detection and humanization",
    version="0.1.0"
)

# Temp storage for processed items
TEMP_DIR = os.path.join(os.getcwd(), "temp_assets")
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/api/detect")
async def api_detect(file: UploadFile = File(...)):
    """
    Passive detection endpoint. Returns JSON schema metrics.
    """
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1]
    temp_path = os.path.join(TEMP_DIR, f"{file_id}{ext}")
    
    try:
        # Save uploaded file
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        # Analyze
        result = DetectorEngine.analyze_asset(temp_path)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        # Clean up raw temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

def _save_upload(file: UploadFile) -> str:
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename or "")[1].lower()
    path = os.path.join(TEMP_DIR, f"{file_id}{ext}")
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return path


@app.post("/api/diagnose")
async def api_diagnose(file: UploadFile = File(...), intensity: float = Form(0.6)):
    """
    Prescriptive diagnosis for an image: fix cards + a base64 Slop X-ray heatmap +
    a generator-fingerprint summary. One call feeds the whole dashboard panel.
    """
    temp_in = _save_upload(file)
    try:
        ext = os.path.splitext(temp_in)[1].lower()
        if ext not in _IMG_EXTS:
            return JSONResponse(status_code=400, content={"error": f"Diagnosis supports images; got {ext}"})
        from quirky.diagnose import diagnose_image, fingerprint_image
        from quirky.diagnose.maps import load_gray_rgb, composite_slop_map, heatmap_png_bytes

        report = diagnose_image(temp_in, intensity=intensity)
        fp = fingerprint_image(temp_in)
        gray, rgb = load_gray_rgb(temp_in)
        comp, _ = composite_slop_map(gray)
        heat_b64 = base64.b64encode(heatmap_png_bytes(rgb, comp)).decode("ascii")

        report["fingerprint"] = {"top": fp["top"], "ranking": fp["ranking"]}
        report["heatmap"] = f"data:image/png;base64,{heat_b64}"
        report["catalog"] = __import__("quirky.diagnose.report", fromlist=["DEFECT_CATALOG"]).DEFECT_CATALOG
        return JSONResponse(content=report)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(temp_in):
            try:
                os.remove(temp_in)
            except Exception:
                pass


@app.post("/api/fingerprint")
async def api_fingerprint(file: UploadFile = File(...)):
    """Guess which generator likely made an image (+ inverse fixes)."""
    temp_in = _save_upload(file)
    try:
        from quirky.diagnose import fingerprint_image
        return JSONResponse(content=fingerprint_image(temp_in))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(temp_in):
            try:
                os.remove(temp_in)
            except Exception:
                pass


@app.post("/api/audit")
async def api_audit(before: UploadFile = File(...), after: UploadFile = File(...)):
    """Honesty check: score before/after against an external oracle (not Quirky's own metric)."""
    b_path = _save_upload(before)
    a_path = _save_upload(after)
    try:
        from quirky.diagnose import audit, EnsembleHeuristicOracle
        return JSONResponse(content=audit(b_path, a_path, EnsembleHeuristicOracle()))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        for p in (b_path, a_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


def _mime_for(ext: str) -> str:
    return {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".webp": "image/webp"}.get(ext.lower(), "application/octet-stream")


@app.post("/api/scan")
async def api_scan(file: UploadFile = File(...)):
    """Read-only metadata report: every embedded field, flagged for generator leaks."""
    temp_in = _save_upload(file)
    try:
        from quirky.clean import scan_metadata
        return JSONResponse(content=scan_metadata(temp_in))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(temp_in):
            try:
                os.remove(temp_in)
            except Exception:
                pass


@app.post("/api/clean")
async def api_clean(file: UploadFile = File(...), attribute: bool = Form(False)):
    """
    Clean: scrub embedded metadata via a clean re-encode. Returns the before/after
    report plus the cleaned file inline as a base64 data URI -- no second download round-trip.
    """
    temp_in = _save_upload(file)
    ext = os.path.splitext(temp_in)[1].lower()
    temp_out = f"{temp_in}.clean{ext}"
    try:
        from quirky.clean import clean_metadata
        res = clean_metadata(temp_in, temp_out, attribute=attribute)
        with open(temp_out, "rb") as f:
            data = f.read()
        res["cleaned_data_uri"] = f"data:{_mime_for(ext)};base64,{base64.b64encode(data).decode('ascii')}"
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        for p in (temp_in, temp_out):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


@app.post("/api/remap")
async def api_remap(before: UploadFile = File(...), after: UploadFile = File(...), intensity: float = Form(0.6)):
    """Re-map: re-diagnose an edited image against the original -- resolved/remaining/new defects + delta heatmap."""
    b_path = _save_upload(before)
    a_path = _save_upload(after)
    try:
        from quirky.diagnose import remap_image
        return JSONResponse(content=remap_image(b_path, a_path, intensity=intensity))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        for p in (b_path, a_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


@app.post("/api/humanize")
async def api_humanize(
    file: UploadFile = File(...),
    intensity: float = Form(0.5),
    gamma: float = Form(0.6),
    delta: float = Form(0.03),
    fixes: str = Form(""),
    lock: bool = Form(False),
    target: float = Form(0.15),
    min_ssim: float = Form(0.86),
):
    """
    Humanization processing endpoint. Returns optimized file.

    `fixes` (comma-separated ids) gates which corrections run on images; empty = all.
    `lock` runs the minimal-edit fidelity loop instead of a single fixed-intensity pass.
    """
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1].lower()
    temp_in = os.path.join(TEMP_DIR, f"{file_id}_in{ext}")
    temp_out = os.path.join(TEMP_DIR, f"{file_id}_out{ext}")
    enabled_fixes = set(f.strip() for f in fixes.split(",") if f.strip()) or None

    try:
        # Save uploaded file
        with open(temp_in, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Humanize based on modality
        if ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
            if lock:
                from quirky.diagnose import humanize_locked
                humanize_locked(temp_in, temp_out, target_ai=target, min_ssim=min_ssim,
                                gamma=gamma, delta=delta, enabled_fixes=enabled_fixes)
            else:
                ImageHumanizer.humanize(temp_in, temp_out, intensity=intensity, gamma=gamma,
                                        delta=delta, enabled_fixes=enabled_fixes)
        elif ext in [".wav"]:
            AudioHumanizer.humanize(temp_in, temp_out, intensity=intensity)
        elif ext in [".txt", ".md", ".json"]:
            with open(temp_in, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            humanized = TextHumanizer.humanize(content, intensity=intensity)
            with open(temp_out, "w", encoding="utf-8") as f:
                f.write(humanized)
        else:
            return JSONResponse(status_code=400, content={"error": f"Unsupported file type: {ext}"})
            
        return FileResponse(temp_out, media_type="application/octet-stream", filename=f"humanized_{file.filename}")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    # We don't delete temp_out immediately because FileResponse needs to stream it.
    # It will be cleared on next server run or scheduled cleanup.

# Serve static web frontend
web_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
os.makedirs(web_dir, exist_ok=True)

# Mount web folder statically
app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")

def start_server(host: str = "127.0.0.1", port: int = 8000):
    # reload=False: the auto-reload watcher spawns a subprocess and is brittle for
    # end users (esp. Windows); developers can run uvicorn --reload directly.
    uvicorn.run("quirky.api.main:app", host=host, port=port, reload=False)

if __name__ == "__main__":
    start_server()
