import os
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

@app.post("/api/humanize")
async def api_humanize(
    file: UploadFile = File(...),
    intensity: float = Form(0.5),
    gamma: float = Form(0.6),
    delta: float = Form(0.03)
):
    """
    Humanization processing endpoint. Returns optimized file.
    """
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1].lower()
    temp_in = os.path.join(TEMP_DIR, f"{file_id}_in{ext}")
    temp_out = os.path.join(TEMP_DIR, f"{file_id}_out{ext}")
    
    try:
        # Save uploaded file
        with open(temp_in, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        # Humanize based on modality
        if ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
            ImageHumanizer.humanize(temp_in, temp_out, intensity=intensity, gamma=gamma, delta=delta)
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
