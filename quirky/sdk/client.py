import httpx
import os
from typing import Dict, Any

class QuirkyClient:
    """
    Production-grade async Python client SDK for the Quirky Human Preference Engine.
    """
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")
        
    async def detect(self, filepath: str) -> Dict[str, Any]:
        """
        Submits an asset to the Quirky Cloud passive detection pipeline.
        Returns a dictionary containing the unified quality score report.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Local file not found at {filepath}")
            
        url = f"{self.base_url}/api/detect"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            with open(filepath, "rb") as f:
                files = {"file": (os.path.basename(filepath), f)}
                response = await client.post(url, files=files)
                
            response.raise_for_status()
            return response.json()
            
    async def humanize(
        self,
        filepath: str,
        output_path: str,
        intensity: float = 0.5,
        gamma: float = 0.6,
        delta: float = 0.03
    ) -> None:
        """
        Submits an asset to the Quirky Cloud humanization pipeline.
        Saves the reconstructed and aligned asset to output_path.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Local file not found at {filepath}")
            
        url = f"{self.base_url}/api/humanize"
        
        data = {
            "intensity": intensity,
            "gamma": gamma,
            "delta": delta
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            with open(filepath, "rb") as f:
                files = {"file": (os.path.basename(filepath), f)}
                response = await client.post(url, files=files, data=data)
                
            response.raise_for_status()
            
            with open(output_path, "wb") as out_f:
                out_f.write(response.content)
