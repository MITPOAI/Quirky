from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from quirky.skills.base import BaseSkill, SkillRegistry
from quirky.skills.text import ClichePrunerSkill, RhythmSculptorSkill, FormattingSanitizerSkill
from quirky.skills.media import AssetOptimizerSkill
from quirky.skills.code import CommentPrunerSkill
from quirky.detector.engine import DetectorEngine

class QuirkyAgent:
    """
    Orchestration agent that analyzes assets and applies skills sequentially
    to remove AI slop while keeping code regions and numbers untouched.
    """
    def __init__(self, skills: Optional[List[BaseSkill]] = None, extra_skills_dir: Optional[str] = None):
        self.registry = SkillRegistry()
        
        default_skills = [
            CommentPrunerSkill(),
            ClichePrunerSkill(),
            RhythmSculptorSkill(),
            FormattingSanitizerSkill(),
            AssetOptimizerSkill()
        ]
        
        if skills is not None:
            for s in skills:
                self.registry.register(s)
        else:
            for s in default_skills:
                self.registry.register(s)
                
        if extra_skills_dir:
            self.registry.load_from_directory(extra_skills_dir)

    @property
    def skills(self) -> List[BaseSkill]:
        return self.registry.list_skills()

    def list_skills(self) -> List[Dict[str, str]]:
        """Lists names and descriptions of all registered skills."""
        return [
            {"name": s.name, "description": s.description}
            for s in self.skills
        ]

    def run(
        self,
        input_data: str,
        is_file: bool = True,
        output_path: Optional[str] = None,
        intensity: float = 0.5
    ) -> Dict[str, Any]:
        """
        Executes the agent on the input. If it is a file, the agent detects
        the file type and applies matching skills.
        """
        logs = []
        applied = []
        
        if is_file:
            abs_path = os.path.abspath(input_data)
            if not os.path.exists(abs_path):
                return {
                    "status": "error",
                    "error": f"File not found at {abs_path}",
                    "logs": ["Failed to find input file."]
                }
                
            ext = os.path.splitext(abs_path)[1].lower()
            logs.append(f"Analyzing file: {abs_path} (format: {ext})")
            
            # Analyze initial state
            initial_report = DetectorEngine.analyze_asset(abs_path)
            initial_score = initial_report.get("metadata", {}).get("ai_score", 0.0)
            logs.append(f"Initial AI slop score: {initial_score}")

            if ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".wav", ".mp4", ".avi", ".mov"]:
                # Binary/media pipeline
                optimizer = self.registry.get("asset_optimizer")
                if not optimizer:
                    return {
                        "status": "error",
                        "error": "AssetOptimizerSkill not registered on agent.",
                        "logs": logs
                    }
                
                logs.append("Routing to media humanization pipeline.")
                try:
                    final_path = optimizer.execute(
                        abs_path,
                        output_path=output_path,
                        intensity=intensity
                    )
                    applied.append("asset_optimizer")
                    logs.append(f"Successfully optimized media file. Saved to: {final_path}")
                    
                    final_report = DetectorEngine.analyze_asset(final_path)
                    final_score = final_report.get("metadata", {}).get("ai_score", 0.0)
                    logs.append(f"Final AI slop score: {final_score}")
                    
                    return {
                        "status": "success",
                        "modality": "media",
                        "original_score": initial_score,
                        "final_score": final_score,
                        "output_path": final_path,
                        "skills_applied": applied,
                        "logs": logs
                    }
                except Exception as e:
                    logs.append(f"Error executing media optimizer: {str(e)}")
                    return {
                        "status": "error",
                        "error": str(e),
                        "logs": logs
                    }
            else:
                # Text file pipeline
                try:
                    with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                        text_content = f.read()
                except Exception as e:
                    return {
                        "status": "error",
                        "error": f"Failed to read file: {str(e)}",
                        "logs": logs
                    }
                
                res = self._run_text_pipeline(text_content, intensity, logs, applied)
                if res["status"] == "success" and output_path:
                    out_abs = os.path.abspath(output_path)
                    try:
                        with open(out_abs, "w", encoding="utf-8") as f:
                            f.write(res["final_content"])
                        res["output_path"] = out_abs
                        logs.append(f"Saved optimized text to: {out_abs}")
                    except Exception as e:
                        logs.append(f"Error saving output file: {str(e)}")
                        res["status"] = "error"
                        res["error"] = str(e)
                return res

        else:
            # Direct text input pipeline
            logs.append("Processing direct text input.")
            return self._run_text_pipeline(input_data, intensity, logs, applied)

    def _run_text_pipeline(
        self,
        text: str,
        intensity: float,
        logs: List[str],
        applied: List[str]
    ) -> Dict[str, Any]:
        # Score initial text
        from quirky.detector.spans import SlopScorer
        scorer = SlopScorer()
        initial_score = scorer.score_text(text).get("score", 0.0)
        logs.append(f"Initial text slop score: {initial_score}")

        current_text = text

        # 1. Comment Pruner
        comment_pruner = self.registry.get("comment_pruner")
        if comment_pruner:
            logs.append("Running comment pruner skill.")
            next_text = comment_pruner.execute(current_text)
            if next_text != current_text:
                logs.append("Comment pruner modified the text (removed redundant comments).")
                applied.append("comment_pruner")
                current_text = next_text
            else:
                logs.append("Comment pruner made no changes.")

        # 2. Cliche Pruner
        cliche_pruner = self.registry.get("cliche_pruner")
        if cliche_pruner:
            logs.append("Running cliché pruner skill.")
            next_text = cliche_pruner.execute(current_text, scorer=scorer)
            if next_text != current_text:
                logs.append("Cliché pruner modified the text (removed AI tropes/hedges).")
                applied.append("cliche_pruner")
                current_text = next_text
            else:
                logs.append("Cliché pruner made no changes.")
        
        # 3. Rhythm Sculptor
        rhythm_sculptor = self.registry.get("rhythm_sculptor")
        if rhythm_sculptor:
            logs.append("Running rhythm sculptor skill.")
            next_text = rhythm_sculptor.execute(current_text, intensity=intensity)
            if next_text != current_text:
                logs.append("Rhythm sculptor modified the text (sculpted burstiness and Zipf rhythm).")
                applied.append("rhythm_sculptor")
                current_text = next_text
            else:
                logs.append("Rhythm sculptor made no changes.")

        # 4. Formatting Sanitizer
        formatting_sanitizer = self.registry.get("formatting_sanitizer")
        if formatting_sanitizer:
            logs.append("Running formatting sanitizer skill.")
            next_text = formatting_sanitizer.execute(current_text)
            if next_text != current_text:
                logs.append("Formatting sanitizer modified the text (cleaned dashes, ellipses, spacing).")
                applied.append("formatting_sanitizer")
                current_text = next_text
            else:
                logs.append("Formatting sanitizer made no changes.")

        # Score final text
        final_score = scorer.score_text(current_text).get("score", 0.0)
        logs.append(f"Final text slop score: {final_score}")

        return {
            "status": "success",
            "modality": "text",
            "original_score": initial_score,
            "final_score": final_score,
            "original_content": text,
            "final_content": current_text,
            "skills_applied": applied,
            "logs": logs
        }
