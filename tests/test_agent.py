import os
import pytest
from unittest.mock import patch, MagicMock
from quirky.agent.core import QuirkyAgent

def test_quirky_agent_text_pipeline():
    agent = QuirkyAgent()
    skills = agent.list_skills()
    assert len(skills) == 5
    
    # Test text pipeline with direct input
    text = "Furthermore, it is important to note that we must optimize this. There are 100 cases."
    result = agent.run(text, is_file=False, intensity=0.8)
    
    assert result["status"] == "success"
    assert result["modality"] == "text"
    assert result["original_score"] >= 0.0
    assert result["final_score"] >= 0.0
    assert "cliche_pruner" in result["skills_applied"]
    assert len(result["logs"]) > 0
    assert "100" in result["final_content"]


@patch("quirky.agent.core.DetectorEngine.analyze_asset")
def test_quirky_agent_media_pipeline(mock_analyze):
    # Mock analysis reports
    mock_analyze.return_value = {"metadata": {"ai_score": 0.95}}
    
    agent = QuirkyAgent()
    
    # Mock AssetOptimizerSkill's execute method
    mock_opt = MagicMock(return_value="mock_output.png")
    agent.registry.get("asset_optimizer").execute = mock_opt
    
    # We patch os.path.exists to return True for our fake image path
    with patch("os.path.exists", return_value=True):
        result = agent.run("fake_image.png", is_file=True)
        
        assert result["status"] == "success"
        assert result["modality"] == "media"
        assert result["original_score"] == 0.95
        assert result["output_path"] == "mock_output.png"
        assert "asset_optimizer" in result["skills_applied"]
        
        mock_opt.assert_called_once_with(
            os.path.abspath("fake_image.png"),
            output_path=None,
            intensity=0.5
        )
