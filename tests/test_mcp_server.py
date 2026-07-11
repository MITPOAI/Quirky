import pytest

from quirky.mcp.server import (
    quirky_critique_text,
    quirky_detect_media,
    quirky_fix_text,
    quirky_humanize_media,
    quirky_score_text,
    quirky_tighten_text,
)


def test_mcp_text_tools():
    pytest.importorskip("fastmcp")
    
    text = "Furthermore, it is important to note that this facilitates optimization."
    
    # 1. score
    res = quirky_score_text(text)
    assert "score" in res
    assert "span_counts" in res
    assert "attribution" in res
    
    # 2. critique
    res = quirky_critique_text(text, max_spans=1)
    assert len(res["spans"]) == 1
    assert "summary" in res
    
    # 3. fix
    res = quirky_fix_text(text)
    assert "fixed_text" in res
    assert "diff" in res
    
    # 4. tighten
    res = quirky_tighten_text(text)
    assert "tightened_text" in res
    assert "Furthermore" not in res["tightened_text"]


def test_detect_media_error():
    pytest.importorskip("fastmcp")
    
    # Missing file path should return an error dict, never raise
    res = quirky_detect_media("non_existent_file.png")
    assert "error" in res
    assert "attribution" in res


def test_humanize_media_error():
    pytest.importorskip("fastmcp")
    
    res = quirky_humanize_media("non_existent_file.png")
    assert "error" in res
    assert "attribution" in res
