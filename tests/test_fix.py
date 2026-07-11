import os
from unittest.mock import patch

import pytest

from quirky.fix.text import apply_rules, fix_spans


def test_bytes_outside_edit_ranges_identical():
    text = "Some prefix code block: ```python\n# keep this\n``` Furthermore, it is important to note that this facilitates optimization. Some suffix."
    res = fix_spans(text)
    
    # Check suffix and prefix are exactly identical
    prefix = "Some prefix code block: ```python\n# keep this\n``` "
    suffix = " Some suffix."
    assert res["fixed_text"].startswith(prefix)
    assert res["fixed_text"].endswith(suffix)


def test_deterministic_behavior():
    text = "Furthermore, it is important to note that this facilitates optimization."
    res1 = fix_spans(text)
    res2 = fix_spans(text)
    assert res1["fixed_text"] == res2["fixed_text"]
    assert res1["diff"] == res2["diff"]


def test_numbers_changed_guard_rejected():
    # If numbers change, the replacement must be rejected and original text retained
    # We enforce a scenario where rule modification changes/drops a number
    text = "Furthermore, it is important to note that 123 packages were updated."
    
    # We patch apply_rules to return a string without the number
    def mock_apply_rules(s):
        return "Honestly, packages were updated.", ["mock"]
        
    with patch("quirky.fix.text.apply_rules", mock_apply_rules):
        res = fix_spans(text)
        assert res["fixed_text"] == text
        assert len(res["rejected"]) == 1
        assert "numbers_changed" in res["rejected"][0]["guard"]


def test_low_similarity_guard_rejected():
    text = "Furthermore, it is important to note that this facilitates optimization."
    
    # We patch apply_rules to return completely unrelated text
    def mock_apply_rules(s):
        return "Completely different text that has zero word overlap.", ["mock"]
        
    with patch("quirky.fix.text.apply_rules", mock_apply_rules):
        # Force Jaccard similarity threshold to high level
        with patch.dict(os.environ, {"QUIRKY_JACCARD_THRESHOLD": "0.9"}):
            res = fix_spans(text)
            assert res["fixed_text"] == text
            assert len(res["rejected"]) == 1
            assert "low_similarity" in res["rejected"][0]["guard"]
