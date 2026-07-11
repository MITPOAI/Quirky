import pytest

from quirky.detector.spans import SlopScorer, segment_text


def test_offset_invariance_and_non_overlapping():
    texts = [
        "Hello world. ```\npython\nprint(123)\n``` How are you today? `x = 10` is good.",
        "Simple sentence. Another one here!",
        "### Header here\nAnd text under it.",
        "Double \n\n newline test.",
        "What is 3.14? A number.",
        "",
        "Single sentence.",
    ]
    
    for text in texts:
        spans = segment_text(text)
        
        # Verify text[start:end] == span.text for every span
        for span in spans:
            assert span["text"] == text[span["start"]:span["end"]], f"Offset mismatch in text: '{text}' for span: {span}"
            
        # Verify sorted and non-overlapping
        for i in range(len(spans) - 1):
            assert spans[i]["end"] <= spans[i + 1]["start"], f"Overlap detected between {spans[i]} and {spans[i+1]}"


def test_slop_scorer_levels():
    scorer = SlopScorer()
    
    # AI Text should trigger red spans
    ai_text = (
        "Furthermore, it is important to note that this approach facilitates optimization. "
        "Moreover, one must utilize structured systems to maximize productivity. "
        "Additionally, it is essential to leverage robust methodologies in order to succeed. "
        "In conclusion, the methodology facilitates correct execution and ensures success."
    )
    spans_ai = scorer.score_spans(ai_text)
    assert any(s["level"] == "red" for s in spans_ai), "Expected at least one red span in slop text"
    
    # Terse human text should be green
    human_text = "I think we can do this. Let's merge the changes now."
    spans_human = scorer.score_spans(human_text)
    assert all(s["level"] == "green" for s in spans_human), "Expected all spans to be green in human text"


def test_code_spans_never_red():
    scorer = SlopScorer()
    text = "Furthermore, it is important to note that: ```python\n# Furthermore, it is important to note\nprint('utilize')\n```"
    spans = scorer.score_spans(text)
    
    for s in spans:
        if s["kind"] == "code":
            assert s["level"] == "green"
            assert "protected" in s["reasons"]


def test_numerical_split_avoidance():
    # 3.14 should not be split into sentence boundary
    text = "Pi is roughly 3.14. It is a constant."
    spans = segment_text(text)
    assert len(spans) == 2
    assert "3.14" in spans[0]["text"]
