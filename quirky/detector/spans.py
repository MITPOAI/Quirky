from __future__ import annotations

import re
from typing import Any, Dict, List, Protocol, Tuple

import numpy as np

from quirky.detector.calibrate import (
    _EXPANDED,
    _EXPANDED_RE,
    calibrated_text_score,
    extract_features,
)
from quirky.detector.lexicons import AI_CLICHES, FILLERS, HEDGES

# Thresholds constants
AMBER_THRESHOLD = 0.45
RED_THRESHOLD = 0.70


class SpanScorerProtocol(Protocol):
    def score_spans(self, text: str) -> List[Dict[str, Any]]:
        ...


def shrink_span(text: str, start: int, end: int) -> Tuple[int, int]:
    """Shrink the span boundaries arithmetically to strip leading/trailing whitespace."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def get_protected_spans(text: str) -> List[Tuple[int, int, str]]:
    """Scan and identify fenced code, inline code, and headings."""
    protected = []
    
    # 1. Fenced code blocks
    for m in re.finditer(r"```[\s\S]*?```", text):
        protected.append((m.start(), m.end(), "code"))
        
    # 2. Inline code (only if not overlapping with already found regions)
    for m in re.finditer(r"`[^`\n]+`", text):
        start, end = m.start(), m.end()
        if not any(s <= start < e or s < end <= e for s, e, _ in protected):
            protected.append((start, end, "code"))
            
    # 3. Headings (only if not overlapping with already found regions)
    for m in re.finditer(r"(?m)^#{1,6}\s+.*$", text):
        start, end = m.start(), m.end()
        if not any(s <= start < e or s < end <= e for s, e, _ in protected):
            protected.append((start, end, "heading"))
            
    protected.sort(key=lambda x: x[0])
    return protected


def segment_text(text: str) -> List[Dict[str, Any]]:
    """Segment text into non-overlapping, sorted spans preserving character offsets."""
    protected = get_protected_spans(text)
    spans = []
    
    # Add protected regions
    for start, end, kind in protected:
        s_start, s_end = shrink_span(text, start, end)
        if s_start < s_end:
            spans.append({
                "text": text[s_start:s_end],
                "start": s_start,
                "end": s_end,
                "kind": kind
            })
            
    # Compute gaps between protected regions
    gaps = []
    last_end = 0
    for start, end, _ in protected:
        if start > last_end:
            gaps.append((last_end, start))
        last_end = end
    if last_end < len(text):
        gaps.append((last_end, len(text)))
        
    # Find sentence boundaries within gaps
    # boundary pattern: [.!?]+(?=[\s"')\]]|$) or double newline (CRLF/LF)
    boundary_re = re.compile(r"[.!?]+(?=[\s\"')\]]|$)|(?:\r?\n){2,}")
    
    for gap_start, gap_end in gaps:
        current_start = gap_start
        gap_str = text[gap_start:gap_end]
        for m in boundary_re.finditer(gap_str):
            b_end = gap_start + m.end()
            s_start, s_end = shrink_span(text, current_start, b_end)
            if s_start < s_end:
                spans.append({
                    "text": text[s_start:s_end],
                    "start": s_start,
                    "end": s_end,
                    "kind": "sentence"
                })
            current_start = b_end
            
        # Remainder of the gap
        s_start, s_end = shrink_span(text, current_start, gap_end)
        if s_start < s_end:
            spans.append({
                "text": text[s_start:s_end],
                "start": s_start,
                "end": s_end,
                "kind": "sentence"
            })
            
    spans.sort(key=lambda x: x["start"])
    return spans


class SlopScorer:
    def __init__(self, calibrator=None, thresholds: Tuple[float, float] = (AMBER_THRESHOLD, RED_THRESHOLD)):
        self.calibrator = calibrator
        self.amber_thresh, self.red_thresh = thresholds

    def score_text(self, text: str) -> Dict[str, Any]:
        """Compute document-level score, features, and span level counts."""
        doc_score, source = calibrated_text_score(text, calibrator=self.calibrator)
        features = extract_features(text)
        
        spans = self.score_spans(text)
        red_count = sum(1 for s in spans if s["level"] == "red")
        amber_count = sum(1 for s in spans if s["level"] == "amber")
        green_count = sum(1 for s in spans if s["level"] == "green")
        
        return {
            "score": round(doc_score, 3),
            "source": source,
            "features": features,
            "span_counts": {
                "red": red_count,
                "amber": amber_count,
                "green": green_count
            }
        }

    def score_spans(self, text: str) -> List[Dict[str, Any]]:
        """Segment and score each individual text span, blending local signals with document prior."""
        raw_spans = segment_text(text)
        if not raw_spans:
            return []
            
        doc_prob, _ = calibrated_text_score(text, calibrator=self.calibrator)
        
        # Calculate length stats for sentence spans in doc
        sentence_lens = [len(s["text"].split()) for s in raw_spans if s["kind"] == "sentence"]
        if len(sentence_lens) > 1:
            mean_len = float(np.mean(sentence_lens))
            std_len = float(np.std(sentence_lens)) + 1e-8
        else:
            mean_len = 15.0
            std_len = 5.0
            
        scored_spans = []
        for span in raw_spans:
            span_text = span["text"]
            kind = span["kind"]
            start = span["start"]
            end = span["end"]
            
            if kind == "code":
                scored_spans.append({
                    "text": span_text,
                    "start": start,
                    "end": end,
                    "kind": kind,
                    "score": 0.0,
                    "level": "green",
                    "reasons": ["protected"]
                })
                continue
                
            # Scan local signals and reasons
            reasons = []
            
            # 1. Clichés
            for c in AI_CLICHES:
                if re.search(r"\b" + re.escape(c) + r"\b", span_text, re.IGNORECASE):
                    reasons.append(f"cliche:{c}")
                    
            # 2. Hedges
            for h in HEDGES:
                if re.search(r"\b" + re.escape(h) + r"\b", span_text, re.IGNORECASE):
                    reasons.append(f"hedge:{h}")
                    
            # 3. Fillers
            for f in FILLERS:
                if re.search(r"\b" + re.escape(f) + r"\b", span_text, re.IGNORECASE):
                    reasons.append(f"filler:{f}")
                    
            # 4. Contractions
            expanded = len(_EXPANDED_RE.findall(span_text))
            contracted = span_text.count("'")
            contraction_deficit = expanded / (expanded + contracted + 1.0)
            for exp in _EXPANDED:
                if re.search(r"\b" + re.escape(exp) + r"\b", span_text, re.IGNORECASE):
                    reasons.append(f"expanded:{exp}")
                    
            # 5. Leak regex
            leak_patterns = [
                (r"(as an AI language model|system instructions|ignore previous instructions)", "leak:ai_model"),
                (r"(Output format:|JSON format:|User:|Assistant:)", "leak:format"),
                (r"(\\n|### Instruction|### Response)", "leak:instruction"),
                (r"(helpful, harmless, and honest)", "leak:alignment")
            ]
            leak_detected = False
            for pat, name in leak_patterns:
                if re.search(pat, span_text, re.IGNORECASE):
                    reasons.append(name)
                    leak_detected = True
                    
            # 6. Length z-score
            span_len = len(span_text.split())
            z_score = (span_len - mean_len) / std_len
            
            # Combine local signals into a score in [0, 1]
            cliche_hits = sum(1 for r in reasons if r.startswith("cliche:"))
            hedge_hits = sum(1 for r in reasons if r.startswith("hedge:"))
            filler_hits = sum(1 for r in reasons if r.startswith("filler:"))
            
            hits_score = (
                0.50 * min(cliche_hits, 1) + 0.15 * max(0, cliche_hits - 1) +
                0.30 * min(hedge_hits, 1) + 0.10 * max(0, hedge_hits - 1) +
                0.30 * min(filler_hits, 1) + 0.10 * max(0, filler_hits - 1)
            )
            cont_score = 0.20 * contraction_deficit
            len_score = 0.15 * (1.0 - min(abs(z_score), 2.0) / 2.0)
            
            local_score = hits_score + cont_score + len_score
            if leak_detected:
                local_score = max(local_score, 0.90)
                
            local_score = float(np.clip(local_score, 0.0, 1.0))
            
            # Blend 70/30 local score with doc-level calibrated prior
            blended_score = 0.7 * local_score + 0.3 * doc_prob
            
            # Determine level
            if blended_score >= self.red_thresh:
                level = "red"
            elif blended_score >= self.amber_thresh:
                level = "amber"
            else:
                level = "green"
                
            scored_spans.append({
                "text": span_text,
                "start": start,
                "end": end,
                "kind": kind,
                "score": round(blended_score, 3),
                "level": level,
                "reasons": reasons if reasons else ["clean"] if kind == "sentence" else ["protected"]
            })
            
        return scored_spans
