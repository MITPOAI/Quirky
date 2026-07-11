from __future__ import annotations

import difflib
import os
import re
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

from quirky.detector.lexicons import FILLERS, HEDGES
from quirky.detector.spans import SlopScorer
from quirky.fix.similarity import similarity
from quirky.text.pipeline import TextHumanizer


def apply_rules(sentence: str) -> Tuple[str, List[str]]:
    """
    Deterministic rule-based rewriting of a single sentence.
    Executes a chain: AI_REPLACEMENTS -> CONTRACTIONS -> Hedges stripping -> Fillers stripping.
    Must NOT call humanize() to avoid stochastic/joining behaviors.
    """
    why = []
    rewritten = sentence
    
    # 1. AI Replacements
    for pattern, replacement in TextHumanizer.AI_REPLACEMENTS.items():
        rx = re.compile(pattern, re.IGNORECASE)
        matches = rx.findall(rewritten)
        if matches:
            rewritten = rx.sub(replacement, rewritten)
            for m in matches:
                match_str = m[0] if isinstance(m, tuple) else m
                clean_match = re.sub(r"[^\w\s-]", "", match_str.lower()).strip()
                why.append(f"cliche:{clean_match}")
                
    # 2. Contractions
    for pattern, contraction in TextHumanizer.CONTRACTIONS.items():
        rx = re.compile(pattern, re.IGNORECASE)
        matches = rx.findall(rewritten)
        if matches:
            rewritten = rx.sub(contraction, rewritten)
            for m in matches:
                match_str = m[0] if isinstance(m, tuple) else m
                clean_match = re.sub(r"[^\w\s-]", "", match_str.lower()).strip()
                why.append(f"contraction:{clean_match}")
                
    # 3. Hedge stripping
    for hedge in sorted(HEDGES, key=len, reverse=True):
        pattern = r"\b" + re.escape(hedge) + r"\b"
        rx = re.compile(pattern, re.IGNORECASE)
        matches = rx.findall(rewritten)
        if matches:
            rewritten = rx.sub("", rewritten)
            for m in matches:
                why.append(f"hedge:{m.lower()}")
                
    # 4. Filler stripping
    for filler in sorted(FILLERS, key=len, reverse=True):
        pattern = r"\b" + re.escape(filler) + r"\b"
        rx = re.compile(pattern, re.IGNORECASE)
        matches = rx.findall(rewritten)
        if matches:
            rewritten = rx.sub("", rewritten)
            for m in matches:
                why.append(f"filler:{m.lower()}")
                
    # Clean up whitespace and punctuation spacing
    rewritten = re.sub(r"\s+", " ", rewritten)
    rewritten = re.sub(r"\s+([.,!?])", r"\1", rewritten)
    rewritten = re.sub(r",\s*,", ",", rewritten)
    rewritten = re.sub(r"\b,\s+([.!?])", r"\1", rewritten)
    rewritten = rewritten.strip()
    
    return rewritten, why


def get_numbers(s: str) -> List[str]:
    """Extract numeric patterns from text."""
    return re.findall(r"\d[\d,.:%\-]*", s)


def fix_spans(
    text: str, scorer: SlopScorer | None = None, levels: Set[str] = {"red", "amber"}
) -> Dict[str, Any]:
    """
    Surgically edit flagged sentence-spans only, leaving other text/bytes identical by construction.
    Applies number-preservation and similarity guards.
    """
    if scorer is None:
        scorer = SlopScorer()
        
    spans = scorer.score_spans(text)
    
    jaccard_thresh = float(os.environ.get("QUIRKY_JACCARD_THRESHOLD", 0.50))
    cosine_thresh = float(os.environ.get("QUIRKY_COSINE_THRESHOLD", 0.75))
    
    fixed_chunks = []
    last_idx = 0
    edits = []
    rejected = []
    
    for span in spans:
        span_text = span["text"]
        start = span["start"]
        end = span["end"]
        kind = span["kind"]
        level = span["level"]
        
        # Splicing untouched bytes before the span
        if start > last_idx:
            fixed_chunks.append(text[last_idx:start])
            
        if kind == "sentence" and level in levels:
            rewritten, why = apply_rules(span_text)
            if rewritten != span_text:
                # Run guards
                before_nums = Counter(get_numbers(span_text))
                after_nums = Counter(get_numbers(rewritten))
                num_ok = before_nums == after_nums
                
                sim_score, sim_method = similarity(span_text, rewritten)
                if sim_method == "cosine":
                    sim_ok = sim_score >= cosine_thresh
                else:
                    sim_ok = sim_score >= jaccard_thresh
                    
                edit_info = {
                    "span": {"start": start, "end": end, "level": level},
                    "before": span_text,
                    "after": rewritten,
                    "why": why,
                    "similarity": {"score": round(sim_score, 3), "method": sim_method},
                }
                
                if num_ok and sim_ok:
                    edit_info["guard"] = "passed"
                    edits.append(edit_info)
                    fixed_chunks.append(rewritten)
                else:
                    failed_list = []
                    if not num_ok:
                        failed_list.append("numbers_changed")
                    if not sim_ok:
                        failed_list.append("low_similarity")
                    edit_info["guard"] = "failed: " + ", ".join(failed_list)
                    rejected.append(edit_info)
                    fixed_chunks.append(span_text)
            else:
                fixed_chunks.append(span_text)
        else:
            fixed_chunks.append(span_text)
            
        last_idx = end
        
    if last_idx < len(text):
        fixed_chunks.append(text[last_idx:])
        
    fixed_text = "".join(fixed_chunks)
    
    # Generate diff
    diff_lines = list(
        difflib.unified_diff(
            text.splitlines(keepends=True),
            fixed_text.splitlines(keepends=True),
            fromfile="before",
            tofile="after",
        )
    )
    diff = "".join(diff_lines)
    
    return {
        "fixed_text": fixed_text,
        "diff": diff,
        "edits": edits,
        "rejected": rejected,
    }


def tighten(text: str) -> str:
    """
    Mask protected regions and numbers, deterministic hedge/filler/in-order-to deletions,
    restore sentinels, verify no corruption.
    """
    patterns = [
        ("fenced", r"```[\s\S]*?```"),
        ("indented", r"(?m)^(?: {4}|\t)+.*$"),
        ("inline", r"`[^`\n]+`"),
        ("cmd", r"(?m)^[$>]\s+.*$"),
        ("num", r"\d[\d,.:%\-]*"),
    ]
    
    # Find all matches
    all_matches = []
    for name, pat in patterns:
        for m in re.finditer(pat, text):
            all_matches.append((m.start(), m.end(), name, m.group(0)))
            
    # Resolve overlaps greedily
    all_matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    non_overlapping = []
    last_end = 0
    for start, end, name, val in all_matches:
        if start >= last_end:
            non_overlapping.append((start, end, name, val))
            last_end = end
            
    # Check for collisions with candidate sentinel templates
    collision_base = False
    for i in range(len(non_overlapping)):
        if f"\x00QK_{i}_\x00" in text:
            collision_base = True
            break
            
    prefix = "\x00QK_"
    if collision_base:
        collision_alt = False
        for i in range(len(non_overlapping)):
            if f"\x00QKALT_{i}_\x00" in text:
                collision_alt = True
                break
        if collision_alt:
            # Double collision: give up safely
            return text
        prefix = "\x00QKALT_"
        
    # Mask regions with sentinels from end to start to preserve offsets
    masked_text = text
    sentinel_map = {}
    for i, (start, end, name, val) in reversed(list(enumerate(non_overlapping))):
        sentinel = f"{prefix}{i}_\x00"
        sentinel_map[sentinel] = (name, val)
        masked_text = masked_text[:start] + sentinel + masked_text[end:]
        
    # Deletions on masked text
    # 1. in order to -> to
    tightened = re.sub(r"\bin order to\b", "to", masked_text, flags=re.IGNORECASE)
    tightened = re.sub(r"\bIn order to\b", "To", tightened, flags=re.IGNORECASE)
    
    # 2. fillers and hedges stripping
    for phrase in sorted(list(FILLERS) + list(HEDGES), key=len, reverse=True):
        pattern = r"\b" + re.escape(phrase) + r"\b"
        tightened = re.sub(pattern, "", tightened, flags=re.IGNORECASE)
        
    # Clean whitespace and spacing (horizontal space only to preserve line-endings)
    tightened = re.sub(r"[ \t]+", " ", tightened)
    tightened = re.sub(r"[ \t]+([.,!?])", r"\1", tightened)
    tightened = re.sub(r",[ \t]*,", ",", tightened)
    tightened = re.sub(r"\b,[ \t]+([.!?])", r"\1", tightened)
    tightened = tightened.strip()
    
    # Restore sentinels
    restored = tightened
    for sentinel, (name, orig_val) in sentinel_map.items():
        if sentinel not in restored or restored.count(sentinel) != 1:
            # Sentinel corrupted or missing: give up safely
            return text
        restored = restored.replace(sentinel, orig_val)
        
    # Verification: re-extract code regions and numbers from output, compare vs input
    output_matches = []
    for name, pat in patterns:
        for m in re.finditer(pat, restored):
            output_matches.append((m.start(), m.end(), name, m.group(0)))
            
    output_matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    output_non_overlapping = []
    last_end = 0
    for start, end, name, val in output_matches:
        if start >= last_end:
            output_non_overlapping.append((start, end, name, val))
            last_end = end
            
    # Compare numbers multiset
    input_nums = Counter([val for start, end, name, val in non_overlapping if name == "num"])
    output_nums = Counter([val for start, end, name, val in output_non_overlapping if name == "num"])
    
    # Compare code regions (in order)
    input_code_cmds = [val for start, end, name, val in non_overlapping if name != "num"]
    output_code_cmds = [val for start, end, name, val in output_non_overlapping if name != "num"]
    
    if input_nums != output_nums or input_code_cmds != output_code_cmds:
        # verification failed: return original text
        return text
        
    return restored
