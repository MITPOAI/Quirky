from __future__ import annotations

import re
import os
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

from quirky.skills.base import BaseSkill
from quirky.detector.spans import segment_text, SlopScorer
from quirky.fix.text import apply_rules, get_numbers
from quirky.fix.similarity import similarity
from quirky.text.pipeline import TextHumanizer

class ClichePrunerSkill(BaseSkill):
    """
    Surgically replaces AI-boilerplate words, cliches, and inserts natural contractions.
    Protects code blocks and numbers using deterministic similarity and number guards.
    """
    @property
    def name(self) -> str:
        return "cliche_pruner"

    @property
    def description(self) -> str:
        return "Surgically replaces AI clichés and inserts contractions while preserving code and numbers."

    def execute(self, content: str, **kwargs) -> str:
        if not isinstance(content, str):
            raise TypeError("ClichePrunerSkill requires string input.")

        # Accept custom levels (e.g. red, amber) or process all sentences
        levels = kwargs.get("levels", {"red", "amber"})
        scorer = kwargs.get("scorer") or SlopScorer()
        
        jaccard_thresh = float(os.environ.get("QUIRKY_JACCARD_THRESHOLD", 0.50))
        cosine_thresh = float(os.environ.get("QUIRKY_COSINE_THRESHOLD", 0.75))

        spans = scorer.score_spans(content)
        fixed_chunks = []
        last_idx = 0

        for span in spans:
            span_text = span["text"]
            start = span["start"]
            end = span["end"]
            kind = span["kind"]
            level = span["level"]

            if start > last_idx:
                fixed_chunks.append(content[last_idx:start])

            if kind == "sentence" and (not levels or level in levels):
                rewritten, _ = apply_rules(span_text)
                if rewritten != span_text:
                    before_nums = Counter(get_numbers(span_text))
                    after_nums = Counter(get_numbers(rewritten))
                    num_ok = before_nums == after_nums

                    sim_score, sim_method = similarity(span_text, rewritten)
                    if sim_method == "cosine":
                        sim_ok = sim_score >= cosine_thresh
                    else:
                        sim_ok = sim_score >= jaccard_thresh

                    if num_ok and sim_ok:
                        fixed_chunks.append(rewritten)
                    else:
                        fixed_chunks.append(span_text)
                else:
                    fixed_chunks.append(span_text)
            else:
                fixed_chunks.append(span_text)

            last_idx = end

        if last_idx < len(content):
            fixed_chunks.append(content[last_idx:])

        return "".join(fixed_chunks)


class RhythmSculptorSkill(BaseSkill):
    """
    Adjusts sentence length variance and punctuation rhythm to fit natural human writing.
    Uses masking to protect code blocks and numbers.
    """
    @property
    def name(self) -> str:
        return "rhythm_sculptor"

    @property
    def description(self) -> str:
        return "Sculpts sentence burstiness and Zipf-Mandelbrot rhythm while protecting code and numbers."

    def execute(self, content: str, **kwargs) -> str:
        if not isinstance(content, str):
            raise TypeError("RhythmSculptorSkill requires string input.")

        intensity = kwargs.get("intensity", 0.5)

        # Sentinel masking system to protect code/commands and numbers
        patterns = [
            ("fenced", r"```[\s\S]*?```"),
            ("indented", r"(?m)^(?: {4}|\t)+.*$"),
            ("inline", r"`[^`\n]+`"),
            ("cmd", r"(?m)^[$>]\s+.*$"),
            ("num", r"\d[\d,.:%\-]*"),
        ]

        all_matches = []
        for name, pat in patterns:
            for m in re.finditer(pat, content):
                all_matches.append((m.start(), m.end(), name, m.group(0)))

        all_matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        non_overlapping = []
        last_end = 0
        for start, end, name, val in all_matches:
            if start >= last_end:
                non_overlapping.append((start, end, name, val))
                last_end = end

        # Double check sentinel collision
        prefix = "\x00QK_R_"
        masked_text = content
        sentinel_map = {}
        for i, (start, end, name, val) in reversed(list(enumerate(non_overlapping))):
            sentinel = f"{prefix}{i}_\x00"
            sentinel_map[sentinel] = (name, val)
            masked_text = masked_text[:start] + sentinel + masked_text[end:]

        # Run text humanization steps on masked text
        processed = TextHumanizer.sculpt_zipf_mandelbrot(masked_text, intensity=intensity)
        processed = TextHumanizer.inject_burstiness(processed, intensity=intensity)
        processed = TextHumanizer.diversify_punctuation(processed, intensity=intensity)

        # Restore sentinels
        restored = processed
        for sentinel, (name, orig_val) in sentinel_map.items():
            if sentinel not in restored or restored.count(sentinel) != 1:
                # Sentinel corrupted or missing: return original content safely
                return content
            restored = restored.replace(sentinel, orig_val)

        # Verify no corruption of numbers or code
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

        input_nums = Counter([val for start, end, name, val in non_overlapping if name == "num"])
        output_nums = Counter([val for start, end, name, val in output_non_overlapping if name == "num"])

        input_code = [val for start, end, name, val in non_overlapping if name != "num"]
        output_code = [val for start, end, name, val in output_non_overlapping if name != "num"]

        if input_nums != output_nums or input_code != output_code:
            # Verification failed: return original content safely
            return content

        return restored


class FormattingSanitizerSkill(BaseSkill):
    """
    Cleans up punctuation tells like em-dashes and ellipses, normalizes quotes and spacing.
    Uses masking to protect code blocks and numbers.
    """
    @property
    def name(self) -> str:
        return "formatting_sanitizer"

    @property
    def description(self) -> str:
        return "Removes em-dashes, ellipses, normalizes spaces/quotes while protecting code and numbers."

    def execute(self, content: str, **kwargs) -> str:
        if not isinstance(content, str):
            raise TypeError("FormattingSanitizerSkill requires string input.")

        # Sentinel masking system to protect code/commands and numbers
        patterns = [
            ("fenced", r"```[\s\S]*?```"),
            ("indented", r"(?m)^(?: {4}|\t)+.*$"),
            ("inline", r"`[^`\n]+`"),
            ("cmd", r"(?m)^[$>]\s+.*$"),
            ("num", r"\d[\d,.:%\-]*"),
        ]

        all_matches = []
        for name, pat in patterns:
            for m in re.finditer(pat, content):
                all_matches.append((m.start(), m.end(), name, m.group(0)))

        all_matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        non_overlapping = []
        last_end = 0
        for start, end, name, val in all_matches:
            if start >= last_end:
                non_overlapping.append((start, end, name, val))
                last_end = end

        prefix = "\x00QK_F_"
        masked_text = content
        sentinel_map = {}
        for i, (start, end, name, val) in reversed(list(enumerate(non_overlapping))):
            sentinel = f"{prefix}{i}_\x00"
            sentinel_map[sentinel] = (name, val)
            masked_text = masked_text[:start] + sentinel + masked_text[end:]

        # Run formatting cleanup on masked text
        processed = TextHumanizer.strip_ai_punctuation(masked_text)

        # Restore sentinels
        restored = processed
        for sentinel, (name, orig_val) in sentinel_map.items():
            if sentinel not in restored or restored.count(sentinel) != 1:
                return content
            restored = restored.replace(sentinel, orig_val)

        # Verify no corruption of numbers or code
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

        input_nums = Counter([val for start, end, name, val in non_overlapping if name == "num"])
        output_nums = Counter([val for start, end, name, val in output_non_overlapping if name == "num"])

        input_code = [val for start, end, name, val in non_overlapping if name != "num"]
        output_code = [val for start, end, name, val in output_non_overlapping if name != "num"]

        if input_nums != output_nums or input_code != output_code:
            return content

        return restored
