from __future__ import annotations

import re
from typing import Any

from quirky.skills.base import BaseSkill

class CommentPrunerSkill(BaseSkill):
    """
    Strips redundant inline comments in source code (Python, JavaScript, TypeScript)
    that merely repeat what the line of code immediately below does.
    """
    @property
    def name(self) -> str:
        return "comment_pruner"

    @property
    def description(self) -> str:
        return "Strips redundant inline comments in source code that repeat the actions of the code."

    def execute(self, content: str, **kwargs) -> str:
        if not isinstance(content, str):
            raise TypeError("CommentPrunerSkill requires string input.")

        lines = content.splitlines(keepends=True)
        cleaned_lines = []
        i = 0
        n = len(lines)

        while i < n:
            line = lines[i]
            stripped = line.lstrip()

            # Identify if it is a single-line comment
            is_py_comment = stripped.startswith("#")
            is_js_comment = stripped.startswith("//")

            if is_py_comment or is_js_comment:
                # Find the next non-empty, non-comment line of code
                next_code_line = ""
                j = i + 1
                while j < n:
                    next_stripped = lines[j].lstrip()
                    if next_stripped and not next_stripped.startswith("#") and not next_stripped.startswith("//") and not next_stripped.startswith("/*") and not next_stripped.startswith("*"):
                        next_code_line = next_stripped
                        break
                    # If we find another comment block or boundary, stop
                    if next_stripped.startswith("#") != is_py_comment and next_stripped.startswith("//") != is_js_comment:
                        break
                    j += 1

                if next_code_line and self._is_redundant(stripped, next_code_line):
                    # Redundant comment: skip it (strip it)
                    i += 1
                    continue

            cleaned_lines.append(line)
            i += 1

        # Preserve line ending structure
        return "".join(cleaned_lines)

    def _is_redundant(self, comment: str, code: str) -> bool:
        # Clean comment prefix
        comment_clean = re.sub(r'^[#/\s*]+', '', comment).strip()
        
        # Keep section headers or short markers
        if len(comment_clean) < 3 or comment_clean.startswith(("-", "=", "#")):
            return False
            
        # Parse alphanumeric words
        def get_words(s: str) -> list[str]:
            return [w.lower() for w in re.findall(r'[a-zA-Z0-9]+', s)]
            
        comm_words = get_words(comment_clean)
        cd_words = get_words(code)
        
        if not comm_words or not cd_words:
            return False
            
        # Stop-words to filter out
        fillers = {
            "a", "an", "the", "to", "for", "in", "of", "and", "or", "is", "by", 
            "with", "from", "on", "at", "we", "need", "should", "must", "please", 
            "make", "sure", "that", "this", "do", "it", "my", "our", "your"
        }
        sig_comm = [w for w in comm_words if w not in fillers]
        
        if not sig_comm:
            return True  # Comment is entirely filler
            
        matches = 0
        for cw in sig_comm:
            if cw in cd_words:
                matches += 1
                continue
            # Handle common abbreviations (e.g. idx -> index, msg -> message)
            matched = False
            for w in cd_words:
                if len(w) > 2 and len(cw) > 2:
                    if w in cw or cw in w:
                        matched = True
                        break
            if matched:
                matches += 1
                
        overlap_ratio = matches / len(sig_comm)
        
        # Trivial verbs that signal redundant description
        trivial_verbs = {
            "set", "return", "call", "run", "print", "increment", "decrement", 
            "assign", "check", "verify", "import", "get", "define", "initialize", 
            "init", "start", "stop", "end", "create", "delete", "remove"
        }
        has_trivial_verb = comm_words[0] in trivial_verbs
        
        # If comment overlaps heavily with the code, it is redundant
        if overlap_ratio >= 0.5:
            return True
        if has_trivial_verb and matches >= 1:
            return True
            
        return False
