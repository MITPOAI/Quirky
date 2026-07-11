from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Module exits 0 silently on ANY unexpected import error or missing deps to avoid blocking the agent.
try:
    from quirky.detector.spans import SlopScorer
    from quirky.fix.text import fix_spans
except Exception:
    sys.exit(0)


def exit_clean():
    sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", required=True, choices=["post-tool-use", "stop"])
    args = parser.parse_args()
    
    # Read stdin payload
    try:
        raw_stdin = sys.stdin.read()
        if not raw_stdin.strip():
            exit_clean()
        payload = json.loads(raw_stdin)
    except Exception:
        exit_clean()
        
    exts_env = os.environ.get("QUIRKY_HOOK_EXTS", ".md,.txt,.mdx,.rst")
    valid_exts = {e.strip().lower() for e in exts_env.split(",") if e.strip()}
    
    threshold = float(os.environ.get("QUIRKY_HOOK_THRESHOLD", 0.6))
    
    def should_check_file(file_path: str) -> bool:
        p = Path(file_path)
        ext = p.suffix.lower()
        if ext not in valid_exts:
            return False
        name = p.name.upper()
        if name.startswith("CHANGELOG") or name.startswith("LICENSE"):
            return False
        return True
        
    if args.event == "post-tool-use":
        tool_input = payload.get("tool_input", {})
        file_path = tool_input.get("file_path")
        if not file_path or not os.path.exists(file_path):
            exit_clean()
            
        if not should_check_file(file_path):
            exit_clean()
            
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            exit_clean()
            
        try:
            scorer = SlopScorer()
            score_res = scorer.score_text(content)
            doc_score = score_res["score"]
            if doc_score > threshold:
                red_spans = score_res["span_counts"]["red"]
                fix_res = fix_spans(content, scorer=scorer)
                diff = fix_res["diff"]
                
                # Build additionalContext details
                add_ctx = f"Quirky detected slop score {doc_score:.2f} in {file_path}.\n"
                if red_spans > 0:
                    add_ctx += f"Flagged spans: {red_spans} red.\n"
                if diff:
                    add_ctx += f"\nSuggested surgical fix diff:\n\n{diff}\n"
                else:
                    add_ctx += "\nNo surgical fixes could be determined.\n"
                    
                # Cap additionalContext to 10k chars
                add_ctx = add_ctx[:9900]
                
                response = {
                    "decision": "block",
                    "reason": f"Quirky: slop score {doc_score:.2f} in {file_path}. {red_spans} red spans.",
                    "hookSpecificOutput": {
                        "hookEventName": "PostToolUse",
                        "additionalContext": add_ctx
                    }
                }
                print(json.dumps(response))
                sys.exit(0)
        except Exception:
            exit_clean()
            
    elif args.event == "stop":
        # Guard stop_hook_active -> exit 0 (loop prevention)
        # Check both direct payload key or inside hookSpecificOutput
        if payload.get("stop_hook_active") or payload.get("hookSpecificOutput", {}).get("stop_hook_active"):
            exit_clean()
            
        # Get list of modified/added files via git diff
        try:
            res = subprocess.run(["git", "diff", "--name-only", "HEAD"], capture_output=True, text=True, check=True)
            files = [f.strip() for f in res.stdout.splitlines() if f.strip()]
        except Exception:
            exit_clean()
            
        flagged_file = None
        flagged_score = 0.0
        flagged_red_spans = 0
        
        for f_path in files:
            if not os.path.exists(f_path):
                continue
            if not should_check_file(f_path):
                continue
                
            try:
                with open(f_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                scorer = SlopScorer()
                score_res = scorer.score_text(content)
                if score_res["score"] > threshold:
                    flagged_file = f_path
                    flagged_score = score_res["score"]
                    flagged_red_spans = score_res["span_counts"]["red"]
                    break
            except Exception:
                pass
                
        if flagged_file:
            response = {
                "decision": "block",
                "reason": f"Quirky: slop score {flagged_score:.2f} in {flagged_file}. {flagged_red_spans} red spans.",
                "hookSpecificOutput": {
                    "stop_hook_active": True
                }
            }
            print(json.dumps(response))
            sys.exit(0)
            
    exit_clean()


if __name__ == "__main__":
    main()
