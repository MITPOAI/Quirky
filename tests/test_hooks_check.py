import json
import os
import sys
import tempfile
from unittest.mock import patch

from quirky.hooks.check import main


def test_post_tool_use_clean():
    # Clean text payload -> exit 0, no output
    payload = {
        "tool_name": "Write",
        "tool_input": {
            "file_path": "clean.md"
        }
    }
    
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".md", encoding="utf-8") as tmp:
        tmp.write("This is clean human text. It is straight to the point.")
        tmp_name = tmp.name
        
    payload["tool_input"]["file_path"] = tmp_name
    
    try:
        # Patch sys.stdin and sys.argv
        with patch("sys.stdin.read", return_value=json.dumps(payload)):
            with patch("sys.argv", ["check.py", "--event", "post-tool-use"]):
                with pytest_raises_system_exit(0) as ex:
                    main()
    finally:
        os.remove(tmp_name)


def test_post_tool_use_sloppy():
    # Sloppy text payload -> exit 0, outputs decision block
    payload = {
        "tool_name": "Write",
        "tool_input": {
            "file_path": "sloppy.md"
        }
    }
    
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".md", encoding="utf-8") as tmp:
        tmp.write(
            "Furthermore, it is important to note that this approach facilitates optimization. "
            "Moreover, one must utilize structured systems in order to succeed."
        )
        tmp_name = tmp.name
        
    payload["tool_input"]["file_path"] = tmp_name
    
    # We catch stdout to check the block payload
    from io import StringIO
    stdout_buf = StringIO()
    
    try:
        with patch("sys.stdin.read", return_value=json.dumps(payload)):
            with patch("sys.argv", ["check.py", "--event", "post-tool-use"]):
                with patch("sys.stdout", stdout_buf):
                    with pytest_raises_system_exit(0):
                        main()
                        
        output = stdout_buf.getvalue().strip()
        assert output != ""
        res = json.loads(output)
        assert res["decision"] == "block"
        assert "PostToolUse" in res["hookSpecificOutput"]["hookEventName"]
        assert "Suggested surgical fix diff:" in res["hookSpecificOutput"]["additionalContext"]
    finally:
        os.remove(tmp_name)


def test_stop_hook_active_short_circuit():
    # loop guard: if stop_hook_active is True -> exit 0 instantly without checking files
    payload = {
        "stop_hook_active": True
    }
    
    with patch("sys.stdin.read", return_value=json.dumps(payload)):
        with patch("sys.argv", ["check.py", "--event", "stop"]):
            with pytest_raises_system_exit(0):
                main()


def test_skip_extensions_and_changelogs():
    payload = {
        "tool_name": "Write",
        "tool_input": {
            "file_path": ""
        }
    }
    
    # CHANGELOG should be skipped even if it contains slop
    tmp_dir = tempfile.mkdtemp()
    changelog_path = os.path.join(tmp_dir, "CHANGELOG.md")
    try:
        with open(changelog_path, "w", encoding="utf-8") as f:
            f.write("Furthermore, it is important to note that this facilitates optimization.")
            
        payload["tool_input"]["file_path"] = changelog_path
        
        from io import StringIO
        stdout_buf = StringIO()
        
        with patch("sys.stdin.read", return_value=json.dumps(payload)):
            with patch("sys.argv", ["check.py", "--event", "post-tool-use"]):
                with patch("sys.stdout", stdout_buf):
                    with pytest_raises_system_exit(0):
                        main()
        # Output should be empty because CHANGELOG is skipped
        assert stdout_buf.getvalue().strip() == ""
    finally:
        try:
            os.remove(changelog_path)
            os.rmdir(tmp_dir)
        except Exception:
            pass


# Helper context manager for system exit checks in testing without requiring pytest dependencies at compile time
class pytest_raises_system_exit:
    def __init__(self, expected_code=0):
        self.expected_code = expected_code
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is SystemExit:
            assert exc_val.code == self.expected_code
            return True
        return False
