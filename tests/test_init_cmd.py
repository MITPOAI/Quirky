import json
from pathlib import Path
import pytest
from typer.testing import CliRunner

from quirky.cli.main import app


def test_quirky_init_creates_files(tmp_path):
    runner = CliRunner()
    
    # Run init
    result = runner.invoke(app, ["init", "--path", str(tmp_path)])
    assert result.exit_code == 0
    assert "Success!" in result.stdout
    
    # Verify files created
    assert (tmp_path / "AGENTS.md").exists()
    assert (tmp_path / "CLAUDE.md").exists()
    assert (tmp_path / ".cursor" / "rules" / "quirky.mdc").exists()
    assert (tmp_path / ".mcp.json").exists()
    
    # Rerun should be unchanged
    result_rerun = runner.invoke(app, ["init", "--path", str(tmp_path)])
    assert result_rerun.exit_code == 0
    assert "[UNCHANGED]" in result_rerun.stdout


def test_user_content_outside_markers_preserved(tmp_path):
    runner = CliRunner()
    
    # Create file with existing user content and markers
    agents_path = tmp_path / "AGENTS.md"
    user_header = "# My Custom Guidelines\nSome user notes here.\n"
    user_footer = "\nMore user notes at footer."
    
    agents_path.write_text(
        f"{user_header}<!-- quirky:begin v1 -->\nOLD CONTENT\n<!-- quirky:end -->{user_footer}",
        encoding="utf-8"
    )
    
    # Run init
    result = runner.invoke(app, ["init", "--path", str(tmp_path)])
    assert result.exit_code == 0
    
    # Verify content replaced inside markers, but preserved outside
    content = agents_path.read_text(encoding="utf-8")
    assert content.startswith(user_header)
    assert content.endswith(user_footer)
    assert "Quirky Agent Guidelines" in content
    assert "OLD CONTENT" not in content


def test_foreign_mcp_keys_preserved(tmp_path):
    runner = CliRunner()
    
    # Pre-create .mcp.json with a foreign server key
    mcp_path = tmp_path / ".mcp.json"
    initial_config = {
        "mcpServers": {
            "my_custom_server": {
                "command": "node",
                "args": ["custom.js"]
            }
        }
    }
    mcp_path.write_text(json.dumps(initial_config, indent=2), encoding="utf-8")
    
    # Run init
    result = runner.invoke(app, ["init", "--path", str(tmp_path)])
    assert result.exit_code == 0
    
    # Verify both keys exist in .mcp.json
    config = json.loads(mcp_path.read_text(encoding="utf-8"))
    assert "my_custom_server" in config["mcpServers"]
    assert "quirky" in config["mcpServers"]
    assert config["mcpServers"]["quirky"]["command"] == "python"
    assert config["mcpServers"]["my_custom_server"]["command"] == "node"


def test_invalid_json_aborts(tmp_path):
    runner = CliRunner()
    
    # Create corrupted JSON
    mcp_path = tmp_path / ".mcp.json"
    mcp_path.write_text("{corrupt json", encoding="utf-8")
    
    # Run init
    result = runner.invoke(app, ["init", "--path", str(tmp_path)])
    assert result.exit_code != 0
    assert "Scaffolding Error" in result.stdout
