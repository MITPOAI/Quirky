from __future__ import annotations

import json
import os
import re
from pathlib import Path

# Get template utility matching Python 3.10+ importlib.resources patterns
try:
    from importlib.resources import files
    def get_template(name: str) -> str:
        return files("quirky.cli.templates").joinpath(name).read_text(encoding="utf-8")
except ImportError:
    import importlib.resources as pkg_resources
    def get_template(name: str) -> str:
        return pkg_resources.read_text("quirky.cli.templates", name)


def init_repo(target_dir: str | os.PathLike, force: bool = False) -> dict[str, str]:
    """
    Scaffolds AGENTS.md, CLAUDE.md, .cursor/rules/quirky.mdc, and .mcp.json into the target repo.
    Returns a status dict: filename -> 'created' | 'updated' | 'unchanged'.
    """
    target_path = Path(target_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 1. Scaffolding Markdown & MDC files (HTML-comment marked blocks)
    files_to_scaffold = [
        ("AGENTS.md", target_path / "AGENTS.md"),
        ("CLAUDE.md", target_path / "CLAUDE.md"),
        ("quirky.mdc", target_path / ".cursor" / "rules" / "quirky.mdc")
    ]
    
    marker_begin = "<!-- quirky:begin v1 -->"
    marker_end = "<!-- quirky:end -->"
    
    for template_name, file_path in files_to_scaffold:
        template_content = get_template(template_name).strip()
        full_marked_content = f"{marker_begin}\n{template_content}\n{marker_end}"
        
        if not file_path.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(full_marked_content, encoding="utf-8")
            results[file_path.name] = "created"
        else:
            existing_content = file_path.read_text(encoding="utf-8")
            if marker_begin in existing_content and marker_end in existing_content:
                # Replace the block
                pattern = re.compile(rf"{re.escape(marker_begin)}[\s\S]*?{re.escape(marker_end)}")
                new_content = pattern.sub(full_marked_content, existing_content)
                if new_content == existing_content:
                    results[file_path.name] = "unchanged"
                else:
                    file_path.write_text(new_content, encoding="utf-8")
                    results[file_path.name] = "updated"
            else:
                if force:
                    file_path.write_text(full_marked_content, encoding="utf-8")
                    results[file_path.name] = "updated"
                else:
                    results[file_path.name] = "unchanged"
                    
    # 2. Scaffolding .mcp.json (JSON-level merge of only mcpServers.quirky)
    mcp_path = target_path / ".mcp.json"
    mcp_template = json.loads(get_template("mcp.json"))
    
    if not mcp_path.exists():
        mcp_path.write_text(json.dumps(mcp_template, indent=2) + "\n", encoding="utf-8")
        results[mcp_path.name] = "created"
    else:
        existing_text = mcp_path.read_text(encoding="utf-8")
        try:
            existing_json = json.loads(existing_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in existing .mcp.json at {mcp_path}: {e}")
            
        mcp_servers = existing_json.setdefault("mcpServers", {})
        quirky_server = mcp_template["mcpServers"]["quirky"]
        
        if mcp_servers.get("quirky") == quirky_server:
            results[mcp_path.name] = "unchanged"
        else:
            mcp_servers["quirky"] = quirky_server
            mcp_path.write_text(json.dumps(existing_json, indent=2) + "\n", encoding="utf-8")
            results[mcp_path.name] = "updated"
            
    return results
