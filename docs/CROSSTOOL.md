# Quirky Cross-Tool Integration Guide

This guide explains how to install and integrate the Quirky active anti-slop layer across different AI assistants and IDE tools.

## 1. Claude Code

To add Quirky's active checks and plugins directly into Claude Code:

### Global Installation
Install Quirky with the `agent` extra:
```bash
uv pip install -e .[agent]
```
Add the repository as a plugin marketplace and install the plugin:
```bash
claude plugin marketplace add MITPOAI/Quirky
claude plugin install quirky
```

### Local Project Installation
Run the scaffolding installer inside your project repository root:
```bash
quirky init
```
This sets up active Claude Code hooks (`PostToolUse` and `Stop`) to prevent writing slop.

---

## 2. Cursor

To integrate Quirky with Cursor:

### Rule Enforcement
Initialize the rules inside your project:
```bash
quirky init
```
This creates `.cursor/rules/quirky.mdc` which automatically targets markdown and text files, forcing Cursor to follow the rules outlined in `AGENTS.md`.

### MCP Server Integration
Add the Quirky FastMCP server to your Cursor settings under **Features > MCP**:
* **Name**: `quirky`
* **Type**: `command`
* **Command**: `python -m quirky.mcp.server`

---

## 3. Codex

Codex natively reads standard open assistant instructions at the repository root:

1. Run the initializer:
   ```bash
   quirky init
   ```
2. Codex will automatically read and follow the custom rules in `AGENTS.md`.
3. To configure the MCP server for Codex, merge the following into your `.mcp.json` or equivalent MCP configuration file:
   ```json
   {
     "mcpServers": {
       "quirky": {
         "command": "python",
         "args": ["-m", "quirky.mcp.server"]
       }
     }
   }
   ```

---

## 4. Hook Requirements & Environment Variables

The hooks require a Python interpreter on your system `PATH` with `quirky` installed. The hook exits silently on missing imports to avoid blocking the agent during workspace setups.

### Environment Variable Overrides
Configure Quirky via the following environment variables:
* `QUIRKY_CALIBRATOR_PATH`: Absolute path to a custom fitted `calibrator.json` model.
* `QUIRKY_EMBED_BACKEND`: Set to `off` to force pure-Jaccard token comparison (disables neural cosine similarity).
* `QUIRKY_JACCARD_THRESHOLD`: Jaccard drift guard limit (default: `0.50`).
* `QUIRKY_COSINE_THRESHOLD`: Cosine similarity drift guard limit (default: `0.75`).
* `QUIRKY_HOOK_THRESHOLD`: Slop threshold at which hooks block edits (default: `0.60`).
* `QUIRKY_HOOK_EXTS`: Comma-separated file extensions checked by hooks (default: `.md,.txt,.mdx,.rst`).
