from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

class BaseSkill:
    """
    Base class for all Quirky agent skills.
    """
    @property
    def name(self) -> str:
        raise NotImplementedError("Skill must define a name.")

    @property
    def description(self) -> str:
        raise NotImplementedError("Skill must define a description.")

    def execute(self, content: Any, **kwargs) -> Any:
        raise NotImplementedError("Skill must implement execute.")


class DynamicSkill(BaseSkill):
    """
    A skill loaded dynamically at runtime from a frontmatter-equipped SKILL.md file.
    """
    def __init__(self, skill_md_path: str):
        self.path = os.path.abspath(skill_md_path)
        self.metadata: Dict[str, Any] = {}
        self.instructions: str = ""
        self._load_skill()

    def _load_skill(self) -> None:
        with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        meta, instructions = self._parse_frontmatter(text)
        self.metadata = meta
        self.instructions = instructions

    def _parse_frontmatter(self, text: str) -> tuple[dict[str, Any], str]:
        meta = {}
        content = text
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                yaml_block = parts[1]
                content = parts[2].strip()
                # Parse key-value lines
                for line in yaml_block.splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        k = k.strip()
                        v = v.strip()
                        if v.startswith(('"', "'")) and v.endswith(('"', "'")):
                            v = v[1:-1]
                        meta[k] = v
        return meta, content

    @property
    def name(self) -> str:
        return self.metadata.get("name", os.path.basename(os.path.dirname(self.path)))

    @property
    def description(self) -> str:
        return self.metadata.get("description", "A dynamic prompt-based skill.")

    def execute(self, content: Any, **kwargs) -> Any:
        """
        Executing a dynamic prompt-based skill returns its instructions and metadata,
        helping agents understand and execute the prompt guidelines on the content.
        """
        return {
            "name": self.name,
            "instructions": self.instructions,
            "metadata": self.metadata,
            "content": content
        }


class SkillRegistry:
    """
    Registry for managing and resolving Quirky skills.
    """
    def __init__(self):
        self._skills: Dict[str, BaseSkill] = {}

    def register(self, skill: BaseSkill) -> None:
        """Registers a skill instance."""
        self._skills[skill.name] = skill

    def get(self, name: str) -> Optional[BaseSkill]:
        """Gets a skill by name."""
        return self._skills.get(name)

    def list_skills(self) -> List[BaseSkill]:
        """Lists all registered skills."""
        return list(self._skills.values())

    def load_from_directory(self, dir_path: str) -> int:
        """
        Scans a directory for folders containing SKILL.md files,
        instantiates them as DynamicSkills, and registers them.
        Returns the number of skills registered.
        """
        count = 0
        p = os.path.abspath(dir_path)
        if not os.path.exists(p) or not os.path.isdir(p):
            return 0

        # Scan subdirectories
        for entry in os.listdir(p):
            sub = os.path.join(p, entry)
            if os.path.isdir(sub):
                skill_file = os.path.join(sub, "SKILL.md")
                if os.path.exists(skill_file):
                    try:
                        skill = DynamicSkill(skill_file)
                        self.register(skill)
                        count += 1
                    except Exception:
                        pass
        return count
