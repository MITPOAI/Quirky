from typing import Any

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
