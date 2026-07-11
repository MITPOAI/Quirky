import pytest
from quirky.skills.text import ClichePrunerSkill, RhythmSculptorSkill, FormattingSanitizerSkill
from quirky.skills.media import AssetOptimizerSkill

def test_cliche_pruner_skill():
    skill = ClichePrunerSkill()
    assert skill.name == "cliche_pruner"
    assert "replaces AI clichés" in skill.description

    text = "Furthermore, it is important to note that 123 packages were updated. Keep this: ```python\n# keep this\n```"
    result = skill.execute(text)
    
    # Check that "Furthermore, it is important to note that" is replaced with natural language
    # but the number 123 and code block are untouched.
    assert "123" in result
    assert "```python\n# keep this\n```" in result
    assert "Furthermore" not in result
    assert "it is important to note that" not in result


def test_rhythm_sculptor_skill():
    skill = RhythmSculptorSkill()
    assert skill.name == "rhythm_sculptor"
    
    # We pass a highly repetitive, non-bursty text to see if it modifies it
    text = "This is a sentence. This is another sentence. This is a third sentence. " \
           "This is a fourth sentence. This is a fifth sentence. This is a sixth sentence. " \
           "This is a seventh sentence. This is an eighth sentence. This is a ninth sentence. " \
           "This is a tenth sentence. Code block: `print(123)`"
    
    result = skill.execute(text, intensity=1.0)
    
    # Code block and numbers inside it must be preserved
    assert "`print(123)`" in result


def test_formatting_sanitizer_skill():
    skill = FormattingSanitizerSkill()
    assert skill.name == "formatting_sanitizer"
    
    text = "Some text — with em dash and some ellipsis... Code: `print(x...y)`"
    result = skill.execute(text)
    
    # Em-dash and ellipsis in plain text must be sanitized
    assert "—" not in result
    assert "ellipsis..." not in result
    # Inline code block must be untouched
    assert "`print(x...y)`" in result


def test_asset_optimizer_skill_missing_file():
    skill = AssetOptimizerSkill()
    with pytest.raises(FileNotFoundError):
        skill.execute("non_existent_file.png")


def test_comment_pruner_skill():
    from quirky.skills.code import CommentPrunerSkill
    skill = CommentPrunerSkill()
    assert skill.name == "comment_pruner"
    
    code = """# Increment count by 1
count += 1
# This is a useful complex fallback mechanism in case of temporary database outage
fallback_port = 8080
# return responses
return responses
"""
    result = skill.execute(code)
    
    # Redundant comments must be stripped
    assert "# Increment count by 1" not in result
    assert "# return responses" not in result
    # Useful comment must be preserved
    assert "# This is a useful complex fallback mechanism" in result

