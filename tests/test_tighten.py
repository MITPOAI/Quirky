from quirky.fix.text import tighten


def test_tighten_masking_integrity():
    text = (
        "Furthermore, we must note that: ```python\n"
        "# in order to optimize\n"
        "print(123)\n"
        "``` and some command line:\n"
        "$ utilize command 45.6\n"
        "And inline code `utilize` works. It is important to note that this is very good."
    )
    
    res = tighten(text)
    
    # 1. Fenced block preserved
    assert "```python\n# in order to optimize\nprint(123)\n```" in res
    
    # 2. Command line preserved
    assert "$ utilize command 45.6" in res
    
    # 3. Inline code preserved
    assert "`utilize`" in res
    
    # 4. Fillers/hedges removed outside protected blocks
    assert "very" not in res
    assert "It is important to note that" not in res
    assert "Furthermore" not in res
    assert "this is good." in res


def test_collision_resilience():
    # Input already containing a candidate sentinel
    text = "Sentinel candidate \x00QK_0_\x00 is here. In order to fix it, do this."
    res = tighten(text)
    assert "\x00QK_0_\x00" in res
    assert "to fix it" in res


def test_double_collision_fallback():
    # If both candidate and alternate sentinels collide, it should safely return original
    text = "Both \x00QK_0_\x00 and \x00QKALT_0_\x00 collide. In order to optimize."
    res = tighten(text)
    assert res == text


def test_idempotence():
    text = "In order to streamline, it is important to note that we really want to optimize."
    first = tighten(text)
    second = tighten(first)
    assert first == second
