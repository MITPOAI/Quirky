---
name: comment_pruner
description: >
  Strips redundant single line comments in Python, JavaScript, and TypeScript
  that just restate the code below them. Keeps block comments and docstrings.
  Trigger: source files with wordy, obvious inline comments.
---

# Comment Pruner Skill

Delete comments that add nothing. A comment earns its place by saying why, not by echoing what the next line already says.

## When to use

- A diff or file is littered with narration comments.
- You're tidying `.py`, `.js`, or `.ts` before commit or review.

## Scope

Only single line comments: `#` for Python, `//` for JavaScript and TypeScript. Block comments and docstrings are read as state and passed through untouched.

## When a comment is redundant

For each `#` or `//` line, the pruner finds the next real line of code and tests three conditions. Any one triggers removal.

1. **Overlap.** Significant words in the comment overlap the code line by 50 percent or more. Filler words (`the`, `to`, `for`, `we`, `make`, `sure`, and friends) don't count toward the total.
2. **Trivial verb.** The comment opens with a mechanical verb and shares at least one word with the code.
3. **Pure filler.** After filler removal, no significant word is left.

```text
Trivial verbs that signal narration:
set, return, call, run, print, increment, decrement, assign, check, verify,
import, get, define, initialize, init, start, stop, end, create, delete, remove
```

## Before and after

```text
BAD:
# increment counter
counter += 1

GOOD:
counter += 1
```

```text
BAD:
// set the user name
user.name = name;

GOOD:
user.name = name;
```

## What it keeps (exceptions)

- **Block comments and docstrings.** Anything inside `/* */`, `"""`, or `'''` stays. The pruner tracks these as open state and never edits within them.
- **Section headers and markers.** A comment that starts with `-`, `=`, or `#` reads as a divider and stays.
- **Very short markers.** Under three characters, it's left alone.
- **The why.** A comment that explains a reason, a caveat, an edge case, or links out survives, because the code alone can't carry that.

```text
KEEP (explains the why, not the what):
# off-by-one guard: the API returns 1-based indexes
counter += 1
```

## Boundary

This skill touches comment lines only. Code, strings, and logic are never rewritten; a redundant comment is dropped whole or the line is left as is.
