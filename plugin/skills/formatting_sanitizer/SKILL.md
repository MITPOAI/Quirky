---
name: formatting_sanitizer
description: >
  Strips synthetic punctuation tells: em and en dashes, ellipses, curly quotes,
  and runs of spaces, while masking code and numbers.
  Trigger: prose with spaced dashes, trailing dots, or smart quotes.
---

# Formatting Sanitizer Skill

Clean punctuation to match how people actually type. The em dash and the trailing ellipsis are two of the loudest AI tells, so they go first.

## When to use

- Text carries spaced em or en dashes between clauses.
- You see `...` or the unicode `…` used for a pause.
- Curly quotes crept in from a paste, or spaces doubled up.

## Rules

### Dashes

A spaced em or en dash becomes a comma, or a period when the next word is capitalized (it was acting as a full stop). A stray dash glued between words becomes a comma plus space.

```text
BAD:  It works - Trust me on this.
GOOD: It works. Trust me on this.

BAD:  We shipped it - finally.
GOOD: We shipped it, finally.
```

### Ellipses

Triple dots and the unicode ellipsis collapse to a single period plus space. The word that follows keeps its original case.

```text
BAD:  Latency dropped... noticeably.
GOOD: Latency dropped. noticeably.
```

### Quotes and spacing

Curly quotes normalize to straight quotes. Doubled punctuation and runs of two or more spaces collapse to one.

```text
BAD:  She said "done".
GOOD: She said "done".
```

## Guards (hard boundaries)

- **Sentinel masking.** Fenced blocks, indented blocks, inline `code`, `$`/`>` command lines, and numbers are masked before the cleanup and restored after. A dash or dot inside code is never rewritten.
- **Verify or revert.** After restore, the skill re extracts code regions and numbers. If any sentinel is missing or duplicated, or the number multiset or code list changed, it returns the original text unchanged.
- **Numbers frozen.** Ranges and decimals like `3-5` or `0.55` sit inside masked spans, so their dashes and dots survive intact.

## Contextual exceptions

- **Real ranges in prose.** A hyphen joining a true range should stay a hyphen. When it lives outside a masked number span, review before flattening it to a comma.
- **Quoted source text.** Preserve punctuation inside a direct quotation when the exact wording matters.
