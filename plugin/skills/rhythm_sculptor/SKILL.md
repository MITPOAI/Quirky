---
name: rhythm_sculptor
description: >
  Restores human sentence rhythm: raises burstiness, sculpts the Zipf word
  curve, and varies punctuation, while masking code and numbers.
  Trigger: prose where every sentence runs the same length or shape.
---

# Rhythm Sculptor Skill

Human prose has burst: short sentences next to long ones. AI prose is flat and even. This skill breaks the flatness.

## When to use

- Sentences march at a uniform length.
- The text feels metronomic or over structured.
- One or two words repeat far past their natural rate.

## What it does

Three steps run on the masked text.

### 1. Burstiness targeting

Human sentence lengths vary with a coefficient of variation around `0.5` to `0.7`. AI text sits well below that. The step measures CV and, when it's under the `0.55` target, lifts it:

- **Fragment** a sentence over 16 words at a mid clause marker: `and`, `but`, `so`, `because`, `which`, `while`, `although`.
- **Merge** a sentence under 9 words into the next with `, and`, `, so`, or `, `.

```text
BAD (flat, low CV):
The model runs fast. The model uses little memory. It needs no GPU. It fits on a CPU.

GOOD (bursty):
The model runs fast, and it uses little memory. It needs no GPU. Fits on a CPU.
```

### 2. Zipf-Mandelbrot sculpting

The step fits the word frequency curve and estimates its exponent. Human text sits near `1.0`. The further off it drifts, the more the step pushes over used tokens toward rarer synonyms, which raises local perplexity variance.

```text
utilize     -> use / try
additionally-> also / besides / plus
provide     -> give / offer / show
ensure      -> make sure / check / verify
leverage    -> use / tap / lean on
facilitate  -> help / ease / smooth
significant -> big / major / real
however     -> but / though / still
therefore   -> so / which means / thus
robust      -> solid / sturdy / tough
```

### 3. Punctuation variety

Some `, and` / `, but` / `, so` joins get promoted to a fresh short sentence starting with the connector, the way people actually talk. No em dash or ellipsis crutch: those are AI tells the sculptor never adds.

```text
BAD:  It compiled, and the tests passed, so we shipped.
GOOD: It compiled. And the tests passed. So we shipped.
```

## Intensity

`intensity` runs `0.0` to `1.0` (default `0.5`). Higher means more fragments, merges, and swaps. The moves stay probabilistic, so rerunning varies the result.

## Guards (hard boundaries)

- **Sentinel masking.** Fenced blocks, indented blocks, inline `code`, `$`/`>` command lines, and numbers are replaced with sentinels before sculpting and restored after.
- **Verify or revert.** After restore, the skill re extracts every code region and number. If any sentinel is missing or duplicated, or the number multiset or code list differs, it returns the original text unchanged.
- **Numbers never move.** A digit inside a masked region is frozen; the sculptor only reshapes plain prose around it.
