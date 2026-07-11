---
name: cliche_pruner
description: >
  Surgically prunes AI boilerplate transitions, safety tropes, hedges, and
  fillers, and folds in contractions, without touching code or numbers.
  Trigger: prose that reads synthetic, padded, or stiff.
---

# Cliche Pruner Skill

Swap AI writing tells for plain, human phrasing. Edit only what's flagged. Leave everything else byte for byte.

## When to use

- A draft leans on stock transitions or hedges.
- Text feels padded, formal, or contraction free.
- You're staging prose for commit or review and want the tells gone.

## What it changes

Four passes run in order on each flagged sentence: cliche swaps, contractions, hedge stripping, filler stripping.

### 1. Cliche and trope swaps

Exact map the fixer applies (case insensitive, comma optional):

```text
Furthermore,               -> Plus,
Moreover,                  -> On top of that,
Additionally,              -> Also,
Firstly,                   -> First,
Lastly,                    -> Finally,
In conclusion,             -> In short,
In summary,                -> Basically,
Consequently,              -> So,
utilize / utilizes         -> use / uses
in order to                -> to
it is essential to         -> you really need to
it is important to note    -> honestly, / Keep in mind that
As an AI language model,    -> As far as I can tell,
As an AI,                  -> Honestly,
```

### 2. Contractions (always prefer)

```text
do not -> don't        cannot -> can't       it is -> it's
should not -> shouldn't we are -> we're       they are -> they're
would not -> wouldn't   I am -> I'm           that is -> that's
there is -> there's     you are -> you're      does not -> doesn't
will not -> won't
```

### 3. Hedges to strip

```text
arguably, potentially, perhaps, possibly, somewhat, relatively, fairly,
to some extent, generally speaking, it seems that, it appears that,
tend to, in most cases, for the most part, could potentially
```

### 4. Fillers to strip

```text
basically, essentially, actually, really, very, just, in fact,
as previously mentioned, needless to say, simply put, in other words,
due to the fact that, at this point in time, first and foremost,
last but not least
```

## Before and after

```text
BAD:  Furthermore, it is important to note that we utilize this in order to scale.
GOOD: Plus, honestly, we use this to scale.

BAD:  It is essential to remember that we do not modify code, and it is a safe default.
GOOD: You really need to remember that we don't modify code, and it's a safe default.
```

## Guards (hard boundaries)

- **Numbers frozen.** The digit multiset before and after must match. Any drift rejects the edit.
- **Similarity gate.** The rewrite must stay close to the source: Jaccard at or above `0.50`, or cosine at or above `0.75` when embeddings are available. Below that, the original stands.
- **Code untouched.** Fenced blocks, inline `code`, indented blocks, and `$`/`>` command lines are masked out and restored verbatim.
- **Scope.** Only sentence spans the scorer marks `red` or `amber` get rewritten. Green prose and headings are left alone.

## Contextual exceptions

- **Security and legal text.** Never reword warnings, disclaimers, or quoted terms. A hedge there can be load bearing, so keep it.
- **Onboarding and first run copy.** A warmer tone is often deliberate. Apply a light touch, or skip.
- **Proper nouns and identifiers.** Product names, API names, and flags stay exact even when they collide with a swap word.
