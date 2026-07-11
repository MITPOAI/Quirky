# Quirky Skills and Agents

Quirky has four core skills designed to restore natural characteristics to synthetic text and media.

## The Skills List

### 1. Cliché Pruner (`cliche_pruner`)
- **Purpose**: Surgically replaces AI boilerplate transitions, clichés, and inserts contractions.
- **Rules**:
  - Replaces words like `Furthermore`, `Moreover`, `utilize`, and `facilitate` with natural alternatives.
  - Inserts contractions (e.g. "do not" to "don't").
  - Preserves exact numbers and code blocks.
  - **Verification**: Uses Jaccard and Cosine similarity checks plus number multiset verification.

### 2. Rhythm Sculptor (`rhythm_sculptor`)
- **Purpose**: Sculpts sentence length variance and punctuation rhythm to fit Zipf-Mandelbrot properties.
- **Rules**:
  - Fragments compound sentences to introduce burstiness.
  - Merges overly short sentences to vary lengths.
  - Promotes clauses to fresh sentences (e.g. ", and" to ". And").
- **Verification**: Protects code blocks, commands, and numbers using sentinel masks.

### 3. Formatting Sanitizer (`formatting_sanitizer`)
- **Purpose**: Cleans up formatting indicators such as em-dashes and ellipses.
- **Rules**:
  - Strips em/en dashes and replaces them with commas or periods.
  - Replaces triple-dot ellipses with a single period.
  - Normalizes curly quotes to straight quotes.
- **Verification**: Masks code blocks, commands, and numbers using sentinels.

### 4. Asset Optimizer (`asset_optimizer`)
- **Purpose**: Automatically handles media file optimizations.
- **Rules**:
  - Detects file type (image, audio, video).
  - Routes images to Poisson-Gaussian noise and CLAHE.
  - Routes audio to glottal tilt drift and shimmer perturbations.
  - Routes video to frame-by-frame drift pipelines.

---

## Comparison: Quirky vs. Caveman Skills

### Caveman Skills
In the Caveman repository, skills are defined as **declarative rules in Markdown files** with YAML frontmatter.
- **How they work**: The LLM ingests these prompt templates directly to modify its behavior for specific tasks (like code review comments or text compression).
- **Pros**: Easy to write and modify without coding.
- **Cons**: Requires LLM calls, costs tokens, lacks deterministic safety checks, and can be slow.

### Quirky Skills
In Quirky, skills are **executable Python modules** guarded by mathematical verification.
- **How they work**: They run local algorithms (such as regex substitutions, Zipf curve fittings, and signal-processing filters). They are exposed as MCP tools with semantic descriptions so that any agent can call them.
- **Pros**: Run in milliseconds, 100% local, no API keys, and have strict guards to protect code blocks and numbers.
- **Cons**: Require Python coding to implement new features.
