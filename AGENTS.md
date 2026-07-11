<!-- quirky:begin v1 -->
# Quirky Agent Guidelines

Cut filler, keep prose edits concise.
Banned clichés: furthermore, moreover, in conclusion, leverage, utilize, facilitate, robust, seamless, cutting-edge, state-of-the-art.
No em/en dashes or ellipses in final outputs.
Contractions are preferred (don't, it's, etc.).
Fix slop - do not pad.
Never alter code, terminal commands, or numbers during prose edits.

Before finalizing any `.md` or `.txt` file, run the `quirky_score_text` tool.
If there are any red spans, run `quirky_fix_text`.
For long documents, run `quirky_tighten_text`.
For media assets, run `quirky_detect_media` or `quirky_humanize_media`.
<!-- quirky:end -->