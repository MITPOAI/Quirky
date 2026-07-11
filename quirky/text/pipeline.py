import re
import numpy as np

ATTRIBUTION = "Powered by Quirky by MITPO"


class TextHumanizer:
    # NOTE: trope patterns must NOT end with ",\b" -- a comma->space transition is not a
    # regex word boundary, so ",\b" never matches. Anchor on the word, comma optional.
    # Longer phrases are listed before the shorter phrases they contain.
    AI_REPLACEMENTS = {
        r"\bthis approach facilitates optimization\b": "this is how you actually make it run faster",
        r"\bfacilitates optimization\b": "makes it run better",
        r"\bAs an AI language model\b,?": "As far as I can tell,",
        r"\bAs an AI\b,?": "Honestly,",
        r"\bit is important to note that\b": "honestly,",
        r"\bIt is important to note that\b": "Keep in mind that",
        r"\bFurthermore\b,?": "Plus,",
        r"\bMoreover\b,?": "On top of that,",
        r"\bAdditionally\b,?": "Also,",
        r"\badditionally\b,?": "also,",
        r"\bFirstly\b,?": "First,",
        r"\bLastly\b,?": "Finally,",
        r"\bIn conclusion\b,?": "In short,",
        r"\bIn summary\b,?": "Basically,",
        r"\bConsequently\b,?": "So,",
        r"\bconsequently\b": "so",
        r"\butilizes\b": "uses",
        r"\butilize\b": "use",
        r"\bit is essential to\b": "you really need to",
        r"\bIt is essential to\b": "You really need to",
        r"\bin order to\b": "to",
    }
    
    CONTRACTIONS = {
        r"\bdo not\b": "don't",
        r"\bcannot\b": "can't",
        r"\bit is\b": "it's",
        r"\bIt is\b": "It's",
        r"\bshould not\b": "shouldn't",
        r"\bwe are\b": "we're",
        r"\bWe are\b": "We're",
        r"\bthey are\b": "they're",
        r"\bThey are\b": "They're",
        r"\bwould not\b": "wouldn't",
        r"\bI am\b": "I'm",
        r"\bthat is\b": "that's",
        r"\bThat is\b": "That's",
        r"\bthere is\b": "there's",
        r"\bThere is\b": "There's",
        r"\byou are\b": "you're",
        r"\bYou are\b": "You're",
        r"\bdoes not\b": "doesn't",
        r"\bwill not\b": "won't",
    }
    
    CONVERSATIONAL_INJECTORS = [
        "But honestly?",
        "Look,",
        "Wait, here's the thing:",
        "Basically,",
        "To be fair,",
        "Surprisingly,"
    ]

    @staticmethod
    def sculpt_zipf_mandelbrot(text: str, intensity: float = 0.5) -> str:
        """
        Calculates the Zipf rank distribution and sculpts/perturbs the text
        to restore the human-like Zipf-Mandelbrot curve.
        """
        # Tokenize words
        words = re.findall(r'\b\w+\b', text)
        if len(words) < 15:
            return text
            
        # Count frequencies
        from collections import Counter
        counts = Counter([w.lower() for w in words])
        
        # Zipf-Mandelbrot parameter estimates
        freqs = np.array(sorted(counts.values(), reverse=True))
        ranks = np.arange(1, len(freqs) + 1)
        
        # Fit log(freq) = A - gamma * log(rank + q)
        # Using a fixed typical q = 2.7, fit linear model
        q = 2.7
        log_ranks_q = np.log(ranks + q)
        log_freqs = np.log(freqs)
        
        try:
            slope, intercept = np.polyfit(log_ranks_q, log_freqs, 1)
            gamma = -slope
        except Exception:
            gamma = 1.0
        
        # The further gamma sits from the human-like ~1.0 exponent, the more the
        # distribution is sculpted. Zipf-guided: swaps target over-frequent tokens,
        # pushing mass toward rarer synonyms to raise local perplexity variance.
        deviation = abs(gamma - 1.0)
        swap_gate = min(1.0, intensity * (0.4 + deviation))
        if np.random.rand() < swap_gate:
            # We want to swap synonymous structures (lower-frequency alternatives)
            synonyms = {
                "utilize": ["use", "try"],
                "additionally": ["also", "besides", "plus"],
                "provide": ["give", "offer", "show"],
                "frequently": ["often", "commonly", "mostly"],
                "subsequent": ["next", "following", "later"],
                "assistance": ["help", "support", "backing"],
                "ensure": ["make sure", "check", "verify"],
                "foster": ["build", "help", "grow"],
                "delve": ["go deep", "look into", "explore"],
                "consequently": ["so", "therefore", "thus"],
                "numerous": ["many", "lots of", "plenty of"],
                "leverage": ["use", "tap", "lean on"],
                "facilitate": ["help", "ease", "smooth"],
                "demonstrate": ["show", "prove", "point to"],
                "significant": ["big", "major", "real"],
                "essential": ["key", "vital", "core"],
                "however": ["but", "though", "still"],
                "therefore": ["so", "which means", "thus"],
                "optimal": ["best", "ideal", "sharpest"],
                "robust": ["solid", "sturdy", "tough"],
            }
            
            # Walk through words and swap with synonyms with some probability
            processed_words = []
            for w in words:
                w_lower = w.lower()
                if w_lower in synonyms and np.random.rand() < (intensity * 0.7):
                    # Select replacement synonym
                    replacement = np.random.choice(synonyms[w_lower])
                    # Match casing
                    if w.isupper():
                        replacement = replacement.upper()
                    elif w[0].isupper():
                        replacement = replacement.capitalize()
                    processed_words.append(replacement)
                else:
                    processed_words.append(w)
            
            # Reconstruct text matching original word boundaries
            word_idx = 0
            def replace_callback(match):
                nonlocal word_idx
                if word_idx < len(processed_words):
                    res = processed_words[word_idx]
                    word_idx += 1
                    return res
                return match.group(0)
            
            sculpted_text = re.sub(r'\b\w+\b', replace_callback, text)
            return sculpted_text
            
        return text

    @staticmethod
    def _split_sentences(text: str):
        """Split into [(sentence, delimiter), ...] preserving terminal punctuation."""
        parts = re.split(r'([.!?]+)', text)
        sents = []
        i = 0
        while i < len(parts):
            s = parts[i].strip()
            if i + 1 < len(parts) and re.match(r'^[.!?]+$', parts[i + 1]):
                delim = parts[i + 1]
                i += 2
            else:
                delim = "."
                i += 1
            if s:
                sents.append((s, delim))
        return sents

    @staticmethod
    def inject_burstiness(text: str, intensity: float = 0.5, target_cv: float = 0.55) -> str:
        """
        Human sentence lengths are ~lognormal with a high coefficient of variation
        (CV ~ 0.5-0.7); AI text is uniform. If the input CV is below target, fragment
        some long sentences at clause markers and merge some short ones, raising
        burstiness -- the single strongest signal separating human from AI prose.
        """
        sents = TextHumanizer._split_sentences(text)
        if len(sents) < 3:
            return text

        lengths = np.array([len(s.split()) for s, _ in sents], dtype=float)
        cv = lengths.std() / (lengths.mean() + 1e-8)
        if cv >= target_cv:
            return text  # already bursty enough

        out = []
        j = 0
        while j < len(sents):
            s, d = sents[j]
            wc = len(s.split())

            # Fragment a long sentence at a mid clause marker
            if wc > 16 and np.random.rand() < intensity:
                markers = list(re.finditer(r'\s+(and|but|so|because|which|while|although)\s+', s))
                if markers:
                    m = markers[len(markers) // 2]
                    first = s[:m.start()].strip().rstrip(',')
                    second = s[m.end():].strip()
                    if first and second:
                        second = second[0].upper() + second[1:]
                        out.append(f"{first}.")
                        out.append(f"{second}{d}")
                        j += 1
                        continue

            # Merge a short sentence with the next into one longer clause
            if wc < 9 and j + 1 < len(sents) and np.random.rand() < intensity:
                s2, d2 = sents[j + 1]
                joiner = np.random.choice([", and ", ", so ", ", "])
                tail = (s2[0].lower() + s2[1:]) if s2 else s2
                out.append(f"{s.rstrip('.')}{joiner}{tail}{d2}")
                j += 2
                continue

            out.append(f"{s}{d}")
            j += 1

        return " ".join(out)

    @staticmethod
    def diversify_punctuation(text: str, intensity: float = 0.5) -> str:
        """
        Vary punctuation rhythm WITHOUT the em-dash/ellipsis crutch (the em-dash is
        itself a notorious AI tell). Occasionally promote a ", and/but/so" clause to a
        fresh short sentence starting with the connector, the way people talk.
        """
        def _promote(m):
            if np.random.rand() < 0.25 * intensity:
                return ". " + m.group(1).capitalize() + " "
            return m.group(0)
        return re.sub(r',\s+(and|but|so)\s+', _promote, text)

    @staticmethod
    def strip_ai_punctuation(text: str) -> str:
        """
        Final cleanup pass: remove ALL em/en dashes and ellipses, normalize curly
        quotes. Dash between clauses becomes a comma; a spaced dash acting as a full
        stop becomes a period when the next word is capitalized.
        """
        # em/en dash between space-separated clauses
        def _dash(m):
            after = m.group(1)
            if after[:1].isupper():
                return ". " + after
            return ", " + after
        text = re.sub(r'\s+[—–-]{1,2}\s+(\S)', lambda m: _dash(m), text)
        # any survivors (word—word, stray dashes)
        text = text.replace("—", ", ").replace("–", ", ")
        # ellipses (unicode + triple-dot) -> period
        text = re.sub(r'(\.{3,}|…)\s*', '. ', text)
        # curly quotes -> straight
        text = (text.replace("“", '"').replace("”", '"')
                    .replace("‘", "'").replace("’", "'"))
        # tidy doubled punctuation/spaces created by the replacements
        text = re.sub(r'\s+([,.!?])', r'\1', text)
        text = re.sub(r',\s*,', ', ', text)
        text = re.sub(r'\.\s*\.', '. ', text)
        return re.sub(r'\s{2,}', ' ', text).strip()

    @staticmethod
    def humanize(text: str, intensity: float = 0.5) -> str:
        """
        Humanizes AI-generated text by adjusting perplexity burstiness,
        removing safety tropes/formatting, inserting contractions, and restructuring sentences.
        """
        if not text.strip():
            return text
            
        # First apply Zipf-Mandelbrot Information Theory Sculpting
        processed = TextHumanizer.sculpt_zipf_mandelbrot(text, intensity=intensity)
        
        # 1. Clean AI Cliches & Tropes (near-deterministic at high intensity: these
        #    boilerplate transitions are the strongest lexical AI tells)
        cliche_gate = min(1.0, 0.5 + intensity)
        for pattern, replacement in TextHumanizer.AI_REPLACEMENTS.items():
            if np.random.rand() < cliche_gate:
                processed = re.sub(pattern, replacement, processed)

        # 2. Insert Contractions
        for pattern, contraction in TextHumanizer.CONTRACTIONS.items():
            if np.random.rand() < cliche_gate:
                processed = re.sub(pattern, contraction, processed)
                
        # 3. Burstiness and Synclastic Restructuring
        # Split into sentences keeping delimiters
        sentences = re.split(r'([.!?]+)', processed)
        
        reconstructed = []
        i = 0
        while i < len(sentences):
            sentence = sentences[i].strip()
            # If it's a delimiter, append it to the previous sentence
            if i + 1 < len(sentences) and re.match(r'^[.!?]+$', sentences[i+1]):
                delimiter = sentences[i+1]
                i += 2
            else:
                delimiter = "."
                i += 1
                
            if not sentence:
                continue
                
            words = sentence.split()
            
            # Restructuring long, compound sentences if intensity is high
            if len(words) > 18 and intensity > 0.4 and np.random.rand() < 0.5:
                # Look for clause splitters like ", and" or ", but" or "because"
                split_points = [m.start() for m in re.finditer(r'(, and|, but|because|since|while)', sentence)]
                if split_points:
                    # Split at the first split point
                    idx = split_points[0]
                    first_part = sentence[:idx].strip()
                    second_part = sentence[idx:].strip()
                    
                    # Clean up split markers
                    second_part = re.sub(r'^,\s*(and|but)\s+', '', second_part)
                    second_part = re.sub(r'^(because|since|while)\s+', '', second_part)
                    
                    # Capitalize first letter of second part
                    if second_part:
                        second_part = second_part[0].upper() + second_part[1:]
                        
                    # Inject a conversational connector to start the second sentence
                    connector = np.random.choice(TextHumanizer.CONVERSATIONAL_INJECTORS)
                    
                    reconstructed.append(f"{first_part}{delimiter}")
                    reconstructed.append(f"{connector} {second_part}.")
                    continue
                    
            # Occasionally inject a short conversational pause/question to vary lengths
            if len(words) > 12 and intensity > 0.6 and np.random.rand() < 0.3:
                connector = np.random.choice(TextHumanizer.CONVERSATIONAL_INJECTORS)
                # Split sentence
                reconstructed.append(f"{connector}")
                
            reconstructed.append(f"{sentence}{delimiter}")
            
        # Re-assemble text
        output = " ".join(reconstructed)

        # 4. Burstiness targeting + human punctuation rhythm (highest-signal steps)
        output = TextHumanizer.inject_burstiness(output, intensity=intensity)
        output = TextHumanizer.diversify_punctuation(output, intensity=intensity)

        # Clean double spaces and punctuation issues
        output = re.sub(r'\s+', ' ', output)
        output = re.sub(r'\s+([.!?])', r'\1', output)

        # 5. Guarantee dash-free, ellipsis-free, straight-quoted output
        output = TextHumanizer.strip_ai_punctuation(output)

        return output
