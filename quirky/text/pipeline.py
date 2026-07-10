import re
import numpy as np

class TextHumanizer:
    AI_REPLACEMENTS = {
        r"\bFurthermore,\b": "Plus,",
        r"\bMoreover,\b": "On top of that,",
        r"\bIn conclusion,\b": "In short,",
        r"\bIn summary,\b": "Basically,",
        r"\bIt is important to note that\b": "Keep in mind that",
        r"\bit is important to note that\b": "honestly,",
        r"\bAs an AI,?\b": "Honestly,",
        r"\bAs an AI language model,?\b": "As far as I can tell,",
        r"\bconsequently\b": "so",
        r"\bConsequently\b": "So",
        r"\butilize\b": "use",
        r"\butilizes\b": "uses",
        r"\bfacilitates optimization\b": "makes it run better",
        r"\bthis approach facilitates optimization\b": "this is how you actually make it run faster",
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
        
        # If the slope indicates an artificial flat or over-steep topology,
        # we perform word substitutions using standard synonyms to sculpt the distribution
        if not (0.75 < gamma < 1.25) and np.random.rand() < intensity:
            # We want to swap synonymous structures
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
                "consequently": ["so", "therefore", "thus"]
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
    def humanize(text: str, intensity: float = 0.5) -> str:
        """
        Humanizes AI-generated text by adjusting perplexity burstiness,
        removing safety tropes/formatting, inserting contractions, and restructuring sentences.
        """
        if not text.strip():
            return text
            
        # First apply Zipf-Mandelbrot Information Theory Sculpting
        processed = TextHumanizer.sculpt_zipf_mandelbrot(text, intensity=intensity)
        
        # 1. Clean AI Cliches & Tropes
        for pattern, replacement in TextHumanizer.AI_REPLACEMENTS.items():
            if np.random.rand() < intensity:
                processed = re.sub(pattern, replacement, processed)
                
        # 2. Insert Contractions
        for pattern, contraction in TextHumanizer.CONTRACTIONS.items():
            if np.random.rand() < intensity:
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
        
        # Clean double spaces and punctuation issues
        output = re.sub(r'\s+', ' ', output)
        output = re.sub(r'\s+([.!?])', r'\1', output)
        
        return output
