import os
import re
from typing import Optional, Tuple

import numpy as np

from quirky.detector.lexicons import AI_CLICHES, HEDGES, FILLERS, STOPWORDS
from quirky.detector.calibrate import _EXPANDED
from quirky.plugins.dl import CACHE_DIR, MODEL_REGISTRY
from quirky.text.pipeline import TextHumanizer

# Compile set of all words involved in the styling lexicons and humanizer rules
STYLE_WORDS = set()
for word_list in (AI_CLICHES, HEDGES, FILLERS, _EXPANDED):
    for phrase in word_list:
        STYLE_WORDS.update(re.findall(r"\b\w+\b", phrase.lower()))

for pat, rep in TextHumanizer.AI_REPLACEMENTS.items():
    STYLE_WORDS.update(re.findall(r"\b\w+\b", pat.lower()))
    STYLE_WORDS.update(re.findall(r"\b\w+\b", rep.lower()))

for pat, rep in TextHumanizer.CONTRACTIONS.items():
    STYLE_WORDS.update(re.findall(r"\b\w+\b", pat.lower()))
    STYLE_WORDS.update(re.findall(r"\b\w+\b", rep.lower()))

_ONNX_SESSION = None
_TOKENIZER = None


def get_cached_model_path(name: str) -> Optional[str]:
    """Search the cache directory for the model filename without downloading."""
    if name not in MODEL_REGISTRY:
        return None
    entry = MODEL_REGISTRY[name]
    filename = os.path.basename(entry["file"])
    if not os.path.exists(CACHE_DIR):
        return None
    for root, _, files in os.walk(CACHE_DIR):
        if filename in files:
            return os.path.join(root, filename)
    return None


def get_jaccard_tokens(s: str) -> set[str]:
    """Tokenize and filter text into content words (excluding stopwords and style words, preserving numbers)."""
    tokens = re.findall(r"\b\w+\b", s.lower())
    filtered = []
    for t in tokens:
        if t in STOPWORDS and not any(c.isdigit() for c in t):
            continue
        if t in STYLE_WORDS:
            continue
        filtered.append(t)
    return set(filtered)


def jaccard_similarity(a: str, b: str) -> float:
    """Calculate the Jaccard similarity between filtered word sets."""
    set_a = get_jaccard_tokens(a)
    set_b = get_jaccard_tokens(b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def get_onnx_embeddings(text: str, model_path: str) -> Optional[np.ndarray]:
    """Get the sentence embedding using the cached ONNX model and transformers tokenizer."""
    global _ONNX_SESSION, _TOKENIZER
    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer
        
        if _ONNX_SESSION is None:
            _ONNX_SESSION = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        if _TOKENIZER is None:
            model_dir = os.path.dirname(model_path)
            # Try to load tokenizer from the model dir or Xenova cache if available
            _TOKENIZER = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            
        inputs = _TOKENIZER(text, padding=True, truncation=True, max_length=512, return_tensors="np")
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        if "token_type_ids" in inputs:
            ort_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)
            
        outputs = _ONNX_SESSION.run(None, ort_inputs)
        token_embeddings = outputs[0]
        input_mask_expanded = np.expand_dims(inputs["attention_mask"], -1).astype(float)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        embedding = sum_embeddings / sum_mask
        
        emb = embedding[0]
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb
    except Exception:
        return None


def similarity(a: str, b: str) -> Tuple[float, str]:
    """
    Calculate similarity between two strings.
    Returns (score, method) where method is either 'cosine' or 'jaccard'.
    """
    if os.environ.get("QUIRKY_EMBED_BACKEND") == "off":
        return jaccard_similarity(a, b), "jaccard"
        
    try:
        # Check if onnxruntime is importable
        import onnxruntime  # noqa: F401
        model_path = get_cached_model_path("minilm_embed")
        if model_path is not None:
            emb_a = get_onnx_embeddings(a, model_path)
            emb_b = get_onnx_embeddings(b, model_path)
            if emb_a is not None and emb_b is not None:
                cosine_sim = float(np.dot(emb_a, emb_b))
                return cosine_sim, "cosine"
    except Exception:
        pass
        
    return jaccard_similarity(a, b), "jaccard"
