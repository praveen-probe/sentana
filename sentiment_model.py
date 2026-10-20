"""Reusable sentiment-analysis module.

Model: cardiffnlp/twitter-roberta-base-sentiment-latest
(RoBERTa transformer for social-media sentiment; NOT original BERT.)
Inference only - no training here.
"""

from __future__ import annotations

import functools
import re
from typing import Dict, Optional, Tuple

MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_CHARS = 2000
MODEL_MAX_TOKENS = 512
VALID_SENTIMENTS = ("POSITIVE", "NEGATIVE", "NEUTRAL")
# Model card documents: 0=negative, 1=neutral, 2=positive.
CANONICAL_ID_ORDER = ("negative", "neutral", "positive")


def validate_text(text: object, max_chars: int = MAX_CHARS) -> Tuple[str, Optional[str]]:
    """Validate raw input. Returns (cleaned, error); error None when valid."""
    if not isinstance(text, str):
        return "", "Please enter some text to analyze."
    cleaned = text.strip()
    if not cleaned:
        return "", "Please enter some text - empty input cannot be analyzed."
    if len(cleaned) > max_chars:
        return "", (
            f"Input is too long ({len(cleaned)} chars). "
            f"Keep it under {max_chars} characters."
        )
    return cleaned, None


def normalize_label(raw_label: str, label_index: Optional[int] = None) -> str:
    """Map a raw model label to POSITIVE / NEGATIVE / NEUTRAL."""
    name = (raw_label or "").strip().lower()
    direct = {"positive": "POSITIVE", "pos": "POSITIVE",
              "negative": "NEGATIVE", "neg": "NEGATIVE",
              "neutral": "NEUTRAL", "neu": "NEUTRAL"}
    if name in direct:
        return direct[name]
    if name.startswith("label_") and label_index is not None:
        try:
            return CANONICAL_ID_ORDER[label_index].upper()
        except IndexError:
            pass
    raise ValueError(f"Unrecognized model label: {raw_label!r}")


def normalize_id2label(id2label: dict) -> Dict[int, str]:
    """Convert HF config id2label to {id: SENTIMENT}; verifies label set."""
    if not id2label:
        raise ValueError("Model config has no id2label mapping.")
    mapped: Dict[int, str] = {}
    for key, raw in id2label.items():
        idx = int(key)
        mapped[idx] = normalize_label(str(raw), label_index=idx)
    if set(mapped.values()) != {"POSITIVE", "NEGATIVE", "NEUTRAL"}:
        raise ValueError(f"Unexpected label set: {mapped}")
    return mapped

# __PART2__
def build_result(predicted: str, positive: float, neutral: float, negative: float) -> dict:
    """Build structured result from per-class probabilities."""
    for value in (positive, neutral, negative):
        if not isinstance(value, float):
            raise TypeError("Probability scores must be floats.")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Probability out of range: {value}")
    total = positive + neutral + negative
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Probabilities must sum to ~1.0, got {total:.4f}")
    if predicted not in VALID_SENTIMENTS:
        raise ValueError(f"Unknown sentiment: {predicted!r}")
    confidence = {"POSITIVE": positive, "NEGATIVE": negative, "NEUTRAL": neutral}[predicted]
    return {
        "sentiment": predicted,
        "confidence": round(float(confidence), 4),
        "scores": {
            "positive": round(float(positive), 4),
            "neutral": round(float(neutral), 4),
            "negative": round(float(negative), 4),
        },
    }


def softmax(logits) -> list:
    """Numerically stable softmax over a 1-D sequence (pure Python)."""
    import math
    if len(logits) == 0:
        raise ValueError("Empty logits.")
    peak = max(logits)
    exps = [math.exp(v - peak) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


@functools.lru_cache(maxsize=1)
def _load_model_cached(model_id: str = MODEL_ID):
    """Load tokenizer + model once per process. Raises RuntimeError on failure."""
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Package 'transformers' missing. Run: pip install -r requirements.txt"
        ) from exc
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Package 'torch' missing. Run: pip install -r requirements.txt"
        ) from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download/load model ({model_id}). Need internet on "
            f"first run; cached afterwards. Details: {type(exc).__name__}: {exc}"
        ) from exc
    model.eval()
    label_map = normalize_id2label(dict(model.config.id2label))
    return tokenizer, model, label_map

# __PART3__
def analyze_sentiment(text: str, model_id: str = MODEL_ID) -> dict:
    """Run transformer inference. Raises ValueError / RuntimeError."""
    cleaned, error = validate_text(text)
    if error:
        raise ValueError(error)
    tokenizer, model, label_map = _load_model_cached(model_id)
    try:
        import torch
        inputs = tokenizer(cleaned, return_tensors="pt",
                           truncation=True, max_length=MODEL_MAX_TOKENS)
        with torch.inference_mode():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits[0], dim=-1).tolist()
    except Exception as exc:
        raise RuntimeError(
            f"Inference failed ({type(exc).__name__}). Try shorter text."
        ) from exc
    ordered = sorted(label_map.items())
    by_sentiment = {sent: probs[i] for i, sent in ordered}
    best = max(by_sentiment, key=by_sentiment.get)
    return build_result(best, positive=float(by_sentiment["POSITIVE"]),
                        neutral=float(by_sentiment["NEUTRAL"]),
                        negative=float(by_sentiment["NEGATIVE"]))


_POS_WORDS = re.compile(
    r"\b(love|loves|loved|amazing|awesome|excellent|wonderful|great|good|"
    r"fantastic|best|happy|glad|beautiful|perfect|brilliant)\b", re.IGNORECASE)
_NEG_WORDS = re.compile(
    r"\b(hate|hates|hated|terrible|awful|horrible|worst|bad|disappoint|"
    r"poor|sad|angry|annoying|useless|waste)\b", re.IGNORECASE)
_POS_EMOJI = set("❤️😍😂🔥😊👍🎉💕😄🙂👏💯🥰😎✨")
_NEG_EMOJI = set("😡👎😢😞💔😠🤬👿😭😤🙁☹️")


def keyword_fallback_sentiment(text: str) -> dict:
    """Keyword heuristic FALLBACK ONLY - always labeled, never silent."""
    cleaned, error = validate_text(text)
    if error:
        raise ValueError(error)
    pos = len(_POS_WORDS.findall(cleaned)) + sum(cleaned.count(e) for e in _POS_EMOJI)
    neg = len(_NEG_WORDS.findall(cleaned)) + sum(cleaned.count(e) for e in _NEG_EMOJI)
    if pos > neg:
        scores = {"positive": 0.70, "neutral": 0.20, "negative": 0.10}
        sent = "POSITIVE"
    elif neg > pos:
        scores = {"positive": 0.10, "neutral": 0.20, "negative": 0.70}
        sent = "NEGATIVE"
    else:
        scores = {"positive": 0.25, "neutral": 0.50, "negative": 0.25}
        sent = "NEUTRAL"
    result = build_result(sent, **scores)
    result["fallback"] = True
    result["model"] = "keyword-fallback"
    return result
