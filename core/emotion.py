"""Text emotion classification via DistilRoBERTa.

Lazy-loads j-hartmann/emotion-english-distilroberta-base on first call.
Returns valence, arousal, discrete class, and full score distribution.
"""

import sys

# Singleton pipeline (loaded once per process)
_pipeline = None

# Class → (valence, arousal) mapping
_EMOTION_VA = {
    "anger":    (-0.6, 0.8),
    "disgust":  (-0.5, 0.5),
    "fear":     (-0.7, 0.7),
    "joy":      ( 0.8, 0.6),
    "neutral":  ( 0.0, 0.2),
    "sadness":  (-0.7, 0.2),
    "surprise": ( 0.3, 0.7),
}

_NEUTRAL_RESULT = {
    "valence": 0.0, "arousal": 0.2, "emotion": "neutral",
    "scores": {k: (1.0 if k == "neutral" else 0.0) for k in _EMOTION_VA},
}


def _get_pipeline():
    """Lazy-load the emotion classification pipeline."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    try:
        from transformers import pipeline as hf_pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        _pipeline = hf_pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            device=device,
            truncation=True,
        )
        return _pipeline
    except Exception as e:
        print(f"[emotion] failed to load model: {e}", file=sys.stderr)
        return None


def _scores_to_result(results: list[dict]) -> dict:
    """Convert pipeline output to {valence, arousal, emotion, scores}."""
    top = max(results, key=lambda r: r["score"])
    valence = sum(r["score"] * _EMOTION_VA.get(r["label"], (0, 0))[0] for r in results)
    arousal = sum(r["score"] * _EMOTION_VA.get(r["label"], (0, 0))[1] for r in results)
    return {
        "valence": round(valence, 3),
        "arousal": round(arousal, 3),
        "emotion": top["label"],
        "scores": {r["label"]: round(r["score"], 3) for r in results},
    }


def classify_emotion(text: str) -> dict:
    """Classify a single text. Returns {valence, arousal, emotion, scores}."""
    pipe = _get_pipeline()
    if pipe is None:
        return dict(_NEUTRAL_RESULT)
    try:
        results = pipe(text[:512])[0]
        return _scores_to_result(results)
    except Exception:
        return dict(_NEUTRAL_RESULT)


def classify_batch(texts: list[str]) -> list[dict]:
    """Batch classify for ingest performance."""
    pipe = _get_pipeline()
    if pipe is None:
        return [dict(_NEUTRAL_RESULT) for _ in texts]
    try:
        truncated = [t[:512] for t in texts]
        all_results = pipe(truncated, batch_size=32)
        return [_scores_to_result(r) for r in all_results]
    except Exception as e:
        print(f"[emotion] batch error: {e}", file=sys.stderr)
        return [dict(_NEUTRAL_RESULT) for _ in texts]
