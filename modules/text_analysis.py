"""
Text Analysis Module
Psychological State Analysis Using Multimodal AI

Implements:
1. Deep Transformer Emotion Classification (DistilRoBERTa 7-class Ekman taxonomy)
2. Traditional Baseline (TF-IDF + Logistic Regression)
3. Clinical Linguistic Pattern Extraction (Pennebaker LIWC first-person pronouns,
   absolutist language, negative emotion density)
4. Scientific interpretation distinguishing momentary emotion from clinical indicators.
"""

import time
import logging
import numpy as np
from typing import Dict, Any, Optional

from config import (
    CANONICAL_EMOTIONS,
    EMOTION_VALENCE_AROUSAL
)
from utils.preprocessing import clean_text, extract_linguistic_features
from utils.model_loader import load_text_emotion_pipeline, load_baseline_models

logger = logging.getLogger("PsychStateAI.TextAnalysis")


def analyze_text(text: str) -> Dict[str, Any]:
    """
    Performs comprehensive text analysis on the input text:
    - Preprocessing & linguistic pattern extraction
    - Neural transformer emotion classification (DistilRoBERTa)
    - Baseline TF-IDF + Logistic Regression classification (if trained)
    - Valence-Arousal dimensional affect projection
    - Academic research indicators (self-referential focus, absolutist bias)

    Returns:
        dict: Complete structured analysis results.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return {
            "success": False,
            "error": "Input text is empty or contains only whitespace.",
            "text": ""
        }

    start_time = time.time()

    # 1. Linguistic Pattern Extraction (Pennebaker LIWC inspired)
    ling_features = extract_linguistic_features(cleaned)

    # 2. Transformer Emotion Classification
    pipe = load_text_emotion_pipeline()
    raw_preds = pipe(cleaned)

    # Normalize transformer output into canonical dictionary
    # Format returned by pipeline with top_k=None: [[{'label': 'sadness', 'score': 0.85}, ...]]
    if isinstance(raw_preds, list) and len(raw_preds) > 0 and isinstance(raw_preds[0], list):
        pred_items = raw_preds[0]
    elif isinstance(raw_preds, list) and len(raw_preds) > 0 and isinstance(raw_preds[0], dict):
        pred_items = raw_preds
    else:
        pred_items = []

    emotion_probs = {e: 0.0 for e in CANONICAL_EMOTIONS}
    for item in pred_items:
        lbl = item.get("label", "").lower()
        score = float(item.get("score", 0.0))
        if lbl in emotion_probs:
            emotion_probs[lbl] = round(score, 4)

    # Dominant emotion
    dominant_emotion = max(emotion_probs.items(), key=lambda x: x[1])[0]
    dominant_confidence = emotion_probs[dominant_emotion]

    # 3. Traditional Baseline Model (TF-IDF + Logistic Regression)
    baseline_result = None
    vec, clf = load_baseline_models()
    if vec is not None and clf is not None:
        try:
            X_vec = vec.transform([cleaned])
            base_pred_label = clf.predict(X_vec)[0]
            if hasattr(clf, "predict_proba"):
                base_probs_arr = clf.predict_proba(X_vec)[0]
                base_classes = list(clf.classes_)
                base_probs_dict = {
                    cls_name: round(float(prob), 4)
                    for cls_name, prob in zip(base_classes, base_probs_arr)
                }
            else:
                base_probs_dict = {base_pred_label: 1.0}

            baseline_result = {
                "available": True,
                "model_name": "TF-IDF + Logistic Regression Baseline",
                "dominant_emotion": base_pred_label,
                "confidence": base_probs_dict.get(base_pred_label, 0.0),
                "probabilities": base_probs_dict
            }
        except Exception as e:
            logger.warning("Baseline model inference error: %s", e)
            baseline_result = {"available": False, "error": str(e)}
    else:
        baseline_result = {
            "available": False,
            "note": "Baseline model not yet trained. Train via 'Model Information & Evaluation' page."
        }

    # 4. Compute Continuous Valence & Arousal Coordinates
    # Weighted sum based on predicted probability distribution
    valence = 0.0
    arousal = 0.0
    total_w = sum(emotion_probs.values()) or 1.0

    for emo, p in emotion_probs.items():
        coords = EMOTION_VALENCE_AROUSAL.get(emo, {"valence": 0.0, "arousal": 0.0})
        valence += (p / total_w) * coords["valence"]
        arousal += (p / total_w) * coords["arousal"]

    # 5. Scientific Interpretation & Research Disclaimers
    # Highlighting clinical linguistics research
    research_indicators = []
    if ling_features["fps_pronoun_ratio"] > 8.0:
        research_indicators.append(
            f"Elevated self-referential pronoun density ({ling_features['fps_pronoun_ratio']}%). "
            "Literature (Pennebaker, Rude et al.) associates high 1st-person singular pronoun use "
            "with heightened internal self-focus and depressive rumination."
        )
    if ling_features["absolutist_ratio"] > 3.0:
        research_indicators.append(
            f"Elevated absolutist language ratio ({ling_features['absolutist_ratio']}%). "
            "Cognitive psychology literature (Al-Mosaiwi et al., 2018) links absolutist phrasing "
            "('always', 'never', 'completely') with black-and-white cognitive distortions."
        )
    if ling_features["negative_word_ratio"] > 5.0:
        research_indicators.append(
            f"High negative emotion lexicon density ({ling_features['negative_word_ratio']}%)."
        )

    latency_ms = round((time.time() - start_time) * 1000, 1)

    return {
        "success": True,
        "input_text": text,
        "cleaned_text": cleaned,
        "dominant_emotion": dominant_emotion,
        "confidence": round(dominant_confidence, 4),
        "emotion_probabilities": emotion_probs,
        "valence": round(valence, 3),
        "arousal": round(arousal, 3),
        "linguistic_features": ling_features,
        "research_indicators": research_indicators,
        "baseline_result": baseline_result,
        "inference_latency_ms": latency_ms,
        "model_used": "DistilRoBERTa (j-hartmann/emotion-english-distilroberta-base)",
        "scientific_disclaimer": (
            "NOTE: This prediction reflects momentary lexical emotional tone. "
            "It does NOT evaluate clinical depression or any psychiatric pathology."
        )
    }
