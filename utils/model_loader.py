"""
Centralized Model Loader with Resource Caching
Psychological State Analysis Using Multimodal AI

Loads transformers pipelines, vision classifiers, ASR models, and baseline
scikit-learn models. Caches in memory for rapid multi-page inference.
"""

import sys
import logging
import joblib
from typing import Optional, Any
from transformers import pipeline

from config import (
    TEXT_EMOTION_MODEL,
    SPEECH_ASR_MODEL,
    FACE_EMOTION_MODEL,
    BASELINE_VECTORIZER_PATH,
    BASELINE_CLASSIFIER_PATH
)

logger = logging.getLogger("PsychStateAI.ModelLoader")

# Try importing Streamlit cache_resource; if running outside Streamlit, use a simple decorator
try:
    import streamlit as st
    cache_resource = st.cache_resource
except Exception:
    def cache_resource(func):
        cache = {}
        def wrapper(*args, **kwargs):
            key = (func.__name__, str(args), str(kwargs))
            if key not in cache:
                cache[key] = func(*args, **kwargs)
            return cache[key]
        return wrapper


@cache_resource
def load_text_emotion_pipeline():
    """
    Loads Hugging Face RoBERTa emotion classification pipeline.
    Output: 7 canonical emotion classes (anger, disgust, fear, joy, neutral, sadness, surprise).
    """
    logger.info("Initializing Text Emotion Pipeline (%s)...", TEXT_EMOTION_MODEL)
    try:
        pipe = pipeline(
            "text-classification",
            model=TEXT_EMOTION_MODEL,
            top_k=None,
            device="cpu"
        )
        return pipe
    except Exception as e:
        logger.error("Failed to load Text Emotion model: %s", e)
        raise RuntimeError(f"Text emotion model could not be loaded: {e}")


@cache_resource
def load_speech_asr_pipeline():
    """
    Loads Hugging Face Whisper-tiny Automatic Speech Recognition pipeline.
    """
    logger.info("Initializing Whisper ASR Pipeline (%s)...", SPEECH_ASR_MODEL)
    try:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=SPEECH_ASR_MODEL,
            device="cpu"
        )
        return pipe
    except Exception as e:
        logger.error("Failed to load Speech ASR model: %s", e)
        raise RuntimeError(f"Whisper ASR model could not be loaded: {e}")


@cache_resource
def load_face_emotion_pipeline():
    """
    Loads Hugging Face Vision Transformer (ViT) facial emotion classification pipeline.
    """
    logger.info("Initializing Face Emotion ViT Pipeline (%s)...", FACE_EMOTION_MODEL)
    try:
        pipe = pipeline(
            "image-classification",
            model=FACE_EMOTION_MODEL,
            top_k=None,
            device="cpu"
        )
        return pipe
    except Exception as e:
        logger.error("Failed to load Facial Emotion model: %s", e)
        raise RuntimeError(f"Facial emotion model could not be loaded: {e}")


@cache_resource
def load_baseline_models():
    """
    Loads the trained TF-IDF vectorizer and Logistic Regression baseline model.
    Returns (vectorizer, classifier) or (None, None) if not yet trained.
    """
    if BASELINE_VECTORIZER_PATH.exists() and BASELINE_CLASSIFIER_PATH.exists():
        try:
            vec = joblib.load(BASELINE_VECTORIZER_PATH)
            clf = joblib.load(BASELINE_CLASSIFIER_PATH)
            return vec, clf
        except Exception as e:
            logger.warning("Could not load baseline models: %s", e)
            return None, None
    return None, None
