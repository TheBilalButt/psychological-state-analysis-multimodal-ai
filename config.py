"""
Application Configuration and Constants
Psychological State Analysis Using Multimodal AI
Academic Research Prototype - University Project
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SAMPLE_AUDIO_DIR = DATA_DIR / "sample_audio"
SAMPLE_IMAGES_DIR = DATA_DIR / "sample_images"
BENCHMARK_DIR = DATA_DIR / "benchmark"

# Hugging Face Model Repositories (Lightweight for local Windows CPU inference)
TEXT_EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
SPEECH_ASR_MODEL = "openai/whisper-tiny"
FACE_EMOTION_MODEL = "dima806/facial_emotions_image_detection"

# Baseline Model Paths
BASELINE_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
BASELINE_CLASSIFIER_PATH = MODELS_DIR / "logistic_regression_baseline.joblib"

# Canonical Emotion Taxonomies (Ekman 6 Basic Emotions + Neutral)
CANONICAL_EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise"
]

# Modality Alias Mappings to Canonical Taxonomy
FACE_LABEL_MAPPING = {
    "happy": "joy",
    "sad": "sadness",
    "angry": "anger",
    "fear": "fear",
    "disgust": "disgust",
    "surprise": "surprise",
    "neutral": "neutral"
}

# Russell's Circumplex Model of Affect (Valence, Arousal coordinates in [-1.0, 1.0])
# Based on James Russell (1980) & Posner et al. (2005) affective computing literature
EMOTION_VALENCE_AROUSAL = {
    "sadness":  {"valence": -0.75, "arousal": -0.45, "description": "Low Pleasure, Low Activation (Depressed/Flat affect)"},
    "fear":     {"valence": -0.65, "arousal":  0.65, "description": "Negative Pleasure, High Activation (Anxious/Apprehensive)"},
    "anger":    {"valence": -0.60, "arousal":  0.75, "description": "Negative Pleasure, High Activation (Agitation/Frustration)"},
    "disgust":  {"valence": -0.55, "arousal":  0.30, "description": "Negative Pleasure, Moderate Activation (Aversive/Distress)"},
    "neutral":  {"valence":  0.00, "arousal":  0.00, "description": "Neutral Pleasure, Baseline Activation (Balanced/Steady)"},
    "surprise": {"valence":  0.20, "arousal":  0.70, "description": "Mild Pleasure, High Activation (Startled/Alert)"},
    "joy":      {"valence":  0.80, "arousal":  0.55, "description": "High Pleasure, Moderate-High Activation (Positive/Elevated)"}
}

# Acoustic Feature Parameters (Librosa)
AUDIO_SAMPLE_RATE = 16000
AUDIO_HOP_LENGTH = 512
PITCH_FMIN = 65.0    # C2 (covers deep male voice)
PITCH_FMAX = 500.0   # B4 (covers high female/child voice)
FRAME_LENGTH = 2048

# Default Fusion Weights (Decision-Level Late Fusion)
DEFAULT_FUSION_WEIGHTS = {
    "text": 0.40,
    "speech_prosody": 0.25,
    "facial_expression": 0.35
}

# Linguistic Markers (Pennebaker LIWC Inspired)
FIRST_PERSON_SINGULAR_PRONOUNS = {
    "i", "me", "my", "myself", "mine"
}

ABSOLUTIST_WORDS = {
    "always", "never", "completely", "totally", "constant", "definitely",
    "entirely", "forever", "nothing", "everything", "all", "none", "must"
}

NEGATIVE_EMOTION_WORDS = {
    "sad", "exhausted", "hopeless", "tired", "lonely", "worthless", "pain",
    "hurt", "depressed", "miserable", "crying", "anxious", "scared", "fear",
    "worried", "awful", "terrible", "empty", "isolated", "failure"
}

# Mandatory Academic & Ethical Disclaimers
DISCLAIMER_TEXT = (
    "ACADEMIC RESEARCH PROTOTYPE DISCLAIMER:\n"
    "This software is an educational and university research prototype designed solely to demonstrate "
    "multimodal machine learning methodologies (NLP, speech processing, and computer vision). "
    "It is NOT a medical device, diagnostic instrument, or clinical decision support tool. "
    "It does NOT diagnose depression, anxiety, or any mental health disorder. "
    "If you or someone you know is experiencing psychological distress, please seek consultation from a "
    "licensed medical professional or contact a national crisis lifeline."
)

PRIVACY_STATEMENT = (
    "PRIVACY GUARANTEE: In compliance with university ethical guidelines, all inference is performed "
    "locally in memory. No user audio, images, or personal text are persisted to external cloud servers "
    "or unmonitored storage without explicit operator consent."
)
