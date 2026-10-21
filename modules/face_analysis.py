"""
Facial Expression Recognition (FER) Module
Psychological State Analysis Using Multimodal AI

Implements:
1. OpenCV Face Detection & Multi-Face Handling
2. Vision Transformer (ViT) Expression Classification
3. Canonical Taxonomy Normalization & Valence-Arousal Projection
4. Academic Disclaimers on Facial Masking & Affective Expression Limits
"""

import time
import logging
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional

from config import (
    CANONICAL_EMOTIONS,
    FACE_LABEL_MAPPING,
    EMOTION_VALENCE_AROUSAL
)
from utils.preprocessing import detect_faces
from utils.model_loader import load_face_emotion_pipeline

logger = logging.getLogger("PsychStateAI.FaceAnalysis")


def analyze_facial_expressions(
    image_input: Any,
    selected_face_idx: int = 0
) -> Dict[str, Any]:
    """
    Detects faces in image/webcam frame and analyzes facial expressions.

    Args:
        image_input: PIL Image, numpy array, bytes, or image file path.
        selected_face_idx: 0-indexed index of face to analyze in detail.

    Returns:
        dict: Complete face detection coordinates, emotion distributions,
              dominant expression, and research notes.
    """
    start_time = time.time()

    # Step 1: Detect faces using OpenCV Haar Cascade
    detected_faces, annotated_image = detect_faces(image_input)

    if not detected_faces:
        return {
            "success": False,
            "face_count": 0,
            "error": "No human faces detected in the provided image/video frame.",
            "annotated_image": annotated_image,
            "latency_ms": round((time.time() - start_time) * 1000, 1)
        }

    total_faces = len(detected_faces)
    # Ensure selected index is within bounds
    target_idx = max(0, min(selected_face_idx, total_faces - 1))
    target_face = detected_faces[target_idx]

    # Step 2: Run Vision Transformer Pipeline on cropped face
    vit_pipe = load_face_emotion_pipeline()
    crop_pil = target_face["crop_pil"]

    raw_preds = vit_pipe(crop_pil)

    # Normalize emotion labels to canonical taxonomy
    emotion_probs = {e: 0.0 for e in CANONICAL_EMOTIONS}

    for item in raw_preds:
        raw_label = item.get("label", "").lower()
        score = float(item.get("score", 0.0))
        # Map label if needed (e.g., happy -> joy, sad -> sadness)
        canon_label = FACE_LABEL_MAPPING.get(raw_label, raw_label)
        if canon_label in emotion_probs:
            emotion_probs[canon_label] = round(score, 4)

    # Re-normalize sum to 1.0
    total_p = sum(emotion_probs.values())
    if total_p > 0:
        emotion_probs = {k: round(v / total_p, 4) for k, v in emotion_probs.items()}

    dominant_emotion = max(emotion_probs.items(), key=lambda x: x[1])[0]
    dominant_confidence = emotion_probs[dominant_emotion]

    # Step 3: Compute continuous Valence & Arousal Coordinates
    valence = 0.0
    arousal = 0.0
    for emo, p in emotion_probs.items():
        coords = EMOTION_VALENCE_AROUSAL.get(emo, {"valence": 0.0, "arousal": 0.0})
        valence += p * coords["valence"]
        arousal += p * coords["arousal"]

    # Step 4: Multi-face metadata
    faces_summary = []
    for f in detected_faces:
        faces_summary.append({
            "index": f["index"],
            "box": f["box"],
            "area": f["area"],
            "is_primary": f["is_primary"],
            "is_selected": (f["index"] - 1 == target_idx)
        })

    latency_ms = round((time.time() - start_time) * 1000, 1)

    return {
        "success": True,
        "face_count": total_faces,
        "selected_face_index": target_idx + 1,
        "selected_face_crop": crop_pil,
        "annotated_image": annotated_image,
        "all_faces_meta": faces_summary,
        "dominant_emotion": dominant_emotion,
        "confidence": round(dominant_confidence, 4),
        "emotion_probabilities": emotion_probs,
        "valence": round(valence, 3),
        "arousal": round(arousal, 3),
        "model_used": "ViT Facial Emotions (dima806/facial_emotions_image_detection)",
        "latency_ms": latency_ms,
        "scientific_disclaimer": (
            "SCIENTIFIC NOTICE: Visible facial expressions reflect surface muscle movement "
            "(Action Units). They CANNOT diagnose internal affective disorders or verify authentic "
            "psychological state, as individuals frequently display social smiling or masking."
        )
    }
