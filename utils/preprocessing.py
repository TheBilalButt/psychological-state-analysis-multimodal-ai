"""
Data Preprocessing Utilities
Psychological State Analysis Using Multimodal AI

Handles text cleaning & linguistic metrics, audio decoding & conditioning,
and OpenCV face detection & bounding box cropping.
"""

import re
import io
import cv2
import numpy as np
import soundfile as sf
import librosa
from PIL import Image
from typing import Tuple, List, Dict, Any, Optional

from config import (
    AUDIO_SAMPLE_RATE,
    FIRST_PERSON_SINGULAR_PRONOUNS,
    ABSOLUTIST_WORDS,
    NEGATIVE_EMOTION_WORDS
)


# ============================================================================
# 1. TEXT PREPROCESSING & LINGUISTIC PATTERN EXTRACTION
# ============================================================================

def clean_text(raw_text: str) -> str:
    """
    Cleans input text by normalizing whitespace, standardizing apostrophes,
    and removing non-printable characters while preserving punctuation.
    """
    if not raw_text or not isinstance(raw_text, str):
        return ""
    # Standardize whitespace and quotes
    text = raw_text.replace("\r\n", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r'[\u2018\u2019]', "'", text)
    text = re.sub(r'[\u201c\u201d]', '"', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_linguistic_features(text: str) -> Dict[str, Any]:
    """
    Extracts cognitive and linguistic markers based on clinical psychology literature:
    - First-person singular pronoun frequency (Pennebaker & Lay, 2002; Rude et al., 2004)
    - Absolutist language frequency (Al-Mosaiwi & Johnstone, 2018)
    - Negative emotion lexicon density
    - Lexical diversity (Type-Token Ratio)
    - Sentence structure statistics
    """
    cleaned = clean_text(text)
    if not cleaned:
        return {
            "char_count": 0,
            "word_count": 0,
            "sentence_count": 0,
            "avg_sentence_length": 0.0,
            "lexical_diversity": 0.0,
            "fps_pronoun_count": 0,
            "fps_pronoun_ratio": 0.0,
            "absolutist_count": 0,
            "absolutist_ratio": 0.0,
            "negative_word_count": 0,
            "negative_word_ratio": 0.0,
            "found_fps_tokens": [],
            "found_absolutist_tokens": [],
            "found_negative_tokens": []
        }

    # Sentence and word tokenization using regex
    sentences = [s.strip() for s in re.split(r'[.!?]+', cleaned) if s.strip()]
    sentence_count = max(len(sentences), 1)

    # Word tokens
    words = re.findall(r"\b[A-Za-z'-]+\b", cleaned.lower())
    total_words = len(words)
    if total_words == 0:
        total_words = 1  # prevent div by zero

    # Lexical Diversity (TTR)
    unique_words = set(words)
    ttr = len(unique_words) / total_words

    # Pennebaker First-Person Singular Pronouns (I, me, my, myself, mine)
    fps_tokens = [w for w in words if w in FIRST_PERSON_SINGULAR_PRONOUNS]
    fps_count = len(fps_tokens)
    fps_ratio = (fps_count / total_words) * 100.0

    # Absolutist Words (always, never, completely, etc.)
    abs_tokens = [w for w in words if w in ABSOLUTIST_WORDS]
    abs_count = len(abs_tokens)
    abs_ratio = (abs_count / total_words) * 100.0

    # Negative Emotion Lexicon
    neg_tokens = [w for w in words if w in NEGATIVE_EMOTION_WORDS]
    neg_count = len(neg_tokens)
    neg_ratio = (neg_count / total_words) * 100.0

    return {
        "char_count": len(cleaned),
        "word_count": len(words),
        "sentence_count": sentence_count,
        "avg_sentence_length": round(len(words) / sentence_count, 1),
        "lexical_diversity": round(ttr, 3),
        "fps_pronoun_count": fps_count,
        "fps_pronoun_ratio": round(fps_ratio, 2),
        "absolutist_count": abs_count,
        "absolutist_ratio": round(abs_ratio, 2),
        "negative_word_count": neg_count,
        "negative_word_ratio": round(neg_ratio, 2),
        "found_fps_tokens": list(set(fps_tokens)),
        "found_absolutist_tokens": list(set(abs_tokens)),
        "found_negative_tokens": list(set(neg_tokens))
    }


# ============================================================================
# 2. AUDIO DECODING & CONDITIONING
# ============================================================================

def load_and_preprocess_audio(
    audio_source: Any,
    target_sr: int = AUDIO_SAMPLE_RATE,
    top_db_trim: int = 25
) -> Tuple[np.ndarray, int, float, Dict[str, Any]]:
    """
    Decodes audio from a file path, BytesIO, or raw bytes into mono float32 at target_sr.
    Performs silence verification, clipping detection, and leading/trailing trimming.

    Returns:
        audio (np.ndarray): 1D float32 audio waveform normalized to [-1, 1].
        sr (int): Sampling rate (target_sr).
        duration (float): Duration in seconds.
        diagnostics (dict): Audio quality metadata (is_silent, peak_amplitude, is_clipped).
    """
    diagnostics = {
        "is_silent": False,
        "peak_amplitude": 0.0,
        "is_clipped": False,
        "original_sr": target_sr,
        "original_duration": 0.0,
        "trimmed_duration": 0.0
    }

    try:
        # Load audio buffer
        if isinstance(audio_source, (str, io.BytesIO, bytes)):
            if isinstance(audio_source, bytes):
                audio_source = io.BytesIO(audio_source)
            
            # Use soundfile or librosa
            try:
                audio, orig_sr = sf.read(audio_source, dtype="float32")
            except Exception:
                # Fallback to librosa
                if hasattr(audio_source, "seek"):
                    audio_source.seek(0)
                audio, orig_sr = librosa.load(audio_source, sr=None, mono=False)
        else:
            raise ValueError("Unsupported audio input type.")

        # Convert stereo to mono
        if audio.ndim > 1:
            if audio.shape[0] < audio.shape[1]:  # channels first
                audio = np.mean(audio, axis=0)
            else:
                audio = np.mean(audio, axis=1)

        diagnostics["original_sr"] = orig_sr
        orig_dur = len(audio) / float(orig_sr) if orig_sr > 0 else 0.0
        diagnostics["original_duration"] = round(orig_dur, 2)

        # Check for silence or near-zero amplitude
        peak_amp = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
        diagnostics["peak_amplitude"] = round(peak_amp, 4)

        if peak_amp < 1e-4 or len(audio) < 100:
            diagnostics["is_silent"] = True
            return np.zeros(target_sr, dtype=np.float32), target_sr, 1.0, diagnostics

        # Check clipping (significant samples near 1.0)
        clipped_ratio = np.mean(np.abs(audio) >= 0.99)
        diagnostics["is_clipped"] = bool(clipped_ratio > 0.02)

        # Resample if needed
        if orig_sr != target_sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

        # Trim leading and trailing silence
        trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db_trim)
        if len(trimmed_audio) >= (target_sr * 0.2):  # at least 200ms
            audio = trimmed_audio

        duration = len(audio) / float(target_sr)
        diagnostics["trimmed_duration"] = round(duration, 2)

        # Peak normalization to prevent distortion
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95

        return audio.astype(np.float32), target_sr, duration, diagnostics

    except Exception as e:
        diagnostics["error"] = str(e)
        diagnostics["is_silent"] = True
        return np.zeros(target_sr, dtype=np.float32), target_sr, 0.0, diagnostics


# ============================================================================
# 3. FACE DETECTION & BOUNDING BOX EXTRACTION
# ============================================================================

# Load Haar cascade once
_HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_FACE_CASCADE = cv2.CascadeClassifier(_HAAR_PATH)

def detect_faces(
    image_input: Any,
    margin_ratio: float = 0.15
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Detects faces in an image using OpenCV Haar Cascade.
    Extracts face bounding boxes with an adjustable margin for optimal FER recognition.

    Args:
        image_input: PIL Image, file path, bytes, or numpy BGR/RGB array.
        margin_ratio: Margin to expand around the detected face bounding box.

    Returns:
        detected_faces (list of dict): Each dict contains:
            - 'box': (x, y, w, h)
            - 'crop_pil': PIL Image cropped to face
            - 'crop_bgr': numpy array cropped to face
            - 'area': int (w * h)
            - 'is_primary': bool (True for largest face)
        annotated_image (np.ndarray): RGB image with bounding boxes drawn.
    """
    # Convert input to numpy RGB and BGR
    if isinstance(image_input, Image.Image):
        rgb_img = np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            rgb_img = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.shape[2] == 4:
            rgb_img = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
        elif image_input.shape[2] == 3:
            rgb_img = image_input.copy()
        else:
            raise ValueError("Invalid numpy image shape.")
    elif isinstance(image_input, (str, io.BytesIO, bytes)):
        if isinstance(image_input, (io.BytesIO, bytes)):
            if isinstance(image_input, io.BytesIO):
                image_input = image_input.getvalue()
            nparr = np.frombuffer(image_input, np.uint8)
            bgr_raw = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            bgr_raw = cv2.imread(image_input)
        if bgr_raw is None:
            return [], np.zeros((224, 224, 3), dtype=np.uint8)
        rgb_img = cv2.cvtColor(bgr_raw, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported image type: {type(image_input)}")

    img_h, img_w = rgb_img.shape[:2]
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    # Detect faces
    faces = _FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    detected_faces = []
    annotated = rgb_img.copy()

    if len(faces) == 0:
        return detected_faces, annotated

    # Sort faces by bounding box area (largest first)
    faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

    for idx, (x, y, w, h) in enumerate(faces_sorted):
        is_primary = (idx == 0)

        # Expand box with margin
        margin_x = int(w * margin_ratio)
        margin_y = int(h * margin_ratio)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(img_w, x + w + margin_x)
        y2 = min(img_h, y + h + margin_y)

        face_crop_rgb = rgb_img[y1:y2, x1:x2]
        face_crop_pil = Image.fromarray(face_crop_rgb)

        detected_faces.append({
            "index": idx + 1,
            "box": (int(x), int(y), int(w), int(h)),
            "crop_box": (int(x1), int(y1), int(x2), int(y2)),
            "crop_pil": face_crop_pil,
            "area": int(w * h),
            "is_primary": is_primary
        })

        # Draw bounding box on annotated image
        # Primary face gets vibrant cyan/blue (#2563EB), others get muted slate
        color = (37, 99, 235) if is_primary else (148, 163, 184)
        thickness = 3 if is_primary else 2
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)

        label = f"Face #{idx + 1}" + (" (Primary)" if is_primary else "")
        cv2.putText(
            annotated, label,
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, color, 2, cv2.LINE_AA
        )

    return detected_faces, annotated
