"""
Vocal Feature Extraction Module (Acoustic Prosody Analysis)
Psychological State Analysis Using Multimodal AI

Extracts measurable acoustic biomarkers using Librosa and Scipy:
1. Fundamental Frequency (F0 / pitch) and pitch variability (prosodic range)
2. Root Mean Square (RMS) energy and loudness dynamics
3. Speaking rate and pause duration estimates (syllable envelope peak tracking)
4. Spectral features: Centroid, Rolloff, Zero-Crossing Rate, and MFCCs
5. Acoustic Arousal mapping for late multimodal fusion
"""

import logging
import numpy as np
import librosa
from scipy.signal import find_peaks, savgol_filter
from typing import Dict, Any, Tuple

from config import (
    AUDIO_SAMPLE_RATE,
    AUDIO_HOP_LENGTH,
    PITCH_FMIN,
    PITCH_FMAX,
    FRAME_LENGTH
)

logger = logging.getLogger("PsychStateAI.AudioFeatures")


def extract_vocal_features(
    audio: np.ndarray,
    sr: int = AUDIO_SAMPLE_RATE
) -> Dict[str, Any]:
    """
    Extracts comprehensive acoustic and prosodic features from conditioned audio.

    Args:
        audio: 1D numpy array of audio samples (float32).
        sr: Sampling rate (default 16000 Hz).

    Returns:
        dict: Extracted acoustic metrics, time-series arrays for plotting,
              and academic research interpretation.
    """
    duration = len(audio) / float(sr)
    if duration < 0.2 or np.max(np.abs(audio)) < 1e-4:
        return {
            "success": False,
            "error": "Audio recording is too brief or silent.",
            "duration": round(duration, 2)
        }

    # ------------------------------------------------------------------------
    # 1. Fundamental Frequency (F0 / Pitch) Tracking via probabilistic YIN (pYIN)
    # ------------------------------------------------------------------------
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=PITCH_FMIN,
            fmax=PITCH_FMAX,
            sr=sr,
            hop_length=AUDIO_HOP_LENGTH,
            frame_length=FRAME_LENGTH
        )
        pitch_times = librosa.times_like(f0, sr=sr, hop_length=AUDIO_HOP_LENGTH)

        # Voiced frames analysis
        valid_f0 = f0[voiced_flag & ~np.isnan(f0)] if voiced_flag is not None else np.array([])

        if len(valid_f0) > 0:
            f0_mean = float(np.mean(valid_f0))
            f0_median = float(np.median(valid_f0))
            f0_std = float(np.std(valid_f0))
            f0_min = float(np.min(valid_f0))
            f0_max = float(np.max(valid_f0))
            voiced_ratio = float(len(valid_f0) / len(f0))
        else:
            f0_mean, f0_median, f0_std, f0_min, f0_max, voiced_ratio = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    except Exception as e:
        logger.warning("F0 estimation encountered an issue: %s", e)
        f0, pitch_times = np.array([]), np.array([])
        f0_mean, f0_median, f0_std, f0_min, f0_max, voiced_ratio = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # ------------------------------------------------------------------------
    # 2. RMS Energy & Loudness Dynamics
    # ------------------------------------------------------------------------
    rms = librosa.feature.rms(
        y=audio,
        frame_length=FRAME_LENGTH,
        hop_length=AUDIO_HOP_LENGTH
    )[0]
    rms_times = librosa.times_like(rms, sr=sr, hop_length=AUDIO_HOP_LENGTH)

    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))
    rms_max = float(np.max(rms)) if len(rms) > 0 else 0.0
    # Dynamic range in decibels
    dynamic_range_db = float(20 * np.log10((rms_max + 1e-6) / (rms_mean + 1e-6)))

    # ------------------------------------------------------------------------
    # 3. Speaking Rate and Pause Analysis (Syllable Envelope Peak Detection)
    # ------------------------------------------------------------------------
    # Smooth RMS envelope to detect syllable nuclei
    speaking_rate_est = 0.0
    pause_ratio = 0.0
    if len(rms) > 10:
        # Window length for Savitzky-Golay filter
        win_len = min(15, len(rms) if len(rms) % 2 == 1 else len(rms) - 1)
        if win_len >= 5:
            smoothed_rms = savgol_filter(rms, window_length=win_len, polyorder=2)
        else:
            smoothed_rms = rms

        # Minimum distance between syllables ~ 160ms (approx 5 frames at 16000 sr / 512 hop)
        min_distance = max(1, int(0.16 * sr / AUDIO_HOP_LENGTH))
        min_height = float(np.mean(smoothed_rms) * 0.5)

        peaks, _ = find_peaks(smoothed_rms, distance=min_distance, height=min_height)
        estimated_syllables = len(peaks)
        speaking_rate_est = float(estimated_syllables / duration) if duration > 0 else 0.0

        # Silence threshold: frames with RMS < 15% of mean RMS
        silence_thresh = 0.15 * rms_mean
        silent_frames = np.sum(rms < silence_thresh)
        pause_ratio = float(silent_frames / len(rms)) if len(rms) > 0 else 0.0

    # ------------------------------------------------------------------------
    # 4. Spectral Features (Timbre, Brightness, Frication)
    # ------------------------------------------------------------------------
    # Spectral Centroid
    cent = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=AUDIO_HOP_LENGTH)[0]
    cent_mean = float(np.mean(cent))

    # Spectral Rolloff (85% energy threshold)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=AUDIO_HOP_LENGTH, roll_percent=0.85)[0]
    rolloff_mean = float(np.mean(rolloff))

    # Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=AUDIO_HOP_LENGTH)[0]
    zcr_mean = float(np.mean(zcr))

    # 13 MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=AUDIO_HOP_LENGTH)
    mfcc_means = [float(np.mean(mfccs[i])) for i in range(13)]

    # ------------------------------------------------------------------------
    # 5. Acoustic Arousal Score Projection (Affective Computing Standard)
    # ------------------------------------------------------------------------
    # In Russell's model, Acoustic Arousal scales with energy, pitch variability, and tempo
    # Standardize components into [-1.0, 1.0] range
    norm_pitch_std = np.clip((f0_std - 25.0) / 35.0, -1.0, 1.0) if f0_mean > 0 else 0.0
    norm_rms = np.clip((rms_mean - 0.08) / 0.08, -1.0, 1.0)
    norm_rate = np.clip((speaking_rate_est - 3.2) / 1.5, -1.0, 1.0)
    norm_pause = np.clip((0.35 - pause_ratio) / 0.25, -1.0, 1.0)

    # Composite acoustic arousal score
    acoustic_arousal = float(np.clip(
        0.35 * norm_rms + 0.30 * norm_pitch_std + 0.20 * norm_rate + 0.15 * norm_pause,
        -1.0, 1.0
    ))

    # Acoustic valence is inherently ambiguous without lexical content (Scherer 2003)
    # Mildly correlate with pitch variability and harmonic stability, centered around 0.0
    acoustic_valence = float(np.clip(0.3 * norm_pitch_std + 0.2 * norm_pause, -0.6, 0.6))

    # ------------------------------------------------------------------------
    # 6. Scientific Research Observations
    # ------------------------------------------------------------------------
    research_observations = []
    if f0_mean > 0 and f0_std < 18.0:
        research_observations.append(
            f"Reduced fundamental frequency variability (F0 SD: {f0_std:.1f} Hz). "
            "In clinical acoustic literature (Mundt et al. 2007; Cummins et al. 2015), "
            "a constricted pitch range ('monopitch' or prosodic flattening) is investigated "
            "as a potential correlate of psychomotor retardation."
        )
    elif f0_std > 45.0:
        research_observations.append(
            f"High pitch variability (F0 SD: {f0_std:.1f} Hz), indicating expressive, "
            "dynamic prosody and high vocal activation."
        )

    if pause_ratio > 0.45:
        research_observations.append(
            f"Prolonged pause ratio ({pause_ratio*100:.1f}% of recording). "
            "Elevated silence latency often reflects cognitive hesitation, reduced articulation tempo, "
            "or low physiological activation."
        )

    if dynamic_range_db < 10.0:
        research_observations.append(
            f"Restricted loudness dynamic range ({dynamic_range_db:.1f} dB), indicating "
            "monotonic vocal volume."
        )

    return {
        "success": True,
        "duration_sec": round(duration, 2),
        "f0_mean": round(f0_mean, 1),
        "f0_median": round(f0_median, 1),
        "f0_std": round(f0_std, 1),
        "f0_min": round(f0_min, 1),
        "f0_max": round(f0_max, 1),
        "voiced_ratio": round(voiced_ratio, 3),
        "rms_mean": round(rms_mean, 4),
        "rms_std": round(rms_std, 4),
        "dynamic_range_db": round(dynamic_range_db, 1),
        "speaking_rate_syllables_per_sec": round(speaking_rate_est, 2),
        "pause_ratio": round(pause_ratio, 3),
        "spectral_centroid_hz": round(cent_mean, 1),
        "spectral_rolloff_hz": round(rolloff_mean, 1),
        "zero_crossing_rate": round(zcr_mean, 4),
        "mfcc_coefficients": [round(c, 2) for c in mfcc_means],
        "acoustic_arousal": round(acoustic_arousal, 3),
        "acoustic_valence": round(acoustic_valence, 3),
        "research_observations": research_observations,
        # Arrays for plotting
        "pitch_times": pitch_times,
        "f0_series": f0,
        "rms_times": rms_times,
        "rms_series": rms,
        "audio_waveform": audio,
        "sample_rate": sr,
        "scientific_disclaimer": (
            "SCIENTIFIC NOTICE: Acoustic and prosodic metrics vary substantially across individuals, "
            "native languages, microphone hardware, and room acoustics. Vocal features alone MUST NEVER "
            "be interpreted as proof of depression or any psychological condition."
        )
    }
