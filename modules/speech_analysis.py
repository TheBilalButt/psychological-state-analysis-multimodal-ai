"""
Speech Analysis Module
Psychological State Analysis Using Multimodal AI

Implements two distinct scientific analysis paths:
Path A: Speech Recognition (Whisper ASR) -> Automatic Transcription -> NLP Affect Pipeline
Path B: Vocal Feature Analysis (Librosa Acoustic Prosody: F0, RMS, Speaking Rate, Spectrogram)
"""

import time
import logging
import numpy as np
from typing import Dict, Any, Optional

from utils.preprocessing import load_and_preprocess_audio
from utils.model_loader import load_speech_asr_pipeline
from modules.audio_features import extract_vocal_features
from modules.text_analysis import analyze_text

logger = logging.getLogger("PsychStateAI.SpeechAnalysis")


def analyze_speech(
    audio_source: Any,
    transcribe: bool = True,
    extract_prosody: bool = True
) -> Dict[str, Any]:
    """
    Executes the dual-path speech analysis workflow.

    Args:
        audio_source: File path, BytesIO, or audio bytes.
        transcribe: Whether to run Whisper ASR (Path A).
        extract_prosody: Whether to extract acoustic biomarkers (Path B).

    Returns:
        dict: Complete results containing transcript, NLP analysis of transcript,
              acoustic features, quality diagnostics, and execution times.
    """
    overall_start = time.time()

    # Step 1: Decode, verify, and condition audio
    audio, sr, duration, diagnostics = load_and_preprocess_audio(audio_source)

    if diagnostics["is_silent"]:
        return {
            "success": False,
            "error": "The provided audio appears to be completely silent, empty, or unreadable.",
            "diagnostics": diagnostics,
            "duration_sec": duration
        }

    results: Dict[str, Any] = {
        "success": True,
        "duration_sec": duration,
        "diagnostics": diagnostics,
        "path_a_transcription": None,
        "path_a_nlp_result": None,
        "path_b_prosody": None
    }

    # ------------------------------------------------------------------------
    # Path A: Speech Transcription via Whisper ASR + Downstream NLP Analysis
    # ------------------------------------------------------------------------
    if transcribe:
        try:
            asr_start = time.time()
            asr_pipe = load_speech_asr_pipeline()

            # Run Whisper on audio array
            asr_output = asr_pipe(audio)
            transcript_text = asr_output.get("text", "").strip()
            asr_time = round(time.time() - asr_start, 2)

            results["path_a_transcription"] = {
                "transcript": transcript_text,
                "latency_sec": asr_time,
                "model_used": "OpenAI Whisper-tiny (Hugging Face)",
                "status": "Transcribed successfully" if transcript_text else "No speech detected"
            }

            # If speech text was recognized, run downstream NLP pipeline
            if transcript_text:
                nlp_res = analyze_text(transcript_text)
                nlp_res["pipeline_tag"] = "Speech transcription followed by text analysis"
                results["path_a_nlp_result"] = nlp_res
            else:
                results["path_a_nlp_result"] = {
                    "success": False,
                    "note": "Transcription returned empty text; downstream NLP skipped."
                }

        except Exception as e:
            logger.error("Whisper transcription error: %s", e)
            results["path_a_transcription"] = {
                "transcript": "",
                "error": f"Transcription error: {str(e)}",
                "status": "Error"
            }
            results["path_a_nlp_result"] = None

    # ------------------------------------------------------------------------
    # Path B: Vocal Feature Analysis (Librosa Acoustic Prosody)
    # ------------------------------------------------------------------------
    if extract_prosody:
        try:
            prosody_res = extract_vocal_features(audio, sr=sr)
            results["path_b_prosody"] = prosody_res
        except Exception as e:
            logger.error("Vocal feature extraction error: %s", e)
            results["path_b_prosody"] = {
                "success": False,
                "error": f"Feature extraction error: {str(e)}"
            }

    results["total_latency_sec"] = round(time.time() - overall_start, 2)
    return results
