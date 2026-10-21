"""
Multimodal Fusion Module (Decision-Level Late Fusion & Dissonance Analysis)
Psychological State Analysis Using Multimodal AI

Implements:
1. Taxonomy-Aligned Decision-Level Late Fusion
2. Continuous 2D Valence-Arousal Affective Space Projection
3. Modality Congruence & Affective Dissonance (Masking) Detection
4. Direct Text-Only vs Multimodal Comparative Analytics
5. Dynamic Reweighting with Graceful Missing Modality Handling
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from config import (
    CANONICAL_EMOTIONS,
    DEFAULT_FUSION_WEIGHTS,
    EMOTION_VALENCE_AROUSAL
)

logger = logging.getLogger("PsychStateAI.MultimodalFusion")


def fuse_multimodal_inputs(
    text_result: Optional[Dict[str, Any]] = None,
    speech_result: Optional[Dict[str, Any]] = None,
    face_result: Optional[Dict[str, Any]] = None,
    custom_weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Executes late fusion across available modalities.

    Args:
        text_result: Output dictionary from modules.text_analysis.analyze_text.
        speech_result: Output dictionary from modules.speech_analysis.analyze_speech.
        face_result: Output dictionary from modules.face_analysis.analyze_facial_expressions.
        custom_weights: Optional dictionary overriding default modality weights.

    Returns:
        dict: Complete multimodal fusion report with comparison, dissonance metrics,
              and academic research discussion.
    """
    weights = dict(DEFAULT_FUSION_WEIGHTS)
    if custom_weights:
        weights.update(custom_weights)

    # ------------------------------------------------------------------------
    # 1. Modality Availability & Verification
    # ------------------------------------------------------------------------
    active_modalities = []
    missing_modalities = []

    has_text = text_result is not None and text_result.get("success", False)
    if has_text:
        active_modalities.append("text")
    else:
        missing_modalities.append("text")

    has_face = face_result is not None and face_result.get("success", False)
    if has_face:
        active_modalities.append("facial_expression")
    else:
        missing_modalities.append("facial_expression")

    has_prosody = (
        speech_result is not None
        and speech_result.get("success", False)
        and speech_result.get("path_b_prosody", {}) is not None
        and speech_result.get("path_b_prosody", {}).get("success", False)
    )
    if has_prosody:
        active_modalities.append("speech_prosody")
    else:
        missing_modalities.append("speech_prosody")

    # If speech transcription text was generated, note if it complements text
    has_transcript = (
        speech_result is not None
        and speech_result.get("path_a_transcription", {}) is not None
        and bool(speech_result.get("path_a_transcription", {}).get("transcript", ""))
    )

    if len(active_modalities) == 0:
        return {
            "success": False,
            "error": "No valid modality outputs were provided for multimodal fusion.",
            "active_modalities": [],
            "missing_modalities": ["text", "facial_expression", "speech_prosody"]
        }

    # ------------------------------------------------------------------------
    # 2. Probability Distribution Alignment & Decision-Level Late Fusion
    # ------------------------------------------------------------------------
    # Collect discrete emotion vectors from text and face
    distribution_weights = {}
    distributions_to_fuse = []

    if has_text:
        t_probs = text_result.get("emotion_probabilities", {})
        t_conf = text_result.get("confidence", 0.5)
        # Weight scales with modality default weight and model confidence
        w_t = weights.get("text", 0.4) * max(0.2, t_conf)
        distribution_weights["text"] = w_t
        distributions_to_fuse.append((w_t, t_probs))

    if has_face:
        f_probs = face_result.get("emotion_probabilities", {})
        f_conf = face_result.get("confidence", 0.5)
        w_f = weights.get("facial_expression", 0.35) * max(0.2, f_conf)
        distribution_weights["facial_expression"] = w_f
        distributions_to_fuse.append((w_f, f_probs))

    # Compute fused discrete emotion probabilities
    fused_probabilities = {e: 0.0 for e in CANONICAL_EMOTIONS}
    total_dist_weight = sum(w for w, _ in distributions_to_fuse)

    if total_dist_weight > 0:
        for w, dist in distributions_to_fuse:
            for e in CANONICAL_EMOTIONS:
                fused_probabilities[e] += (w / total_dist_weight) * dist.get(e, 0.0)

        # Normalize sum to 1.0
        norm_sum = sum(fused_probabilities.values()) or 1.0
        fused_probabilities = {k: round(v / norm_sum, 4) for k, v in fused_probabilities.items()}
    elif has_prosody:
        # If ONLY audio prosody is available, infer discrete affect from acoustic arousal/valence
        prosody_data = speech_result["path_b_prosody"]
        a_aro = prosody_data.get("acoustic_arousal", 0.0)
        a_val = prosody_data.get("acoustic_valence", 0.0)
        # Approximate discrete distribution via Euclidean distance to anchors
        distances = {}
        for emo, coord in EMOTION_VALENCE_AROUSAL.items():
            dist = np.sqrt((a_val - coord["valence"])**2 + (a_aro - coord["arousal"])**2)
            distances[emo] = 1.0 / (dist + 0.1)
        sum_sim = sum(distances.values())
        fused_probabilities = {k: round(v / sum_sim, 4) for k, v in distances.items()}

    dominant_fused_emotion = max(fused_probabilities.items(), key=lambda x: x[1])[0]
    fused_confidence = fused_probabilities[dominant_fused_emotion]

    # ------------------------------------------------------------------------
    # 3. Continuous Valence-Arousal Affective Space Projection
    # ------------------------------------------------------------------------
    modality_points = {}
    va_weights = {}

    if has_text:
        modality_points["text"] = {
            "valence": text_result.get("valence", 0.0),
            "arousal": text_result.get("arousal", 0.0)
        }
        va_weights["text"] = weights.get("text", 0.40)

    if has_face:
        modality_points["facial_expression"] = {
            "valence": face_result.get("valence", 0.0),
            "arousal": face_result.get("arousal", 0.0)
        }
        va_weights["facial_expression"] = weights.get("facial_expression", 0.35)

    if has_prosody:
        prosody_data = speech_result["path_b_prosody"]
        modality_points["speech_prosody"] = {
            "valence": prosody_data.get("acoustic_valence", 0.0),
            "arousal": prosody_data.get("acoustic_arousal", 0.0)
        }
        va_weights["speech_prosody"] = weights.get("speech_prosody", 0.25)

    # Compute fused continuous point
    sum_va_w = sum(va_weights.values()) or 1.0
    fused_valence = sum((va_weights[m] / sum_va_w) * modality_points[m]["valence"] for m in modality_points)
    fused_arousal = sum((va_weights[m] / sum_va_w) * modality_points[m]["arousal"] for m in modality_points)

    fused_point = {
        "valence": round(float(fused_valence), 3),
        "arousal": round(float(fused_arousal), 3)
    }

    # ------------------------------------------------------------------------
    # 4. Modality Congruence & Affective Dissonance Analysis
    # ------------------------------------------------------------------------
    congruence_info = analyze_congruence(
        has_text, text_result,
        has_face, face_result,
        has_prosody, speech_result
    )

    # ------------------------------------------------------------------------
    # 5. Direct Comparative Analysis: Text-Only vs Multimodal
    # ------------------------------------------------------------------------
    comparison_notes = []
    if has_text:
        text_dom = text_result.get("dominant_emotion", "unknown")
        text_conf = text_result.get("confidence", 0.0)
        text_val = text_result.get("valence", 0.0)
        text_aro = text_result.get("arousal", 0.0)

        if text_dom != dominant_fused_emotion:
            comparison_notes.append(
                f"MODALITY SHIFT: Text-only model indicated '{text_dom.upper()}' ({text_conf*100:.1f}%), "
                f"whereas Multimodal Integration updated the dominant state to '{dominant_fused_emotion.upper()}' "
                f"({fused_confidence*100:.1f}%). This occurs because non-verbal modalities "
                f"(facial expression and/or vocal prosody) altered the probability balance."
            )
        else:
            comparison_notes.append(
                f"CORROBORATION: Both text-only and multimodal models agree on '{dominant_fused_emotion.upper()}'. "
                f"Non-verbal signals reinforced the textual sentiment, resulting in a cohesive affective reading."
            )

        # Valence/Arousal Delta
        val_delta = fused_point["valence"] - text_val
        aro_delta = fused_point["arousal"] - text_aro
        if abs(val_delta) > 0.25 or abs(aro_delta) > 0.25:
            comparison_notes.append(
                f"AFFECTIVE DISPLACEMENT: Multimodal fusion shifted the emotional coordinate in Russell's "
                f"Circumplex by Delta-Valence={val_delta:+.2f} and Delta-Arousal={aro_delta:+.2f}. "
                "This demonstrates how non-verbal cues refine semantic meaning beyond lexical keywords alone."
            )
    else:
        comparison_notes.append(
            "Text modality was not provided. Multimodal output is driven entirely by non-verbal channels."
        )

    return {
        "success": True,
        "active_modalities": active_modalities,
        "missing_modalities": missing_modalities,
        "has_all_modalities": (len(active_modalities) == 3),
        "dominant_emotion": dominant_fused_emotion,
        "confidence": round(fused_confidence, 4),
        "emotion_probabilities": fused_probabilities,
        "fused_point": fused_point,
        "modality_points": modality_points,
        "congruence": congruence_info,
        "comparison_notes": comparison_notes,
        "modality_weights_applied": {
            m: round(va_weights.get(m, 0.0) / sum_va_w, 3) for m in active_modalities
        },
        "scientific_disclaimer": (
            "SCIENTIFIC NOTICE: This late fusion demonstrates mathematical combination of multimodal signals "
            "for research and demonstration purposes. It does NOT represent a clinically validated diagnostic system."
        )
    }


def analyze_congruence(
    has_text: bool, text_result: Optional[Dict],
    has_face: bool, face_result: Optional[Dict],
    has_prosody: bool, speech_result: Optional[Dict]
) -> Dict[str, Any]:
    """
    Evaluates whether non-verbal modalities confirm or contradict lexical sentiment.
    Detects affective dissonance (e.g. smiling face with sad text, or flat prosody with agitated text).
    """
    findings = []
    congruence_level = "Single Modality"
    dissonance_score = 0.0  # 0.0 = perfectly congruent, 1.0 = maximal dissonance

    if has_text and has_face:
        t_dom = text_result.get("dominant_emotion", "")
        f_dom = face_result.get("dominant_emotion", "")
        t_val = text_result.get("valence", 0.0)
        f_val = face_result.get("valence", 0.0)

        # Distance in Valence
        val_diff = abs(t_val - f_val)

        if (t_dom in ["sadness", "fear", "anger"] and f_dom in ["joy"]) or (t_val < -0.3 and f_val > 0.4):
            congruence_level = "High Dissonance (Affective Masking Detected)"
            dissonance_score = 0.85
            findings.append(
                "AFFECTIVE DISSONANCE: Discrepancy between negative text sentiment and positive/smiling "
                "facial expression. In clinical psychology (Ekman 2003, Rottenberg 2005), this often reflects "
                "social masking ('smiling depression' or facade behavior), where outward facial affect masks "
                "internal dysphoria."
            )
        elif (t_dom in ["joy"] and f_dom in ["sadness", "anger", "fear"]) or (t_val > 0.4 and f_val < -0.3):
            congruence_level = "High Dissonance (Incongruent Expressivity)"
            dissonance_score = 0.80
            findings.append(
                "INCONGRUENT EXPRESSIVITY: Positive lexical text paired with negative facial affect. "
                "May indicate sarcasm, masked frustration, or conflicting communicative intent."
            )
        elif t_dom == f_dom:
            congruence_level = "High Congruence"
            dissonance_score = 0.10
            findings.append(
                f"STRONG CONGRUENCE: Text sentiment and visible facial expression consistently express '{t_dom.upper()}'."
            )
        else:
            congruence_level = "Moderate Congruence"
            dissonance_score = round(min(1.0, val_diff), 2)
            findings.append(
                f"PARTIAL ALIGNMENT: Text displays '{t_dom}' while facial expression indicates '{f_dom}'."
            )

    if has_prosody and (has_text or has_face):
        prosody = speech_result["path_b_prosody"]
        aro = prosody.get("acoustic_arousal", 0.0)
        f0_std = prosody.get("f0_std", 30.0)

        if has_text and text_result.get("dominant_emotion") in ["anger", "fear", "surprise"] and aro < -0.4:
            findings.append(
                "PROSODIC DAMPENING: High-arousal text paired with blunted/low vocal acoustic energy."
            )
        elif has_text and text_result.get("dominant_emotion") == "sadness" and f0_std < 18.0:
            findings.append(
                "ACOUSTIC FLAT AFFECT: Vocal prosodic flattening (constricted F0 range) aligns with "
                "depressive/sad text sentiment, reinforcing low physiological activation."
            )

    if not findings:
        findings.append("Single modality active. Provide additional modalities to evaluate multimodal congruence.")

    return {
        "status": congruence_level,
        "dissonance_score": round(dissonance_score, 2),
        "observations": findings
    }
