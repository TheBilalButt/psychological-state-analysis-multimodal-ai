"""
Unit tests for multimodal fusion logic, Russell circumplex coordinates, and dissonance detection.
"""

import unittest
from modules.multimodal_fusion import fuse_multimodal_inputs


class TestMultimodalFusion(unittest.TestCase):

    def setUp(self):
        self.text_sad = {
            "success": True,
            "dominant_emotion": "sadness",
            "confidence": 0.88,
            "emotion_probabilities": {
                "sadness": 0.88, "neutral": 0.04, "fear": 0.03,
                "anger": 0.02, "disgust": 0.01, "joy": 0.01, "surprise": 0.01
            },
            "valence": -0.68,
            "arousal": -0.40
        }

        self.speech_flat = {
            "success": True,
            "path_b_prosody": {
                "success": True,
                "f0_mean": 115.0,
                "f0_std": 9.5,
                "acoustic_arousal": -0.45,
                "acoustic_valence": -0.20
            }
        }

        self.face_happy = {
            "success": True,
            "dominant_emotion": "joy",
            "confidence": 0.92,
            "emotion_probabilities": {
                "joy": 0.92, "neutral": 0.03, "surprise": 0.02,
                "sadness": 0.01, "fear": 0.01, "anger": 0.005, "disgust": 0.005
            },
            "valence": 0.75,
            "arousal": 0.52
        }

        self.face_sad = {
            "success": True,
            "dominant_emotion": "sadness",
            "confidence": 0.85,
            "emotion_probabilities": {
                "sadness": 0.85, "neutral": 0.05, "fear": 0.04,
                "anger": 0.02, "disgust": 0.02, "joy": 0.01, "surprise": 0.01
            },
            "valence": -0.65,
            "arousal": -0.38
        }

    def test_fusion_congruent(self):
        # Sad text + Flat prosody + Sad face
        fusion = fuse_multimodal_inputs(self.text_sad, self.speech_flat, self.face_sad)
        self.assertTrue(fusion["success"])
        self.assertEqual(fusion["dominant_emotion"], "sadness")
        self.assertLess(fusion["fused_point"]["valence"], -0.3)
        self.assertIn("Congruence", fusion["congruence"]["status"])

    def test_fusion_affective_dissonance(self):
        # Sad text + Happy smiling face -> Masking
        fusion = fuse_multimodal_inputs(self.text_sad, self.speech_flat, self.face_happy)
        self.assertTrue(fusion["success"])
        self.assertIn("Dissonance", fusion["congruence"]["status"])
        self.assertGreater(fusion["congruence"]["dissonance_score"], 0.5)

    def test_missing_modality_handling(self):
        # Only text supplied
        fusion = fuse_multimodal_inputs(text_result=self.text_sad)
        self.assertTrue(fusion["success"])
        self.assertEqual(fusion["active_modalities"], ["text"])
        self.assertEqual(fusion["dominant_emotion"], "sadness")

        # Empty inputs
        fusion_empty = fuse_multimodal_inputs()
        self.assertFalse(fusion_empty["success"])


if __name__ == "__main__":
    unittest.main()
