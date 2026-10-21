"""
Unit tests for text analysis module (RoBERTa pipeline, baseline comparison, and linguistic extraction).
"""

import unittest
from modules.text_analysis import analyze_text


class TestTextAnalysis(unittest.TestCase):

    def test_analyze_text_valid(self):
        text = "I feel so grateful and happy about my new university achievement."
        res = analyze_text(text)
        self.assertTrue(res["success"])
        self.assertEqual(res["dominant_emotion"], "joy")
        self.assertGreater(res["confidence"], 0.5)
        self.assertIn("valence", res)
        self.assertIn("arousal", res)
        self.assertGreater(res["valence"], 0.0)
        # Check sum of probabilities
        prob_sum = sum(res["emotion_probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=2)

    def test_analyze_text_empty(self):
        res = analyze_text("")
        self.assertFalse(res["success"])
        self.assertIn("error", res)

    def test_analyze_text_depressive_markers(self):
        text = "I am completely exhausted and I feel hopeless. My life has no purpose."
        res = analyze_text(text)
        self.assertTrue(res["success"])
        self.assertIn(res["dominant_emotion"], ["sadness", "fear"])
        self.assertLess(res["valence"], 0.0)
        # Verify research indicators generated
        self.assertGreater(len(res["research_indicators"]), 0)


if __name__ == "__main__":
    unittest.main()
