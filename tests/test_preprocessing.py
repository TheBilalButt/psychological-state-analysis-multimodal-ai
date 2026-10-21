"""
Unit tests for preprocessing utilities: text cleaning, LIWC markers, and audio conditioning.
"""

import unittest
import numpy as np
from utils.preprocessing import clean_text, extract_linguistic_features, load_and_preprocess_audio


class TestPreprocessing(unittest.TestCase):

    def test_clean_text(self):
        raw = "  Hello   world! \r\n This is a ‘test’ with “smart quotes”.  "
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "Hello world! This is a 'test' with \"smart quotes\".")

    def test_clean_text_empty(self):
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text("   \n\t "), "")
        self.assertEqual(clean_text(None), "")

    def test_extract_linguistic_features(self):
        text = "I feel completely exhausted and sad. I am always tired and my life is hopeless."
        ling = extract_linguistic_features(text)

        self.assertGreaterEqual(ling["fps_pronoun_count"], 3)  # I, I, my
        self.assertIn("i", ling["found_fps_tokens"])
        self.assertIn("my", ling["found_fps_tokens"])
        self.assertGreaterEqual(ling["absolutist_count"], 2)   # completely, always
        self.assertGreaterEqual(ling["negative_word_count"], 2) # exhausted, sad, hopeless
        self.assertGreater(ling["fps_pronoun_ratio"], 0.0)
        self.assertGreater(ling["absolutist_ratio"], 0.0)

    def test_extract_linguistic_features_empty(self):
        ling = extract_linguistic_features("")
        self.assertEqual(ling["word_count"], 0)
        self.assertEqual(ling["fps_pronoun_count"], 0)

    def test_audio_preprocessing_silent(self):
        sr = 16000
        silent_audio = np.zeros(sr * 2, dtype=np.float32)
        conditioned, out_sr, dur, diag = load_and_preprocess_audio(silent_audio.tobytes())
        self.assertTrue(diag["is_silent"])
        self.assertEqual(out_sr, sr)


if __name__ == "__main__":
    unittest.main()
