"""
Unit tests for vocal feature extraction (pitch, RMS energy, speaking rate, acoustic arousal).
"""

import unittest
import numpy as np
from modules.audio_features import extract_vocal_features


class TestAudioFeatures(unittest.TestCase):

    def test_extract_vocal_features_valid(self):
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # Synthetic harmonic signal at 150 Hz
        signal = 0.2 * np.sin(2 * np.pi * 150 * t) + 0.1 * np.sin(2 * np.pi * 300 * t)

        res = extract_vocal_features(signal, sr=sr)
        self.assertTrue(res["success"])
        self.assertAlmostEqual(res["duration_sec"], duration, delta=0.1)
        self.assertGreater(res["f0_mean"], 100.0)
        self.assertLess(res["f0_mean"], 200.0)
        self.assertGreater(res["rms_mean"], 0.0)
        self.assertIn("acoustic_arousal", res)
        self.assertGreaterEqual(res["acoustic_arousal"], -1.0)
        self.assertLessEqual(res["acoustic_arousal"], 1.0)
        self.assertEqual(len(res["mfcc_coefficients"]), 13)

    def test_extract_vocal_features_silent(self):
        sr = 16000
        silent = np.zeros(sr * 2, dtype=np.float32)
        res = extract_vocal_features(silent, sr=sr)
        self.assertFalse(res["success"])
        self.assertIn("error", res)


if __name__ == "__main__":
    unittest.main()
