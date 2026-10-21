"""
Psychological State Analysis Using Multimodal AI
macOS & Linear Inspired AI Research Station
University Academic Research Prototype
"""

import sys
import io
import time
import queue
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Ensure workspace is on sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import (
    CANONICAL_EMOTIONS,
    DEFAULT_FUSION_WEIGHTS,
    DISCLAIMER_TEXT,
    PRIVACY_STATEMENT,
    SAMPLE_AUDIO_DIR,
    SAMPLE_IMAGES_DIR,
    FACE_LABEL_MAPPING,
    EMOTION_VALENCE_AROUSAL
)
from modules.text_analysis import analyze_text
from modules.speech_analysis import analyze_speech
from modules.face_analysis import analyze_facial_expressions
from modules.multimodal_fusion import fuse_multimodal_inputs
from modules.dataset_evaluation import (
    load_or_create_benchmark_dataset,
    train_and_evaluate_baseline,
    evaluate_pretrained_transformer_on_benchmark,
    DATASET_DOCUMENTATION
)
from utils.model_loader import load_face_emotion_pipeline
from utils.preprocessing import detect_faces
from utils.visualizer import (
    plot_emotion_distribution,
    plot_emotion_radar,
    plot_audio_wave_and_energy,
    plot_pitch_contour,
    plot_circumplex_affect,
    plot_modality_comparison_bar,
    plot_confusion_matrix_heatmap
)

# Optional WebRTC Live Camera Support
try:
    from streamlit_webrtc import (
        webrtc_streamer,
        VideoProcessorBase,
        WebRtcMode,
        RTCConfiguration
    )
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False


# ============================================================================
# 1. PAGE CONFIGURATION & SESSION STATE
# ============================================================================

st.set_page_config(
    page_title="Psychological State Analysis",
    page_icon="assets/favicon.ico" if (BASE_DIR / "assets/favicon.ico").exists() else None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Persistent Session State
if "theme" not in st.session_state:
    st.session_state.theme = "dark"  # Refined dark mode default
if "page" not in st.session_state:
    st.session_state.page = "Overview"
if "text_result" not in st.session_state:
    st.session_state.text_result = None
if "speech_result" not in st.session_state:
    st.session_state.speech_result = None
if "face_result" not in st.session_state:
    st.session_state.face_result = None
if "input_text_val" not in st.session_state:
    st.session_state.input_text_val = ""
if "selected_audio_path" not in st.session_state:
    st.session_state.selected_audio_path = str(SAMPLE_AUDIO_DIR / "sample_sad_monotone.wav")
if "selected_image_path" not in st.session_state:
    st.session_state.selected_image_path = str(SAMPLE_IMAGES_DIR / "sample_face_joy.jpg")
if "baseline_eval_results" not in st.session_state:
    st.session_state.baseline_eval_results = None
if "transformer_eval_results" not in st.session_state:
    st.session_state.transformer_eval_results = None
if "session_history" not in st.session_state:
    st.session_state.session_history = []
if "fusion_weights" not in st.session_state:
    st.session_state.fusion_weights = dict(DEFAULT_FUSION_WEIGHTS)

theme = st.session_state.theme


# ============================================================================
# 2. MACOS & LINEAR DESIGN SYSTEM STYLESHEET (NO EMOJIS, CLEAN APPLE AESTHETIC)
# ============================================================================

if theme == "dark":
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        :root {
            --bg-canvas: #0A0A0C;
            --bg-sidebar: #121215;
            --bg-card: #18181D;
            --bg-card-hover: #202026;
            --border-hairline: rgba(255, 255, 255, 0.08);
            --border-subtle: rgba(255, 255, 255, 0.14);
            --text-primary: #F5F5F7;
            --text-secondary: #8E8E93;
            --text-muted: #636366;
            --accent: #0A84FF;
            --accent-hover: #0071E3;
            --badge-bg: rgba(255, 255, 255, 0.06);
            --badge-border: rgba(255, 255, 255, 0.10);
            --green: #30D158;
            --orange: #FF9F0A;
            --red: #FF453A;
            --purple: #BF5AF2;
        }

        html, body, [class*="st-emotion-cache"] {
            background-color: var(--bg-canvas) !important;
            color: var(--text-primary) !important;
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", Inter, sans-serif !important;
            letter-spacing: -0.011em;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: var(--bg-sidebar) !important;
            border-right: 1px solid var(--border-hairline) !important;
        }
        section[data-testid="stSidebar"] .stRadio > div {
            gap: 2px !important;
        }
        section[data-testid="stSidebar"] .stRadio label {
            background: transparent;
            padding: 7px 12px !important;
            border-radius: 6px !important;
            font-size: 13px !important;
            font-weight: 500 !important;
            color: var(--text-secondary) !important;
            transition: all 0.15s ease-in-out;
        }
        section[data-testid="stSidebar"] .stRadio label:hover {
            background: rgba(255, 255, 255, 0.04) !important;
            color: var(--text-primary) !important;
        }

        /* Top Bar macOS Style */
        .macos-topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 18px;
            background: var(--bg-card);
            border: 1px solid var(--border-hairline);
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .mac-traffic-lights {
            display: flex;
            gap: 6px;
            align-items: center;
        }
        .mac-dot {
            width: 11px;
            height: 11px;
            border-radius: 50%;
            display: inline-block;
        }
        .mac-dot.red { background: #FF5F56; border: 1px solid #E0443E; }
        .mac-dot.yellow { background: #FFBD2E; border: 1px solid #DEA123; }
        .mac-dot.green { background: #27C93F; border: 1px solid #1AAB29; }
        
        .macos-title {
            font-size: 13px;
            font-weight: 600;
            color: var(--text-secondary);
            letter-spacing: -0.01em;
        }
        
        /* Linear / Apple Cards */
        .apple-card {
            background: var(--bg-card);
            border: 1px solid var(--border-hairline);
            border-radius: 12px;
            padding: 18px 20px;
            margin-bottom: 16px;
            transition: border-color 0.2s ease, background-color 0.2s ease;
        }
        .apple-card:hover {
            border-color: var(--border-subtle);
            background: var(--bg-card-hover);
        }
        .apple-card-title {
            font-size: 11px;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 6px;
        }
        .apple-card-value {
            font-size: 22px;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.02em;
        }
        .apple-card-sub {
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 4px;
        }

        /* Pill status badges */
        .apple-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 3px 10px;
            font-size: 11px;
            font-weight: 500;
            border-radius: 9999px;
            background: var(--badge-bg);
            border: 1px solid var(--badge-border);
            color: var(--text-secondary);
        }
        .apple-pill.active {
            color: var(--green);
            background: rgba(48, 209, 88, 0.10);
            border-color: rgba(48, 209, 88, 0.25);
        }
        .apple-pill.standby {
            color: var(--orange);
            background: rgba(255, 159, 10, 0.10);
            border-color: rgba(255, 159, 10, 0.25);
        }
        .apple-pill.blue {
            color: var(--accent);
            background: rgba(10, 132, 255, 0.10);
            border-color: rgba(10, 132, 255, 0.25);
        }

        /* Buttons */
        div.stButton > button {
            border-radius: 8px !important;
            font-size: 13px !important;
            font-weight: 500 !important;
            padding: 6px 16px !important;
            border: 1px solid var(--border-hairline) !important;
            background: var(--bg-card) !important;
            color: var(--text-primary) !important;
            transition: all 0.15s ease !important;
        }
        div.stButton > button:hover {
            border-color: var(--border-subtle) !important;
            background: var(--bg-card-hover) !important;
        }
        div.stButton > button[kind="primary"] {
            background: var(--accent) !important;
            border-color: var(--accent) !important;
            color: #FFFFFF !important;
        }
        div.stButton > button[kind="primary"]:hover {
            background: var(--accent-hover) !important;
        }

        /* Text area and inputs */
        textarea, input {
            background-color: var(--bg-card) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-hairline) !important;
            border-radius: 8px !important;
            font-size: 13px !important;
        }
        textarea:focus, input:focus {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 1px var(--accent) !important;
        }

        /* Info & Warning Panels */
        .info-panel {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-hairline);
            border-radius: 8px;
            padding: 12px 14px;
            font-size: 12px;
            color: var(--text-secondary);
            margin: 12px 0;
            line-height: 1.5;
        }
        .alert-panel {
            background: rgba(255, 69, 58, 0.08);
            border: 1px solid rgba(255, 69, 58, 0.25);
            border-radius: 8px;
            padding: 12px 14px;
            font-size: 12px;
            color: #FF6961;
            margin: 12px 0;
            line-height: 1.5;
        }
    </style>
    """
else:  # Light Mode (Apple iOS / macOS Light)
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        :root {
            --bg-canvas: #F5F5F7;
            --bg-sidebar: #EBEBF0;
            --bg-card: #FFFFFF;
            --bg-card-hover: #FBFBFC;
            --border-hairline: rgba(0, 0, 0, 0.08);
            --border-subtle: rgba(0, 0, 0, 0.14);
            --text-primary: #1D1D1F;
            --text-secondary: #6E6E73;
            --text-muted: #86868B;
            --accent: #0071E3;
            --accent-hover: #0077ED;
            --badge-bg: rgba(0, 0, 0, 0.04);
            --badge-border: rgba(0, 0, 0, 0.08);
            --green: #34C759;
            --orange: #FF9500;
            --red: #FF3B30;
            --purple: #AF52DE;
        }

        html, body, [class*="st-emotion-cache"] {
            background-color: var(--bg-canvas) !important;
            color: var(--text-primary) !important;
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", Inter, sans-serif !important;
            letter-spacing: -0.011em;
        }

        section[data-testid="stSidebar"] {
            background-color: var(--bg-sidebar) !important;
            border-right: 1px solid var(--border-hairline) !important;
        }
        section[data-testid="stSidebar"] .stRadio label {
            color: var(--text-secondary) !important;
            font-size: 13px !important;
            font-weight: 500 !important;
        }

        .macos-topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 18px;
            background: var(--bg-card);
            border: 1px solid var(--border-hairline);
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }
        .mac-traffic-lights {
            display: flex;
            gap: 6px;
            align-items: center;
        }
        .mac-dot {
            width: 11px;
            height: 11px;
            border-radius: 50%;
            display: inline-block;
        }
        .mac-dot.red { background: #FF5F56; border: 1px solid #E0443E; }
        .mac-dot.yellow { background: #FFBD2E; border: 1px solid #DEA123; }
        .mac-dot.green { background: #27C93F; border: 1px solid #1AAB29; }
        
        .macos-title {
            font-size: 13px;
            font-weight: 600;
            color: var(--text-secondary);
        }

        .apple-card {
            background: var(--bg-card);
            border: 1px solid var(--border-hairline);
            border-radius: 12px;
            padding: 18px 20px;
            margin-bottom: 16px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        }
        .apple-card:hover {
            border-color: var(--border-subtle);
            background: var(--bg-card-hover);
        }
        .apple-card-title {
            font-size: 11px;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 6px;
        }
        .apple-card-value {
            font-size: 22px;
            font-weight: 700;
            color: var(--text-primary);
        }
        .apple-card-sub {
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 4px;
        }

        .apple-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 3px 10px;
            font-size: 11px;
            font-weight: 500;
            border-radius: 9999px;
            background: var(--badge-bg);
            border: 1px solid var(--badge-border);
            color: var(--text-secondary);
        }
        .apple-pill.active {
            color: #15803D;
            background: #DCFCE7;
            border-color: #BBF7D0;
        }
        .apple-pill.standby {
            color: #B45309;
            background: #FEF3C7;
            border-color: #FDE68A;
        }
        .apple-pill.blue {
            color: #1D4ED8;
            background: #DBEAFE;
            border-color: #BFDBFE;
        }

        div.stButton > button {
            border-radius: 8px !important;
            font-size: 13px !important;
            font-weight: 500 !important;
            padding: 6px 16px !important;
            border: 1px solid var(--border-hairline) !important;
            background: var(--bg-card) !important;
            color: var(--text-primary) !important;
        }
        div.stButton > button[kind="primary"] {
            background: var(--accent) !important;
            border-color: var(--accent) !important;
            color: #FFFFFF !important;
        }

        .info-panel {
            background: rgba(0, 0, 0, 0.02);
            border: 1px solid var(--border-hairline);
            border-radius: 8px;
            padding: 12px 14px;
            font-size: 12px;
            color: var(--text-secondary);
            margin: 12px 0;
            line-height: 1.5;
        }
        .alert-panel {
            background: #FEF2F2;
            border: 1px solid #FECACA;
            border-radius: 8px;
            padding: 12px 14px;
            font-size: 12px;
            color: #B91C1C;
            margin: 12px 0;
            line-height: 1.5;
        }
    </style>
    """

st.markdown(css, unsafe_allow_html=True)


# ============================================================================
# 3. WEBRTC LIVE VIDEO PROCESSOR (REAL-TIME FACE & EMOTION DETECTION)
# ============================================================================

class LiveFacialVideoProcessor(VideoProcessorBase if HAS_WEBRTC else object):
    """
    Real-time WebRTC frame transformer that detects faces with OpenCV and runs
    throttled ViT emotion predictions in the background without freezing the stream.
    """
    def __init__(self):
        self._frame_count = 0
        self._last_emotion = "Scanning"
        self._last_confidence = 0.0
        self._last_box = None
        self._vit_pipeline = None
        self._lock = threading.Lock()
        self._latest_frame_bgr = None

    def _get_pipeline(self):
        if self._vit_pipeline is None:
            self._vit_pipeline = load_face_emotion_pipeline()
        return self._vit_pipeline

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self._frame_count += 1
        h, w = img.shape[:2]

        with self._lock:
            self._latest_frame_bgr = img.copy()

        # Run face detection and classification every 12 frames (~2.5 times per second)
        # to ensure 60fps video rendering without CPU bottleneck
        if self._frame_count % 12 == 1:
            try:
                rgb_small = cv2.cvtColor(cv2.resize(img, (320, 240)), cv2.COLOR_BGR2RGB)
                faces, _ = detect_faces(rgb_small, margin_ratio=0.15)
                if faces:
                    prim = faces[0]
                    # Scale box back to full frame
                    scale_x = w / 320.0
                    scale_y = h / 240.0
                    bx, by, bw, bh = prim["box"]
                    self._last_box = (
                        int(bx * scale_x), int(by * scale_y),
                        int(bw * scale_x), int(bh * scale_y)
                    )

                    # Quick ViT inference on cropped face
                    pipe = self._get_pipeline()
                    preds = pipe(prim["crop_pil"])
                    top = preds[0] if isinstance(preds[0], dict) else preds[0][0]
                    raw_lbl = top["label"].lower()
                    canon_lbl = FACE_LABEL_MAPPING.get(raw_lbl, raw_lbl)
                    self._last_emotion = canon_lbl.capitalize()
                    self._last_confidence = float(top["score"])
                else:
                    self._last_box = None
                    self._last_emotion = "No Face Detected"
                    self._last_confidence = 0.0
            except Exception:
                pass

        # Draw clean macOS hairline bounding box
        if self._last_box:
            x, y, bw, bh = self._last_box
            # Hairline Apple Blue box
            cv2.rectangle(img, (x, y), (x + bw, y + bh), (255, 132, 10), 2)  # BGR
            tag = f"{self._last_emotion} {int(self._last_confidence*100)}%" if self._last_confidence > 0 else self._last_emotion
            cv2.putText(
                img, tag, (x, max(22, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 132, 10), 2, cv2.LINE_AA
            )
        else:
            # Subtle top scanner bar
            cv2.putText(
                img, "Live Face Tracker Active", (16, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1, cv2.LINE_AA
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ============================================================================
# 4. SIDEBAR NAVIGATION & MODALITY STATUS
# ============================================================================

with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 16px 0;">
        <div style="font-size: 14px; font-weight: 700; letter-spacing: -0.01em;">Psychological State AI</div>
        <div style="font-size: 11px; color: var(--text-muted);">Multimodal Research Station</div>
    </div>
    """, unsafe_allow_html=True)

    nav_options = [
        "Overview",
        "Text Analysis",
        "Speech Analysis",
        "Facial Analysis",
        "Multimodal Results",
        "Settings"
    ]
    
    # Selected navigation page
    current_index = nav_options.index(st.session_state.page) if st.session_state.page in nav_options else 0
    selected_page = st.radio("Navigation", nav_options, index=current_index, label_visibility="collapsed")
    if selected_page != st.session_state.page:
        st.session_state.page = selected_page
        st.rerun()

    st.markdown("<hr style='border:none; border-top: 1px solid var(--border-hairline); margin: 20px 0 14px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 11px; font-weight: 600; text-transform: uppercase; color: var(--text-muted); margin-bottom: 8px;'>Active Modalities</div>", unsafe_allow_html=True)

    # Text status pill
    if st.session_state.text_result:
        t_dom = st.session_state.text_result.get("dominant_emotion", "").capitalize()
        st.markdown(f"<div style='margin-bottom:6px;'><span class='apple-pill active'>● Text: {t_dom}</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='margin-bottom:6px;'><span class='apple-pill standby'>○ Text: Standby</span></div>", unsafe_allow_html=True)

    # Speech status pill
    if st.session_state.speech_result and st.session_state.speech_result.get("path_b_prosody", {}).get("success", False):
        aro = st.session_state.speech_result["path_b_prosody"].get("acoustic_arousal", 0.0)
        st.markdown(f"<div style='margin-bottom:6px;'><span class='apple-pill active'>● Speech: Arousal {aro:+.2f}</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='margin-bottom:6px;'><span class='apple-pill standby'>○ Speech: Standby</span></div>", unsafe_allow_html=True)

    # Face status pill
    if st.session_state.face_result and st.session_state.face_result.get("success", False):
        f_dom = st.session_state.face_result.get("dominant_emotion", "").capitalize()
        st.markdown(f"<div style='margin-bottom:6px;'><span class='apple-pill active'>● Face: {f_dom}</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='margin-bottom:6px;'><span class='apple-pill standby'>○ Face: Standby</span></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Reset Session Data", use_container_width=True):
        st.session_state.text_result = None
        st.session_state.speech_result = None
        st.session_state.face_result = None
        st.rerun()


# ============================================================================
# 5. TOP BAR (MACOS CHROME & THEME TOGGLE)
# ============================================================================

top_col1, top_col2 = st.columns([3, 1])
with top_col1:
    st.markdown(f"""
    <div class="macos-topbar">
        <div style="display:flex; align-items:center; gap:12px;">
            <div class="mac-traffic-lights">
                <span class="mac-dot red"></span>
                <span class="mac-dot yellow"></span>
                <span class="mac-dot green"></span>
            </div>
            <div class="macos-title">Psychological State Analysis &bull; {st.session_state.page}</div>
        </div>
        <div style="display:flex; align-items:center; gap:10px;">
            <span class="apple-pill {'active' if st.session_state.face_result else 'standby'}">
                {'● Camera Ready' if st.session_state.face_result else '○ Camera Standby'}
            </span>
            <span class="apple-pill blue">Local Inference</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with top_col2:
    # Minimal Theme Toggle Button
    theme_label = "Light Mode" if theme == "dark" else "Dark Mode"
    if st.button(theme_label, use_container_width=True):
        st.session_state.theme = "light" if theme == "dark" else "dark"
        st.rerun()


# ============================================================================
# 6. PAGE: OVERVIEW (MAIN DASHBOARD)
# ============================================================================

if st.session_state.page == "Overview":
    st.markdown("<h2 style='font-size:24px; font-weight:700; margin:0 0 4px 0;'>Psychological State Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:13px; color:var(--text-secondary); margin:0 0 20px 0;'>Explore linguistic, vocal and visual indicators across three independent modalities.</p>", unsafe_allow_html=True)

    # 3 Compact Modality Cards
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="apple-card">
            <div class="apple-card-title">Modality 01</div>
            <div class="apple-card-value">Text Analysis</div>
            <div class="apple-card-sub">DistilRoBERTa emotion classification, Pennebaker LIWC pronouns, and TF-IDF baseline.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Text Module", key="btn_to_text", use_container_width=True):
            st.session_state.page = "Text Analysis"
            st.rerun()

    with c2:
        st.markdown("""
        <div class="apple-card">
            <div class="apple-card-title">Modality 02</div>
            <div class="apple-card-value">Speech Prosody</div>
            <div class="apple-card-sub">Whisper ASR transcription coupled with Librosa acoustic pitch (F0), RMS, and speaking rate.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Speech Module", key="btn_to_speech", use_container_width=True):
            st.session_state.page = "Speech Analysis"
            st.rerun()

    with c3:
        st.markdown("""
        <div class="apple-card">
            <div class="apple-card-title">Modality 03</div>
            <div class="apple-card-value">Facial Expression</div>
            <div class="apple-card-sub">Live webcam stream or portrait upload with OpenCV face tracking and Vision Transformer FER.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Camera Module", key="btn_to_face", use_container_width=True):
            st.session_state.page = "Facial Analysis"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Start Analysis Action & Status
    has_any = (st.session_state.text_result or st.session_state.speech_result or st.session_state.face_result)
    btn_col1, btn_col2 = st.columns([1, 3])
    with btn_col1:
        if st.button("View Multimodal Results", type="primary", use_container_width=True):
            st.session_state.page = "Multimodal Results"
            st.rerun()
    with btn_col2:
        active_cnt = sum([bool(st.session_state.text_result), bool(st.session_state.speech_result), bool(st.session_state.face_result)])
        st.markdown(f"<div style='padding-top:8px; font-size:12px; color:var(--text-secondary);'>{active_cnt} of 3 modalities loaded for decision-level late fusion.</div>", unsafe_allow_html=True)

    # Recent Analysis Section (Only if real session history exists)
    if st.session_state.session_history:
        st.markdown("<hr style='border:none; border-top: 1px solid var(--border-hairline); margin: 24px 0 16px 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:12px; font-weight:600; text-transform:uppercase; color:var(--text-muted); margin-bottom:10px;'>Recent Session Inferences</div>", unsafe_allow_html=True)
        for item in reversed(st.session_state.session_history[-3:]):
            st.markdown(f"""
            <div class="apple-card" style="padding:12px 16px; margin-bottom:8px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="font-size:13px; font-weight:600;">{item['modality']} &mdash; <span style="color:var(--accent);">{item['summary']}</span></div>
                    <div style="font-size:11px; color:var(--text-muted);">{item['timestamp']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Minimal Disclaimers
    st.markdown("""
    <div class="info-panel">
        <strong>Academic Notice:</strong> This prototype evaluates affective and linguistic indicators for university research demonstration. 
        It does not diagnose depression or any mental health disorder. All processing is computed locally in memory.
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# 7. PAGE: TEXT ANALYSIS
# ============================================================================

elif st.session_state.page == "Text Analysis":
    st.markdown("<h2 style='font-size:22px; font-weight:700; margin:0 0 4px 0;'>Text Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:13px; color:var(--text-secondary); margin:0 0 16px 0;'>Transformer emotion classification and Pennebaker LIWC cognitive linguistic markers.</p>", unsafe_allow_html=True)

    if "txt_input_area" not in st.session_state:
        st.session_state.txt_input_area = st.session_state.input_text_val

    # Preset quick-fill buttons
    preset_cols = st.columns(4)
    if preset_cols[0].button("Depressed Sample", use_container_width=True):
        st.session_state.txt_input_area = "I feel completely exhausted, hopeless, and empty inside. Everything I try to do feels like a heavy burden."
        st.session_state.input_text_val = st.session_state.txt_input_area
        st.rerun()
    if preset_cols[1].button("Anxiety Sample", use_container_width=True):
        st.session_state.txt_input_area = "My heart is pounding and I cannot catch my breath. I am terrified that something catastrophic is going to happen."
        st.session_state.input_text_val = st.session_state.txt_input_area
        st.rerun()
    if preset_cols[2].button("Joyful Sample", use_container_width=True):
        st.session_state.txt_input_area = "I am so excited and genuinely grateful today! Our team's paper was accepted at the conference."
        st.session_state.input_text_val = st.session_state.txt_input_area
        st.rerun()
    if preset_cols[3].button("Masked Sample", use_container_width=True):
        st.session_state.txt_input_area = "I always tell everyone that everything is fine and totally normal, but inside I feel completely isolated."
        st.session_state.input_text_val = st.session_state.txt_input_area
        st.rerun()

    user_text = st.text_area(
        "Enter text to analyze",
        key="txt_input_area",
        height=110,
        placeholder="Enter text to analyze",
        label_visibility="collapsed"
    )

    t_btn1, t_btn2 = st.columns([1, 4])
    if t_btn1.button("Analyze", type="primary", use_container_width=True) and user_text.strip():
        st.session_state.input_text_val = user_text
        with st.spinner("Analyzing text..."):
            res = analyze_text(user_text)
            st.session_state.text_result = res
            st.session_state.session_history.append({
                "modality": "Text",
                "summary": f"{res['dominant_emotion'].capitalize()} ({res['confidence']*100:.0f}%)",
                "timestamp": time.strftime("%H:%M:%S")
            })

    # Results Panel
    if st.session_state.text_result:
        res = st.session_state.text_result
        st.markdown("<hr style='border:none; border-top: 1px solid var(--border-hairline); margin: 20px 0;'>", unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        dom = res["dominant_emotion"].capitalize()
        conf = res["confidence"]
        val = res["valence"]
        aro = res["arousal"]

        m1.markdown(f"""
        <div class="apple-card">
            <div class="apple-card-title">Predicted Category</div>
            <div class="apple-card-value">{dom}</div>
            <div class="apple-card-sub">Confidence: {conf*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        m2.markdown(f"""
        <div class="apple-card">
            <div class="apple-card-title">Continuous Valence</div>
            <div class="apple-card-value">{val:+.2f}</div>
            <div class="apple-card-sub">Pleasure axis (Russell Circumplex)</div>
        </div>
        """, unsafe_allow_html=True)

        m3.markdown(f"""
        <div class="apple-card">
            <div class="apple-card-title">Continuous Arousal</div>
            <div class="apple-card-value">{aro:+.2f}</div>
            <div class="apple-card-sub">Activation axis (Russell Circumplex)</div>
        </div>
        """, unsafe_allow_html=True)

        # Plot and Linguistic Indicators
        c_p1, c_p2 = st.columns([3, 2])
        with c_p1:
            fig = plot_emotion_distribution(res["emotion_probabilities"], height=250, theme=theme)
            st.plotly_chart(fig, use_container_width=True)
        with c_p2:
            ling = res["linguistic_features"]
            st.markdown(f"""
            <div class="apple-card" style="height:100%;">
                <div class="apple-card-title">Linguistic Indicators</div>
                <div style="font-size:12px; line-height:1.7; color:var(--text-secondary);">
                    &bull; <b>First-Person Pronouns (I, me):</b> {ling['fps_pronoun_ratio']}%<br>
                    &bull; <b>Absolutist Terms (always, never):</b> {ling['absolutist_ratio']}%<br>
                    &bull; <b>Negative Lexicon:</b> {ling['negative_word_ratio']}%<br>
                    &bull; <b>Lexical Diversity (TTR):</b> {ling['lexical_diversity']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        if res["research_indicators"]:
            for ind in res["research_indicators"]:
                st.markdown(f"<div class='info-panel'>&bull; {ind}</div>", unsafe_allow_html=True)


# ============================================================================
# 8. PAGE: SPEECH ANALYSIS
# ============================================================================

elif st.session_state.page == "Speech Analysis":
    st.markdown("<h2 style='font-size:22px; font-weight:700; margin:0 0 4px 0;'>Speech &amp; Vocal Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:13px; color:var(--text-secondary); margin:0 0 16px 0;'>Whisper speech recognition and Librosa vocal prosody feature extraction.</p>", unsafe_allow_html=True)

    # Preset quick-fill buttons
    sp_cols = st.columns(4)
    if sp_cols[0].button("Monotone Prosody", use_container_width=True):
        st.session_state.selected_audio_path = str(SAMPLE_AUDIO_DIR / "sample_sad_monotone.wav")
        st.rerun()
    if sp_cols[1].button("Dynamic Voice", use_container_width=True):
        st.session_state.selected_audio_path = str(SAMPLE_AUDIO_DIR / "sample_joy_expressive.wav")
        st.rerun()
    if sp_cols[2].button("Anxious Voice", use_container_width=True):
        st.session_state.selected_audio_path = str(SAMPLE_AUDIO_DIR / "sample_anxious_arousal.wav")
        st.rerun()
    if sp_cols[3].button("Neutral Speech", use_container_width=True):
        st.session_state.selected_audio_path = str(SAMPLE_AUDIO_DIR / "sample_neutral_speech.wav")
        st.rerun()

    source_tab = st.radio("Source", ["Preset Audio", "Upload Audio", "Record Audio"], horizontal=True, label_visibility="collapsed")

    audio_bytes = None
    if source_tab == "Preset Audio":
        p = Path(st.session_state.selected_audio_path)
        if p.exists():
            with open(p, "rb") as f:
                audio_bytes = f.read()
            st.caption(f"Loaded: {p.name}")
    elif source_tab == "Upload Audio":
        up_aud = st.file_uploader("Upload audio file", type=["wav", "mp3", "ogg"], label_visibility="collapsed")
        if up_aud:
            audio_bytes = up_aud.read()
    elif source_tab == "Record Audio":
        mic = st.audio_input("Record audio", label_visibility="collapsed")
        if mic:
            audio_bytes = mic.read()

    if audio_bytes:
        st.audio(audio_bytes)
        if st.button("Analyze Speech", type="primary"):
            with st.spinner("Transcribing and extracting acoustic features..."):
                res = analyze_speech(audio_bytes, transcribe=True, extract_prosody=True)
                st.session_state.speech_result = res
                st.session_state.session_history.append({
                    "modality": "Speech",
                    "summary": f"Duration {res['duration_sec']:.1f}s",
                    "timestamp": time.strftime("%H:%M:%S")
                })

    # Results Panel
    if st.session_state.speech_result:
        res = st.session_state.speech_result
        st.markdown("<hr style='border:none; border-top: 1px solid var(--border-hairline); margin: 20px 0;'>", unsafe_allow_html=True)

        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("<div class='apple-card-title'>Transcript &bull; Whisper ASR</div>", unsafe_allow_html=True)
            transcript = res.get("path_a_transcription", {}).get("transcript", "")
            if transcript:
                st.markdown(f"<div class='apple-card' style='font-size:14px; font-style:italic;'>\"{transcript}\"</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='apple-card' style='font-size:13px; color:var(--text-muted);'>No speech transcript detected.</div>", unsafe_allow_html=True)

        with sc2:
            st.markdown("<div class='apple-card-title'>Vocal Prosody Metrics</div>", unsafe_allow_html=True)
            prosody = res.get("path_b_prosody", {})
            if prosody and prosody.get("success", False):
                m_row1, m_row2 = st.columns(2)
                m_row1.metric("Pitch (F0)", f"{prosody['f0_mean']:.0f} Hz", f"SD: ±{prosody['f0_std']:.0f} Hz")
                m_row2.metric("Loudness Dynamic", f"{prosody['dynamic_range_db']:.1f} dB", f"RMS: {prosody['rms_mean']:.3f}")
                m_row1.metric("Speaking Rate", f"{prosody['speaking_rate_syllables_per_sec']:.1f} syl/s")
                m_row2.metric("Pause Ratio", f"{prosody['pause_ratio']*100:.0f}%")

        prosody = res.get("path_b_prosody", {})
        if prosody and prosody.get("success", False):
            f_wave = plot_audio_wave_and_energy(
                prosody["audio_waveform"], prosody["sample_rate"],
                prosody["rms_times"], prosody["rms_series"],
                theme=theme
            )
            st.plotly_chart(f_wave, use_container_width=True)

            if len(prosody["pitch_times"]) > 0:
                f_pitch = plot_pitch_contour(
                    prosody["pitch_times"], prosody["f0_series"],
                    prosody["f0_mean"], prosody["f0_std"],
                    theme=theme
                )
                st.plotly_chart(f_pitch, use_container_width=True)

        st.markdown("<div class='info-panel'>Acoustic features reflect physical voice characteristics and are not direct evidence of psychological disorders.</div>", unsafe_allow_html=True)


# ============================================================================
# 9. PAGE: FACIAL EXPRESSION ANALYSIS (REAL-TIME CAMERA STREAM)
# ============================================================================

elif st.session_state.page == "Facial Analysis":
    st.markdown("<h2 style='font-size:22px; font-weight:700; margin:0 0 4px 0;'>Facial Expression Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:13px; color:var(--text-secondary); margin:0 0 16px 0;'>Live webcam stream, face detection, and Vision Transformer expression classification.</p>", unsafe_allow_html=True)

    face_mode = st.radio("Camera Mode", ["Live Real-Time Camera", "Preset Photo", "Upload Photo"], horizontal=True, label_visibility="collapsed")

    # 1. Live Real-Time WebRTC Camera Stream
    if face_mode == "Live Real-Time Camera":
        if not HAS_WEBRTC:
            st.error("Live streaming requires streamlit-webrtc. Use Upload Photo mode.")
        else:
            st.markdown("""
            <div class="apple-card" style="padding:14px; margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-size:13px; font-weight:600;">Real-Time Video Feed</div>
                        <div style="font-size:11px; color:var(--text-muted);">Face detection bounding boxes and visible expression tracking.</div>
                    </div>
                    <span class="apple-pill active">Camera Ready</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            webrtc_ctx = webrtc_streamer(
                key="facial_live_stream",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                video_processor_factory=LiveFacialVideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True
            )

            # Freeze / Capture Frame Action
            if webrtc_ctx and webrtc_ctx.video_processor:
                cap_col1, cap_col2 = st.columns([1, 3])
                if cap_col1.button("Capture & Commit Frame", type="primary", use_container_width=True):
                    with webrtc_ctx.video_processor._lock:
                        frame_bgr = webrtc_ctx.video_processor._latest_frame_bgr
                    if frame_bgr is not None:
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(frame_rgb)
                        with st.spinner("Analyzing captured frame..."):
                            res = analyze_facial_expressions(pil_img)
                            st.session_state.face_result = res
                            st.session_state.session_history.append({
                                "modality": "Face",
                                "summary": f"{res['dominant_emotion'].capitalize()} ({res['confidence']*100:.0f}%)",
                                "timestamp": time.strftime("%H:%M:%S")
                            })
                            st.success("Frame committed to multimodal state.")

    # 2. Preset Photo
    elif face_mode == "Preset Photo":
        ps_cols = st.columns(3)
        if ps_cols[0].button("Smiling Portrait", use_container_width=True):
            st.session_state.selected_image_path = str(SAMPLE_IMAGES_DIR / "sample_face_joy.jpg")
            st.rerun()
        if ps_cols[1].button("Somber Portrait", use_container_width=True):
            st.session_state.selected_image_path = str(SAMPLE_IMAGES_DIR / "sample_face_sad.jpg")
            st.rerun()
        if ps_cols[2].button("Neutral Portrait", use_container_width=True):
            st.session_state.selected_image_path = str(SAMPLE_IMAGES_DIR / "sample_face_neutral.jpg")
            st.rerun()

        img_p = Path(st.session_state.selected_image_path)
        if img_p.exists():
            pil_img = Image.open(img_p)
            c_v1, c_v2 = st.columns([1, 2])
            with c_v1:
                st.image(pil_img, use_container_width=True, caption=img_p.name)
            with c_v2:
                if st.button("Analyze Facial Expression", type="primary"):
                    with st.spinner("Classifying facial Action Units..."):
                        res = analyze_facial_expressions(pil_img)
                        st.session_state.face_result = res

    # 3. Upload Photo
    elif face_mode == "Upload Photo":
        up_f = st.file_uploader("Upload portrait photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if up_f:
            pil_img = Image.open(up_f)
            c_v1, c_v2 = st.columns([1, 2])
            with c_v1:
                st.image(pil_img, use_container_width=True)
            with c_v2:
                if st.button("Analyze Uploaded Photo", type="primary"):
                    with st.spinner("Detecting faces and classifying..."):
                        res = analyze_facial_expressions(pil_img)
                        st.session_state.face_result = res

    # Results Panel
    if st.session_state.face_result:
        res = st.session_state.face_result
        if res.get("success", False):
            st.markdown("<hr style='border:none; border-top: 1px solid var(--border-hairline); margin: 20px 0;'>", unsafe_allow_html=True)
            rf1, rf2, rf3 = st.columns([1, 1, 2])
            with rf1:
                st.image(res["annotated_image"], caption=f"Detected Faces ({res['face_count']})", use_container_width=True)
            with rf2:
                st.image(res["selected_face_crop"], caption="Analyzed Face", use_container_width=True)
            with rf3:
                f_dom = res["dominant_emotion"].capitalize()
                f_conf = res["confidence"]
                st.markdown(f"""
                <div class="apple-card">
                    <div class="apple-card-title">Visible Facial Expression</div>
                    <div class="apple-card-value">{f_dom}</div>
                    <div class="apple-card-sub">Confidence: {f_conf*100:.1f}% | Valence: {res['valence']:+.2f}</div>
                </div>
                """, unsafe_allow_html=True)

            f_bar = plot_emotion_distribution(res["emotion_probabilities"], title="Visible Expression Probabilities", height=240, theme=theme)
            st.plotly_chart(f_bar, use_container_width=True)

        st.markdown("<div class='info-panel'>Visible expressions reflect surface facial muscle actions and do not indicate internal psychiatric status.</div>", unsafe_allow_html=True)


# ============================================================================
# 10. PAGE: MULTIMODAL RESULTS (COMPARISON & FUSION)
# ============================================================================

elif st.session_state.page == "Multimodal Results":
    st.markdown("<h2 style='font-size:22px; font-weight:700; margin:0 0 4px 0;'>Multimodal Results</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:13px; color:var(--text-secondary); margin:0 0 16px 0;'>Decision-level late fusion, Russell Circumplex affect space, and cross-modal dissonance detection.</p>", unsafe_allow_html=True)

    # 3-Column Modality Summary
    col_t, col_s, col_f = st.columns(3)

    with col_t:
        st.markdown("<div class='apple-card-title'>Text Modality</div>", unsafe_allow_html=True)
        if st.session_state.text_result:
            t = st.session_state.text_result
            st.markdown(f"""
            <div class="apple-card" style="border-left: 3px solid #0A84FF;">
                <div style="font-size:16px; font-weight:700;">{t['dominant_emotion'].capitalize()}</div>
                <div style="font-size:12px; color:var(--text-secondary); margin-top:2px;">Confidence: {t['confidence']*100:.1f}%</div>
                <div style="font-size:11px; color:var(--text-muted); margin-top:4px;">Valence: {t['valence']:+.2f} | Arousal: {t['arousal']:+.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<div class='apple-card' style='color:var(--text-muted); font-size:12px;'>No text input provided</div>", unsafe_allow_html=True)

    with col_s:
        st.markdown("<div class='apple-card-title'>Speech Prosody</div>", unsafe_allow_html=True)
        if st.session_state.speech_result and st.session_state.speech_result.get("path_b_prosody", {}).get("success", False):
            sp = st.session_state.speech_result["path_b_prosody"]
            st.markdown(f"""
            <div class="apple-card" style="border-left: 3px solid #FF9F0A;">
                <div style="font-size:16px; font-weight:700;">Arousal: {sp['acoustic_arousal']:+.2f}</div>
                <div style="font-size:12px; color:var(--text-secondary); margin-top:2px;">F0: {sp['f0_mean']:.0f} Hz (±{sp['f0_std']:.0f})</div>
                <div style="font-size:11px; color:var(--text-muted); margin-top:4px;">Tempo: {sp['speaking_rate_syllables_per_sec']:.1f} syl/s</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<div class='apple-card' style='color:var(--text-muted); font-size:12px;'>No speech recording provided</div>", unsafe_allow_html=True)

    with col_f:
        st.markdown("<div class='apple-card-title'>Facial Expression</div>", unsafe_allow_html=True)
        if st.session_state.face_result and st.session_state.face_result.get("success", False):
            fc = st.session_state.face_result
            st.markdown(f"""
            <div class="apple-card" style="border-left: 3px solid #30D158;">
                <div style="font-size:16px; font-weight:700;">{fc['dominant_emotion'].capitalize()}</div>
                <div style="font-size:12px; color:var(--text-secondary); margin-top:2px;">Confidence: {fc['confidence']*100:.1f}%</div>
                <div style="font-size:11px; color:var(--text-muted); margin-top:4px;">Valence: {fc['valence']:+.2f} | Arousal: {fc['arousal']:+.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<div class='apple-card' style='color:var(--text-muted); font-size:12px;'>No facial frame provided</div>", unsafe_allow_html=True)

    # Late Fusion Execution
    fusion = fuse_multimodal_inputs(
        text_result=st.session_state.text_result,
        speech_result=st.session_state.speech_result,
        face_result=st.session_state.face_result,
        custom_weights=st.session_state.fusion_weights
    )

    if not fusion.get("success", False):
        st.markdown("<div class='info-panel'>Provide at least one modality (Text, Speech, or Face) to compute multimodal comparisons.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<hr style='border:none; border-top: 1px solid var(--border-hairline); margin: 20px 0;'>", unsafe_allow_html=True)

        fc1, fc2, fc3 = st.columns(3)
        fc1.markdown(f"""
        <div class="apple-card">
            <div class="apple-card-title">Combined Affective State</div>
            <div class="apple-card-value">{fusion['dominant_emotion'].capitalize()}</div>
            <div class="apple-card-sub">Confidence: {fusion['confidence']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        f_pt = fusion["fused_point"]
        fc2.markdown(f"""
        <div class="apple-card">
            <div class="apple-card-title">Fused 2D Coordinates</div>
            <div class="apple-card-value">V: {f_pt['valence']:+.2f} | A: {f_pt['arousal']:+.2f}</div>
            <div class="apple-card-sub">Russell Circumplex Affective Centroid</div>
        </div>
        """, unsafe_allow_html=True)

        cong = fusion["congruence"]
        fc3.markdown(f"""
        <div class="apple-card">
            <div class="apple-card-title">Modality Alignment</div>
            <div class="apple-card-value">{cong['status'].split('(')[0]}</div>
            <div class="apple-card-sub">Dissonance Index: {cong['dissonance_score']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Russell 2D Circumplex Plot
        fig_circ = plot_circumplex_affect(
            modality_points=fusion["modality_points"],
            fused_point=fusion["fused_point"],
            theme=theme
        )
        st.plotly_chart(fig_circ, use_container_width=True)

        # Text vs Multimodal Comparison
        if st.session_state.text_result:
            fig_cmp = plot_modality_comparison_bar(
                st.session_state.text_result["emotion_probabilities"],
                fusion["emotion_probabilities"],
                theme=theme
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

            if fusion.get("comparison_notes"):
                for note in fusion["comparison_notes"]:
                    st.markdown(f"<div class='info-panel'>&bull; {note}</div>", unsafe_allow_html=True)

        # Affective Dissonance / Masking Notice
        if cong.get("observations"):
            for obs in cong["observations"]:
                if "DISSONANCE" in obs or "INCONGRUENT" in obs:
                    st.markdown(f"<div class='alert-panel'><strong>Notice:</strong> {obs}</div>", unsafe_allow_html=True)

        st.markdown(f"<div class='info-panel'>{fusion['scientific_disclaimer']}</div>", unsafe_allow_html=True)


# ============================================================================
# 11. PAGE: SETTINGS & MODEL BENCHMARK
# ============================================================================

elif st.session_state.page == "Settings":
    st.markdown("<h2 style='font-size:22px; font-weight:700; margin:0 0 4px 0;'>Settings &amp; Model Information</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:13px; color:var(--text-secondary); margin:0 0 16px 0;'>System configuration, empirical baseline metrics, and research dataset documentation.</p>", unsafe_allow_html=True)

    tab_set, tab_bench, tab_data = st.tabs(["Configuration", "Empirical Evaluation", "Public Datasets"])

    with tab_set:
        st.markdown("<div class='apple-card-title'>Fusion Modality Weights</div>", unsafe_allow_html=True)
        col_w1, col_w2, col_w3 = st.columns(3)
        wt = col_w1.slider("Text Weight", 0.0, 1.0, st.session_state.fusion_weights["text"], 0.05)
        ws = col_w2.slider("Speech Prosody Weight", 0.0, 1.0, st.session_state.fusion_weights["speech_prosody"], 0.05)
        wf = col_w3.slider("Facial Expression Weight", 0.0, 1.0, st.session_state.fusion_weights["facial_expression"], 0.05)
        st.session_state.fusion_weights = {"text": wt, "speech_prosody": ws, "facial_expression": wf}

        st.markdown("<hr style='border:none; border-top: 1px solid var(--border-hairline); margin: 20px 0;'>", unsafe_allow_html=True)
        st.markdown("<div class='apple-card-title'>Data & Privacy</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='info-panel'>{PRIVACY_STATEMENT}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='info-panel'>{DISCLAIMER_TEXT}</div>", unsafe_allow_html=True)

    with tab_bench:
        st.markdown("<div class='apple-card-title'>Empirical Benchmark Evaluation</div>", unsafe_allow_html=True)
        if st.button("Run Benchmark Evaluation"):
            with st.spinner("Evaluating baseline vs transformer on held-out test split..."):
                b_res = train_and_evaluate_baseline()
                t_res = evaluate_pretrained_transformer_on_benchmark(b_res["X_test"], b_res["y_test"])
                st.session_state.baseline_eval_results = b_res
                st.session_state.transformer_eval_results = t_res

        if st.session_state.baseline_eval_results and st.session_state.transformer_eval_results:
            b_res = st.session_state.baseline_eval_results
            t_res = st.session_state.transformer_eval_results
            edf = pd.DataFrame([
                {
                    "Model": b_res["model_name"],
                    "Accuracy": f"{b_res['accuracy']*100:.1f}%",
                    "Macro F1": f"{b_res['macro_f1']*100:.1f}%",
                    "Weighted F1": f"{b_res['weighted_f1']*100:.1f}%"
                },
                {
                    "Model": t_res["model_name"],
                    "Accuracy": f"{t_res['accuracy']*100:.1f}%",
                    "Macro F1": f"{t_res['macro_f1']*100:.1f}%",
                    "Weighted F1": f"{t_res['weighted_f1']*100:.1f}%"
                }
            ])
            st.table(edf)

            cm1, cm2 = st.columns(2)
            with cm1:
                st.plotly_chart(plot_confusion_matrix_heatmap(b_res["confusion_matrix"], b_res["classes"], title="Baseline TF-IDF", theme=theme), use_container_width=True)
            with cm2:
                st.plotly_chart(plot_confusion_matrix_heatmap(t_res["confusion_matrix"], t_res["classes"], title="DistilRoBERTa", theme=theme), use_container_width=True)

    with tab_data:
        for dname, dinfo in DATASET_DOCUMENTATION.items():
            st.markdown(f"""
            <div class="apple-card">
                <div style="font-size:14px; font-weight:700;">{dname}</div>
                <div style="font-size:11px; color:var(--accent); margin:2px 0 6px 0;">{dinfo['type']} &bull; {dinfo['source']}</div>
                <div style="font-size:12px; color:var(--text-secondary);">{dinfo['description']}</div>
                <div style="font-size:11px; color:var(--text-muted); margin-top:6px;"><b>Leakage Prevention:</b> {dinfo['protocol']}</div>
            </div>
            """, unsafe_allow_html=True)
