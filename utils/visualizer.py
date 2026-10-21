"""
Interactive Plotly Visualizations for Multimodal Psychological State Analysis
Apple macOS & Linear Inspired Theme Engine (Dark & Light Mode Support)

Generates publication-quality charts for:
1. Emotion probability bars & radar profiles
2. Audio waveform, RMS energy, and pitch (F0) contours
3. Mel-spectrogram heatmap
4. Russell's 2D Circumplex Model of Affect (Valence vs Arousal)
5. Text vs Multimodal comparison charts
6. Evaluation confusion matrix
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any

from config import CANONICAL_EMOTIONS, EMOTION_VALENCE_AROUSAL

# Apple System Color Tokens
COLOR_MAP = {
    "sadness": "#0A84FF",    # Apple System Blue
    "joy": "#30D158",        # Apple System Green
    "fear": "#BF5AF2",       # Apple System Purple
    "anger": "#FF453A",      # Apple System Red
    "surprise": "#FF9F0A",   # Apple System Orange
    "disgust": "#64D2FF",    # Apple System Teal/Cyan
    "neutral": "#8E8E93"     # Apple System Gray
}


def _get_theme_tokens(theme: str = "dark"):
    """Returns typography, grid, and background tokens for active theme."""
    if theme == "light":
        return {
            "text": "#1D1D1F",
            "muted": "#6E6E73",
            "grid": "rgba(0, 0, 0, 0.06)",
            "border": "rgba(0, 0, 0, 0.10)",
            "plot_bg": "rgba(245, 245, 247, 0.5)",
            "quad_anx": "rgba(255, 59, 48, 0.04)",
            "quad_dep": "rgba(0, 113, 227, 0.05)",
            "quad_joy": "rgba(52, 199, 89, 0.04)",
            "quad_calm": "rgba(255, 149, 0, 0.04)"
        }
    else:  # Dark
        return {
            "text": "#F5F5F7",
            "muted": "#86868B",
            "grid": "rgba(255, 255, 255, 0.06)",
            "border": "rgba(255, 255, 255, 0.12)",
            "plot_bg": "rgba(24, 24, 29, 0.5)",
            "quad_anx": "rgba(255, 69, 58, 0.08)",
            "quad_dep": "rgba(10, 132, 255, 0.08)",
            "quad_joy": "rgba(48, 209, 88, 0.08)",
            "quad_calm": "rgba(255, 159, 10, 0.08)"
        }


def plot_emotion_distribution(
    probs: Dict[str, float],
    title: str = "Emotion Probabilities",
    height: int = 280,
    theme: str = "dark"
) -> go.Figure:
    """
    Renders an Apple-inspired horizontal bar chart of emotion probabilities.
    """
    t = _get_theme_tokens(theme)
    sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0].capitalize() for item in sorted_items]
    values = [item[1] for item in sorted_items]
    colors = [COLOR_MAP.get(item[0].lower(), "#0A84FF") for item in sorted_items]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color=t["border"], width=1)
        ),
        text=[f"{v*100:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(size=11, color=t["text"]),
        cliponaxis=False
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=t["text"], family="-apple-system, BlinkMacSystemFont, sans-serif")),
        xaxis=dict(
            title="",
            range=[0, max(max(values) * 1.25, 0.25)],
            tickformat=".0%",
            gridcolor=t["grid"],
            tickfont=dict(size=10, color=t["muted"]),
            zeroline=False
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=12, color=t["text"], family="-apple-system, BlinkMacSystemFont, sans-serif")
        ),
        margin=dict(l=70, r=40, t=35, b=25),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=t["plot_bg"]
    )
    return fig


def plot_emotion_radar(
    probs: Dict[str, float],
    title: str = "Affective Radar Profile",
    height: int = 300,
    theme: str = "dark"
) -> go.Figure:
    """
    Renders a polar radar chart showing the multi-dimensional emotion distribution.
    """
    t = _get_theme_tokens(theme)
    categories = [e.capitalize() for e in CANONICAL_EMOTIONS]
    values = [probs.get(e, 0.0) for e in CANONICAL_EMOTIONS]
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(10, 132, 255, 0.20)',
        line=dict(color='#0A84FF', width=1.8),
        name="Probability"
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=t["text"])),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(values) * 1.25, 0.25)],
                tickformat=".0%",
                gridcolor=t["grid"],
                tickfont=dict(size=9, color=t["muted"])
            ),
            angularaxis=dict(
                tickfont=dict(size=10, color=t["text"]),
                gridcolor=t["grid"]
            ),
            bgcolor=t["plot_bg"]
        ),
        margin=dict(l=35, r=35, t=45, b=25),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_audio_wave_and_energy(
    audio: np.ndarray,
    sr: int,
    rms_times: np.ndarray,
    rms_values: np.ndarray,
    height: int = 280,
    theme: str = "dark"
) -> go.Figure:
    """
    Plots the audio waveform alongside the RMS energy contour in clean Linear style.
    """
    t = _get_theme_tokens(theme)
    duration = len(audio) / sr
    step = max(1, len(audio) // 2000)
    time_wave = np.linspace(0, duration, len(audio))[::step]
    wave_down = audio[::step]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=("Acoustic Waveform", "RMS Energy Dynamics")
    )

    # Waveform
    fig.add_trace(
        go.Scatter(
            x=time_wave, y=wave_down,
            mode='lines',
            line=dict(color='#0A84FF', width=1),
            name="Waveform"
        ),
        row=1, col=1
    )

    # RMS Energy
    fig.add_trace(
        go.Scatter(
            x=rms_times, y=rms_values,
            mode='lines',
            line=dict(color='#FF9F0A', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(255, 159, 10, 0.12)',
            name="RMS"
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=height,
        margin=dict(l=45, r=20, t=35, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=t["plot_bg"],
        showlegend=False,
        font=dict(family="-apple-system, BlinkMacSystemFont, sans-serif", color=t["text"])
    )
    fig.update_xaxes(title_text="Time (s)", row=2, col=1, gridcolor=t["grid"], tickfont=dict(color=t["muted"], size=10))
    fig.update_yaxes(title_text="Amp", row=1, col=1, gridcolor=t["grid"], tickfont=dict(color=t["muted"], size=10))
    fig.update_yaxes(title_text="RMS", row=2, col=1, gridcolor=t["grid"], tickfont=dict(color=t["muted"], size=10))

    return fig


def plot_pitch_contour(
    pitch_times: np.ndarray,
    pitch_values: np.ndarray,
    f0_mean: float,
    f0_std: float,
    height: int = 250,
    theme: str = "dark"
) -> go.Figure:
    """
    Plots the estimated Fundamental Frequency (F0 / pitch) contour over time.
    """
    t = _get_theme_tokens(theme)
    clean_pitch = [float(p) if not np.isnan(p) and p > 0 else None for p in pitch_values]

    fig = go.Figure()

    # Mean F0 horizontal line
    if f0_mean > 0:
        duration = pitch_times[-1] if len(pitch_times) > 0 else 1.0
        fig.add_trace(go.Scatter(
            x=[0, duration], y=[f0_mean, f0_mean],
            mode='lines',
            line=dict(color=t["muted"], width=1.2, dash='dash'),
            name=f"Mean F0 ({f0_mean:.0f} Hz)"
        ))

        # Variability band
        fig.add_trace(go.Scatter(
            x=[0, duration, duration, 0],
            y=[f0_mean - f0_std, f0_mean - f0_std, f0_mean + f0_std, f0_mean + f0_std],
            fill='toself',
            fillcolor='rgba(142, 142, 147, 0.08)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f"SD (±{f0_std:.0f} Hz)"
        ))

    # Pitch contour
    fig.add_trace(go.Scatter(
        x=pitch_times, y=clean_pitch,
        mode='lines+markers',
        marker=dict(size=2.5, color='#BF5AF2'),
        line=dict(color='#BF5AF2', width=1.6),
        name="Pitch (F0)",
        connectgaps=False
    ))

    fig.update_layout(
        title=dict(text="Vocal Pitch (F0) Tracking Over Time", font=dict(size=13, color=t["text"])),
        xaxis=dict(title="Time (s)", gridcolor=t["grid"], tickfont=dict(color=t["muted"], size=10)),
        yaxis=dict(title="Hz", gridcolor=t["grid"], tickfont=dict(color=t["muted"], size=10)),
        height=height,
        margin=dict(l=45, r=25, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=t["plot_bg"],
        font=dict(family="-apple-system, BlinkMacSystemFont, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, color=t["muted"]))
    )
    return fig


def plot_circumplex_affect(
    modality_points: Dict[str, Dict[str, float]],
    fused_point: Optional[Dict[str, float]] = None,
    height: int = 420,
    theme: str = "dark"
) -> go.Figure:
    """
    Renders Russell's 2D Circumplex Model of Affect (Valence on X vs Arousal on Y).
    Visualizes modality positions, affective dissonance/distance, and combined fusion centroid.
    """
    t = _get_theme_tokens(theme)
    fig = go.Figure()

    # 1. Background Quadrant Colors
    fig.add_shape(type="rect", x0=-1, y0=0, x1=0, y1=1,
                  fillcolor=t["quad_anx"], line=dict(width=0))
    fig.add_shape(type="rect", x0=-1, y0=-1, x1=0, y1=0,
                  fillcolor=t["quad_dep"], line=dict(width=0))
    fig.add_shape(type="rect", x0=0, y0=0, x1=1, y1=1,
                  fillcolor=t["quad_joy"], line=dict(width=0))
    fig.add_shape(type="rect", x0=0, y0=-1, x1=1, y1=0,
                  fillcolor=t["quad_calm"], line=dict(width=0))

    # Center Axes
    fig.add_shape(type="line", x0=-1, y0=0, x1=1, y1=0, line=dict(color=t["border"], width=1, dash="dash"))
    fig.add_shape(type="line", x0=0, y0=-1, x1=0, y1=1, line=dict(color=t["border"], width=1, dash="dash"))

    # Quadrant Annotations
    fig.add_annotation(x=-0.5, y=0.88, text="Agitation / Anxiety",
                       showarrow=False, font=dict(size=10, color="#FF453A", family="-apple-system"))
    fig.add_annotation(x=-0.5, y=-0.88, text="Depression / Flat Affect",
                       showarrow=False, font=dict(size=10, color="#0A84FF", family="-apple-system"))
    fig.add_annotation(x=0.5, y=0.88, text="Joy / High Activation",
                       showarrow=False, font=dict(size=10, color="#30D158", family="-apple-system"))
    fig.add_annotation(x=0.5, y=-0.88, text="Calm / Low Activation",
                       showarrow=False, font=dict(size=10, color="#FF9F0A", family="-apple-system"))

    # Canonical Emotion Anchors
    for emo, coord in EMOTION_VALENCE_AROUSAL.items():
        fig.add_trace(go.Scatter(
            x=[coord["valence"]], y=[coord["arousal"]],
            mode="markers+text",
            marker=dict(size=6, color=t["muted"], symbol="circle-open"),
            text=[emo.capitalize()],
            textposition="top center",
            textfont=dict(size=9, color=t["muted"]),
            hoverinfo="text",
            hovertext=f"{emo.capitalize()} ({coord['description']})",
            showlegend=False
        ))

    # Modality Point Markers
    mod_colors = {
        "text": "#0A84FF",              # Apple Blue
        "speech_prosody": "#FF9F0A",    # Apple Orange
        "facial_expression": "#30D158"  # Apple Green
    }
    mod_names = {
        "text": "Text Modality",
        "speech_prosody": "Prosody Modality",
        "facial_expression": "Facial Modality"
    }

    pts_x = []
    pts_y = []

    for mod_key, coords in modality_points.items():
        v = coords.get("valence", 0.0)
        a = coords.get("arousal", 0.0)
        pts_x.append(v)
        pts_y.append(a)

        fig.add_trace(go.Scatter(
            x=[v], y=[a],
            mode="markers+text",
            marker=dict(
                size=12,
                color=mod_colors.get(mod_key, "#BF5AF2"),
                line=dict(color="#000000" if theme=="dark" else "#FFFFFF", width=1.5)
            ),
            text=[mod_names.get(mod_key, mod_key)],
            textposition="bottom center",
            textfont=dict(size=10, color=t["text"]),
            name=mod_names.get(mod_key, mod_key)
        ))

    # Connection lines for distance/dissonance
    if len(pts_x) >= 2:
        for i in range(len(pts_x)):
            for j in range(i + 1, len(pts_x)):
                fig.add_trace(go.Scatter(
                    x=[pts_x[i], pts_x[j]],
                    y=[pts_y[i], pts_y[j]],
                    mode="lines",
                    line=dict(color=t["border"], width=1, dash="dot"),
                    hoverinfo="none",
                    showlegend=False
                ))

    # Multimodal Fused Point
    if fused_point:
        fv = fused_point.get("valence", 0.0)
        fa = fused_point.get("arousal", 0.0)
        fig.add_trace(go.Scatter(
            x=[fv], y=[fa],
            mode="markers+text",
            marker=dict(
                size=16,
                color="#FF453A",
                symbol="star",
                line=dict(color="#FFFFFF" if theme=="dark" else "#000000", width=1.5)
            ),
            text=["Fused State"],
            textposition="top center",
            textfont=dict(size=11, color=t["text"]),
            name="Multimodal Centroid"
        ))

    fig.update_layout(
        title=dict(
            text="Russell's Circumplex Affective Space (Valence-Arousal)",
            font=dict(size=13, color=t["text"])
        ),
        xaxis=dict(
            title="Valence (Pleasure: Negative ↔ Positive)",
            range=[-1.1, 1.1],
            zeroline=False,
            gridcolor=t["grid"],
            tickfont=dict(color=t["muted"], size=10)
        ),
        yaxis=dict(
            title="Arousal (Activation: Low ↔ High)",
            range=[-1.1, 1.1],
            zeroline=False,
            gridcolor=t["grid"],
            tickfont=dict(color=t["muted"], size=10)
        ),
        height=height,
        margin=dict(l=45, r=35, t=45, b=45),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=t["plot_bg"],
        font=dict(family="-apple-system, BlinkMacSystemFont, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5, font=dict(size=10, color=t["muted"]))
    )
    return fig


def plot_modality_comparison_bar(
    text_probs: Dict[str, float],
    fused_probs: Dict[str, float],
    height: int = 300,
    theme: str = "dark"
) -> go.Figure:
    """
    Renders a grouped bar chart comparing Text-Only vs Multimodal Emotion probabilities.
    """
    t = _get_theme_tokens(theme)
    emotions = [e.capitalize() for e in CANONICAL_EMOTIONS]
    t_vals = [text_probs.get(e, 0.0) for e in CANONICAL_EMOTIONS]
    f_vals = [fused_probs.get(e, 0.0) for e in CANONICAL_EMOTIONS]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=emotions,
        y=t_vals,
        name="Text-Only",
        marker=dict(color="#0A84FF", line=dict(color=t["border"], width=1))
    ))

    fig.add_trace(go.Bar(
        x=emotions,
        y=f_vals,
        name="Multimodal",
        marker=dict(color="#30D158", line=dict(color=t["border"], width=1))
    ))

    fig.update_layout(
        title=dict(text="Text-Only vs Multimodal Distribution", font=dict(size=13, color=t["text"])),
        barmode='group',
        xaxis=dict(title="", gridcolor=t["grid"], tickfont=dict(color=t["text"], size=11)),
        yaxis=dict(title="", range=[0, 1.05], tickformat=".0%", gridcolor=t["grid"], tickfont=dict(color=t["muted"], size=10)),
        height=height,
        margin=dict(l=40, r=20, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=t["plot_bg"],
        font=dict(family="-apple-system, BlinkMacSystemFont, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, color=t["muted"]))
    )
    return fig


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    classes: List[str],
    title: str = "Confusion Matrix",
    theme: str = "dark"
) -> go.Figure:
    """
    Renders a normalized confusion matrix heatmap with numerical annotations.
    """
    t = _get_theme_tokens(theme)
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    text_annotations = []
    for i in range(len(classes)):
        row_ann = []
        for j in range(len(classes)):
            row_ann.append(f"{cm[i, j]}<br>({cm_norm[i, j]*100:.0f}%)")
        text_annotations.append(row_ann)

    colorscale = [[0, "#18181D"], [1, "#0A84FF"]] if theme == "dark" else [[0, "#F5F5F7"], [1, "#0071E3"]]

    fig = go.Figure(data=go.Heatmap(
        z=cm_norm,
        x=[c.capitalize() for c in classes],
        y=[c.capitalize() for c in classes],
        text=text_annotations,
        texttemplate="%{text}",
        colorscale=colorscale,
        showscale=False
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=t["text"])),
        xaxis=dict(title="Predicted", tickangle=-25, tickfont=dict(color=t["text"], size=10)),
        yaxis=dict(title="Ground Truth", autorange="reversed", tickfont=dict(color=t["text"], size=10)),
        height=380,
        margin=dict(l=65, r=20, t=40, b=45),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system, BlinkMacSystemFont, sans-serif")
    )
    return fig
