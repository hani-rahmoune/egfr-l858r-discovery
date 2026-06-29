"""
Presentation-only theming for the Streamlit dashboard.

A single ``inject()`` call drops a light, minimal style block into the page:
white canvas, soft rounded cards, one calm teal accent, quiet alert styling, and
a tidy sidebar. This module contains NO behaviour, NO data access, and NO page
logic. It is imported once by app.py and applied at the top of main(), so every
page (including the Discovery Copilot) inherits the same look.

Selectors target Streamlit's stable ``data-testid`` attributes (Streamlit 1.5x)
and degrade gracefully: if a selector does not match a future build, the page
still renders, it just falls back to the default theme for that element.
"""

from __future__ import annotations

import streamlit as st

# Accent + neutral palette (kept in sync with .streamlit/config.toml)
_ACCENT = "#2C7A7B"  # calm teal
_ACCENT_DARK = "#246668"  # hover/pressed
_INK = "#16242E"  # headings
_TEXT = "#1E2A32"  # body
_MUTED = "#5B6B76"  # captions / secondary
_BORDER = "#E7ECF0"  # hairline borders
_SURFACE = "#FFFFFF"  # card surface
_CANVAS_SOFT = "#F7F9FB"  # sidebar / subtle fills
_SHADOW = "0 1px 2px rgba(16,40,60,0.04)"

_CSS = f"""
<style>
/* ── Typography ──────────────────────────────────────────────────────────── */
html, body, [class*="css"], .stMarkdown, .stButton, input, textarea, button, select {{
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
        Helvetica, Arial, sans-serif;
}}
h1, h2, h3, h4 {{
    color: {_INK};
    font-weight: 650;
    letter-spacing: -0.01em;
}}
h1 {{ font-size: 1.95rem; }}
h2 {{ font-size: 1.4rem; margin-top: 0.3rem; }}
h3 {{ font-size: 1.08rem; color: #2A3A45; }}
.stMarkdown p, .stMarkdown li {{ color: {_TEXT}; line-height: 1.62; }}
[data-testid="stCaptionContainer"], .stCaption {{ color: {_MUTED} !important; }}

/* ── Generous whitespace + comfortable reading width ─────────────────────── */
.stMain .block-container {{
    padding-top: 2.6rem;
    padding-bottom: 4rem;
    max-width: 1180px;
}}

/* ── Buttons: rounded, restrained, teal accent ───────────────────────────── */
.stButton > button, .stDownloadButton > button {{
    border-radius: 9px;
    border: 1px solid {_BORDER};
    background: {_SURFACE};
    color: #25323B;
    font-weight: 550;
    padding: 0.46rem 1.05rem;
    transition: border-color .15s ease, color .15s ease, box-shadow .15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    border-color: {_ACCENT};
    color: {_ACCENT};
    box-shadow: {_SHADOW};
}}
.stButton > button:focus, .stDownloadButton > button:focus {{
    box-shadow: 0 0 0 3px rgba(44,122,123,0.18);
}}
.stButton > button[kind="primary"] {{
    background: {_ACCENT};
    border-color: {_ACCENT};
    color: #FFFFFF;
}}
.stButton > button[kind="primary"]:hover {{
    background: {_ACCENT_DARK};
    border-color: {_ACCENT_DARK};
    color: #FFFFFF;
}}

/* ── Metric tiles as soft cards ──────────────────────────────────────────── */
[data-testid="stMetric"] {{
    background: {_SURFACE};
    border: 1px solid {_BORDER};
    border-radius: 12px;
    padding: 1rem 1.1rem;
    box-shadow: {_SHADOW};
}}
[data-testid="stMetricLabel"] p {{ color: {_MUTED}; font-weight: 500; }}
[data-testid="stMetricValue"] {{ color: {_INK}; font-weight: 650; }}

/* ── Expanders = card panels (Evidence / Warnings) ───────────────────────── */
[data-testid="stExpander"] {{
    border: 1px solid {_BORDER};
    border-radius: 12px;
    background: {_SURFACE};
    box-shadow: {_SHADOW};
    overflow: hidden;
}}
[data-testid="stExpander"] summary {{
    font-weight: 550;
    color: #2A3A45;
    padding: 0.15rem 0;
}}
[data-testid="stExpander"] summary:hover {{ color: {_ACCENT}; }}

/* ── Chat message = Answer card ──────────────────────────────────────────── */
[data-testid="stChatMessage"] {{
    background: {_SURFACE};
    border: 1px solid {_BORDER};
    border-radius: 14px;
    padding: 0.5rem 1.1rem;
    box-shadow: {_SHADOW};
}}
[data-testid="stChatInput"] {{ border-radius: 12px; }}

/* ── Alerts: quiet + elegant, keep semantic hue, soften the box ──────────── */
[data-testid="stAlert"] {{
    border-radius: 10px;
    border: 1px solid {_BORDER};
    border-left-width: 4px;
    box-shadow: none;
    padding: 0.85rem 1rem;
}}
[data-testid="stAlert"] p {{ font-size: 0.95rem; }}

/* ── Tables / dataframes: clean rounded frame ────────────────────────────── */
[data-testid="stDataFrame"], [data-testid="stTable"] {{
    border: 1px solid {_BORDER};
    border-radius: 10px;
    overflow: hidden;
}}

/* ── Inputs ──────────────────────────────────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-baseweb="input"], [data-baseweb="textarea"] {{
    border-radius: 8px !important;
}}
[data-testid="stCode"], pre {{ border-radius: 8px; }}

/* ── Sidebar: soft surface, hairline divider, tidy spacing ───────────────── */
section[data-testid="stSidebar"] {{
    background: {_CANVAS_SOFT};
    border-right: 1px solid {_BORDER};
}}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1.6rem; }}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
    font-weight: 550;
    color: #34434E;
}}
section[data-testid="stSidebar"] hr {{ border-color: {_BORDER}; }}

/* ── Misc polish ─────────────────────────────────────────────────────────── */
hr {{ border-color: {_BORDER}; }}
[data-testid="stHeader"] {{ background: transparent; }}
</style>
"""


def inject() -> None:
    """Inject the dashboard style block. Presentation only, safe to call once."""
    st.markdown(_CSS, unsafe_allow_html=True)
