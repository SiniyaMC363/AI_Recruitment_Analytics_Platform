"""Analytics page — Phase 4: hiring funnel and recruitment KPIs."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.components.sidebar import render_sidebar_header

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")

render_sidebar_header()

st.title("Recruitment Analytics")
st.info(
    "Phase 4 — Hiring funnel, time-to-hire, and source effectiveness "
    "visualizations will be added here."
)

st.markdown(
    """
    **Planned capabilities:**
    - Hiring funnel by stage
    - Time-to-hire trends
    - Source quality comparison
    - Exportable summary reports
    """
)
