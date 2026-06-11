"""Streamlit entry point for the Recruitment Analytics Platform.

Run with: streamlit run app/main.py
"""

import sys
from pathlib import Path

# Ensure project root is on the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from config.settings import settings

settings.ensure_directories()

st.set_page_config(
    page_title=settings.app_title,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title(settings.app_title)
st.markdown(
    """
    Welcome to the **AI-Powered Recruitment Analytics Platform**.

    Use the sidebar to navigate between modules. Features are being built
    incrementally — see the README for the development roadmap.
    """
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Phase", "1 — Foundation", help="Config, data loaders, app shell")

with col2:
    st.metric("NLP Module", "Planned", help="Resume parsing and skill extraction")

with col3:
    st.metric("ML Module", "Planned", help="Candidate ranking and matching")

st.divider()

st.subheader("Getting Started")
st.markdown(
    """
    1. Place sample data in `data/raw/` (see `candidates.csv` and `job_postings.csv`)
    2. Open **Dashboard** from the sidebar to preview loaded data
    3. Build out NLP, ML, and analytics modules phase by phase
    """
)

with st.sidebar:
    st.header("Navigation")
    st.page_link("main.py", label="Home", icon="🏠")
    st.page_link("pages/01_dashboard.py", label="Dashboard", icon="📈")
    st.page_link("pages/02_resume_analysis.py", label="Resume Analysis", icon="📄")
    st.page_link("pages/03_candidate_ranking.py", label="Candidate Ranking", icon="🎯")
    st.page_link("pages/04_analytics.py", label="Analytics", icon="📊")
