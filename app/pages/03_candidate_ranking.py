"""Candidate Ranking page — Phase 3: ML-based candidate–job matching."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.components.sidebar import render_sidebar_header

st.set_page_config(page_title="Candidate Ranking", page_icon="🎯", layout="wide")

render_sidebar_header()

st.title("Candidate Ranking")
st.info(
    "Phase 3 — Feature engineering and ranking models will be integrated here. "
    "Candidates will be scored against selected job postings."
)

st.markdown(
    """
    **Planned capabilities:**
    - Select a job posting
    - Rank candidates by relevance score
    - Explain top matches (skill overlap, experience fit)
    """
)
