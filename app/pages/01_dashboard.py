"""Dashboard page — Phase 1: data overview and summary metrics."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.components.sidebar import render_sidebar_header
from config.settings import settings
from src.analytics.reports import generate_summary_report
from src.data.loaders import load_candidates, load_job_postings

st.set_page_config(page_title="Dashboard", page_icon="📈", layout="wide")

render_sidebar_header()

st.title("Dashboard")
st.markdown("Overview of candidates and job postings.")

settings.ensure_directories()

candidates_path = settings.data_raw_dir / "candidates.csv"
jobs_path = settings.data_raw_dir / "job_postings.csv"

if not candidates_path.exists() or not jobs_path.exists():
    st.warning(
        "Sample data not found. Ensure `candidates.csv` and `job_postings.csv` "
        "exist in `data/raw/`."
    )
    st.stop()

try:
    candidates = load_candidates()
    jobs = load_job_postings()
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

report = generate_summary_report(candidates, jobs)

col1, col2 = st.columns(2)
col1.metric("Total Candidates", report["total_candidates"])
col2.metric("Total Job Postings", report["total_job_postings"])

st.subheader("Candidates")
st.dataframe(candidates, use_container_width=True)

st.subheader("Job Postings")
st.dataframe(jobs, use_container_width=True)
