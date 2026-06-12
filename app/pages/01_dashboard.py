"""Dashboard page — data overview and summary metrics."""

from app.bootstrap import ensure_project_root

ensure_project_root()

import streamlit as st

from app.components.sidebar import render_sidebar_header
from app.components.ui import (
    inject_custom_css,
    render_empty_state,
    render_friendly_error,
    render_kpi_row,
    render_page_header,
    render_section_divider,
)
from app.services.cached_loaders import load_candidates_cached, load_job_postings_cached
from config.settings import settings
from src.analytics.reports import generate_summary_report

st.set_page_config(page_title="Dashboard | AI Recruit Pro", page_icon="📈", layout="wide")

inject_custom_css()
render_sidebar_header()

render_page_header(
    "Dashboard",
    subtitle="Overview of candidates and job postings in your recruitment pipeline.",
    icon="📈",
)

settings.ensure_directories()

candidates_path = settings.data_raw_dir / "candidates.csv"
jobs_path = settings.data_raw_dir / "job_postings.csv"

if not candidates_path.exists() or not jobs_path.exists():
    missing = []
    if not candidates_path.exists():
        missing.append("`candidates.csv`")
    if not jobs_path.exists():
        missing.append("`job_postings.csv`")
    render_empty_state(
        f"Sample data not found. Add {', '.join(missing)} to `data/raw/`.",
        icon="📂",
    )
    st.info(
        "Place CSV files with the required columns in the data/raw/ folder. "
        "See the README for file format details."
    )
    st.stop()

try:
    with st.spinner("Loading recruitment data..."):
        candidates = load_candidates_cached(str(candidates_path))
        jobs = load_job_postings_cached(str(jobs_path))
except FileNotFoundError as exc:
    render_friendly_error(
        "We couldn't find the data files needed for this page.",
        suggestion=str(exc),
    )
    st.stop()
except ValueError as exc:
    render_friendly_error(
        "The data files are missing required columns.",
        suggestion=f"Check your CSV format: {exc}",
    )
    st.stop()
except Exception:
    render_friendly_error(
        "Something went wrong while loading the data.",
        suggestion="Verify that your CSV files are valid and not open in another program.",
    )
    st.stop()

report = generate_summary_report(candidates, jobs)

render_section_divider("Summary Metrics")
render_kpi_row(
    [
        {
            "label": "Total Candidates",
            "value": report["total_candidates"],
            "help": "Number of candidate records loaded from data/raw/candidates.csv",
        },
        {
            "label": "Total Job Postings",
            "value": report["total_job_postings"],
            "help": "Number of job posting records loaded from data/raw/job_postings.csv",
        },
        {
            "label": "Data Status",
            "value": "Ready",
            "help": "Both datasets loaded and validated successfully",
        },
    ]
)

render_section_divider("Candidates")
st.caption(f"Showing {len(candidates)} candidate record(s).")
st.dataframe(candidates, use_container_width=True, hide_index=True)

render_section_divider("Job Postings")
st.caption(f"Showing {len(jobs)} job posting(s).")
st.dataframe(jobs, use_container_width=True, hide_index=True)
