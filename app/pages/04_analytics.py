"""Analytics page — Phase 4: recruitment KPI dashboard and reporting."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import plotly.express as px
import streamlit as st

from app.components.sidebar import render_sidebar_header
from config.settings import settings
from src.analytics.data_analyzer import (
    build_analytics_dataset,
    filter_analytics_data,
    get_category_distribution,
    get_education_comparison,
    get_experience_distribution,
    get_score_distribution,
    get_top_candidates,
    load_analytics_dataset,
    save_analytics_dataset,
)
from src.analytics.kpi_calculator import calculate_all_kpis
from src.analytics.report_generator import analytics_to_csv_string, generate_analytics_report
from src.data.loaders import load_candidates, load_job_postings
from src.ml.feature_engineering import SUITABILITY_LABELS
from src.ml.predictor import model_is_available

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")

render_sidebar_header()

st.title("Recruitment Analytics Dashboard")
st.markdown(
    "Explore candidate suitability trends, skill alignment, and experience "
    "patterns across your recruitment pipeline."
)

settings.ensure_directories()

# --- Load source data ---
candidates_path = settings.data_raw_dir / "candidates.csv"
jobs_path = settings.data_raw_dir / "job_postings.csv"

if not candidates_path.exists():
    st.warning("Candidate data not found. Add `candidates.csv` to `data/raw/`.")
    st.stop()

try:
    candidates = load_candidates()
    jobs = load_job_postings() if jobs_path.exists() else None
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.stop()


@st.cache_data(show_spinner="Building analytics dataset...")
def get_analytics_data(refresh_token: int):
    """Build or load the processed analytics dataset."""
    saved = load_analytics_dataset()
    if not saved.empty and refresh_token == 0:
        return saved

    analytics_df = build_analytics_dataset(candidates, job_postings=jobs)
    save_analytics_dataset(analytics_df)
    return analytics_df


if "analytics_refresh_token" not in st.session_state:
    st.session_state["analytics_refresh_token"] = 0

refresh_col, model_col = st.columns([1, 2])
with refresh_col:
    if st.button("Refresh Analytics Data", type="secondary"):
        st.session_state["analytics_refresh_token"] += 1
        get_analytics_data.clear()
        st.rerun()

with model_col:
    if model_is_available():
        st.caption("ML model loaded — predictions use the trained Random Forest ranker.")
    else:
        st.caption(
            "ML model not found — using rule-based suitability labels. "
            "Train with: `python -m src.ml.train_model`"
        )

analytics_df = get_analytics_data(st.session_state["analytics_refresh_token"])

if analytics_df.empty:
    st.info("No candidate records available for analytics.")
    st.stop()

# --- Sidebar filters ---
st.sidebar.header("Filters")
selected_categories = st.sidebar.multiselect(
    "Candidate category",
    options=SUITABILITY_LABELS,
    default=SUITABILITY_LABELS,
)
min_experience = st.sidebar.slider(
    "Minimum experience (years)",
    min_value=0.0,
    max_value=25.0,
    value=0.0,
    step=0.5,
)
min_skill_match = st.sidebar.slider(
    "Minimum skill match score",
    min_value=0.0,
    max_value=100.0,
    value=0.0,
    step=5.0,
)

filtered_df = filter_analytics_data(
    analytics_df,
    categories=selected_categories,
    min_experience=min_experience,
    min_skill_match=min_skill_match,
)

if filtered_df.empty:
    st.warning("No candidates match the selected filters.")
    st.stop()

# --- KPI cards ---
kpis = calculate_all_kpis(filtered_df)

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
kpi_col1.metric("Total Candidates", kpis["total_candidates"])
kpi_col2.metric("Highly Suitable", kpis["highly_suitable_count"])
kpi_col3.metric("Avg Match Score", f"{kpis['average_skill_match_score']}%")
kpi_col4.metric("Avg Experience", f"{kpis['average_experience_years']} yrs")

with st.expander("Additional KPIs"):
    extra_col1, extra_col2, extra_col3 = st.columns(3)
    extra_col1.metric("Suitable Candidates", kpis["suitable_count"])
    extra_col2.metric("Less Suitable Candidates", kpis["less_suitable_count"])
    extra_col3.metric("Avg Skills Detected", kpis["average_skills_count"])

    pct_col1, pct_col2, pct_col3 = st.columns(3)
    pct_col1.metric("Highly Suitable %", f"{kpis['highly_suitable_percentage']}%")
    pct_col2.metric("Suitable %", f"{kpis['suitable_percentage']}%")
    pct_col3.metric("Less Suitable %", f"{kpis['less_suitable_percentage']}%")

st.divider()

# --- Charts ---
chart_row1_col1, chart_row1_col2 = st.columns(2)

category_distribution = get_category_distribution(filtered_df)
category_fig = px.bar(
    category_distribution,
    x="category",
    y="count",
    color="category",
    title="Candidate Category Distribution",
    labels={"category": "Suitability Category", "count": "Candidates"},
    color_discrete_map={
        "Highly Suitable": "#2ecc71",
        "Suitable": "#3498db",
        "Less Suitable": "#e74c3c",
    },
)
category_fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Count")
chart_row1_col1.plotly_chart(category_fig, use_container_width=True)

score_distribution = get_score_distribution(filtered_df, bins=8)
score_fig = px.bar(
    score_distribution,
    x="score_range",
    y="count",
    title="Skill Match Score Distribution",
    labels={"score_range": "Score Range", "count": "Candidates"},
    color_discrete_sequence=["#8e44ad"],
)
score_fig.update_layout(xaxis_tickangle=-30)
chart_row1_col2.plotly_chart(score_fig, use_container_width=True)

chart_row2_col1, chart_row2_col2 = st.columns(2)

experience_distribution = get_experience_distribution(filtered_df, bins=6)
experience_fig = px.bar(
    experience_distribution,
    x="experience_range",
    y="count",
    title="Experience Distribution",
    labels={"experience_range": "Experience Range (years)", "count": "Candidates"},
    color_discrete_sequence=["#16a085"],
)
experience_fig.update_layout(xaxis_tickangle=-30)
chart_row2_col1.plotly_chart(experience_fig, use_container_width=True)

education_comparison = get_education_comparison(filtered_df)
education_fig = px.bar(
    education_comparison,
    x="education_level",
    y="avg_match_score",
    text="avg_match_score",
    title="Education Level vs Average Match Score",
    labels={
        "education_level": "Education Level",
        "avg_match_score": "Average Match Score (%)",
    },
    color="education_level",
)
education_fig.update_traces(texttemplate="%{text}%", textposition="outside")
education_fig.update_layout(showlegend=False)
chart_row2_col2.plotly_chart(education_fig, use_container_width=True)

top_candidates = get_top_candidates(filtered_df, top_n=min(10, len(filtered_df)))
ranking_fig = px.bar(
    top_candidates,
    x="skill_match_score",
    y="candidate_name",
    orientation="h",
    color="predicted_suitability_category",
    title="Top Candidate Ranking by Skill Match Score",
    labels={
        "skill_match_score": "Skill Match Score (%)",
        "candidate_name": "Candidate",
        "predicted_suitability_category": "Category",
    },
    color_discrete_map={
        "Highly Suitable": "#2ecc71",
        "Suitable": "#3498db",
        "Less Suitable": "#e74c3c",
    },
)
ranking_fig.update_layout(yaxis={"categoryorder": "total ascending"})
st.plotly_chart(ranking_fig, use_container_width=True)

st.divider()

# --- Candidate table ---
st.subheader("Candidate Analytics Table")
st.dataframe(filtered_df, use_container_width=True, hide_index=True)

# --- Reporting / export ---
st.subheader("Export Report")
report = generate_analytics_report(filtered_df)
st.caption(
    f"Report generated at {report['generated_at_utc']} UTC — "
    f"{report['record_count']} candidate record(s)."
)

csv_data = analytics_to_csv_string(filtered_df)
st.download_button(
    label="Download Analytics CSV",
    data=csv_data,
    file_name="candidate_analytics_report.csv",
    mime="text/csv",
    type="primary",
)
