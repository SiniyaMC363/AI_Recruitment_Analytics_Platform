"""Streamlit entry point for AI Recruit Pro.

Run with: streamlit run app/main.py
"""
from bootstrap import ensure_project_root
ensure_project_root()

from config.settings import settings

from app.components.ui import (
    inject_custom_css,
    render_feature_highlights,
    render_hero,
    render_kpi_row,
    render_section_divider,
    render_sidebar_navigation,
    render_tech_stack,
    render_workflow_steps,
)

import streamlit as st

settings.ensure_directories()

st.set_page_config(
    page_title=settings.app_title,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()
render_sidebar_navigation()

render_hero(settings.app_title)

st.markdown(
    """
    **AI Recruit Pro** helps recruiters and hiring teams analyze resumes, match
    candidates to job requirements, rank suitability with machine learning, and
    visualize recruitment metrics — all in one interactive platform.
    """
)

render_section_divider("Platform Status")

render_kpi_row(
    [
        {
            "label": "Foundation",
            "value": "Complete",
            "help": "Configuration, data loaders, and application architecture",
        },
        {
            "label": "NLP Module",
            "value": "Complete",
            "help": "Resume parsing, preprocessing, and skill extraction",
        },
        {
            "label": "ML Module",
            "value": "Complete",
            "help": "Random Forest candidate ranking and suitability prediction",
        },
        {
            "label": "Analytics",
            "value": "Complete",
            "help": "KPI dashboard, visualizations, filters, and CSV export",
        },
    ]
)

render_section_divider("End-to-End Workflow")
render_workflow_steps()

render_section_divider("Feature Highlights")
render_feature_highlights()

render_section_divider("Technology Stack")
render_tech_stack()

render_section_divider("Get Started")

st.markdown(
    """
    1. **Dashboard** — Preview loaded candidates and job postings
    2. **Resume Analysis** — Upload a PDF or paste text to extract and match skills
    3. **Candidate Ranking** — Run ML suitability predictions with confidence scores
    4. **Analytics** — Explore KPIs, charts, and export reports as CSV

    Use the sidebar to navigate to any module.
    """
)
