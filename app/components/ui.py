"""Shared Streamlit UI components, styling, and layout helpers."""

from __future__ import annotations

import streamlit as st

from app.constants import APP_TAGLINE, CATEGORY_COLORS, NAV_PAGES
from config.settings import settings


def inject_custom_css() -> None:
    """Apply lightweight custom CSS for a consistent professional theme."""
    st.markdown(
        """
        <style>
        /* Page header styling */
        .app-hero {
            background: linear-gradient(135deg, #1E3A5F 0%, #2563EB 100%);
            padding: 2rem 2.5rem;
            border-radius: 12px;
            color: white;
            margin-bottom: 1.5rem;
        }
        .app-hero h1 {
            color: white !important;
            font-size: 2rem !important;
            margin-bottom: 0.25rem !important;
        }
        .app-hero p {
            color: rgba(255,255,255,0.9) !important;
            font-size: 1.05rem;
            margin: 0;
        }

        /* Section headers */
        .section-header {
            border-left: 4px solid #2563EB;
            padding-left: 0.75rem;
            margin: 1.5rem 0 1rem 0;
        }
        .section-header h3 {
            margin: 0;
            color: #1E293B;
        }
        .section-header p {
            margin: 0.25rem 0 0 0;
            color: #64748B;
            font-size: 0.9rem;
        }

        /* KPI cards */
        div[data-testid="stMetric"] {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 0.75rem 1rem;
        }
        div[data-testid="stMetric"] label {
            color: #64748B !important;
            font-size: 0.85rem !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #1E293B !important;
            font-weight: 600 !important;
        }

        /* Feature cards on landing page */
        .feature-card {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 1.25rem;
            height: 100%;
        }
        .feature-card h4 {
            color: #1E293B;
            margin: 0 0 0.5rem 0;
        }
        .feature-card p {
            color: #64748B;
            font-size: 0.9rem;
            margin: 0;
        }

        /* Workflow steps */
        .workflow-step {
            text-align: center;
            padding: 1rem;
        }
        .workflow-step .step-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        .workflow-step .step-title {
            font-weight: 600;
            color: #1E293B;
        }
        .workflow-step .step-desc {
            color: #64748B;
            font-size: 0.85rem;
        }
        .workflow-arrow {
            display: flex;
            align-items: center;
            justify-content: center;
            color: #94A3B8;
            font-size: 1.5rem;
        }

        /* Skill tags */
        .skill-tag {
            display: inline-block;
            background: #EFF6FF;
            color: #1D4ED8;
            border: 1px solid #BFDBFE;
            border-radius: 16px;
            padding: 0.25rem 0.75rem;
            margin: 0.2rem;
            font-size: 0.85rem;
        }
        .skill-tag-matched {
            background: #ECFDF5;
            color: #047857;
            border-color: #A7F3D0;
        }
        .skill-tag-missing {
            background: #FEF2F2;
            color: #B91C1C;
            border-color: #FECACA;
        }

        /* Sidebar branding */
        .sidebar-brand {
            font-size: 0.95rem;
            color: #64748B;
            margin-bottom: 0.5rem;
        }

        /* Empty state */
        .empty-state {
            text-align: center;
            padding: 2rem;
            color: #64748B;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, subtitle: str = "", icon: str = "") -> None:
    """Render a consistent page header with optional icon and subtitle."""
    display_title = f"{icon} {title}" if icon else title
    st.markdown(
        f"""
        <div class="section-header">
            <h3>{display_title}</h3>
            {"<p>" + subtitle + "</p>" if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str = APP_TAGLINE) -> None:
    """Render the landing page hero banner."""
    st.markdown(
        f"""
        <div class="app-hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_divider(label: str = "") -> None:
    """Render a labeled section divider."""
    if label:
        st.markdown(f"##### {label}")
    st.divider()


def render_kpi_row(items: list[dict]) -> None:
    """Render a row of KPI metric cards.

    Each item dict supports: label, value, delta (optional), help (optional).
    """
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            st.metric(
                label=item["label"],
                value=item["value"],
                delta=item.get("delta"),
                help=item.get("help"),
            )


def render_skill_tags(skills: list[str], tag_class: str = "skill-tag") -> None:
    """Render skills as styled HTML tags."""
    if not skills:
        st.caption("None detected.")
        return
    tags_html = "".join(f'<span class="{tag_class}">{skill}</span>' for skill in skills)
    st.markdown(tags_html, unsafe_allow_html=True)


def render_sidebar_navigation() -> None:
    """Render consistent sidebar branding and page navigation."""
    with st.sidebar:
        st.markdown(f"### {settings.app_title}")
        st.markdown(
            f'<p class="sidebar-brand">{APP_TAGLINE}</p>',
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown("**Navigation**")
        for page in NAV_PAGES:
            st.page_link(page["path"], label=page["label"], icon=page["icon"])


def render_empty_state(message: str, icon: str = "📭") -> None:
    """Display a friendly empty-state message."""
    st.markdown(
        f"""
        <div class="empty-state">
            <div style="font-size: 2.5rem;">{icon}</div>
            <p>{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_friendly_error(message: str, suggestion: str = "") -> None:
    """Display a user-friendly error with an optional recovery suggestion."""
    st.error(message)
    if suggestion:
        st.info(f"💡 **Suggestion:** {suggestion}")


def render_workflow_steps() -> None:
    """Render the end-to-end recruitment workflow on the landing page."""
    steps = [
        ("📄", "Resume Upload", "Import PDF or paste resume text"),
        ("🔍", "NLP Analysis", "Extract skills and match to job requirements"),
        ("🎯", "Candidate Ranking", "ML-powered suitability prediction"),
        ("📊", "Analytics Dashboard", "KPIs, charts, and CSV reporting"),
    ]
    cols = st.columns(len(steps) * 2 - 1)
    col_idx = 0
    for i, (icon, title, desc) in enumerate(steps):
        with cols[col_idx]:
            st.markdown(
                f"""
                <div class="workflow-step">
                    <div class="step-icon">{icon}</div>
                    <div class="step-title">{title}</div>
                    <div class="step-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        col_idx += 1
        if i < len(steps) - 1:
            with cols[col_idx]:
                st.markdown(
                    '<div class="workflow-arrow">→</div>',
                    unsafe_allow_html=True,
                )
            col_idx += 1


def render_feature_highlights() -> None:
    """Render feature highlight cards on the landing page."""
    features = [
        ("📄", "Resume Parsing", "Extract text from PDF resumes and preprocess content for analysis."),
        ("🧠", "NLP Skill Extraction", "Identify technical skills using spaCy and a curated skills database."),
        ("🔗", "Job Matching", "Compare candidate skills against job requirements with match scoring."),
        ("🤖", "ML Candidate Ranking", "Random Forest model predicts suitability with confidence scores."),
        ("📊", "Analytics Dashboard", "Interactive KPIs, charts, filters, and candidate insights."),
        ("📥", "CSV Reporting", "Export filtered analytics data for offline review and sharing."),
    ]
    row1 = features[:3]
    row2 = features[3:]
    for row in (row1, row2):
        cols = st.columns(3)
        for col, (icon, title, desc) in zip(cols, row):
            with col:
                st.markdown(
                    f"""
                    <div class="feature-card">
                        <h4>{icon} {title}</h4>
                        <p>{desc}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_tech_stack() -> None:
    """Render the technology stack section on the landing page."""
    stack = [
        ("Python", "Core language"),
        ("Streamlit", "Web application"),
        ("spaCy / NLTK", "NLP processing"),
        ("Scikit-learn", "Machine learning"),
        ("Pandas", "Data manipulation"),
        ("Plotly", "Interactive charts"),
    ]
    cols = st.columns(3)
    for i, (tech, role) in enumerate(stack):
        with cols[i % 3]:
            st.markdown(f"**{tech}** — {role}")


def get_category_color_map() -> dict[str, str]:
    """Return the standard suitability category color map for charts."""
    return dict(CATEGORY_COLORS)


def confidence_label(confidence: float) -> str:
    """Return a human-readable confidence level label."""
    if confidence >= 80:
        return "High confidence"
    if confidence >= 60:
        return "Moderate confidence"
    return "Low confidence — review manually"


def category_badge_color(category: str) -> str:
    """Return the hex color for a suitability category badge."""
    return CATEGORY_COLORS.get(category, "#64748B")
