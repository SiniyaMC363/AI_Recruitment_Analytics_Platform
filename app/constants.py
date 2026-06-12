"""Shared constants for the Streamlit application."""

EDUCATION_OPTIONS = [
    "PhD",
    "Master",
    "Bachelor",
    "Associate",
    "High School",
    "Other",
]

APP_TAGLINE = "Intelligent Recruitment Analytics Platform"

NAV_PAGES = [
    {"path": "main.py", "label": "Home", "icon": "🏠"},
    {"path": "pages/01_dashboard.py", "label": "Dashboard", "icon": "📈"},
    {"path": "pages/02_resume_analysis.py", "label": "Resume Analysis", "icon": "📄"},
    {"path": "pages/03_candidate_ranking.py", "label": "Candidate Ranking", "icon": "🎯"},
    {"path": "pages/04_analytics.py", "label": "Analytics", "icon": "📊"},
]

CATEGORY_COLORS = {
    "Highly Suitable": "#059669",
    "Suitable": "#2563EB",
    "Less Suitable": "#DC2626",
}
