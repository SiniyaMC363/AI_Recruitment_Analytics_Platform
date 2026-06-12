"""Shared sidebar components for Streamlit pages."""

from app.components.ui import render_sidebar_navigation


def render_sidebar_header() -> None:
    """Render a consistent sidebar header with navigation across pages."""
    render_sidebar_navigation()
