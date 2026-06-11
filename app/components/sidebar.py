"""Shared sidebar components for Streamlit pages."""

import streamlit as st

from config.settings import settings


def render_sidebar_header() -> None:
    """Render a consistent sidebar header across pages."""
    with st.sidebar:
        st.title(settings.app_title)
        st.caption("Modular recruitment analytics")
