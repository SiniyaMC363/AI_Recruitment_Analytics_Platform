"""Recruitment analytics and reporting.

Phase 4 module — KPIs, funnel metrics, and dashboard data.
"""

from src.analytics.metrics import compute_hiring_funnel, compute_time_to_hire
from src.analytics.reports import generate_summary_report

__all__ = ["compute_hiring_funnel", "compute_time_to_hire", "generate_summary_report"]
