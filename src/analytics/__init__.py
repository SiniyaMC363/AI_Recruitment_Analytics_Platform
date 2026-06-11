"""Recruitment analytics and reporting.

Phase 4 module — KPIs, data analysis, and dashboard exports.
"""

from src.analytics.data_analyzer import (
    ANALYTICS_COLUMNS,
    ANALYTICS_DATASET_FILENAME,
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
from src.analytics.metrics import compute_hiring_funnel, compute_time_to_hire
from src.analytics.report_generator import (
    analytics_to_csv_string,
    export_analytics_to_csv,
    generate_analytics_report,
)
from src.analytics.reports import generate_summary_report

__all__ = [
    "ANALYTICS_COLUMNS",
    "ANALYTICS_DATASET_FILENAME",
    "analytics_to_csv_string",
    "build_analytics_dataset",
    "calculate_all_kpis",
    "compute_hiring_funnel",
    "compute_time_to_hire",
    "export_analytics_to_csv",
    "filter_analytics_data",
    "generate_analytics_report",
    "generate_summary_report",
    "get_category_distribution",
    "get_education_comparison",
    "get_experience_distribution",
    "get_score_distribution",
    "get_top_candidates",
    "load_analytics_dataset",
    "save_analytics_dataset",
]
