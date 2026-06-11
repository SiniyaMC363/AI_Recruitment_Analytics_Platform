"""Report generation and export utilities for recruitment analytics."""

from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from src.analytics.data_analyzer import ANALYTICS_COLUMNS, get_analytics_dataset_path
from src.analytics.kpi_calculator import calculate_all_kpis


def export_analytics_to_csv(
    analytics_df: pd.DataFrame,
    output_path: Path | None = None,
) -> Path:
    """Export the analytics dataset to a CSV file.

    Args:
        analytics_df: Processed candidate analytics data.
        output_path: Optional custom output path.

    Returns:
        Path to the written CSV file.
    """
    path = output_path or get_analytics_dataset_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    export_df = _prepare_export_dataframe(analytics_df)
    export_df.to_csv(path, index=False)
    return path


def analytics_to_csv_string(analytics_df: pd.DataFrame) -> str:
    """Convert analytics data to a CSV string for Streamlit downloads."""
    export_df = _prepare_export_dataframe(analytics_df)
    buffer = StringIO()
    export_df.to_csv(buffer, index=False)
    return buffer.getvalue()


def generate_analytics_report(analytics_df: pd.DataFrame) -> dict[str, Any]:
    """Generate a structured analytics summary report.

    Args:
        analytics_df: Processed candidate analytics data.

    Returns:
        Dictionary containing KPIs and export metadata.
    """
    kpis = calculate_all_kpis(analytics_df)
    generated_at = datetime.now(timezone.utc).isoformat()

    return {
        "generated_at_utc": generated_at,
        "record_count": len(analytics_df) if analytics_df is not None else 0,
        "columns": ANALYTICS_COLUMNS,
        "kpis": kpis,
        "status": "ready",
    }


def _prepare_export_dataframe(analytics_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure exported data uses the standard analytics column order."""
    if analytics_df is None or analytics_df.empty:
        return pd.DataFrame(columns=ANALYTICS_COLUMNS)

    export_df = analytics_df.copy()
    for column in ANALYTICS_COLUMNS:
        if column not in export_df.columns:
            export_df[column] = pd.NA

    return export_df[ANALYTICS_COLUMNS]
