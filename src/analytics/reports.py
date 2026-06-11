"""Report generation for recruitment analytics.

Phase 4: Export summaries as HTML, PDF, or structured JSON.
"""

from typing import Any

import pandas as pd


def generate_summary_report(
    candidates: pd.DataFrame,
    job_postings: pd.DataFrame,
) -> dict[str, Any]:
    """Generate a high-level recruitment summary report.

    Args:
        candidates: Candidate records.
        job_postings: Job posting records.

    Returns:
        Summary statistics dictionary.
    """
    return {
        "total_candidates": len(candidates),
        "total_job_postings": len(job_postings),
        "status": "Phase 4 — full reporting not yet implemented",
    }
