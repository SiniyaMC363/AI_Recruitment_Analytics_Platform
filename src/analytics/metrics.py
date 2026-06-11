"""Recruitment KPI and funnel metric calculations.

Phase 4: Implement hiring funnel, time-to-hire, and source effectiveness.
"""

import pandas as pd


def compute_hiring_funnel(applications: pd.DataFrame) -> pd.DataFrame:
    """Compute stage counts for the hiring funnel.

    Args:
        applications: Application records with a 'stage' column.

    Returns:
        DataFrame with stage counts.

    Raises:
        NotImplementedError: Funnel metrics are planned for Phase 4.
    """
    raise NotImplementedError("Hiring funnel metrics will be implemented in Phase 4.")


def compute_time_to_hire(applications: pd.DataFrame) -> pd.Series:
    """Compute time-to-hire per successful application.

    Args:
        applications: Application records with date columns.

    Returns:
        Series of time-to-hire values in days.

    Raises:
        NotImplementedError: Time-to-hire metrics are planned for Phase 4.
    """
    raise NotImplementedError("Time-to-hire metrics will be implemented in Phase 4.")
