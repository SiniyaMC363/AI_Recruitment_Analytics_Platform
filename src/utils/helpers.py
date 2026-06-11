"""General-purpose helper utilities."""

import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Configure basic logging for scripts and notebooks."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a decimal ratio as a percentage string."""
    return f"{value * 100:.{decimals}f}%"
