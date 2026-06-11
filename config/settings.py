"""Central configuration for paths, environment variables, and constants."""

from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

APP_TITLE = os.getenv("APP_TITLE", "AI Recruitment Analytics Platform")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

DATA_RAW_DIR = PROJECT_ROOT / os.getenv("DATA_RAW_DIR", "data/raw")
DATA_PROCESSED_DIR = PROJECT_ROOT / os.getenv("DATA_PROCESSED_DIR", "data/processed")
DATA_EXTERNAL_DIR = PROJECT_ROOT / "data/external"
MODELS_DIR = PROJECT_ROOT / os.getenv("MODELS_DIR", "models")
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")


class Settings:
    """Grouped settings object for convenient imports."""

    project_root = PROJECT_ROOT
    app_title = APP_TITLE
    debug = DEBUG
    data_raw_dir = DATA_RAW_DIR
    data_processed_dir = DATA_PROCESSED_DIR
    data_external_dir = DATA_EXTERNAL_DIR
    models_dir = MODELS_DIR
    notebooks_dir = NOTEBOOKS_DIR
    spacy_model = SPACY_MODEL

    def ensure_directories(self) -> None:
        """Create required data and model directories if they do not exist."""
        for directory in (
            self.data_raw_dir,
            self.data_processed_dir,
            self.data_external_dir,
            self.models_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
