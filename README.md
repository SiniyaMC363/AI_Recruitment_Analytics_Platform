# AI-Powered Recruitment Analytics Platform

A modular Python platform for recruitment analytics using NLP, machine learning, and Streamlit.

## Project Structure

```
AI_Recruitment_Analytics/
├── app/                    # Streamlit application
├── config/                 # Configuration and settings
├── data/                   # Data storage (raw, processed, external)
├── models/                 # Trained ML model artifacts
├── notebooks/              # Exploratory analysis
├── src/                    # Core source code
│   ├── analytics/          # Metrics and reporting
│   ├── data/               # Data loading and validation
│   ├── ml/                 # Feature engineering and models
│   ├── nlp/                # Text processing and extraction
│   └── utils/              # Shared utilities
└── tests/                  # Unit tests
```

## Module Roadmap

Build and integrate features in this order:

| Phase | Module | Description |
|-------|--------|-------------|
| 1 | **Foundation** | Config, data loaders, Streamlit shell |
| 2 | **NLP** | Resume parsing, text preprocessing, skill extraction |
| 3 | **ML** | Feature engineering, candidate ranking models |
| 4 | **Analytics** | KPIs, dashboards, hiring funnel metrics |
| 5 | **Integration** | End-to-end workflows in the Streamlit app |

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Copy `.env.example` to `.env` and adjust paths as needed.

## Run the App

```bash
streamlit run app/main.py
```

## Development

- Add new logic under `src/` — keep UI code in `app/`
- Place raw datasets in `data/raw/`
- Save trained models to `models/`
- Run tests: `pytest tests/`
