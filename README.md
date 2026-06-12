# AI Recruit Pro — Intelligent Recruitment Analytics Platform

An end-to-end recruitment analytics platform that combines NLP resume parsing, machine learning candidate ranking, and interactive dashboards — built with Python and Streamlit.

## Overview

Recruiters spend significant time manually reviewing resumes and comparing candidates against job requirements. **AI Recruit Pro** automates key parts of this workflow: it extracts skills from resumes, matches them to job requirements, ranks candidates with a trained Random Forest model, and presents actionable insights through an interactive analytics dashboard.

This project demonstrates a modular data science pipeline — from text processing and feature engineering to model deployment and reporting — packaged in a professional, portfolio-ready Streamlit application.

## Features

| Module | Description |
|--------|-------------|
| **Resume Parsing** | Extract text from PDF resumes using pdfplumber; support plain-text paste input |
| **NLP Skill Extraction** | Identify technical skills via spaCy preprocessing and a curated skills database |
| **Job Matching** | Compare candidate skills against job requirements with match percentage scoring |
| **ML Candidate Ranking** | Random Forest model predicts suitability (Highly Suitable / Suitable / Less Suitable) with confidence scores |
| **Analytics Dashboard** | Interactive KPIs, Plotly charts, sidebar filters, and candidate tables |
| **CSV Reporting** | Export filtered analytics data for offline review and sharing |

## Technology Stack

| Technology | Role |
|------------|------|
| **Python** | Core language |
| **Streamlit** | Web application framework |
| **spaCy / NLTK** | NLP text preprocessing and tokenization |
| **Scikit-learn** | Random Forest classification model |
| **Pandas** | Data manipulation and CSV handling |
| **Plotly** | Interactive chart visualizations |
| **pdfplumber** | PDF resume text extraction |
| **joblib** | Model serialization |

## Project Architecture

```
AI_Recruitment_Analytics/
├── app/                        # Streamlit application layer
│   ├── main.py                 # Landing page and entry point
│   ├── bootstrap.py            # Project path setup
│   ├── constants.py            # Shared UI constants
│   ├── components/             # Reusable UI components (sidebar, theme)
│   ├── pages/                  # Streamlit multi-page modules
│   │   ├── 01_dashboard.py
│   │   ├── 02_resume_analysis.py
│   │   ├── 03_candidate_ranking.py
│   │   └── 04_analytics.py
│   └── services/               # Cached data/model loaders
├── config/                     # Environment and path configuration
├── data/
│   ├── raw/                    # Source CSV datasets
│   ├── processed/              # Generated analytics datasets
│   └── external/               # External data sources
├── models/                     # Trained ML model artifacts (.pkl)
├── src/                        # Core business logic
│   ├── analytics/              # KPI calculation, reporting, data analysis
│   ├── data/                   # CSV loaders and validation
│   ├── ml/                     # Feature engineering, training, prediction
│   ├── nlp/                    # Resume parsing, preprocessing, skill extraction
│   └── utils/                  # Shared helper utilities
├── tests/                      # Unit tests (pytest)
└── .streamlit/                 # Streamlit theme configuration
```

## Installation Instructions

### Prerequisites

- Python 3.10 or later
- Windows PowerShell (or any terminal)

### Setup (Windows)

```powershell
# Clone the repository
git clone https://github.com/your-username/AI_Recruitment_Analytics.git
cd AI_Recruitment_Analytics

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm

# Copy environment configuration (optional)
copy .env.example .env
```

### Train the ML Model

The candidate ranking module requires a trained model artifact:

```powershell
python -m src.ml.train_model
```

This generates synthetic training data, trains a Random Forest classifier, and saves the model to `models/candidate_ranker.pkl`.

## How to Use

Start the application:

```powershell
streamlit run app/main.py
```

### Home (Landing Page)

Overview of the platform, feature highlights, technology stack, and end-to-end workflow.

### Dashboard

Preview loaded candidates and job postings from `data/raw/`. Summary metrics show total record counts and data readiness status.

### Resume Analysis

1. Upload a PDF resume or paste resume text
2. Enter required skills or paste a job description
3. Review extracted skills, matched/missing skills, and match score
4. Optionally send results to Candidate Ranking via session state

### Candidate Ranking

1. Enter candidate details manually, or use data from Resume Analysis
2. Click **Predict Suitability** to run the Random Forest model
3. Review predicted category, confidence score, feature values, and recruiter-friendly explanation

### Analytics Dashboard

1. Explore KPI cards (total candidates, suitability breakdown, averages)
2. Use sidebar filters for category, experience, and skill match score
3. Review interactive charts (category distribution, score histograms, experience, education)
4. Download filtered results as CSV

## Machine Learning Workflow

```
Input Data → Feature Engineering → Random Forest Model → Suitability Prediction
```

**Input features:**
- Skill match score (%)
- Matched skills count
- Total skills count
- Years of experience
- Education score (mapped from degree level)

**Output labels:**
- Highly Suitable
- Suitable
- Less Suitable

The model is trained on synthetically generated candidate profiles with rule-based labels. In production, this would be replaced with real labeled hiring outcomes.

## Screenshots

> Add screenshots to a `docs/screenshots/` folder and update the paths below.

| Page | Placeholder |
|------|-------------|
| Homepage | `docs/screenshots/homepage.png` |
| Resume Analysis | `docs/screenshots/resume_analysis.png` |
| Candidate Ranking | `docs/screenshots/candidate_ranking.png` |
| Analytics Dashboard | `docs/screenshots/analytics_dashboard.png` |

## Development

```powershell
# Run unit tests
pytest tests/

# Train or retrain the ML model
python -m src.ml.train_model
```

- Add business logic under `src/` — keep UI code in `app/`
- Place raw datasets in `data/raw/`
- Save trained models to `models/`
- Processed analytics are written to `data/processed/`

## Deployment Checklist

Use this checklist before pushing to GitHub or deploying to [Streamlit Community Cloud](https://streamlit.io/cloud):

- [ ] Virtual environment is **not** committed (`venv/` in `.gitignore`)
- [ ] `.env` file is **not** committed (secrets excluded)
- [ ] `requirements.txt` is complete and up to date
- [ ] spaCy model download step is documented in README
- [ ] ML model training step is documented (or model artifact strategy decided)
- [ ] Sample data exists in `data/raw/`
- [ ] All tests pass: `pytest tests/`
- [ ] Application runs locally: `streamlit run app/main.py`
- [ ] Screenshots captured for portfolio README
- [ ] `.gitignore` excludes `__pycache__/`, `*.pkl`, processed data, and temp files

### Streamlit Cloud Deployment

1. Push the repository to GitHub
2. Connect the repo at [share.streamlit.io](https://share.streamlit.io)
3. Set main file path to `app/main.py`
4. Add a `packages.txt` or post-install script if spaCy model download is needed:
   ```
   # In requirements or setup script
   python -m spacy download en_core_web_sm
   ```
5. Train the model locally and commit strategy: either run training in a setup script or document that users must train locally

## Future Improvements

- Real-world resume datasets for training and evaluation
- Deep learning-based semantic matching (e.g., sentence transformers)
- Database integration (PostgreSQL / SQLite) for persistent candidate storage
- User authentication and role-based access control
- Cloud deployment with CI/CD pipeline (GitHub Actions)
- DOCX resume support and multi-format document ingestion
- A/B testing of ranking models and recruiter feedback loops
- Integration with ATS (Applicant Tracking System) APIs

## License

This project is intended for portfolio and educational use.
