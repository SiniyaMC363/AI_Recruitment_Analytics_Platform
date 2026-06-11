"""Candidate Ranking page — Phase 3: ML-based suitability prediction."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.components.sidebar import render_sidebar_header
from src.ml.feature_engineering import (
    build_candidate_features,
    build_features_from_skill_match,
    education_level_to_score,
)
from src.ml.predictor import model_is_available, predict_candidate

st.set_page_config(page_title="Candidate Ranking", page_icon="🎯", layout="wide")

render_sidebar_header()

st.title("Candidate Ranking")
st.markdown(
    "Predict candidate suitability using a trained Random Forest model. "
    "Enter details manually or reuse outputs from Resume Analysis."
)

if not model_is_available():
    st.error(
        "No trained model found. Run `python -m src.ml.train_model` from the project "
        "root to generate data and train the candidate ranker."
    )
    st.stop()

EDUCATION_OPTIONS = ["PhD", "Master", "Bachelor", "Associate", "High School", "Other"]

manual_tab, nlp_tab = st.tabs(["Manual Entry", "From Resume Analysis"])

run_manual_prediction = False
run_nlp_prediction = False

# Defaults used when no NLP session data is available.
default_name = "John Doe"
default_skill_match = 65.0
default_matched = 3
default_skills = 6
default_experience = 3.0
default_education = "Bachelor"

nlp_data = st.session_state.get("ranking_candidate_data")
if nlp_data:
    default_name = nlp_data.get("candidate_name", default_name)
    default_skill_match = float(nlp_data.get("skill_match_percentage", default_skill_match))
    default_matched = int(nlp_data.get("matched_skills_count", default_matched))
    default_skills = int(nlp_data.get("skills_count", default_skills))
    default_experience = float(nlp_data.get("experience_years", default_experience))
    default_education = nlp_data.get("education_level", default_education)

with manual_tab:
    st.subheader("Enter Candidate Details")
    form_col1, form_col2 = st.columns(2)

    with form_col1:
        candidate_name = st.text_input("Candidate name", value=default_name)
        skill_match = st.slider(
            "Skill match percentage",
            min_value=0.0,
            max_value=100.0,
            value=float(default_skill_match),
            step=0.5,
        )
        matched_skills_count = st.number_input(
            "Matched skills count",
            min_value=0,
            max_value=20,
            value=int(default_matched),
            step=1,
        )

    with form_col2:
        skills_count = st.number_input(
            "Total candidate skills",
            min_value=0,
            max_value=20,
            value=int(default_skills),
            step=1,
        )
        experience_years = st.number_input(
            "Years of experience",
            min_value=0.0,
            max_value=40.0,
            value=float(default_experience),
            step=0.5,
        )
        education_level = st.selectbox(
            "Education level",
            options=EDUCATION_OPTIONS,
            index=EDUCATION_OPTIONS.index(default_education)
            if default_education in EDUCATION_OPTIONS
            else 2,
        )

    run_manual_prediction = st.button("Predict Suitability", type="primary", key="manual_predict")

with nlp_tab:
    st.subheader("Use NLP Resume Analysis Results")

    if nlp_data:
        st.success("Resume analysis data loaded from session.")
        st.json(nlp_data)

        nlp_name = st.text_input("Candidate name", value=default_name, key="nlp_name")
        nlp_experience = st.number_input(
            "Years of experience (estimate if not in resume)",
            min_value=0.0,
            max_value=40.0,
            value=float(default_experience),
            step=0.5,
            key="nlp_experience",
        )
        nlp_education = st.selectbox(
            "Education level",
            options=EDUCATION_OPTIONS,
            index=EDUCATION_OPTIONS.index(default_education)
            if default_education in EDUCATION_OPTIONS
            else 2,
            key="nlp_education",
        )
        run_nlp_prediction = st.button("Predict from NLP Data", type="primary", key="nlp_predict")
    else:
        st.info(
            "No resume analysis data found. Open **Resume Analysis**, run an analysis, "
            "then click **Send to Candidate Ranking**."
        )

# --- Run prediction ---
prediction = None

if run_manual_prediction:
    features = build_candidate_features(
        skill_match_percentage=skill_match,
        matched_skills_count=matched_skills_count,
        skills_count=skills_count,
        experience_years=experience_years,
        education_level=education_level,
    )
    prediction = predict_candidate(candidate_name=candidate_name, features=features)

if run_nlp_prediction and nlp_data:
    features = build_features_from_skill_match(
        match_percentage=float(nlp_data.get("skill_match_percentage", 0)),
        matched_skills=nlp_data.get("matched_skills", []),
        extracted_skills=nlp_data.get("extracted_skills", []),
        experience_years=nlp_experience,
        education_level=nlp_education,
    )
    prediction = predict_candidate(candidate_name=nlp_name, features=features)

if prediction:
    st.divider()
    st.subheader("Prediction Results")

    header_col1, header_col2, header_col3 = st.columns(3)
    header_col1.metric("Candidate", prediction.candidate_name)
    header_col2.metric("Predicted Category", prediction.predicted_category)
    header_col3.metric("Confidence", f"{prediction.confidence:.1f}%")

    with st.container(border=True):
        st.markdown("#### Feature Values Used by the Model")
        feature_cols = st.columns(5)
        feature_cols[0].metric("Skill Match", f"{prediction.features['skill_match_score']}%")
        feature_cols[1].metric("Matched Skills", prediction.features["matched_skills_count"])
        feature_cols[2].metric("Total Skills", prediction.features["skills_count"])
        feature_cols[3].metric("Experience", f"{prediction.features['experience_years']} yrs")
        feature_cols[4].metric(
            "Education Score",
            f"{prediction.features['education_score']:.2f}",
        )

    explain_col1, explain_col2 = st.columns([1.2, 1])

    with explain_col1:
        st.markdown("#### Explanation")
        st.markdown(prediction.explanation)

    with explain_col2:
        st.markdown("#### Class Probabilities")
        for label, probability in sorted(
            prediction.probabilities.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            st.progress(
                min(probability / 100, 1.0),
                text=f"{label}: {probability:.1f}%",
            )

else:
    with st.expander("Preview education score mapping"):
        for level in EDUCATION_OPTIONS:
            st.write(f"- **{level}** → {education_level_to_score(level):.2f}")
