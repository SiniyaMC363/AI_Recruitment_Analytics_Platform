"""Candidate Ranking page — ML-based suitability prediction."""

from app.bootstrap import ensure_project_root

ensure_project_root()

import streamlit as st

from app.components.sidebar import render_sidebar_header
from app.components.ui import (
    confidence_label,
    inject_custom_css,
    render_empty_state,
    render_friendly_error,
    render_kpi_row,
    render_page_header,
    render_section_divider,
)
from app.constants import EDUCATION_OPTIONS
from src.ml.feature_engineering import (
    build_candidate_features,
    build_features_from_skill_match,
    education_level_to_score,
)
from src.ml.predictor import model_is_available, predict_candidate

st.set_page_config(
    page_title="Candidate Ranking | AI Recruit Pro",
    page_icon="🎯",
    layout="wide",
)

inject_custom_css()
render_sidebar_header()

render_page_header(
    "Candidate Ranking",
    subtitle="Predict candidate suitability using a trained Random Forest model.",
    icon="🎯",
)

if not model_is_available():
    render_friendly_error(
        "No trained model found. The ranking module requires a trained model artifact.",
        suggestion="Run `python -m src.ml.train_model` from the project root to generate "
        "training data and train the candidate ranker.",
    )
    st.stop()

manual_tab, nlp_tab = st.tabs(["Manual Entry", "From Resume Analysis"])

run_manual_prediction = False
run_nlp_prediction = False

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
    st.markdown("##### Enter Candidate Details")
    st.caption("Provide skill match metrics and background information for the model.")

    form_col1, form_col2 = st.columns(2)

    with form_col1:
        candidate_name = st.text_input(
            "Candidate name",
            value=default_name,
            help="Display name shown in prediction results.",
        )
        skill_match = st.slider(
            "Skill match percentage",
            min_value=0.0,
            max_value=100.0,
            value=float(default_skill_match),
            step=0.5,
            help="Percentage of required job skills matched by the candidate.",
        )
        matched_skills_count = st.number_input(
            "Matched skills count",
            min_value=0,
            max_value=20,
            value=int(default_matched),
            step=1,
            help="Number of required skills found in the candidate profile.",
        )

    with form_col2:
        skills_count = st.number_input(
            "Total candidate skills",
            min_value=0,
            max_value=20,
            value=int(default_skills),
            step=1,
            help="Total skills detected in the candidate profile.",
        )
        experience_years = st.number_input(
            "Years of experience",
            min_value=0.0,
            max_value=40.0,
            value=float(default_experience),
            step=0.5,
            help="Total relevant professional experience in years.",
        )
        education_level = st.selectbox(
            "Education level",
            options=EDUCATION_OPTIONS,
            index=EDUCATION_OPTIONS.index(default_education)
            if default_education in EDUCATION_OPTIONS
            else 2,
            help="Highest completed degree or equivalent.",
        )

    run_manual_prediction = st.button("Predict Suitability", type="primary", key="manual_predict")

with nlp_tab:
    st.markdown("##### Use NLP Resume Analysis Results")

    if nlp_data:
        st.success("Resume analysis data loaded from session.")
        with st.expander("View loaded NLP data"):
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
        render_empty_state(
            "No resume analysis data found. Run an analysis on the Resume Analysis page first.",
            icon="📄",
        )
        st.info(
            "Open **Resume Analysis**, upload a resume, then click "
            "**Send to Candidate Ranking** to transfer results here."
        )

prediction = None

if run_manual_prediction:
    try:
        with st.spinner("Running suitability prediction..."):
            features = build_candidate_features(
                skill_match_percentage=skill_match,
                matched_skills_count=matched_skills_count,
                skills_count=skills_count,
                experience_years=experience_years,
                education_level=education_level,
            )
            prediction = predict_candidate(candidate_name=candidate_name, features=features)
    except Exception:
        render_friendly_error(
            "Prediction failed.",
            suggestion="Ensure the model artifact exists in the models/ directory.",
        )

if run_nlp_prediction and nlp_data:
    try:
        with st.spinner("Running suitability prediction from NLP data..."):
            features = build_features_from_skill_match(
                match_percentage=float(nlp_data.get("skill_match_percentage", 0)),
                matched_skills=nlp_data.get("matched_skills", []),
                extracted_skills=nlp_data.get("extracted_skills", []),
                experience_years=nlp_experience,
                education_level=nlp_education,
            )
            prediction = predict_candidate(candidate_name=nlp_name, features=features)
    except Exception:
        render_friendly_error(
            "Prediction from NLP data failed.",
            suggestion="Return to Resume Analysis and resend candidate data.",
        )

if prediction:
    render_section_divider("Prediction Results")

    conf_label = confidence_label(prediction.confidence)
    render_kpi_row(
        [
            {"label": "Candidate", "value": prediction.candidate_name},
            {"label": "Predicted Category", "value": prediction.predicted_category},
            {
                "label": "Confidence",
                "value": f"{prediction.confidence:.1f}%",
                "help": conf_label,
            },
        ]
    )

    st.caption(f"Assessment: **{conf_label}** — higher confidence means the model is more certain.")

    with st.container(border=True):
        st.markdown("##### Feature Values Used by the Model")
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
        st.markdown("##### Explanation for Recruiters")
        st.markdown(prediction.explanation)
        st.caption(
            "Categories: **Highly Suitable** (strong match), "
            "**Suitable** (moderate match), **Less Suitable** (weak match)."
        )

    with explain_col2:
        st.markdown("##### Class Probabilities")
        st.caption("How likely the model considers each suitability category.")
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
        st.caption("Education levels are converted to numeric scores for the ML model.")
        for level in EDUCATION_OPTIONS:
            st.write(f"- **{level}** → {education_level_to_score(level):.2f}")
