"""Resume Analysis page — Phase 2: NLP-powered resume parsing and skill matching."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.components.sidebar import render_sidebar_header
from src.nlp.preprocessing import preprocess_text
from src.nlp.resume_parser import extract_text_from_pdf
from src.nlp.skill_extractor import extract_skills, match_skills, parse_skills_input
from src.nlp.skills_database import PREDEFINED_SKILLS

st.set_page_config(page_title="Resume Analysis", page_icon="📄", layout="wide")

render_sidebar_header()

st.title("Resume Analysis")
st.markdown(
    "Upload a PDF resume or paste resume text to extract skills and compare them "
    "against job requirements."
)

# --- Input section ---
input_col, skills_col = st.columns([1.2, 1])

with input_col:
    st.subheader("Resume Input")
    uploaded_file = st.file_uploader(
        "Upload PDF resume",
        type=["pdf"],
        help="Supported format: PDF. Text-based PDFs work best.",
    )

    pasted_text = st.text_area(
        "Or paste resume text",
        placeholder="Paste resume content here if you do not have a PDF file...",
        height=180,
    )

with skills_col:
    st.subheader("Job Requirements")
    job_skills_input = st.text_area(
        "Required skills",
        value="Python, SQL, Machine Learning, Git, AWS",
        help="Enter comma-separated skills required for the job.",
        height=100,
    )
    job_description = st.text_area(
        "Job description (optional)",
        placeholder="Paste a job description to auto-detect required skills...",
        height=80,
    )

# --- Process resume text ---
raw_text = ""
extraction_error = None
page_count = 0

if uploaded_file is not None:
    pdf_result = extract_text_from_pdf(uploaded_file)
    if pdf_result.success:
        raw_text = pdf_result.text
        page_count = pdf_result.page_count
    else:
        extraction_error = pdf_result.error

if pasted_text.strip():
    # Pasted text takes precedence when both inputs are provided.
    raw_text = pasted_text.strip()

if extraction_error:
    st.error(f"PDF extraction failed: {extraction_error}")

if not raw_text:
    st.info("Upload a PDF or paste resume text to begin analysis.")
    with st.expander("Supported skills database"):
        st.write(", ".join(PREDEFINED_SKILLS))
    st.stop()

# --- NLP processing ---
cleaned_text = preprocess_text(raw_text)
extracted_skills = extract_skills(raw_text)

required_skills = parse_skills_input(job_skills_input)
if job_description.strip():
    # Merge manually entered skills with skills detected from the job description.
    detected_job_skills = extract_skills(job_description)
    merged = {skill.lower(): skill for skill in required_skills}
    for skill in detected_job_skills:
        merged.setdefault(skill.lower(), skill)
    required_skills = sorted(merged.values(), key=str.lower)

match_result = match_skills(raw_text, required_skills)

# --- Summary metrics ---
st.divider()
st.subheader("Analysis Summary")

metric_cols = st.columns(4)
metric_cols[0].metric("Skills Found", match_result.skill_count)
metric_cols[1].metric("Required Skills", len(match_result.required_skills))
metric_cols[2].metric("Matched Skills", len(match_result.matched_skills))
metric_cols[3].metric("Match Score", f"{match_result.match_percentage}%")

if page_count:
    st.caption(f"PDF pages processed: {page_count}")

# --- Text previews ---
preview_col1, preview_col2 = st.columns(2)

with preview_col1:
    st.subheader("Resume Text Preview")
    st.text_area(
        "Original text",
        value=raw_text[:3000] + ("..." if len(raw_text) > 3000 else ""),
        height=220,
        disabled=True,
        label_visibility="collapsed",
    )

with preview_col2:
    st.subheader("Cleaned Resume Preview")
    st.text_area(
        "Preprocessed text",
        value=cleaned_text[:3000] + ("..." if len(cleaned_text) > 3000 else ""),
        height=220,
        disabled=True,
        label_visibility="collapsed",
    )

# --- Skill results ---
st.divider()
skills_col1, skills_col2, skills_col3 = st.columns(3)

with skills_col1:
    st.subheader("Extracted Skills")
    if extracted_skills:
        for skill in extracted_skills:
            st.markdown(f"- **{skill}**")
    else:
        st.caption("No predefined skills detected in this resume.")

with skills_col2:
    st.subheader("Matched Skills")
    if match_result.matched_skills:
        for skill in match_result.matched_skills:
            st.success(skill)
    else:
        st.caption("No required skills matched yet.")

with skills_col3:
    st.subheader("Missing Skills")
    if match_result.missing_skills:
        for skill in match_result.missing_skills:
            st.warning(skill)
    else:
        st.caption("No missing skills — great match!")

# --- Match progress bar ---
if match_result.required_skills:
    st.subheader("Skill Match Progress")
    st.progress(
        min(match_result.match_percentage / 100, 1.0),
        text=f"{match_result.match_percentage}% of required skills found in resume",
    )
else:
    st.info("Add required skills or a job description to calculate match percentage.")

with st.expander("View full predefined skills database"):
    st.write(", ".join(PREDEFINED_SKILLS))
