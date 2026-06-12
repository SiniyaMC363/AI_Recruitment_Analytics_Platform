"""Resume Analysis page — NLP-powered resume parsing and skill matching."""

from app.bootstrap import ensure_project_root

ensure_project_root()

import streamlit as st

from app.components.sidebar import render_sidebar_header
from app.components.ui import (
    inject_custom_css,
    render_empty_state,
    render_friendly_error,
    render_kpi_row,
    render_page_header,
    render_section_divider,
    render_skill_tags,
)
from app.constants import EDUCATION_OPTIONS
from src.nlp.preprocessing import preprocess_text
from src.nlp.resume_parser import extract_text_from_pdf
from src.nlp.skill_extractor import extract_skills, match_skills, parse_skills_input
from src.nlp.skills_database import PREDEFINED_SKILLS

st.set_page_config(
    page_title="Resume Analysis | AI Recruit Pro",
    page_icon="📄",
    layout="wide",
)

inject_custom_css()
render_sidebar_header()

render_page_header(
    "Resume Analysis",
    subtitle="Upload a PDF resume or paste text to extract skills and compare against job requirements.",
    icon="📄",
)

input_col, skills_col = st.columns([1.2, 1])

with input_col:
    st.markdown("##### Resume Input")
    uploaded_file = st.file_uploader(
        "Upload PDF resume",
        type=["pdf"],
        help="Supported format: PDF. Text-based PDFs work best; scanned images may not extract well.",
    )

    pasted_text = st.text_area(
        "Or paste resume text",
        placeholder="Paste resume content here if you do not have a PDF file...",
        height=180,
        help="Plain-text resumes can be pasted directly for analysis.",
    )

with skills_col:
    st.markdown("##### Job Requirements")
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
        help="Skills mentioned in the job description will be merged with your manual list.",
    )

raw_text = ""
extraction_error = None
page_count = 0

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        pdf_result = extract_text_from_pdf(uploaded_file)
    if pdf_result.success:
        raw_text = pdf_result.text
        page_count = pdf_result.page_count
        st.success(f"PDF processed successfully — {page_count} page(s) extracted.")
    else:
        extraction_error = pdf_result.error

if pasted_text.strip():
    raw_text = pasted_text.strip()

if extraction_error:
    render_friendly_error(
        f"PDF extraction failed: {extraction_error}",
        suggestion="Try pasting the resume text directly, or use a text-based PDF export.",
    )

if not raw_text:
    render_empty_state(
        "Upload a PDF or paste resume text to begin analysis.",
        icon="📄",
    )
    with st.expander("Supported skills database"):
        st.write(", ".join(PREDEFINED_SKILLS))
    st.stop()

try:
    with st.spinner("Running NLP analysis..."):
        cleaned_text = preprocess_text(raw_text)
        extracted_skills = extract_skills(raw_text)

        required_skills = parse_skills_input(job_skills_input)
        if job_description.strip():
            detected_job_skills = extract_skills(job_description)
            merged = {skill.lower(): skill for skill in required_skills}
            for skill in detected_job_skills:
                merged.setdefault(skill.lower(), skill)
            required_skills = sorted(merged.values(), key=str.lower)

        match_result = match_skills(raw_text, required_skills)
except Exception:
    render_friendly_error(
        "NLP processing encountered an error.",
        suggestion="Ensure the spaCy model is installed: python -m spacy download en_core_web_sm",
    )
    st.stop()

render_section_divider("Analysis Summary")

render_kpi_row(
    [
        {
            "label": "Skills Found",
            "value": match_result.skill_count,
            "help": "Total predefined skills detected in the resume",
        },
        {
            "label": "Required Skills",
            "value": len(match_result.required_skills),
            "help": "Skills specified in job requirements",
        },
        {
            "label": "Matched Skills",
            "value": len(match_result.matched_skills),
            "help": "Required skills found in the resume",
        },
        {
            "label": "Match Score",
            "value": f"{match_result.match_percentage}%",
            "help": "Percentage of required skills matched",
        },
    ]
)

if page_count:
    st.caption(f"PDF pages processed: {page_count}")

preview_col1, preview_col2 = st.columns(2)

with preview_col1:
    st.markdown("##### Resume Text Preview")
    st.text_area(
        "Original text",
        value=raw_text[:3000] + ("..." if len(raw_text) > 3000 else ""),
        height=220,
        disabled=True,
        label_visibility="collapsed",
    )

with preview_col2:
    st.markdown("##### Cleaned Resume Preview")
    st.text_area(
        "Preprocessed text",
        value=cleaned_text[:3000] + ("..." if len(cleaned_text) > 3000 else ""),
        height=220,
        disabled=True,
        label_visibility="collapsed",
    )

render_section_divider("Skill Analysis")

skills_col1, skills_col2, skills_col3 = st.columns(3)

with skills_col1:
    st.markdown("##### Extracted Skills")
    render_skill_tags(extracted_skills)

with skills_col2:
    st.markdown("##### Matched Skills")
    render_skill_tags(match_result.matched_skills, tag_class="skill-tag skill-tag-matched")

with skills_col3:
    st.markdown("##### Missing Skills")
    render_skill_tags(match_result.missing_skills, tag_class="skill-tag skill-tag-missing")

if match_result.required_skills:
    render_section_divider("Skill Match Progress")
    st.progress(
        min(match_result.match_percentage / 100, 1.0),
        text=f"{match_result.match_percentage}% of required skills found in resume",
    )
else:
    st.info("Add required skills or a job description to calculate match percentage.")

with st.expander("View full predefined skills database"):
    st.write(", ".join(PREDEFINED_SKILLS))

render_section_divider("Send to Candidate Ranking")
st.caption("Transfer this analysis to the ML ranking module for suitability prediction.")

ranking_col1, ranking_col2 = st.columns([1, 1])
with ranking_col1:
    candidate_name = st.text_input(
        "Candidate name for ranking",
        value="John Doe",
        key="ranking_candidate_name",
        help="Display name used in prediction results.",
    )
with ranking_col2:
    experience_years = st.number_input(
        "Estimated years of experience",
        min_value=0.0,
        max_value=40.0,
        value=3.0,
        step=0.5,
        key="ranking_experience_years",
        help="Estimate if not clearly stated in the resume.",
    )

education_level = st.selectbox(
    "Education level",
    options=EDUCATION_OPTIONS,
    index=2,
    key="ranking_education_level",
    help="Highest completed degree or equivalent.",
)

if st.button("Send to Candidate Ranking", type="primary"):
    st.session_state["ranking_candidate_data"] = {
        "candidate_name": candidate_name,
        "skill_match_percentage": match_result.match_percentage,
        "matched_skills_count": len(match_result.matched_skills),
        "skills_count": match_result.skill_count,
        "matched_skills": match_result.matched_skills,
        "extracted_skills": extracted_skills,
        "experience_years": experience_years,
        "education_level": education_level,
    }
    st.success(
        "Candidate data saved. Open **Candidate Ranking** from the sidebar to run ML prediction."
    )
