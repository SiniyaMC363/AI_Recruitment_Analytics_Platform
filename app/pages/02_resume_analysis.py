"""Resume Analysis page — Phase 2: NLP-powered resume parsing."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.components.sidebar import render_sidebar_header
from src.nlp.preprocessing import clean_text
from src.nlp.skill_extractor import extract_skills

st.set_page_config(page_title="Resume Analysis", page_icon="📄", layout="wide")

render_sidebar_header()

st.title("Resume Analysis")
st.info("Phase 2 — NLP resume parsing and skill extraction will be built here.")

resume_text = st.text_area(
    "Paste resume text",
    placeholder="Enter resume content to preview basic text cleaning...",
    height=200,
)

known_skills = st.text_input(
    "Known skills (comma-separated)",
    value="python, machine learning, sql, nlp, streamlit",
)

if resume_text:
    cleaned = clean_text(resume_text)
    skills = extract_skills(
        cleaned,
        known_skills=[s.strip() for s in known_skills.split(",") if s.strip()],
    )

    st.subheader("Cleaned Text")
    st.text(cleaned[:500] + ("..." if len(cleaned) > 500 else ""))

    st.subheader("Matched Skills")
    if skills:
        st.write(", ".join(skills))
    else:
        st.caption("No skills matched yet. Full extraction comes in Phase 2.")
