"""Predefined skill vocabulary for keyword-based extraction."""

# Canonical skill names used for matching and display.
PREDEFINED_SKILLS: list[str] = [
    "Python",
    "Java",
    "SQL",
    "Machine Learning",
    "Deep Learning",
    "Data Analysis",
    "Power BI",
    "Excel",
    "Tableau",
    "Git",
    "Docker",
    "AWS",
    "Azure",
    "Cloud Computing",
]

# Lowercase lookup map for fast normalization.
SKILL_LOOKUP: dict[str, str] = {skill.lower(): skill for skill in PREDEFINED_SKILLS}
