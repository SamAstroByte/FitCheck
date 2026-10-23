"""Create sample resume PDFs and job-description text files in samples/.

The four resumes are built to cover the interesting cases when scored against
the sample Data Scientist job description:
  strong      -> covers almost every required skill
  partial     -> covers about half
  web dev     -> unrelated field
  switcher    -> web developer who has started learning data science
"""
import random
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from generate_data import build_resume, jd_text
from skills import CATEGORY_SKILLS

OUT = Path("samples")
HEADINGS = {"SUMMARY", "SKILLS", "EXPERIENCE", "PROJECTS", "EDUCATION"}


def write_pdf(path, text):
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10, leading=13)
    head = ParagraphStyle("head", parent=styles["Heading3"], spaceBefore=8, spaceAfter=2)
    title = ParagraphStyle("title", parent=styles["Title"], fontSize=18, spaceAfter=0)
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=50, rightMargin=50, topMargin=45, bottomMargin=45)
    story = []
    for i, line in enumerate(text.split("\n")):
        if not line.strip():
            story.append(Spacer(1, 4))
        elif i == 0:
            story.append(Paragraph(escape(line), title))
        elif line.strip() in HEADINGS:
            story.append(Paragraph(escape(line), head))
        else:
            story.append(Paragraph(escape(line), body))
    doc.build(story)


def main():
    OUT.mkdir(exist_ok=True)
    rng = random.Random(7)

    ds = CATEGORY_SKILLS["Data Science"]
    ds_req, ds_nice = ds[:12], ds[12:15]
    web = CATEGORY_SKILLS["Web Development"]
    web_req, web_nice = web[:11], web[11:14]

    (OUT / "jd_data_scientist.txt").write_text(
        jd_text("Data Scientist", "Quantix Labs", ds_req, ds_nice), encoding="utf-8")
    (OUT / "jd_web_developer.txt").write_text(
        jd_text("Full Stack Developer", "Pixelforge Studio", web_req, web_nice), encoding="utf-8")

    resumes = {
        "resume_strong_data_scientist.pdf": ("Aarav Mehta", "Data Scientist", ds_req[:10] + ds_nice[:1], 6, []),
        "resume_partial_data_scientist.pdf": ("Diya Nair", "Data Analyst", ds_req[:5], 3, ["html", "git"]),
        "resume_web_developer.pdf": ("Rohan Iyer", "Full Stack Developer", web_req[:9] + web_nice[:1], 5, []),
        "resume_career_switcher.pdf": ("Sneha Kulkarni", "Web Developer", web_req[:6] + ["python", "pandas", "sql"], 4, ["machine learning"]),
    }
    for fname, (name, role, skills, years, extra) in resumes.items():
        write_pdf(OUT / fname, build_resume(skills, role, rng, name=name, years=years, extra=extra))
    print("Wrote", len(resumes), "resume PDFs and 2 job descriptions to", OUT)


if __name__ == "__main__":
    main()
