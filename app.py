"""FitCheck - Streamlit demo: upload resume PDF(s), paste or pick a job description, get a scorecard.

Run:  streamlit run app.py
"""
import io
from pathlib import Path

import pandas as pd
import streamlit as st

from extract import extract_text_from_pdf
from matcher import ResumeMatcher

st.set_page_config(page_title="FitCheck", page_icon="✅", layout="wide")


@st.cache_resource
def get_matcher():
    return ResumeMatcher()


@st.cache_data
def load_sample_jds():
    p = Path("data/job_descriptions.csv")
    if not p.exists():
        return pd.DataFrame(columns=["id", "title", "category", "text"])
    return pd.read_csv(p)


def pills(items, color):
    if not items:
        return "<span style='opacity:.6'>none</span>"
    style = (f"background:{color}22;color:{color};border:1px solid {color}66;"
             "padding:2px 10px;margin:2px 4px 2px 0;border-radius:999px;display:inline-block;font-size:0.85rem")
    return "".join(f"<span style='{style}'>{i}</span>" for i in items)


def read_resume(upload):
    return extract_text_from_pdf(io.BytesIO(upload.getvalue()))


def render_result(res):
    c1, c2, c3 = st.columns(3)
    c1.metric("Match score", f"{res['score']:.0f}%")
    c2.metric("Required skills matched", f"{res['n_matched_required']}/{res['n_required']}")
    c3.metric("Missing skills", len(res["missing_required"]))
    st.progress(min(res["score"] / 100, 1.0))

    msg = f"**Recommendation: {res['tier']}**"
    if res["tier"].startswith("Strong"):
        st.success(msg)
    elif res["tier"].startswith("Good"):
        st.info(msg)
    elif res["tier"].startswith("Partial"):
        st.warning(msg)
    else:
        st.error(msg)

    st.markdown("**Matched skills**")
    st.markdown(pills(res["matched_required"], "#16a34a"), unsafe_allow_html=True)
    st.markdown("**Missing skills**")
    st.markdown(pills(res["missing_required"], "#dc2626"), unsafe_allow_html=True)
    if res["matched_preferred"] or res["missing_preferred"]:
        st.markdown("**Nice-to-have**")
        st.markdown(pills(res["matched_preferred"], "#16a34a") + pills(res["missing_preferred"], "#9ca3af"),
                    unsafe_allow_html=True)

    with st.expander("How was this score computed?"):
        c = res["components"]
        st.write(pd.DataFrame({
            "component": ["Skill overlap", "TF-IDF text similarity", "Category alignment"],
            "value": [c["skill_overlap"], c["text_similarity"], c["category_alignment"]],
            "weight": [matcher.weights["skills"], matcher.weights["similarity"], matcher.weights["category"]],
        }))
        st.caption(f"Raw cosine similarity: {c['raw_cosine']:.2f}. "
                   "Resume looks like: " + ", ".join(f"{k} ({p:.0%})" for k, p in res["resume_categories"])
                   + f". Job looks like: {res['jd_category']}.")
        if not res["jd_has_skills"]:
            st.warning("No known skills were found in the job description, so the skill component was ignored.")


try:
    matcher = get_matcher()
except Exception:
    st.error("No trained model found. Run `python generate_data.py` then `python train.py` first.")
    st.stop()

st.title("✅ FitCheck")
st.caption("AI resume screening & job matching: TF-IDF + skill extraction + a trained classifier. "
           "Scores support a human decision; they should not replace it.")

# ---- job description input ----
with st.sidebar:
    st.header("Job description")
    samples = load_sample_jds()
    source = st.radio("Source", ["Paste / type", "Sample from dataset"], horizontal=True)
    if source == "Sample from dataset" and len(samples):
        labels = [f"{r.id} - {r.title} ({r.category})" for r in samples.itertuples()]
        choice = st.selectbox("Pick a sample JD", labels)
        jd_default = samples.iloc[labels.index(choice)]["text"]
    else:
        jd_default = Path("samples/jd_data_scientist.txt").read_text() if Path("samples/jd_data_scientist.txt").exists() else ""
    jd_text = st.text_area("Job description text", jd_default, height=380,
                           help="Put 'Nice to have:' skills under their own heading so they count as optional.")

tab1, tab2 = st.tabs(["Single candidate", "Rank multiple candidates"])

with tab1:
    up = st.file_uploader("Resume (PDF)", type="pdf", key="single")
    if up and jd_text.strip():
        text = read_resume(up)
        if len(text.strip()) < 50:
            st.error("Couldn't read text from this PDF (it may be a scanned image). Try a text-based PDF.")
        else:
            render_result(matcher.score(text, jd_text))
            with st.expander("Extracted resume text"):
                st.text(text)
    elif up:
        st.info("Add a job description in the sidebar.")
    else:
        st.info("Upload a resume PDF to see its match score.")

with tab2:
    ups = st.file_uploader("Resumes (PDF, several)", type="pdf", accept_multiple_files=True, key="batch")
    if ups and jd_text.strip():
        names, texts = [], []
        for u in ups:
            t = read_resume(u)
            if len(t.strip()) >= 50:
                names.append(u.name)
                texts.append(t)
            else:
                st.warning(f"Skipped {u.name}: no extractable text.")
        if texts:
            results = matcher.score_batch(texts, jd_text)
            table = pd.DataFrame({
                "Candidate": names,
                "Score": [r["score"] for r in results],
                "Recommendation": [r["tier"] for r in results],
                "Required matched": [f"{r['n_matched_required']}/{r['n_required']}" for r in results],
                "Missing skills": [", ".join(r["missing_required"]) for r in results],
            }).sort_values("Score", ascending=False).reset_index(drop=True)
            table.index += 1
            st.dataframe(table)
            st.download_button("Download ranking (CSV)", table.to_csv(index_label="Rank"),
                               "ranking.csv", "text/csv")
    else:
        st.info("Upload several resume PDFs to rank them against the job description.")
