"""Hybrid resume <-> job-description matcher.

Final score (0-100) = weighted blend of three signals:
  skills      (default 0.60)  share of the JD's skills found in the resume
                              (required skills count 1.0, "nice to have" count 0.5)
  similarity  (default 0.20)  TF-IDF cosine similarity between resume and JD,
                              divided by a calibrated reference value and capped at 1
  category    (default 0.20)  overlap between the classifier's category probabilities
                              for the resume and for the JD

Why a blend? Skill overlap is explainable but brittle; TF-IDF similarity captures
wording beyond the skill list; the classifier adds a "is this the right kind of
profile?" signal learned from labeled resumes. Weights live in models/config.json.
"""
import json
import re
from pathlib import Path

import joblib
import numpy as np

from extract import clean_text
from skills import extract_skills

DEFAULT_WEIGHTS = {"skills": 0.60, "similarity": 0.20, "category": 0.20}

# ---------- job-description parsing (required vs. nice-to-have) ----------
_PREF = re.compile(
    r"^(?:[-•]\s*)?(?P<kw>nice[- ]to[- ]have|preferred(?: qualifications| skills)?|"
    r"good to have|bonus(?: points)?|desirable)(?P<sep>\s*:?\s*)(?P<rest>.*)$",
    re.I,
)


def _match_pref(line):
    """If `line` is a 'nice to have' heading return its inline remainder, else None."""
    m = _PREF.match(line.strip().strip("*#_ "))
    if m and (":" in m.group("sep") or not m.group("rest").strip()):
        return m.group("rest").strip()
    return None


def _is_heading(line):
    s = line.strip()
    if not s:
        return False
    if s.startswith("#"):
        return True
    core = s.strip("*_ ")
    if core.startswith(("-", "•")):
        return False
    words = core.rstrip(":").split()
    if not 0 < len(words) <= 6:
        return False
    return core.endswith(":") or (core.isupper() and len(words) <= 5)


def split_jd(text):
    """Split a JD into (required_text, preferred_text). Everything under a
    'Nice to have / Preferred' heading, up to the next heading, is preferred."""
    req, pref, state = [], [], "req"
    for line in text.splitlines():
        inline = _match_pref(line)
        if inline is not None:
            state = "pref"
            if inline:
                pref.append(inline)
            continue
        if state == "pref" and _is_heading(line):
            state = "req"
        (req if state == "req" else pref).append(line)
    return "\n".join(req), "\n".join(pref)


def tier(score):
    if score >= 75:
        return "Strong Candidate"
    if score >= 60:
        return "Good Fit - Shortlist"
    if score >= 40:
        return "Partial Match"
    return "Weak Match"


class ResumeMatcher:
    def __init__(self, model_dir="models"):
        d = Path(model_dir)
        self.pipe = joblib.load(d / "best_model.joblib")
        self.config = json.loads((d / "config.json").read_text())
        self.weights = {**DEFAULT_WEIGHTS, **self.config.get("weights", {})}
        self.cos_ref = float(self.config.get("cos_ref") or 0.3)
        self.vec = self.pipe.named_steps["tfidf"]
        self.clf = self.pipe.named_steps["clf"]
        self.classes = list(self.clf.classes_)

    # ----- helpers -----
    def top_categories(self, proba, k=3):
        idx = np.argsort(-proba)[:k]
        return [(self.classes[i], float(proba[i])) for i in idx]

    def analyze_jd(self, jd_text):
        req_text, pref_text = split_jd(jd_text)
        required = extract_skills(req_text)
        preferred = extract_skills(pref_text) - required
        vec = self.vec.transform([clean_text(jd_text)])
        proba = self.clf.predict_proba(vec)[0]
        return {"required": required, "preferred": preferred, "vec": vec, "proba": proba,
                "category": self.classes[int(np.argmax(proba))]}

    # ----- scoring -----
    def score_batch(self, resume_texts, jd):
        """Score many resumes against one JD (text or the dict from analyze_jd)."""
        if not isinstance(jd, dict):
            jd = self.analyze_jd(jd)
        cleaned = [clean_text(t) for t in resume_texts]
        X = self.vec.transform(cleaned)
        sims = (X @ jd["vec"].T).toarray().ravel()  # TF-IDF rows are L2-normalised -> cosine
        P = self.clf.predict_proba(X)

        req, pref = jd["required"], jd["preferred"]
        results = []
        for i, text in enumerate(cleaned):
            have = extract_skills(text)
            m_req, x_req = sorted(req & have), sorted(req - have)
            m_pref, x_pref = sorted(pref & have), sorted(pref - have)

            denom = len(req) + 0.5 * len(pref)
            skill = (len(m_req) + 0.5 * len(m_pref)) / denom if denom else 0.0
            sim = min(float(sims[i]) / self.cos_ref, 1.0)
            cat = float(P[i] @ jd["proba"])

            w = dict(self.weights)
            if not denom:  # JD mentions no known skills -> ignore the skill component
                w["skills"] = 0.0
            score = 100 * (w["skills"] * skill + w["similarity"] * sim + w["category"] * cat) / sum(w.values())

            results.append({
                "score": round(score, 1),
                "tier": tier(score),
                "n_required": len(req),
                "n_matched_required": len(m_req),
                "matched_required": m_req,
                "missing_required": x_req,
                "matched_preferred": m_pref,
                "missing_preferred": x_pref,
                "extra_skills": sorted(have - req - pref),
                "components": {
                    "skill_overlap": round(skill, 3),
                    "text_similarity": round(sim, 3),
                    "raw_cosine": round(float(sims[i]), 3),
                    "category_alignment": round(cat, 3),
                },
                "resume_categories": self.top_categories(P[i]),
                "jd_category": jd["category"],
                "jd_has_skills": bool(denom),
            })
        return results

    def score(self, resume_text, jd_text):
        return self.score_batch([resume_text], jd_text)[0]


def format_scorecard(res, width=36):
    """ASCII scorecard like the one in the project proposal, plus details."""
    miss = len(res["missing_required"])
    lines = [
        f"Match Score: {res['score']:.0f}%",
        f"Skills Matched: {res['n_matched_required']}/{res['n_required']}",
        f"Missing Skills: {miss}",
        "",
        "Recommendation:",
        res["tier"],
    ]
    box = ["┌" + "─" * (width + 2) + "┐"]
    box += [f"│ {l:<{width}} │" for l in lines]
    box += ["└" + "─" * (width + 2) + "┘"]
    c = res["components"]
    details = [
        "",
        f"Matched : {', '.join(res['matched_required']) or '-'}",
        f"Missing : {', '.join(res['missing_required']) or '-'}",
    ]
    if res["matched_preferred"] or res["missing_preferred"]:
        details.append(f"Nice-to-have matched: {', '.join(res['matched_preferred']) or '-'}"
                       f" | missing: {', '.join(res['missing_preferred']) or '-'}")
    details += [
        "",
        f"Score breakdown: skills {c['skill_overlap']:.0%} | text similarity {c['text_similarity']:.0%}"
        f" (raw cosine {c['raw_cosine']:.2f}) | category alignment {c['category_alignment']:.0%}",
        "Resume looks like: " + ", ".join(f"{k} ({p:.0%})" for k, p in res["resume_categories"]),
        f"Job looks like   : {res['jd_category']}",
    ]
    return "\n".join(box + details)
