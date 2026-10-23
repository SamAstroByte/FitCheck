"""Generate a synthetic, labeled resume dataset plus job descriptions.

Why synthetic? Real labeled resume data is hard to get and contains personal
information. This generator gives the project a self-contained dataset so the
whole pipeline runs out of the box. Resumes are made deliberately messy:
  * each resume only lists 25-60% of its category's core skills
  * 3-8 skills from unrelated categories are mixed in
  * ~25% are "career switchers" with a large block of another category's skills
  * skills are sometimes written as aliases (ML, k8s, JS, ...)
  * only ~30% of resumes carry a category-specific job title (the rest are generic,
    otherwise the title alone would leak the label)

Because the data is synthetic, reported accuracy is optimistic. To use real data
(e.g. the Kaggle "Resume Dataset"), run train.py with --data/--text-col/--label-col.

Usage:  python generate_data.py --per-class 150 --seed 42
"""
import argparse
import random
from pathlib import Path

import pandas as pd

from skills import CATEGORY_SKILLS, SKILL_SYNONYMS

ROLES = {
    "Data Science": ["Data Scientist", "Machine Learning Engineer", "Data Analyst"],
    "Web Development": ["Full Stack Developer", "Frontend Developer", "Web Developer"],
    "Mobile Development": ["Mobile App Developer", "Android Developer", "Flutter Developer"],
    "DevOps": ["DevOps Engineer", "Site Reliability Engineer", "Cloud Engineer"],
    "Java Backend": ["Java Backend Developer", "Backend Engineer", "Software Engineer"],
    "Software Testing": ["QA Engineer", "Test Automation Engineer", "Software Tester"],
    "Cybersecurity": ["Security Analyst", "Penetration Tester", "Cybersecurity Engineer"],
    "Data Engineering": ["Data Engineer", "ETL Developer", "Big Data Engineer"],
}

GENERIC_ROLES = [
    "Software Engineer", "Software Developer", "Technical Associate",
    "Systems Engineer", "Application Developer", "Engineer",
]

COMPANIES = [
    "Northwind Technologies", "BlueOrbit Systems", "Quantix Labs",
    "Brightpath Software", "Helios Digital", "Cedar Peak Solutions",
    "Nimbus Works", "Ironleaf Analytics", "Pixelforge Studio", "Vertex Cloud",
]

EDUCATION = [
    "B.Tech in Computer Science", "B.E. in Information Technology",
    "M.Tech in Computer Science", "BCA, Computer Applications",
    "B.Sc in Computer Science", "MCA, Computer Applications",
]

BULLETS = [
    "Developed and maintained solutions using {a} and {b}, improving delivery speed by {n}%",
    "Collaborated with cross-functional teams to ship features built on {a}",
    "Optimized {a} workflows, reducing turnaround time by {n}%",
    "Designed and implemented components with {a}, {b} and {c}",
    "Mentored junior team members and conducted code reviews focused on {a}",
    "Automated repetitive tasks using {a}, saving roughly {n} hours per week",
    "Documented processes and presented results to stakeholders",
    "Participated in sprint planning and daily stand-ups with the wider team",
]

PROJECTS = [
    "Capstone project: built an end-to-end solution using {a} and {b}",
    "Personal project: implemented and documented a working prototype with {a}, {b} and {c}",
    "Open-source contribution: added features and fixed issues involving {a}",
]

SPECIAL = {
    "ci/cd": "CI/CD", "a/b testing": "A/B Testing", "scikit-learn": "Scikit-learn",
    "node.js": "Node.js", "next.js": "Next.js", "mysql": "MySQL",
    "postgresql": "PostgreSQL", "mongodb": "MongoDB", "javascript": "JavaScript",
    "typescript": "TypeScript", "tensorflow": "TensorFlow", "pytorch": "PyTorch",
    "graphql": "GraphQL", "bigquery": "BigQuery", "testng": "TestNG",
    "junit": "JUnit", "jmeter": "JMeter", "gitlab ci": "GitLab CI",
    "github": "GitHub", "nlp": "NLP", "sqlite": "SQLite", "iso 27001": "ISO 27001",
    "rest api": "REST API", "power bi": "Power BI", "ios": "iOS", "aws": "AWS",
    "sql": "SQL", "html": "HTML", "css": "CSS", "etl": "ETL", "jpa": "JPA",
    "mvvm": "MVVM", "owasp": "OWASP", "siem": "SIEM", "dbt": "dbt",
    "k8s": "K8s", "js": "JS", "ml": "ML", "xcode": "Xcode",
}


def pretty(skill):
    if skill in SPECIAL:
        return SPECIAL[skill]
    return " ".join(w.capitalize() for w in skill.split(" "))


def display(skill, rng):
    """Sometimes write a skill using one of its aliases (e.g. 'ML', 'k8s')."""
    aliases = SKILL_SYNONYMS.get(skill, [])
    if aliases and rng.random() < 0.25:
        return rng.choice(aliases)
    return skill


def build_resume(skills, role, rng, name=None, years=None, extra=()):
    """Render a plain-text resume that mentions `skills` (+ `extra` noise skills)."""
    skills = list(skills)
    pool = skills or ["teamwork"]
    years = years if years is not None else rng.randint(1, 9)

    def pick():
        return pretty(display(rng.choice(pool), rng))

    def fill(template):
        return template.format(a=pick(), b=pick(), c=pick(), n=rng.choice([15, 20, 25, 30, 35, 40]))

    all_skills = skills + [s for s in extra if s not in skills]
    rng.shuffle(all_skills)
    skills_line = ", ".join(pretty(display(s, rng)) for s in all_skills)

    lines = []
    if name:
        lines.append(name)
    lines.append(role)
    lines += [
        "", "SUMMARY",
        f"{role} with {years} years of experience building and shipping real projects. "
        f"Hands-on with {pick()}, {pick()} and {pick()}. Comfortable working in a fast-paced team environment.",
        "", "SKILLS", skills_line,
        "", "EXPERIENCE",
    ]
    y_mid = 2026 - max(1, years // 2)
    jobs = [(role, f"{y_mid} - Present")]
    if years >= 3:
        jobs.append((f"Junior {role}", f"{2026 - years} - {y_mid}"))
    for title, span in jobs:
        lines.append(f"{title}, {rng.choice(COMPANIES)} ({span})")
        for b in rng.sample(BULLETS, 3):
            lines.append(f"- {fill(b)}")
        lines.append("")
    lines += ["PROJECTS"] + [f"- {fill(p)}" for p in rng.sample(PROJECTS, 2)]
    lines += ["", "EDUCATION", rng.choice(EDUCATION)]
    return "\n".join(lines)


def make_resumes(per_class, rng):
    rows = []
    for cat, core in CATEGORY_SKILLS.items():
        noise_pool = sorted({s for c, sk in CATEGORY_SKILLS.items() if c != cat for s in sk if s not in core})
        for _ in range(per_class):
            k = max(4, rng.randint(int(len(core) * 0.25), int(len(core) * 0.6)))
            own = rng.sample(core, k)
            noise = rng.sample(noise_pool, rng.randint(3, 8))
            if rng.random() < 0.25:  # career switcher
                other = rng.choice([c for c in CATEGORY_SKILLS if c != cat])
                extra = rng.sample(CATEGORY_SKILLS[other], int(len(CATEGORY_SKILLS[other]) * 0.45))
                noise = sorted(set(noise) | set(s for s in extra if s not in core))
            # most real resumes use vague titles; an exact title would leak the label
            role = rng.choice(ROLES[cat]) if rng.random() < 0.3 else rng.choice(GENERIC_ROLES)
            text = build_resume(own, role, rng, extra=noise)
            rows.append({"category": cat, "text": text, "skills": "|".join(own + noise)})
    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["id"] = f"R{i:04d}"
    return pd.DataFrame(rows)[["id", "category", "text", "skills"]]


def jd_text(title, company, req, nice):
    lines = [
        f"Job Title: {title}",
        f"Company: {company}",
        "",
        "About the role:",
        f"We are looking for a {title} to join our growing team. You will work closely with "
        "product and engineering colleagues to deliver reliable, high-quality solutions.",
        "",
        "Responsibilities:",
        f"- Design, build and maintain solutions using {pretty(req[0])} and {pretty(req[1])}",
        "- Collaborate with cross-functional teams to plan and deliver projects",
        "- Write clean, well-documented work and take part in code reviews",
        "",
        "Required skills:",
    ]
    lines += [f"- {pretty(s)}" for s in req]
    lines += ["", "Nice to have:"]
    lines += [f"- {pretty(s)}" for s in nice]
    return "\n".join(lines)


def make_jds(per_class, rng):
    rows = []
    for cat, core in CATEGORY_SKILLS.items():
        for _ in range(per_class):
            n_req = rng.randint(8, min(12, len(core) - 3))
            req = rng.sample(core, n_req)
            nice = rng.sample([s for s in core if s not in req], 3)
            title = rng.choice(ROLES[cat])
            company = rng.choice(COMPANIES)
            rows.append({
                "category": cat, "title": title,
                "text": jd_text(title, company, req, nice),
                "required": "|".join(req), "preferred": "|".join(nice),
            })
    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["id"] = f"J{i:03d}"
    return pd.DataFrame(rows)[["id", "category", "title", "text", "required", "preferred"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=150)
    ap.add_argument("--jds-per-class", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    resumes = make_resumes(args.per_class, rng)
    jds = make_jds(args.jds_per_class, rng)
    resumes.to_csv(out / "resumes.csv", index=False)
    jds.to_csv(out / "job_descriptions.csv", index=False)
    print(f"Wrote {len(resumes)} resumes ({resumes.category.nunique()} categories) "
          f"and {len(jds)} job descriptions to {out}/")


if __name__ == "__main__":
    main()
