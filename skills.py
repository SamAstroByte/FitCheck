"""Skill taxonomy and rule-based skill extraction.

SKILL_SYNONYMS maps a canonical skill name to its aliases, so that
"ML", "machine learning" and "Machine-Learning" all count as one skill.
CATEGORY_SKILLS lists the core skills of each job category and is used by the
synthetic data generator.

To extend the system, add skills / aliases here -- nothing else needs to change.
"""
import re

SKILL_SYNONYMS = {
    # --- languages ---
    "python": [],
    "java": [],
    "javascript": ["js", "ecmascript"],
    "typescript": ["ts"],
    "kotlin": [],
    "swift": [],
    "dart": [],
    "scala": [],
    "bash": ["shell scripting", "shell script"],
    "sql": [],
    "html": ["html5"],
    "css": ["css3"],
    # --- data science / ML ---
    "machine learning": ["ml"],
    "deep learning": [],
    "tensorflow": [],
    "pytorch": [],
    "keras": [],
    "scikit-learn": ["sklearn", "scikit learn"],
    "pandas": [],
    "numpy": [],
    "statistics": ["statistical analysis"],
    "data visualization": ["data visualisation"],
    "matplotlib": [],
    "nlp": ["natural language processing"],
    "jupyter": ["jupyter notebook", "jupyter notebooks"],
    "feature engineering": [],
    "tableau": [],
    "power bi": ["powerbi"],
    "a/b testing": ["ab testing", "a b testing"],
    "computer vision": [],
    # --- web ---
    "react": ["react.js", "reactjs"],
    "angular": ["angularjs"],
    "vue": ["vue.js", "vuejs"],
    "node.js": ["nodejs", "node js", "node"],
    "express": ["express.js", "expressjs"],
    "rest api": ["restful api", "rest apis", "restful apis", "restful services"],
    "mongodb": ["mongo"],
    "tailwind": ["tailwind css", "tailwindcss"],
    "redux": [],
    "next.js": ["nextjs", "next js"],
    "webpack": [],
    "graphql": [],
    "responsive design": ["responsive web design"],
    # --- mobile ---
    "android": [],
    "ios": [],
    "flutter": [],
    "react native": ["react-native"],
    "firebase": [],
    "sqlite": [],
    "xcode": [],
    "android studio": [],
    "material design": [],
    "jetpack compose": [],
    "mvvm": [],
    "push notifications": ["fcm"],
    # --- devops / cloud ---
    "docker": [],
    "kubernetes": ["k8s"],
    "aws": ["amazon web services"],
    "azure": ["microsoft azure"],
    "terraform": [],
    "jenkins": [],
    "ci/cd": ["cicd", "ci cd", "continuous integration"],
    "linux": [],
    "ansible": [],
    "git": ["github"],
    "prometheus": [],
    "grafana": [],
    "gitlab ci": ["gitlab-ci"],
    "helm": [],
    "monitoring": [],
    "nginx": [],
    # --- java backend ---
    "spring boot": ["springboot"],
    "hibernate": [],
    "maven": [],
    "microservices": ["microservice"],
    "mysql": [],
    "junit": [],
    "kafka": ["apache kafka"],
    "redis": [],
    "jpa": [],
    "spring security": [],
    "postgresql": ["postgres"],
    # --- testing ---
    "selenium": [],
    "manual testing": [],
    "test automation": ["automation testing"],
    "testng": [],
    "jira": [],
    "postman": [],
    "api testing": [],
    "cypress": [],
    "agile": [],
    "bug tracking": ["defect tracking"],
    "test cases": ["test case"],
    "performance testing": [],
    "jmeter": [],
    # --- security ---
    "network security": [],
    "penetration testing": ["pentesting", "pen testing", "pen-testing"],
    "siem": [],
    "firewalls": ["firewall"],
    "vulnerability assessment": ["vulnerability scanning"],
    "owasp": ["owasp top 10"],
    "wireshark": [],
    "cryptography": ["encryption"],
    "incident response": [],
    "splunk": [],
    "kali linux": ["kali"],
    "nmap": [],
    "iso 27001": ["iso27001"],
    "ethical hacking": [],
    "burp suite": ["burpsuite"],
    # --- data engineering ---
    "spark": ["apache spark", "pyspark"],
    "hadoop": [],
    "airflow": ["apache airflow"],
    "etl": ["elt", "etl pipelines", "etl pipeline"],
    "data warehousing": ["data warehouse", "data warehouses"],
    "snowflake": [],
    "bigquery": ["big query"],
    "dbt": [],
    "data modeling": ["data modelling"],
}

CATEGORY_SKILLS = {
    "Data Science": [
        "python", "machine learning", "deep learning", "tensorflow", "pytorch",
        "scikit-learn", "pandas", "numpy", "statistics", "data visualization",
        "matplotlib", "nlp", "sql", "jupyter", "feature engineering", "tableau",
        "a/b testing", "computer vision", "keras", "power bi",
    ],
    "Web Development": [
        "html", "css", "javascript", "typescript", "react", "angular", "vue",
        "node.js", "express", "rest api", "mongodb", "sql", "git", "tailwind",
        "redux", "next.js", "webpack", "graphql", "responsive design", "docker",
    ],
    "Mobile Development": [
        "android", "kotlin", "java", "swift", "ios", "flutter", "dart",
        "react native", "firebase", "rest api", "sqlite", "xcode",
        "android studio", "material design", "git", "jetpack compose", "mvvm",
        "push notifications",
    ],
    "DevOps": [
        "docker", "kubernetes", "aws", "terraform", "jenkins", "ci/cd", "linux",
        "ansible", "git", "bash", "prometheus", "grafana", "azure", "gitlab ci",
        "helm", "python", "monitoring", "nginx",
    ],
    "Java Backend": [
        "java", "spring boot", "hibernate", "maven", "microservices", "rest api",
        "sql", "mysql", "junit", "kafka", "git", "docker", "redis", "jpa",
        "spring security", "postgresql",
    ],
    "Software Testing": [
        "selenium", "manual testing", "test automation", "junit", "testng",
        "jira", "postman", "api testing", "cypress", "python", "sql", "jenkins",
        "agile", "bug tracking", "test cases", "performance testing", "jmeter",
        "git",
    ],
    "Cybersecurity": [
        "network security", "penetration testing", "siem", "firewalls", "linux",
        "python", "vulnerability assessment", "owasp", "wireshark",
        "cryptography", "incident response", "splunk", "kali linux", "nmap",
        "iso 27001", "ethical hacking", "burp suite",
    ],
    "Data Engineering": [
        "python", "sql", "spark", "hadoop", "airflow", "kafka", "etl", "aws",
        "data warehousing", "snowflake", "scala", "postgresql", "docker", "git",
        "bigquery", "dbt", "data modeling", "mongodb",
    ],
}

# Every skill used in a category must exist in the synonym table.
for _cat, _skills in CATEGORY_SKILLS.items():
    for _s in _skills:
        assert _s in SKILL_SYNONYMS, f"'{_s}' ({_cat}) missing from SKILL_SYNONYMS"


def _build_patterns():
    patterns = {}
    for skill, aliases in SKILL_SYNONYMS.items():
        names = sorted({skill, *aliases}, key=len, reverse=True)
        body = "|".join(re.escape(n) for n in names)
        # whole-token match: 'java' must not match inside 'javascript', 'sql' not in 'mysql'
        patterns[skill] = re.compile(rf"(?<![a-z0-9])(?:{body})(?![a-z0-9])")
    return patterns


_PATTERNS = _build_patterns()


def extract_skills(text):
    """Return the set of canonical skills mentioned in `text`."""
    # collapse whitespace so multi-word skills survive PDF line wrapping
    t = re.sub(r"\s+", " ", text.lower())
    return {skill for skill, pat in _PATTERNS.items() if pat.search(t)}
