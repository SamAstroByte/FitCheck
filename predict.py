"""Score one resume against one job description from the command line.

Usage:
  python predict.py samples/resume_strong_data_scientist.pdf samples/jd_data_scientist.txt
Both arguments can be a .pdf or a plain-text file.
"""
import argparse
from pathlib import Path

from extract import extract_text_from_pdf
from matcher import ResumeMatcher, format_scorecard


def read_any(path):
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        return extract_text_from_pdf(str(p))
    return p.read_text(encoding="utf-8", errors="ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("resume", help="resume file (.pdf or .txt)")
    ap.add_argument("jd", help="job description file (.pdf or .txt)")
    ap.add_argument("--model-dir", default="models")
    args = ap.parse_args()

    matcher = ResumeMatcher(args.model_dir)
    result = matcher.score(read_any(args.resume), read_any(args.jd))
    print(format_scorecard(result))


if __name__ == "__main__":
    main()
