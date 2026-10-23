"""Evaluate the matcher as a RANKING system on held-out resumes.

Proxy ground truth: for each job description, a held-out resume is "relevant" if
it belongs to the JD's category. For every JD we rank all test resumes and report
precision@10, MAP and NDCG@10 for four scorers (an ablation):
  skills only | TF-IDF cosine only | category classifier only | hybrid (final score)

This is a PROXY: category membership is not the same as "good fit for this exact
job". For a stronger check, label real pairs by hand -- this script writes
data/hand_labels_template.csv; fill in the `label` column (0 weak / 1 partial /
2 strong), save it as data/hand_labels.csv and re-run to get Spearman correlation
between your labels and the hybrid score.

Usage:  python evaluate.py
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from matcher import ResumeMatcher  # noqa: E402
from skills import extract_skills  # noqa: E402
from train import load_data, split_data  # noqa: E402


def rank_metrics(scores, rel, k=10):
    order = np.argsort(-scores, kind="stable")
    r = rel[order]
    p_at_k = r[:k].mean()
    pos = np.nonzero(r == 1)[0]
    ap = float(np.mean(np.cumsum(r)[pos] / (pos + 1))) if len(pos) else 0.0
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((r[:k] * disc[: len(r[:k])]).sum())
    ideal = np.sort(rel)[::-1][:k]
    idcg = float((ideal * disc[: len(ideal)]).sum())
    return p_at_k, ap, (dcg / idcg if idcg else 0.0)


def write_label_template(test, jds, matcher, path, n=40, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n):
        jd = jds.iloc[rng.randint(len(jds))]
        same = i % 2 == 0  # half the pairs from the same category, half from others
        pool = test[(test["category"] == jd["category"]) == same]
        res = pool.iloc[rng.randint(len(pool))]
        rows.append({
            "resume_id": res["id"], "jd_id": jd["id"], "jd_title": jd["title"],
            "jd_required_skills": ", ".join(sorted(matcher.analyze_jd(jd["text"])["required"])),
            "resume_skills": ", ".join(sorted(extract_skills(res["text"]))),
            "label": "",
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/resumes.csv")
    ap.add_argument("--jds", default="data/job_descriptions.csv")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    matcher = ResumeMatcher()
    _, test = split_data(load_data(args.data))
    jds = pd.read_csv(args.jds)
    texts = test["text"].tolist()
    print(f"Ranking {len(test)} held-out resumes for each of {len(jds)} job descriptions "
          f"(model: {matcher.config['best_model']})\n")

    metrics = {k: [] for k in ["Skill overlap only", "TF-IDF cosine only", "Category classifier only", "Hybrid (final score)"]}
    for _, jd in jds.iterrows():
        analysed = matcher.analyze_jd(jd["text"])
        res = matcher.score_batch(texts, analysed)
        rel = (test["category"] == jd["category"]).to_numpy().astype(int)
        scorers = {
            "Skill overlap only": [r["components"]["skill_overlap"] for r in res],
            "TF-IDF cosine only": [r["components"]["raw_cosine"] for r in res],
            "Category classifier only": [r["components"]["category_alignment"] for r in res],
            "Hybrid (final score)": [r["score"] for r in res],
        }
        for name, s in scorers.items():
            metrics[name].append(rank_metrics(np.asarray(s, dtype=float), rel))

    table = pd.DataFrame(
        {name: np.mean(v, axis=0) for name, v in metrics.items()},
        index=["precision@10", "MAP", "NDCG@10"],
    ).T.round(3)
    table.to_csv(out / "ranking_eval.csv")
    print(table.to_string(), "\n")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(table))
    ax.bar(x - 0.2, table["MAP"], 0.4, label="MAP")
    ax.bar(x + 0.2, table["precision@10"], 0.4, label="precision@10")
    ax.set_xticks(x)
    ax.set_xticklabels(table.index, rotation=15)
    ax.set_ylim(max(0, table[["MAP", "precision@10"]].min().min() - 0.1), 1.02)
    ax.set_title("Ranking quality by scorer (held-out resumes, proxy relevance)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "ranking_ablation.png", dpi=150)
    plt.close(fig)

    template = Path("data/hand_labels_template.csv")
    write_label_template(test, jds, matcher, template)
    print(f"Wrote {template} (fill in `label` 0/1/2 and save as data/hand_labels.csv)")

    labels_path = Path("data/hand_labels.csv")
    if labels_path.exists():
        lab = pd.read_csv(labels_path).dropna(subset=["label"])
        by_id, jd_by_id = test.set_index("id"), jds.set_index("id")
        preds = [matcher.score(by_id.loc[r.resume_id, "text"], jd_by_id.loc[r.jd_id, "text"])["score"]
                 for r in lab.itertuples()]
        rho = pd.Series(preds).corr(lab["label"].reset_index(drop=True), method="spearman")
        print(f"Spearman correlation with hand labels (n={len(lab)}): {rho:.3f}")


if __name__ == "__main__":
    main()
