"""Train and compare resume-category classifiers on TF-IDF features.

Models: Logistic Regression, Multinomial Naive Bayes, linear SVM, Random Forest.
For each model: 5-fold stratified CV on the training split (macro-F1 / accuracy),
then a held-out test evaluation. Saves comparison table + plots to outputs/ and
fitted pipelines to models/. The best model (by CV macro-F1) is stored as
models/best_model.joblib together with models/config.json.

Usage:
  python train.py
  python train.py --data kaggle_resumes.csv --text-col Resume --label-col Category
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,  # noqa: E402
                             classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split  # noqa: E402
from sklearn.naive_bayes import MultinomialNB  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from extract import clean_text  # noqa: E402

SEED = 42
DEFAULT_WEIGHTS = {"skills": 0.60, "similarity": 0.20, "category": 0.20}


def load_data(path, text_col="text", label_col="category"):
    df = pd.read_csv(path).rename(columns={text_col: "text", label_col: "category"})
    df = df.dropna(subset=["text", "category"]).copy()
    df["clean"] = df["text"].map(clean_text)
    return df.reset_index(drop=True)


def split_data(df, seed=SEED, test_size=0.2):
    """Stratified train/test split. Shared with evaluate.py so both see the same split."""
    tr, te = train_test_split(np.arange(len(df)), test_size=test_size,
                              stratify=df["category"], random_state=seed)
    return df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)


def make_models():
    return {
        "Logistic Regression": LogisticRegression(C=10, max_iter=2000, random_state=SEED),
        "Naive Bayes": MultinomialNB(alpha=0.1),
        # linear SVM + probability calibration (needed for predict_proba in the matcher)
        "SVM (linear)": CalibratedClassifierCV(SVC(kernel="linear", C=1.0, random_state=SEED),
                                               ensemble=False, cv=3),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    }


def build_pipeline(clf):
    tfidf = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2),
                            sublinear_tf=True, min_df=2, max_features=8000)
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def slug(name):
    return "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")


def calibrate_cosine(pipe, train_df, jds_path):
    """Pick cos_ref = 90th percentile of resume<->JD cosine for same-category pairs.
    The matcher divides raw cosine by this value so a 'typical good match' maps to ~1."""
    if not Path(jds_path).exists():
        return 0.30, None
    jds = pd.read_csv(jds_path)
    vec = pipe.named_steps["tfidf"]
    X = vec.transform(train_df["clean"])
    same, diff = [], []
    for _, jd in jds.iterrows():
        sims = (X @ vec.transform([clean_text(jd["text"])]).T).toarray().ravel()
        mask = (train_df["category"] == jd["category"]).to_numpy()
        same.extend(sims[mask])
        diff.extend(sims[~mask])
    if not same:
        return 0.30, None
    stats = {"same_category_median": float(np.median(same)), "other_category_median": float(np.median(diff))}
    return float(np.percentile(same, 90)), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/resumes.csv")
    ap.add_argument("--jds", default="data/job_descriptions.csv")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="category")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--model-dir", default="models")
    args = ap.parse_args()

    out, mdir = Path(args.out_dir), Path(args.model_dir)
    out.mkdir(exist_ok=True)
    mdir.mkdir(exist_ok=True)

    df = load_data(args.data, args.text_col, args.label_col)
    train, test = split_data(df)
    print(f"{len(df)} resumes | {df.category.nunique()} categories | train {len(train)} / test {len(test)}\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, fitted, preds = [], {}, {}
    for name, clf in make_models().items():
        t0 = time.time()
        pipe = build_pipeline(clf)
        cvr = cross_validate(pipe, train["clean"], train["category"], cv=cv,
                             scoring=["accuracy", "f1_macro"], n_jobs=1)
        pipe.fit(train["clean"], train["category"])
        y_pred = pipe.predict(test["clean"])
        p, r, f1, _ = precision_recall_fscore_support(test["category"], y_pred, average="macro", zero_division=0)
        rows.append({
            "model": name,
            "cv_accuracy": cvr["test_accuracy"].mean(), "cv_f1_macro": cvr["test_f1_macro"].mean(),
            "cv_f1_std": cvr["test_f1_macro"].std(),
            "test_accuracy": accuracy_score(test["category"], y_pred),
            "test_precision": p, "test_recall": r, "test_f1_macro": f1,
            "seconds": time.time() - t0,
        })
        fitted[name], preds[name] = pipe, y_pred
        joblib.dump(pipe, mdir / f"{slug(name)}.joblib")
        print(f"{name:20s} CV F1 {rows[-1]['cv_f1_macro']:.3f} (+/-{rows[-1]['cv_f1_std']:.3f}) | "
              f"test acc {rows[-1]['test_accuracy']:.3f} | test F1 {f1:.3f}")

    res = pd.DataFrame(rows).sort_values("cv_f1_macro", ascending=False).reset_index(drop=True)
    res.round(4).to_csv(out / "model_comparison.csv", index=False)
    best = res.loc[0, "model"]
    print(f"\nBest by CV macro-F1: {best}")

    # --- plots / reports ---
    labels = sorted(df["category"].unique())
    cm = confusion_matrix(test["category"], preds[best], labels=labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, xticks_rotation=45, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion matrix - {best} (test set)")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(res))
    ax.bar(x - 0.2, res["cv_f1_macro"], 0.4, label="CV macro-F1 (train)")
    ax.bar(x + 0.2, res["test_f1_macro"], 0.4, label="Test macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(res["model"], rotation=15)
    lo = max(0.0, min(res["cv_f1_macro"].min(), res["test_f1_macro"].min()) - 0.05)
    ax.set_ylim(lo, 1.0)
    ax.set_ylabel("macro-F1")
    ax.set_title("Model comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=150)
    plt.close(fig)

    (out / "classification_report.txt").write_text(
        f"Model: {best}\n\n" + classification_report(test["category"], preds[best], zero_division=0))

    # --- save best model + matcher config ---
    best_pipe = fitted[best]
    joblib.dump(best_pipe, mdir / "best_model.joblib")
    cos_ref, cos_stats = calibrate_cosine(best_pipe, train, args.jds)
    config = {
        "best_model": best,
        "classes": list(best_pipe.named_steps["clf"].classes_),
        "weights": DEFAULT_WEIGHTS,
        "cos_ref": cos_ref,
        "cosine_stats": cos_stats,
    }
    (mdir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"cos_ref (90th pct same-category cosine) = {cos_ref:.3f}  {cos_stats or ''}")
    print(f"\nSaved models to {mdir}/, tables and plots to {out}/")


if __name__ == "__main__":
    main()
