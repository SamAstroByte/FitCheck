# FitCheck: AI-Powered Resume Screening & Job Matching

FitCheck scores how well a resume (PDF) matches a job description and explains why.

```
Resume PDF -> text extraction -> skill extraction -> TF-IDF features -> ML model
                                                                          |
   Match score, skills matched / missing, recommendation  <---------------+
```

## Quick start

```bash
pip install -r requirements.txt

python generate_data.py          # synthetic labeled resumes + job descriptions -> data/
python make_sample_pdfs.py       # 4 sample resume PDFs + 2 JDs -> samples/
python train.py                  # trains LR / Naive Bayes / SVM / Random Forest -> models/, outputs/
python evaluate.py               # ranking evaluation + ablation -> outputs/

python predict.py samples/resume_strong_data_scientist.pdf samples/jd_data_scientist.txt
streamlit run app.py             # FitCheck web app (single candidate + rank many candidates)
```

## How it works

| Step | File | What happens |
|---|---|---|
| Text extraction | `extract.py` | `pdfplumber` reads the PDF; cleaning removes URLs, emails and phone numbers |
| Skill extraction | `skills.py` | Taxonomy of ~120 skills with aliases (`ML` = machine learning, `k8s` = Kubernetes); whole-token regex matching |
| Features + models | `train.py` | TF-IDF (1-2 grams, sublinear tf) into Logistic Regression, Naive Bayes, linear SVM, Random Forest |
| Scoring | `matcher.py` | Hybrid score (below) |
| Evaluation | `evaluate.py` | Ranking metrics with an ablation of each signal |
| Demo | `app.py`, `predict.py` | Streamlit UI and command-line scorecard |

### The score

`score = 100 x (0.6 x skill overlap + 0.2 x text similarity + 0.2 x category alignment)`

- **Skill overlap**: share of the JD's skills found in the resume. Skills under a "Nice to have / Preferred" heading count half.
- **Text similarity**: TF-IDF cosine between resume and JD, divided by a reference value calibrated on training data (the 90th percentile of same-category cosine) and capped at 1.
- **Category alignment**: dot product of the classifier's category probabilities for the resume and for the JD.

Tiers: >=75 Strong Candidate, 60-75 Good Fit, 40-60 Partial Match, <40 Weak Match. Weights and tiers are judgment calls, not fitted values; change them in `models/config.json` / `matcher.py`.

## Results (on the synthetic dataset)

1,200 resumes, 8 categories, 80/20 stratified split, 5-fold CV on the training part.

| Model | CV macro-F1 | Test accuracy |
|---|---|---|
| Naive Bayes | 0.999 | 1.000 |
| SVM (linear) | 0.995 | 0.996 |
| Random Forest | 0.994 | 1.000 |
| Logistic Regression | 0.992 | 1.000 |

Plots: `outputs/confusion_matrix.png`, `outputs/model_comparison.png`, `outputs/ranking_ablation.png`.

**Read these numbers carefully.**
- The data is synthetic and built from keyword lists, so it is almost perfectly separable. The four models are statistically indistinguishable here. The comparison only becomes meaningful on real resumes (see below).
- In the ranking ablation, "category classifier only" scores 1.000 by construction, because the proxy relevance (same category) is exactly what that classifier was trained to predict. The result that does carry information: skill overlap alone ranks worst (P@10 0.78) because skills like Python, SQL and Git are shared across categories, and the hybrid fixes that.
- Neither result says anything about how well the score agrees with a human recruiter. For that, use the hand-labeling step below.

## Do these before you submit

1. **Run on real data.** Download the Kaggle "Resume Dataset" (`UpdatedResumeDataSet.csv`, columns `Category`, `Resume`) and run:
   ```bash
   python train.py --data UpdatedResumeDataSet.csv --text-col Resume --label-col Category
   ```
   The model comparison, confusion matrix and classification report will then reflect real, messy text. (Category calibration against the sample JDs is skipped if categories don't match, and a default is used.) You may want to extend `skills.py` with that dataset's categories (HR, Sales, Advocate...).
2. **Hand-label ~40 pairs.** `python evaluate.py` writes `data/hand_labels_template.csv`. Fill the `label` column (0 weak, 1 partial, 2 strong), save as `data/hand_labels.csv`, re-run to get the Spearman correlation between your labels and the score. Ask a friend to label independently and report agreement.
3. **Try a few of your own real PDFs** (with permission) to see where extraction or skill matching breaks.

## Limitations and fairness (worth a section in your report)

- Text-only PDFs; scanned resumes need OCR. Multi-column layouts can scramble text order.
- Skill matching is a fixed taxonomy: unknown skills are invisible, and mentioning a skill is not the same as having it.
- The classifier is overconfident (probabilities near 0 or 1), so the category component is close to binary.
- No experience, seniority or education modeling.
- Screening is a high-stakes area. Names, gender, age, schools and addresses can act as proxies for protected attributes. This system never uses names (the generator excludes them from training text), but real PDFs contain them, so the extractor should be extended to redact them before any real use. Treat the score as decision support for a human, not a decision.

## Report outline

1. Problem and motivation
2. Data (synthetic generator design, why it saturates; real dataset)
3. Preprocessing: PDF extraction, cleaning, skill taxonomy
4. Features: TF-IDF settings
5. Models and comparison (CV, test, confusion matrix)
6. Hybrid scoring design and weight choices
7. Evaluation: ranking metrics, ablation, hand-labeled pairs
8. Demo (screenshots of the app)
9. Limitations, bias and ethics
10. Future work: sentence embeddings, experience parsing, name redaction, OCR
