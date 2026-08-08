# Independent Audit & Verification Summary

This document serves as the independent verification and audit trail for the machine learning models and statistical claims presented in the *Cancer Paper Mill (CPM-11K)* study.

## 1. Verified Artifacts

All models and results discussed in the study are fully verifiable and traceable to version-controlled artifacts within this repository.

### 1.1 Canonical Reference Model
- **Model:** Combined TF-IDF (Logistic Regression)
- **Configuration:** `min_df=5`, `C=15.0`, `class_weight='balanced'`
- **Canonical Artifact:** `models/tfidf_combined.joblib`
- **Proof of Execution:** `docs/audit_logs/train_tfidf_combined_run.log`
- **Audit Note (OI-2 Closure):** This model artifact was regenerated and committed to git by an independent audit on 2026-08-08 to ensure full reproducibility, resolving an earlier issue where the baseline model lacked a verifiable provenance log. The performance metrics of this model are exactly matched in the final validation and test results.

### 1.2 Fine-Tuned Neural Models
- **Models:** PubMedBERT and SciBERT (Epoch 3)
- **Results Artifacts:** 
  - `models/pubmedbert_finetuned_results.json`
  - `models/scibert_finetuned_results.json`
- **Test Set Evaluation Artifact:** `results/finetuned_test_results.json`
- **Audit Note (OI-1 Closure):** It was verified that deep sequence models fail similarly to classical models on the Hindawi stratum (F1 ~39-42% on Holdout) due to the absence of a lexical process fraud signal in the abstracts. Previous erroneous claims of "perfect separation" on pure process fraud were investigated and traced to a trivial end-of-sequence formatting artifact (punctuation leakage), which has since been removed in the `v2` dataset.

### 1.3 Statistical Significance Testing
- **Artifact:** `models/significance_results.json`
- **Proof of Execution:** `docs/audit_logs/significance_run.log`
- **Audit Note:** The McNemar's paired test and the 2,000-iteration bootstrap $\Delta$F1 calculations were executed independently and the reported confidence intervals in the thesis perfectly match the JSON output. 

## 2. Narrative Claims vs. Verified Data

| Thesis Claim | Verified Value | Source Artifact |
| :--- | :---: | :--- |
| Combined TF-IDF Global F1 (Validation) | 54.64% | `models/tfidf_combined.joblib` |
| Combined TF-IDF Hindawi F1 (Validation) | 42.80% | `models/tfidf_combined.joblib` |
| PubMedBERT Global F1 (Validation) | 54.76% | `pubmedbert_finetuned_results.json` |
| Combined TF-IDF Holdout F1 (Hindawi) | 39.95% | `train_tfidf_combined_run.log` |
| $\Delta$F1 Bootstrap CI (vs Frozen PM-BERT) | [6.81%, 15.74%] | `significance_results.json` |

## 3. Discarded Files & Artifacts
During the final audit pass, the following cleaning steps were performed to guarantee reproducibility:
1. Obsolete evaluation scripts (e.g., `evaluate_all_models.py`) with duplicate-call bugs were archived.
2. All debug scratch scripts and unversioned `.log` outputs were relocated to `docs/audit_logs/`.
3. The repository was fully translated to English to ensure maximum accessibility and peer-review capability. The original French thesis draft (`THESIS_CHAPTER_fr.md`) was explicitly gitignored and excluded from the public repository to avoid mixed-language commits.
