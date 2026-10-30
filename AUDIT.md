# Audit and Provenance

Where every reported number comes from, and what was corrected during the reproducibility audits. All scripts run from any directory and read the canonical v2 splits in `data/processed/`.

## 1. Provenance map

| Reported quantity | Script | Output |
|---|---|---|
| C sweep and per-publisher sub-sweep table (thesis §4.1) | `scripts/train/exhaustive_search.py` | `results/hyperparameter_sweep_full.json` |
| Canonical Combined TF-IDF model (C=15.0, t=0.35) | `scripts/train/train_tfidf_combined.py` | `models/tfidf_combined.joblib` |
| Combined TF-IDF val / test / holdout metrics | `scripts/eval/eval_tfidf_combined.py` | `results/tfidf_combined_results.json` |
| Linear SVM, XGBoost | `scripts/train/train_classical_baselines.py` | `results/classical_baselines_results.json` |
| MiniLM and S-PubMedBert frozen embeddings | `scripts/eval/eval_frozen_embeddings.py` | `results/frozen_embeddings_results.json` |
| Fine-tuned PubMedBERT / SciBERT, validation | `scripts/train/finetune_transformer.py` | `results/{pubmedbert,scibert}_finetuned_results.json` |
| TF-IDF baseline, char n-grams, McNemar, Holm, bootstrap ΔF1 | `scripts/eval/significance_testing.py` | `results/significance_results.json` |
| Fine-tuned test / holdout, holdout significance | `scripts/eval/eval_final_test.py` | `results/finetuned_test_results.json` |
| Stratum CIs, probability statistics, retraction causes, dose-response, permutation test | `scripts/eval/hindawi_analysis.py` | `results/hindawi_analysis_results.json` |
| Exploratory train + holdout run | `scripts/eval/exploratory_hindawi_training_volume.py` | `results/exploratory_training_volume/` |
| v1 → v2 title normalisation | `scripts/data_build/strip_title_punctuation.py` | `data/processed/` |
| All figures | `scripts/viz/*.py` | `docs/figures/` |

Console logs of the regenerating runs are in `docs/audit_logs/`.

## 2. Reproduction status (re-run on 2026-10-30, CPU, package versions in `requirements.txt`)

**Reproduced exactly** (identical to the previously committed JSON, or to the thesis to the reported precision):

- C sweep grid (all 800 (C, t) pairs), including every row of the thesis sub-sweep table.
- Combined TF-IDF. Retraining gives predictions identical to the committed `tfidf_combined.joblib` on val/test/holdout. All val/test/holdout metrics match; vocabulary sizes are 104,138 (min_df=2) and 36,929 (min_df=5).
- Linear SVM and XGBoost, overall and stratified.
- Fine-tuned models at their validation thresholds on test and holdout.
- McNemar and bootstrap for TF-IDF baseline, char n-grams and both fine-tuned models.
- Retraction-reason shares (239/312, 291/291, 180/291) and subgroup sizes (82/52).
- The 50-seed permutation test (mean 19.39%, sd 2.19%, 16.1σ / 14.2σ).
- Spandidos bootstrap CIs.
- Fine-tuned subgroup-B AUCs (62.54% / 66.44%).
- Dataset counts, split × publisher and split × year tables, and abstract-length statistics.
- v2 data from the archived v1 files: every record of all splits matches.

**Re-derived, replacing earlier values that could not be reproduced:**

| Quantity | Earlier value | Reproducible value | Likely cause |
|---|---|---|---|
| Char n-grams val P / R / F1 / AUC | 51.66 / 43.25 / 47.08 / 78.21 | 46.25 / 46.43 / 46.34 / 77.49 | unversioned earlier configuration; the significance test already used the reproducible model |
| MiniLM val F1 / AUC | 39.61 / 69.28 | 39.74 / 68.83 | embedding/library version |
| Frozen PubMedBERT val F1 / AUC | 45.62 / 78.22 | 44.51 / 74.71 | embedding/library version |
| Frozen PubMedBERT McNemar χ², ΔF1 | 28.56, +11.07 [6.81, 15.74] | 40.51, +10.21 [5.89, 14.54] | as above (JSON before re-run: 38.06, +10.17) |
| Char n-grams McNemar χ² in thesis prose | 0.025 (p = 0.873) | 1.59 (p = 0.207) | prose never updated from an older run |
| Holm-adjusted p (char, fine-tuned) | 0.873 / 0.791 / 0.887 | 0.622 / 0.622 / 0.622 | earlier values were not monotone Holm adjustments |
| Fine-tuned P / R / PR-AUC (val) | 48.02 / 63.89 / 56.12 (PubMedBERT), 52.61 / 59.13 / 58.41 (SciBERT) | 47.26 / 65.08 / 58.15, 51.70 / 60.32 / 59.19 | thesis table not synced with training JSON |
| Fine-tuned test/holdout at t = 0.35 | e.g. PubMedBERT test F1 49.52, holdout 42.07 | 49.24, 42.76 | no script had produced the t = 0.35 numbers; `eval_final_test.py` now does |
| Hindawi / Others bootstrap CIs | e.g. Hindawi F1 [35.15, 50.20] | [34.98, 50.18] | unversioned resampling code |
| Dose-response AUCs (TF-IDF A / B) | 63.39 / 63.07 | 62.89 / 62.98 | stale C=10 model |
| Feature weights / case probabilities (§5.4) | e.g. `microrna` +8.04 | +8.97 | stale C=10 model |
| Char n-gram vocabulary | 143,989 | 143,742 | |
| Crossref-sourced positives | 22 (98.5% PubMed) | 42 (98.0% PubMed) | |

**Not re-verified:** the §5.9 formatting-artifact ablation (run on v1 data with the C=10 model through `archive/diagnostics/`; the thesis now says so), the funding-term audit in the dataset card, and the purge-cascade counts (taken from the dataset-build run logs; the upstream step is not versioned).

## 3. Earlier audit items (OI-1 … OI-5)

- **Model selection leakage (C=10.0).** The original Combined TF-IDF configuration was reconstructed by matching test and holdout metrics. It was replaced by C=15.0, selected on validation only by `exhaustive_search.py`. C=10.0 numbers survive only as marked DO-NOT-CITE notes.
- **Model provenance (OI-2).** `models/tfidf_combined.joblib` was regenerated with a logged run (`docs/audit_logs/train_tfidf_combined_run.log`, 2026-08-08); this audit confirms that retraining reproduces it.
- **Fine-tuned thresholds (OI-4).** Test/holdout evaluation used a hardcoded t = 0.36; it now reads each model's stored validation threshold, and also reports the common t = 0.35.
- **Punctuation artifact (§5.8).** The v1 period asymmetry produced 100% AUC. The v2 fix is now a versioned script that reproduces the canonical data.

## 4. Repository clean-up in this audit

- Removed scratch scripts (`docs/audit_logs/inspect_bak*.py`, `check_new_joblib.py`) and superseded UTF-16 logs.
- Merged the duplicated scripts: `exhaustive_search_logging.py` into `exhaustive_search.py`, and `finetune_{pubmedbert,scibert}.py` into `finetune_transformer.py`. `run_holdout_significance.py` (hardcoded threshold, no saved output) was folded into `eval_final_test.py`.
- Moved deprecated C=10 diagnostics (`compute_test_strata_bootstrap_ci.py`, `others_ci_verification.json`) and the v1 result file `all_evaluation_results.json` into `archive/`.
- Unified data paths (`data/final*` → `data/interim/` and `data/processed/`) and factored shared code into `scripts/common.py`.
- Regenerated every figure from code. Previously the permutation histogram was a simulated normal sample, the dose-response and retraction-cause figures were hard-coded, and the forest plot clipped negative CIs at −1.
- Corrected the fine-tuning batch size: the effective batch is 8 (4 × 2 accumulation), not 16.
