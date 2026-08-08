> **Discarded exploratory branch — superseded by min_df=5 search, not cited in final results, retained for historical transparency.**

#### Regularization Sweep Table $C$ (TF-IDF, min_df=2)

| $C$ Value | Optimal Decision Threshold | Global F1 (Validation) | Hindawi F1 | Spandidos F1 | Others F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1.0 (Baseline)** | 0.37 | 46.91% | 39.15% | 84.00% | 47.77% |
| **10.0** | 0.29 | 52.50% | **41.58%** | 85.11% | 57.63% |
| **30.0** | 0.29 | 53.92% | 40.00% | 85.00% | 62.55% |
| **100.0** | 0.10 | 51.44% | 40.00% | 85.11% | 56.31% |
| **300.0** | 0.15 | **55.30%** | 40.77% | 93.02% | **63.78%** |
| **1,000.0** | 0.08 | 53.07% | 37.40% | 93.02% | 61.13% |
| **3,000.0** | 0.02 | 52.17% | 35.44% | 93.02% | 60.29% |
| **10,000.0** | 0.01 | 52.44% | 35.98% | 93.02% | 60.52% |

> **Reproducibility Note (min_df=2 sweep table):** The script that produced this table is not recoverable from the available Git history. This 8-point sweep with strata breakdown (Hindawi / Spandidos / Others) cannot be independently reproduced from versioned artifacts. The presented values should not be cited as verifiable results; they serve as exploration context data with unverifiable provenance.

*Analysis of the inflection point:* The global F1 score reaches its maximum at $C=300$ (55.30%), driven by the improvement of the "Others" stratum. However, beyond $C=10$, performance on the Hindawi stratum continuously declines (dropping from 41.58% to 35.98%). Furthermore, for $C \ge 100$ values, the optimal decision threshold collapses to aberrant values (0.10, 0.02, 0.01), indicating that predicted probabilities are abnormally concentrated near zero due to massive overfitting.