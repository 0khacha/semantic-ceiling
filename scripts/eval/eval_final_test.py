"""
Final test-set and Hindawi-holdout evaluation of the fine-tuned transformers.

Each model (epoch-3 checkpoint) is evaluated at two operating points, both
fixed before the test/holdout data are scored:
  * "val_optimal": the model's own F1-optimal validation threshold, read from
    results/{model}_finetuned_results.json (PubMedBERT 0.28, SciBERT 0.23);
  * "common":      the shared threshold t = 0.35 frozen for Combined TF-IDF,
    used for the like-for-like comparison tables in the thesis (§5.5, §5.6).
AUC and PR-AUC do not depend on the threshold.

On the Hindawi holdout each model is also compared with Combined TF-IDF
(t=0.35) at its val-optimal threshold: McNemar's test and a 2,000-iteration
paired bootstrap of dF1 = F1(model) - F1(Combined TF-IDF).

Output: results/finetuned_test_results.json
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    FINETUNED, MODELS_DIR, RESULTS_DIR, PublisherLookup, finetuned_probabilities, labels,
    load_split, mcnemar, metrics_at, save_json, stratified_metrics, texts,
)

COMMON_THRESHOLD = 0.35
N_BOOT = 2000
SEED = 42


def evaluate(probs, y, strata, threshold):
    return {
        "test_global": metrics_at(probs["test"], y["test"], threshold),
        "test_stratified": stratified_metrics(probs["test"], y["test"], strata, threshold),
        "holdout": metrics_at(probs["holdout"], y["holdout"], threshold),
    }


def holdout_significance(y, preds_model, preds_ref):
    rng = np.random.RandomState(SEED)
    diffs = []
    for _ in range(N_BOOT):
        idx = rng.choice(len(y), len(y), replace=True)
        diffs.append(f1_score(y[idx], preds_model[idx]) - f1_score(y[idx], preds_ref[idx]))
    return {
        "mcnemar": mcnemar(y, preds_model, preds_ref),
        "delta_f1": {"mean": float(np.mean(diffs)),
                     "lower": float(np.percentile(diffs, 2.5)),
                     "upper": float(np.percentile(diffs, 97.5))},
    }


def main():
    records = {s: load_split(s) for s in ("test", "holdout")}
    y = {s: labels(r) for s, r in records.items()}
    strata = PublisherLookup().strata(records["test"])

    tfidf = joblib.load(MODELS_DIR / "tfidf_combined.joblib")
    preds_tfidf = (tfidf["classifier"].predict_proba(
        tfidf["vectorizer"].transform(texts(records["holdout"])))[:, 1]
        >= tfidf["threshold"]).astype(int)

    output = {"common_threshold": COMMON_THRESHOLD}
    for key in FINETUNED:
        with open(RESULTS_DIR / f"{key}_finetuned_results.json", encoding="utf-8") as f:
            val_threshold = json.load(f)["best_threshold"]

        print(f"--- {key}: scoring test and holdout with {key}_epoch_3.pt")
        probs = {s: finetuned_probabilities(key, s) for s in records}

        output[key] = {
            "val_optimal_threshold": val_threshold,
            "val_optimal": evaluate(probs, y, strata, val_threshold),
            "common": evaluate(probs, y, strata, COMMON_THRESHOLD),
            "holdout_vs_combined_tfidf": holdout_significance(
                y["holdout"], (probs["holdout"] >= val_threshold).astype(int), preds_tfidf),
        }
        for point in ("val_optimal", "common"):
            res = output[key][point]
            g, h = res["test_global"], res["holdout"]
            print(f"  [{point:<11} t={g['threshold']:.2f}] test F1={g['f1']:.4f} AUC={g['auc']:.4f} | "
                  f"holdout F1={h['f1']:.4f} AUC={h['auc']:.4f}", flush=True)
        sig = output[key]["holdout_vs_combined_tfidf"]
        print(f"  holdout vs Combined TF-IDF: McNemar p={sig['mcnemar']['pvalue']:.4f}, "
              f"dF1={100 * sig['delta_f1']['mean']:+.2f} pp "
              f"[{100 * sig['delta_f1']['lower']:+.2f}, {100 * sig['delta_f1']['upper']:+.2f}]")

    save_json(output, RESULTS_DIR / "finetuned_test_results.json")


if __name__ == "__main__":
    main()
