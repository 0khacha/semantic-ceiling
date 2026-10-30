"""
Validation, test and Hindawi-holdout evaluation of the canonical Combined
TF-IDF model (models/tfidf_combined.joblib: min_df=5, C=15.0, balanced LR).

The decision threshold is the model's F1-optimal validation threshold
(stored in the joblib, t=0.35); it is applied unchanged to test and holdout.
Stratified metrics (Hindawi / Spandidos / Others) are reported for val and test.

Output: results/tfidf_combined_results.json
"""
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    MODELS_DIR, RESULTS_DIR, PublisherLookup, best_threshold, labels, load_split, metrics_at,
    save_json, stratified_metrics, texts,
)


def main():
    model = joblib.load(MODELS_DIR / "tfidf_combined.joblib")
    vec, clf, threshold = model["vectorizer"], model["classifier"], model["threshold"]
    lookup = PublisherLookup()

    results = {
        "model": {
            "classifier": "LogisticRegression",
            "C": clf.C,
            "class_weight": clf.class_weight,
            "vectorizer": "TfidfVectorizer",
            "ngram_range": list(vec.ngram_range),
            "min_df": vec.min_df,
            "sublinear_tf": vec.sublinear_tf,
            "vocabulary_size": len(vec.vocabulary_),
        },
        "frozen_threshold": threshold,
    }
    for split in ("val", "test", "holdout"):
        records = load_split(split)
        y = labels(records)
        probs = clf.predict_proba(vec.transform(texts(records)))[:, 1]
        if split == "val":
            assert best_threshold(probs, y) == threshold, "stored threshold is not val-optimal"
        results[split] = {"overall": metrics_at(probs, y, threshold)}
        if split != "holdout":
            results[split]["stratified"] = stratified_metrics(
                probs, y, lookup.strata(records), threshold)

        o = results[split]["overall"]
        print(f"{split:<8} t={threshold:.2f} P={o['precision']:.4f} R={o['recall']:.4f} "
              f"F1={o['f1']:.4f} AUC={o['auc']:.4f} PR-AUC={o['pr_auc']:.4f}")
        for s, m in results[split].get("stratified", {}).items():
            print(f"  {s:<10} ({m['n_pos']}/{m['n_neg']}) F1={m['f1']:.4f} AUC={m['auc']:.4f}")

    save_json(results, RESULTS_DIR / "tfidf_combined_results.json")


if __name__ == "__main__":
    main()
