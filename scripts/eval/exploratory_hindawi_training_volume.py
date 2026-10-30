"""
EXPLORATORY - not part of the main evaluation protocol.

Does more Hindawi training data lift the Hindawi ceiling? Combined TF-IDF
(same hyperparameters as the canonical model) is re-trained on train + the
whole Hindawi holdout, its threshold is re-selected on validation, and it is
scored on the validation and test Hindawi strata. Because the holdout is used
for training here, these numbers must not be compared with the holdout results.

Output: results/exploratory_training_volume/augmented_hindawi_results.json
"""
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    RESULTS_DIR, PublisherLookup, best_threshold, labels, load_split, metrics_at, save_json,
    texts, trivial_f1,
)


def main():
    train, val, test, holdout = (load_split(s) for s in ("train", "val", "test", "holdout"))
    augmented = train + holdout
    print(f"Training on train ({len(train)}) + holdout ({len(holdout)}) = {len(augmented)} records")

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True)
    clf = LogisticRegression(C=15.0, class_weight="balanced", max_iter=1000,
                             random_state=123, solver="lbfgs")
    clf.fit(vec.fit_transform(texts(augmented)), labels(augmented))

    lookup = PublisherLookup()
    probs_val = clf.predict_proba(vec.transform(texts(val)))[:, 1]
    t = best_threshold(probs_val, labels(val))
    print(f"Validation-optimal threshold: {t:.2f}")

    results = {
        "experiment": "Exploratory Hindawi training-volume check (train + holdout)",
        "note": "EXPLORATORY - not part of the main protocol; the holdout is used for training.",
        "hyperparameters": {
            "C": 15.0,
            "class_weight": "balanced",
            "vectorizer": "TfidfVectorizer(ngram_range=(1,2), min_df=5, sublinear_tf=True)",
            "optimal_val_threshold": t,
        },
    }
    for name, records, probs in (("val", val, probs_val),
                                 ("test", test, clf.predict_proba(vec.transform(texts(test)))[:, 1])):
        mask = lookup.strata(records) == "hindawi"
        y = labels(records)[mask]
        res = metrics_at(probs[mask], y, t)
        res["trivial_baseline_f1"] = trivial_f1(y)
        results[f"{name}_hindawi"] = res
        print(f"{name:<5} Hindawi ({res['n_pos']}/{res['n_neg']}): F1={res['f1']:.4f} "
              f"AUC={res['auc']:.4f} trivial F1={res['trivial_baseline_f1']:.4f}")

    save_json(results, RESULTS_DIR / "exploratory_training_volume" / "augmented_hindawi_results.json")


if __name__ == "__main__":
    main()
