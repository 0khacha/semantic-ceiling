"""
Train the canonical Combined TF-IDF model.

    TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True)
    LogisticRegression(C=15.0, class_weight='balanced', max_iter=1000, random_state=123)

C=15.0 is the validation-only selection of scripts/train/exhaustive_search.py.
The decision threshold is the F1-optimal validation threshold (t=0.35); it is
stored with the model and reused unchanged for test and holdout evaluation.

Output: models/tfidf_combined.joblib  ({"vectorizer", "classifier", "threshold"})
"""
import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    MODELS_DIR, PublisherLookup, best_threshold, labels, load_split, metrics_at,
    stratified_metrics, texts,
)


def main():
    train, val = load_split("train"), load_split("val")
    y_train, y_val = labels(train), labels(val)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True)
    X_train = vec.fit_transform(texts(train))
    clf = LogisticRegression(C=15.0, class_weight="balanced", max_iter=1000,
                             random_state=123, solver="lbfgs")
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(vec.transform(texts(val)))[:, 1]
    t = best_threshold(probs, y_val)
    o = metrics_at(probs, y_val, t)
    print(f"Vocabulary: {len(vec.vocabulary_):,} features")
    print(f"Validation t={t:.2f} P={o['precision']:.4%} R={o['recall']:.4%} F1={o['f1']:.4%} "
          f"AUC={o['auc']:.4%} PR-AUC={o['pr_auc']:.4%}")
    for s, m in stratified_metrics(probs, y_val, PublisherLookup().strata(val), t).items():
        print(f"  {s:<10} ({m['n_pos']}/{m['n_neg']}) F1={m['f1']:.4%} AUC={m['auc']:.4%}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"vectorizer": vec, "classifier": clf, "threshold": t},
                MODELS_DIR / "tfidf_combined.joblib")
    print(f"Saved {MODELS_DIR / 'tfidf_combined.joblib'}")


if __name__ == "__main__":
    main()
