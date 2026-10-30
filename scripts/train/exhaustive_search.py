"""
Validation-only regularisation sweep for the Combined TF-IDF model.

TF-IDF (word 1-2 grams, min_df=5, sublinear_tf) + LogisticRegression
(class_weight='balanced'). For every C the decision threshold is swept over
[0.10, 0.89] on the validation split; no test or holdout data are read.
The (C, threshold) pair with the highest validation F1 is the configuration
frozen in scripts/train/train_tfidf_combined.py (C=15.0, t=0.35).

Output: results/hyperparameter_sweep_full.json
    grid   - validation F1 for every (C, threshold) pair
    per_C  - best threshold per C, with global and per-publisher metrics
    best   - the selected (C, threshold)
"""
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    RESULTS_DIR, THRESHOLD_GRID, PublisherLookup, labels, load_split, metrics_at, save_json,
    stratified_metrics, texts,
)

C_VALUES = [1.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 100.0, 300.0]


def main():
    train, val = load_split("train"), load_split("val")
    y_train, y_val = labels(train), labels(val)
    strata = PublisherLookup().strata(val)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True)
    X_train = vec.fit_transform(texts(train))
    X_val = vec.transform(texts(val))
    print(f"Vocabulary (min_df=5): {len(vec.vocabulary_):,} features")

    grid, per_c = [], []
    for C in C_VALUES:
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced",
                                 random_state=123, solver="lbfgs")
        probs = clf.fit(X_train, y_train).predict_proba(X_val)[:, 1]

        best_t, best_f1 = None, -1.0
        for t in THRESHOLD_GRID:
            t = round(float(t), 2)
            f1 = float(f1_score(y_val, (probs >= t).astype(int), zero_division=0))
            grid.append({"C": C, "threshold": t, "F1": f1})
            if f1 > best_f1:
                best_t, best_f1 = t, f1

        per_c.append({
            "C": C,
            "overall": metrics_at(probs, y_val, best_t),
            "stratified": stratified_metrics(probs, y_val, strata, best_t),
        })
        s = per_c[-1]["stratified"]
        print(f"C={C:<6} t={best_t:.2f} F1={best_f1:.4%} | Hindawi={s['hindawi']['f1']:.2%} "
              f"Spandidos={s['spandidos']['f1']:.2%} Others={s['others']['f1']:.2%}")

    best = max(per_c, key=lambda r: r["overall"]["f1"])
    selected = {"C": best["C"], "threshold": best["overall"]["threshold"], "F1": best["overall"]["f1"]}
    print(f"Selected: C={selected['C']} t={selected['threshold']} F1={selected['F1']:.4%}")
    save_json({
        "config": {"vectorizer": "TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True)",
                   "classifier": "LogisticRegression(class_weight='balanced', max_iter=2000, "
                                 "random_state=123)",
                   "vocabulary_size": len(vec.vocabulary_)},
        "grid": grid,
        "per_C": per_c,
        "best": selected,
    }, RESULTS_DIR / "hyperparameter_sweep_full.json")


if __name__ == "__main__":
    main()
