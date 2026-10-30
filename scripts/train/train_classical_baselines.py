"""
Linear SVM and XGBoost baselines on the Combined TF-IDF vocabulary.

Features: TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True).
  * LinearSVC(class_weight='balanced'); the threshold is swept over the
    decision function in [-2.0, 2.0) with step 0.05.
  * XGBClassifier(100 trees, depth 6, lr 0.1, scale_pos_weight = neg/pos);
    the threshold is swept over probabilities in [0.10, 0.89].
Thresholds are chosen on validation; no test or holdout data are read.

Output: results/classical_baselines_results.json
"""
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    RESULTS_DIR, PublisherLookup, best_threshold, labels, load_split, metrics_at, save_json,
    stratified_metrics, texts,
)


def report(name, scores, y, strata, grid):
    t = best_threshold(scores, y, grid)
    res = {"overall": metrics_at(scores, y, t), "stratified": stratified_metrics(scores, y, strata, t)}
    o, s = res["overall"], res["stratified"]
    print(f"{name:<8} t={t:+.2f} P={o['precision']:.4f} R={o['recall']:.4f} F1={o['f1']:.4f} "
          f"AUC={o['auc']:.4f} | Hindawi F1={s['hindawi']['f1']:.4f} "
          f"Spandidos F1={s['spandidos']['f1']:.4f} Others F1={s['others']['f1']:.4f}")
    return res


def main():
    train, val = load_split("train"), load_split("val")
    y_train, y_val = labels(train), labels(val)
    strata = PublisherLookup().strata(val)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True)
    X_train = vec.fit_transform(texts(train))
    X_val = vec.transform(texts(val))

    svm = LinearSVC(class_weight="balanced", random_state=42, dual="auto").fit(X_train, y_train)
    svm_scores = svm.decision_function(X_val)

    xgb_clf = xgb.XGBClassifier(
        random_state=42, eval_metric="logloss", n_estimators=100, max_depth=6,
        learning_rate=0.1, scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum())
    xgb_probs = xgb_clf.fit(X_train, y_train).predict_proba(X_val)[:, 1]

    save_json({
        "svm": report("SVM", svm_scores, y_val, strata, np.arange(-2.0, 2.0, 0.05)),
        "xgboost": report("XGBoost", xgb_probs, y_val, strata, np.arange(0.1, 0.9, 0.01)),
    }, RESULTS_DIR / "classical_baselines_results.json")


if __name__ == "__main__":
    main()
