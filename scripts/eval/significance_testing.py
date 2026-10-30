"""
Paired significance tests on the validation split, reference = Combined TF-IDF.

For each compared model:
  * McNemar's test with continuity correction on paired correct/incorrect
    decisions (b = compared model correct & reference wrong, c = the reverse);
  * Holm-Bonferroni adjustment across the five comparisons;
  * 2,000-iteration paired bootstrap of dF1 = F1(Combined TF-IDF) - F1(model),
    so a positive dF1 favours Combined TF-IDF and a negative one the other model.

Decision thresholds: Combined TF-IDF uses its frozen t=0.35; TF-IDF Baseline
t=0.37; Char N-grams and frozen PubMedBERT use their validation-optimal
thresholds; fine-tuned models use the thresholds stored at training time.

Also stores full validation metrics (overall + per publisher) of the two
lexical baselines trained here (TF-IDF Baseline, Char N-grams).

Output: results/significance_results.json
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    FROZEN_PUBMEDBERT_NAME, MODELS_DIR, RESULTS_DIR, PublisherLookup, best_threshold,
    finetuned_probabilities, frozen_embedding_probs, holm, labels,
    load_split, mcnemar, metrics_at, save_json, stratified_metrics, texts,
)

N_BOOT = 2000
SEED = 42


def tfidf_lr_probs(vectorizer, train, val):
    X_train = vectorizer.fit_transform(texts(train))
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)
    clf.fit(X_train, labels(train))
    return clf.predict_proba(vectorizer.transform(texts(val)))[:, 1]


def main():
    train, val = load_split("train"), load_split("val")
    y = labels(val)

    probs, thresholds = {}, {}
    probs["TF-IDF Baseline"] = tfidf_lr_probs(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), train, val)
    thresholds["TF-IDF Baseline"] = 0.37

    model = joblib.load(MODELS_DIR / "tfidf_combined.joblib")
    probs["Combined TF-IDF"] = model["classifier"].predict_proba(
        model["vectorizer"].transform(texts(val)))[:, 1]
    thresholds["Combined TF-IDF"] = model["threshold"]

    probs["Char N-grams"] = tfidf_lr_probs(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True),
        train, val)
    probs["PubMedBERT (frozen)"] = frozen_embedding_probs(FROZEN_PUBMEDBERT_NAME, "val")
    for name in ("Char N-grams", "PubMedBERT (frozen)"):
        thresholds[name] = best_threshold(probs[name], y)

    for name, key in (("PubMedBERT (fine-tuned)", "pubmedbert"), ("SciBERT (fine-tuned)", "scibert")):
        with open(RESULTS_DIR / f"{key}_finetuned_results.json", encoding="utf-8") as f:
            thresholds[name] = json.load(f)["best_threshold"]
        probs[name] = finetuned_probabilities(key, "val")

    preds = {m: (probs[m] >= thresholds[m]).astype(int) for m in probs}
    ref = preds["Combined TF-IDF"]
    compared = ["TF-IDF Baseline", "Char N-grams", "PubMedBERT (frozen)",
                "PubMedBERT (fine-tuned)", "SciBERT (fine-tuned)"]

    tests = {m: mcnemar(y, preds[m], ref) for m in compared}
    for m, adj in holm({m: tests[m]["pvalue"] for m in compared}).items():
        tests[m]["pvalue_holm"] = adj

    np.random.seed(SEED)
    diffs = {m: [] for m in compared}
    for _ in range(N_BOOT):
        idx = np.random.choice(len(y), size=len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        f1_ref = f1_score(y[idx], ref[idx])
        for m in compared:
            diffs[m].append(f1_ref - f1_score(y[idx], preds[m][idx]))
    boot = {m: {"mean": float(np.mean(d)),
                "lower": float(np.percentile(d, 2.5)),
                "upper": float(np.percentile(d, 97.5))} for m, d in diffs.items()}

    print(f"{'Model':<26}{'F1':>8}{'chi2':>9}{'p':>11}{'p_holm':>11}  dF1 [95% CI]")
    for m in compared:
        t, bt = tests[m], boot[m]
        print(f"{m:<26}{f1_score(y, preds[m]):>8.4f}{t['chi2']:>9.3f}{t['pvalue']:>11.3g}"
              f"{t['pvalue_holm']:>11.3g}  {100 * bt['mean']:+.2f} "
              f"[{100 * bt['lower']:+.2f}, {100 * bt['upper']:+.2f}]")

    strata = PublisherLookup().strata(val)
    baselines = {m: {"overall": metrics_at(probs[m], y, thresholds[m]),
                     "stratified": stratified_metrics(probs[m], y, strata, thresholds[m])}
                 for m in ("TF-IDF Baseline", "Char N-grams")}

    save_json({
        "reference": "Combined TF-IDF",
        "baseline_val_metrics": baselines,
        "val_f1": {m: float(f1_score(y, preds[m])) for m in preds},
        "mcnemar": tests,
        "bootstrap_ci": boot,
        "thresholds_used": {m: float(t) for m, t in thresholds.items()},
    }, RESULTS_DIR / "significance_results.json")


if __name__ == "__main__":
    main()
