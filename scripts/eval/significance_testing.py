import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.stats import chi2

def main():
    print("="*80)
    # Load data
    print("Loading datasets...")
    with open("data/processed/cancer_pm_train.json", "r", encoding="utf-8") as f:
        train = json.load(f)
    with open("data/processed/cancer_pm_val.json", "r", encoding="utf-8") as f:
        val = json.load(f)

    train_texts = [r["title"] + " " + r["abstract"] for r in train]
    train_labels = np.array([r["label"] for r in train])
    val_texts = [r["title"] + " " + r["abstract"] for r in val]
    val_labels = np.array([r["label"] for r in val])

    print(f"Train size: {len(train_texts)}, Val size: {len(val_texts)}")

    # 1. Train TF-IDF Baseline
    print("\nTraining TF-IDF Baseline (min_df=2, C=1.0)...")
    vec_base = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train_base = vec_base.fit_transform(train_texts)
    X_val_base = vec_base.transform(val_texts)
    clf_base = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)
    clf_base.fit(X_train_base, train_labels)
    probs_base = clf_base.predict_proba(X_val_base)[:, 1]
    preds_base = (probs_base >= 0.37).astype(int)

    # 2. TF-IDF Combiné — load canonical C=15.0 model from joblib (do not re-train)
    print("Loading TF-IDF Combiné C=15.0 from models/tfidf_combined.joblib...")
    import joblib as _joblib
    _model_data = _joblib.load("models/tfidf_combined.joblib")
    vec_comb = _model_data["vectorizer"]
    clf_comb = _model_data["classifier"]
    _frozen_thresh = _model_data["threshold"]
    X_val_comb = vec_comb.transform(val_texts)
    probs_comb = clf_comb.predict_proba(X_val_comb)[:, 1]
    preds_comb = (probs_comb >= _frozen_thresh).astype(int)

    # 3. Train Character N-grams
    print("Training Character N-grams (char_wb 3-5, C=1.0)...")
    vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    X_train_char = vec_char.fit_transform(train_texts)
    X_val_char = vec_char.transform(val_texts)
    clf_char = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)
    clf_char.fit(X_train_char, train_labels)
    probs_char = clf_char.predict_proba(X_val_char)[:, 1]
    # Val-optimal threshold sweep for Char N-grams (do not hardcode)
    _char_best_thresh, _char_best_f1 = 0.5, 0.0
    for _t in np.arange(0.1, 0.9, 0.01):
        _p = (probs_char >= _t).astype(int)
        _f = f1_score(val_labels, _p, zero_division=0)
        if _f > _char_best_f1:
            _char_best_f1 = _f
            _char_best_thresh = round(float(_t), 2)
    print(f"  Char N-grams val-optimal threshold: {_char_best_thresh} (F1={_char_best_f1:.4f})")
    preds_char = (probs_char >= _char_best_thresh).astype(int)

    # 4. Train PubMedBERT (frozen)
    print("Extracting PubMedBERT frozen embeddings and training Logistic Regression (C=1.0)...")
    model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO", local_files_only=True)
    X_train_pm = model.encode(train_texts, batch_size=32, show_progress_bar=True)
    X_val_pm = model.encode(val_texts, batch_size=32, show_progress_bar=True)
    clf_pm = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)
    clf_pm.fit(X_train_pm, train_labels)
    probs_pm = clf_pm.predict_proba(X_val_pm)[:, 1]
    
    # Val-optimal threshold sweep for PubMedBERT (do not hardcode)
    _pm_best_thresh, _pm_best_f1 = 0.5, 0.0
    for _t in np.arange(0.1, 0.9, 0.01):
        _p = (probs_pm >= _t).astype(int)
        _f = f1_score(val_labels, _p, zero_division=0)
        if _f > _pm_best_f1:
            _pm_best_f1 = _f
            _pm_best_thresh = round(float(_t), 2)
    print(f"  PubMedBERT val-optimal threshold:   {_pm_best_thresh} (F1={_pm_best_f1:.4f})")
    preds_pm = (probs_pm >= _pm_best_thresh).astype(int)

    print("\n" + "="*50)
    print("REPLICATED METRICS (Validation)")
    print("="*50)
    print(f"TF-IDF Baseline: F1 = {f1_score(val_labels, preds_base):.4%}")
    print(f"TF-IDF Combiné:  F1 = {f1_score(val_labels, preds_comb):.4%}")
    print(f"Char N-grams:    F1 = {f1_score(val_labels, preds_char):.4%}")
    print(f"PubMedBERT:      F1 = {f1_score(val_labels, preds_pm):.4%}")

    # =========================================================================
    # McNemar's Test (paired contingency tables)
    # =========================================================================
    print("\n" + "="*50)
    print("MCNEMAR'S TEST RESULTS (vs. TF-IDF Combiné)")
    print("="*50)
    
    def run_mcnemar(preds_a, preds_b, name_a, name_b):
        b = int(((preds_a == val_labels) & (preds_b != val_labels)).sum())  # A correct, B wrong
        c = int(((preds_a != val_labels) & (preds_b == val_labels)).sum())  # A wrong, B correct
        # Edwards' continuity correction
        if b + c > 0:
            chi2_stat = ((abs(b - c) - 1.0) ** 2) / (b + c)
            p_val = chi2.sf(chi2_stat, 1)
        else:
            chi2_stat = 0.0
            p_val = 1.0
        print(f"McNemar: {name_b} vs {name_a}")
        print(f"  Contingency cell b (A correct, B wrong): {b}")
        print(f"  Contingency cell c (A wrong, B correct): {c}")
        print(f"  Chi2 statistic: {chi2_stat:.4f}")
        print(f"  p-value:        {p_val:.6f}")
        print(f"  Significant (p < 0.05): {p_val < 0.05}")
        return {"b": b, "c": c, "chi2": float(chi2_stat), "pvalue": float(p_val)}

    mcnemar_base = run_mcnemar(preds_base, preds_comb, "TF-IDF Baseline", "TF-IDF Combiné")
    mcnemar_char = run_mcnemar(preds_char, preds_comb, "Char N-grams", "TF-IDF Combiné")
    mcnemar_pm = run_mcnemar(preds_pm, preds_comb, "PubMedBERT (frozen)", "TF-IDF Combiné")

    # =========================================================================
    # Bootstrapping F1 Difference (2,000 iterations)
    # =========================================================================
    print("\n" + "="*50)
    print("BOOTSTRAP F1 DIFFERENCE (95% Confidence Intervals, 2000 iterations)")
    print("="*50)
    
    np.random.seed(42)
    n_iterations = 2000
    n_samples = len(val_labels)
    
    diff_base_list = []
    diff_char_list = []
    diff_pm_list = []
    
    for i in range(n_iterations):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_y = val_labels[idx]
        
        if len(np.unique(boot_y)) < 2:
            continue
            
        f1_comb = f1_score(boot_y, preds_comb[idx])
        f1_base = f1_score(boot_y, preds_base[idx])
        f1_char = f1_score(boot_y, preds_char[idx])
        f1_pm = f1_score(boot_y, preds_pm[idx])
        
        diff_base_list.append(f1_comb - f1_base)
        diff_char_list.append(f1_comb - f1_char)
        diff_pm_list.append(f1_comb - f1_pm)
        
    def get_ci(diff_list):
        lo = np.percentile(diff_list, 2.5)
        hi = np.percentile(diff_list, 97.5)
        mean_diff = np.mean(diff_list)
        return mean_diff, lo, hi

    mean_b, lo_b, hi_b = get_ci(diff_base_list)
    print(f"TF-IDF Combiné vs TF-IDF Baseline dF1:")
    print(f"  Mean Difference: {mean_b*100:+.2f}%")
    print(f"  95% CI: [{lo_b*100:.2f}%, {hi_b*100:.2f}%]")
    print(f"  Distinguishable from 0: {not (lo_b <= 0 <= hi_b)}")
    
    mean_c, lo_c, hi_c = get_ci(diff_char_list)
    print(f"TF-IDF Combiné vs Char N-grams dF1:")
    print(f"  Mean Difference: {mean_c*100:+.2f}%")
    print(f"  95% CI: [{lo_c*100:.2f}%, {hi_c*100:.2f}%]")
    print(f"  Distinguishable from 0: {not (lo_c <= 0 <= hi_c)}")
    
    mean_p, lo_p, hi_p = get_ci(diff_pm_list)
    print(f"TF-IDF Combiné vs PubMedBERT (frozen) dF1:")
    print(f"  Mean Difference: {mean_p*100:+.2f}%")
    print(f"  95% CI: [{lo_p*100:.2f}%, {hi_p*100:.2f}%]")
    print(f"  Distinguishable from 0: {not (lo_p <= 0 <= hi_p)}")

    # Save results
    results = {
        "mcnemar": {
            "TF-IDF Baseline": mcnemar_base,
            "Char N-grams": mcnemar_char,
            "PubMedBERT (frozen)": mcnemar_pm
        },
        "bootstrap_ci": {
            "TF-IDF Baseline": {"mean": float(mean_b), "lower": float(lo_b), "upper": float(hi_b)},
            "Char N-grams": {"mean": float(mean_c), "lower": float(lo_c), "upper": float(hi_c)},
            "PubMedBERT (frozen)": {"mean": float(mean_p), "lower": float(lo_p), "upper": float(hi_p)}
        }
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/significance_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSignificance testing results saved to results/significance_results.json")

if __name__ == "__main__":
    main()
