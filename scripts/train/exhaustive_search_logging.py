import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def get_best_threshold_and_log(clf, vec, train_texts, val_texts, train_labels, val_labels, C_val, sweep_log):
    X_train = vec.fit_transform(train_texts)
    X_val = vec.transform(val_texts)
    clf.fit(X_train, train_labels)
    probs_val = clf.predict_proba(X_val)[:, 1]
    
    best_thresh = 0.5
    best_f1 = 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        t_rounded = float(round(t, 2))
        preds = (probs_val >= t_rounded).astype(int)
        f1 = float(f1_score(val_labels, preds, zero_division=0))
        
        # Log every single triple
        sweep_log.append({"C": C_val, "threshold": t_rounded, "F1": f1})
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t_rounded
    return best_thresh, best_f1

def main():
    print("Loading data...")
    with open("data/processed/cancer_pm_train.json", "r", encoding="utf-8") as f:
        train = json.load(f)
    with open("data/processed/cancer_pm_val.json", "r", encoding="utf-8") as f:
        val = json.load(f)

    train_texts = [r["title"] + " " + r["abstract"] for r in train]
    val_texts = [r["title"] + " " + r["abstract"] for r in val]

    train_labels = np.array([int(r["label"]) for r in train])
    val_labels = np.array([int(r["label"]) for r in val])

    results = []
    sweep_log = []

    print("Running exhaustive search for C with class_weight='balanced'...")
    for C in [1.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 100.0, 300.0]:
        vec = TfidfVectorizer(ngram_range=(1,2), min_df=5, sublinear_tf=True)
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=123, solver="lbfgs")
        thresh, f1 = get_best_threshold_and_log(clf, vec, train_texts, val_texts, train_labels, val_labels, C, sweep_log)
        results.append((f"C={C}", thresh, f1))

    print(f"\n{'Config Variant':<20} | {'Threshold':<10} | {'F1 Score':<10}")
    print("-" * 46)
    for name, thresh, f1 in results:
        print(f"{name:<20} | {thresh:<10.2f} | {f1:<10.4%}")
        
    os.makedirs("results", exist_ok=True)
    out_path = "results/hyperparameter_sweep_full.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sweep_log, f, indent=4)
    print(f"\nFull sweep logged to {out_path}")

if __name__ == "__main__":
    main()
