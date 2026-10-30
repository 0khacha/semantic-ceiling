"""
Validation-set analyses behind thesis §5.1-5.3 and the §5.8 permutation check.

All scores come from the canonical Combined TF-IDF model
(models/tfidf_combined.joblib, C=15.0, t=0.35) unless stated otherwise.

  stratum_ci            - bootstrap 95% CIs of F1 / AUC per publisher stratum
  probability_stats     - predicted-probability summaries per stratum and class
  retraction_causes     - RWDB retraction reasons of train+val positives (by DOI)
  dose_response         - Hindawi positives split into subgroup A (RWDB reason
                          mentions computer-aided/-generated content) and
                          subgroup B (all others); AUC vs all Hindawi negatives
                          with bootstrap CIs, for Combined TF-IDF, frozen
                          PubMedBERT (S-PubMedBert-MS-MARCO + LR) and the two
                          fine-tuned transformers
  permutation_test      - 50 label permutations (numpy default_rng seeds 0-49)
                          of the validation labels against fixed predictions

Output: results/hindawi_analysis_results.json
"""
import csv
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    FROZEN_PUBMEDBERT_NAME, MODELS_DIR, RESULTS_DIR, RWDB_CSV, STRATA, PublisherLookup,
    finetuned_probabilities, frozen_embedding_probs, labels, load_split, save_json, texts,
)

N_BOOT = 2000
SEED = 42
N_PERMUTATIONS = 50
CAUSES = {
    "image_duplication": ("duplication of/in image",),
    "peer_review_manipulation": ("peer review",),
    "citation_manipulation": ("referencing/attributions", "citation"),
    "computer_generated_content": ("computer-aided content", "computer-generated content"),
}


def bootstrap_ci(y, probs, threshold=None, seed=SEED):
    """Percentile 95% CIs; F1 only when a threshold is given."""
    rng = np.random.RandomState(seed)
    preds = None if threshold is None else (probs >= threshold).astype(int)
    f1s, aucs = [], []
    for _ in range(N_BOOT):
        idx = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], probs[idx]))
        if preds is not None:
            f1s.append(f1_score(y[idx], preds[idx], zero_division=0))
    out = {"auc": float(roc_auc_score(y, probs)),
           "auc_ci": [float(v) for v in np.percentile(aucs, [2.5, 97.5])]}
    if preds is not None:
        out["f1"] = float(f1_score(y, preds, zero_division=0))
        out["f1_ci"] = [float(v) for v in np.percentile(f1s, [2.5, 97.5])]
    return out


def summary(p):
    q25, q75 = np.percentile(p, [25, 75])
    return {"n": int(len(p)), "mean": float(p.mean()), "median": float(np.median(p)),
            "std": float(p.std()), "q25": float(q25), "q75": float(q75),
            "share_below_0.10": float((p < 0.10).mean())}


def rwdb_reasons():
    reasons = {}
    with open(RWDB_CSV, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            doi = row.get("OriginalPaperDOI", "").strip().lower()
            if doi:
                reasons[doi] = row.get("Reason", "").lower()
    return reasons


def has_cause(reason, cause):
    return any(k in reason for k in CAUSES[cause])


def main():
    lookup = PublisherLookup()
    reasons = rwdb_reasons()
    train, val = load_split("train"), load_split("val")
    y = labels(val)
    strata = lookup.strata(val)

    model = joblib.load(MODELS_DIR / "tfidf_combined.joblib")
    threshold = model["threshold"]
    probs = model["classifier"].predict_proba(model["vectorizer"].transform(texts(val)))[:, 1]
    preds = (probs >= threshold).astype(int)
    results = {"model": "models/tfidf_combined.joblib", "threshold": threshold}

    results["stratum_ci"] = {
        s: bootstrap_ci(y[strata == s], probs[strata == s], threshold) for s in STRATA}

    results["probability_stats"] = {
        s: {"positives": summary(probs[(strata == s) & (y == 1)]),
            "negatives": summary(probs[(strata == s) & (y == 0)])} for s in STRATA}

    causes = {}
    for pub in ("Hindawi", "Spandidos"):
        found = [reasons.get(r["doi"].strip().lower()) for r in train + val
                 if r["label"] == 1 and lookup.publisher(r) == pub]
        matched = [x for x in found if x is not None]
        causes[pub.lower()] = {"n_positives": len(found), "n_matched": len(matched)}
        for cause in CAUSES:
            k = sum(has_cause(x, cause) for x in matched)
            causes[pub.lower()][cause] = {"count": k, "share": k / len(matched)}
    results["retraction_causes"] = causes

    hindawi = np.where(strata == "hindawi")[0]
    in_a = np.array([y[i] == 1 and has_cause(reasons.get(val[i]["doi"].strip().lower(), ""),
                                             "computer_generated_content") for i in hindawi])
    neg = hindawi[y[hindawi] == 0]
    groups = {"A_computer_generated": hindawi[in_a],
              "B_other_process_fraud": hindawi[(y[hindawi] == 1) & ~in_a]}
    model_probs = {"combined_tfidf": probs,
                   "pubmedbert_frozen": frozen_embedding_probs(FROZEN_PUBMEDBERT_NAME, "val"),
                   "pubmedbert_finetuned": finetuned_probabilities("pubmedbert", "val"),
                   "scibert_finetuned": finetuned_probabilities("scibert", "val")}
    dose = {}
    for m, p in model_probs.items():
        dose[m] = {}
        for g, pos in groups.items():
            idx = np.concatenate([pos, neg])
            dose[m][g] = {"n_pos": int(len(pos)), "n_neg": int(len(neg)),
                          **bootstrap_ci(y[idx], p[idx])}
        (a_lo, a_hi), (b_lo, b_hi) = (dose[m][g]["auc_ci"] for g in groups)
        dose[m]["ci_overlap"] = [max(a_lo, b_lo), min(a_hi, b_hi)]
    results["dose_response"] = dose

    perm_f1, perm_auc = [], []
    for seed in range(N_PERMUTATIONS):
        y_perm = np.random.default_rng(seed).permutation(y)
        perm_f1.append(float(f1_score(y_perm, preds)))
        perm_auc.append(float(roc_auc_score(y_perm, probs)))
    f1_obs, auc_obs = float(f1_score(y, preds)), float(roc_auc_score(y, probs))
    results["permutation_test"] = {
        "seeds": list(range(N_PERMUTATIONS)),
        "observed": {"f1": f1_obs, "auc": auc_obs},
        "f1": perm_f1,
        "auc": perm_auc,
        "z_f1": (f1_obs - np.mean(perm_f1)) / np.std(perm_f1),
        "z_auc": (auc_obs - np.mean(perm_auc)) / np.std(perm_auc),
        "empirical_p_f1": float(np.mean(np.array(perm_f1) >= f1_obs)),
        "empirical_p_auc": float(np.mean(np.array(perm_auc) >= auc_obs)),
    }

    for s, r in results["stratum_ci"].items():
        print(f"{s:<10} F1={r['f1']:.4f} {np.round(r['f1_ci'], 4)} AUC={r['auc']:.4f} "
              f"{np.round(r['auc_ci'], 4)}")
    for m, d in dose.items():
        for g in groups:
            print(f"{m:<18} {g:<22} N={d[g]['n_pos']:<3} AUC={d[g]['auc']:.4f} "
                  f"{np.round(d[g]['auc_ci'], 4)}")
    pt = results["permutation_test"]
    print(f"Permutation: F1 mean={np.mean(perm_f1):.4f} sd={np.std(perm_f1):.4f} "
          f"z={pt['z_f1']:.1f}; AUC mean={np.mean(perm_auc):.4f} z={pt['z_auc']:.1f}")
    save_json(results, RESULTS_DIR / "hindawi_analysis_results.json")


if __name__ == "__main__":
    main()
