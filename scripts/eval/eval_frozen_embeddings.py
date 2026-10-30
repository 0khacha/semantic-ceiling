"""
Validation metrics for the frozen-embedding baselines.

Each sentence-transformer encodes title + abstract; a LogisticRegression
(C=1.0, class_weight='balanced') is fitted on the train split and its decision
threshold is swept on validation.
  * all-MiniLM-L6-v2            (general domain, 384-d)
  * S-PubMedBert-MS-MARCO       (biomedical, 768-d; "PubMedBERT (frozen)")

Output: results/frozen_embeddings_results.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    FROZEN_PUBMEDBERT_NAME, MINILM_NAME, RESULTS_DIR, PublisherLookup, best_threshold,
    frozen_embedding_probs, labels, load_split, metrics_at, save_json, stratified_metrics,
)

MODELS = {"minilm": MINILM_NAME, "pubmedbert_frozen": FROZEN_PUBMEDBERT_NAME}


def main():
    val = load_split("val")
    y = labels(val)
    strata = PublisherLookup().strata(val)

    results = {}
    for key, name in MODELS.items():
        probs = frozen_embedding_probs(name, "val")
        t = best_threshold(probs, y)
        results[key] = {
            "model": name,
            "overall": metrics_at(probs, y, t),
            "stratified": stratified_metrics(probs, y, strata, t),
        }
        o = results[key]["overall"]
        print(f"{key:<18} t={t:.2f} P={o['precision']:.4f} R={o['recall']:.4f} "
              f"F1={o['f1']:.4f} AUC={o['auc']:.4f} PR-AUC={o['pr_auc']:.4f}")

    save_json(results, RESULTS_DIR / "frozen_embeddings_results.json")


if __name__ == "__main__":
    main()
