"""
Model-comparison figures (validation split), drawn only from results/*.json.

Outputs (docs/figures/):
    global_comparison.png          global F1 and ROC-AUC of all models
    stratified_comparison.png      F1 per publisher stratum, classical models
    finetuned_models.png           F1 per stratum including fine-tuned transformers
    significance_forest_plot.png   bootstrap dF1 CIs and McNemar p-values

Run after: significance_testing.py, eval_frozen_embeddings.py,
train_classical_baselines.py, eval_tfidf_combined.py, finetune_transformer.py.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    FIGURES_DIR, RESULTS_DIR, STRATA, PublisherLookup, labels, load_split, trivial_f1,
)

STRATUM_STYLE = {"hindawi": ("Hindawi (in-pool)", "#E06666"),
                 "spandidos": ("Spandidos", "#6FA8DC"),
                 "others": ("Pooled others", "#8FCE00")}

plt.style.use("seaborn-v0_8-whitegrid")


def load(name):
    with open(RESULTS_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def collect_models():
    """[(label, overall, stratified)] with metrics as fractions."""
    sig = load("significance_results.json")["baseline_val_metrics"]
    frozen = load("frozen_embeddings_results.json")
    classical = load("classical_baselines_results.json")
    tfidf = load("tfidf_combined_results.json")["val"]

    val = load_split("val")
    y, strata = labels(val), PublisherLookup().strata(val)
    trivial = {"overall": {"f1": trivial_f1(y), "auc": 0.5},
               "stratified": {s: {"f1": trivial_f1(y[strata == s])} for s in STRATA}}

    models = [("Trivial baseline", trivial),
              ("TF-IDF baseline", sig["TF-IDF Baseline"]),
              ("Combined TF-IDF", tfidf),
              ("Char n-grams", sig["Char N-grams"]),
              ("MiniLM (frozen)", frozen["minilm"]),
              ("PubMedBERT (frozen)", frozen["pubmedbert_frozen"]),
              ("Linear SVM", classical["svm"]),
              ("XGBoost", classical["xgboost"])]
    finetuned = [("PubMedBERT (fine-tuned)", load("pubmedbert_finetuned_results.json")),
                 ("SciBERT (fine-tuned)", load("scibert_finetuned_results.json"))]
    return models, finetuned


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=300)
    plt.close(fig)
    print(f"Saved docs/figures/{name}")


def global_comparison(models):
    names = [m for m, _ in models]
    f1 = [100 * r["overall"]["f1"] for _, r in models]
    auc = [100 * r["overall"]["auc"] for _, r in models]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12, 6))
    for offset, vals, label, color in ((-0.175, f1, "F1 (%)", "#E06666"),
                                       (0.175, auc, "ROC-AUC (%)", "#6FA8DC")):
        bars = ax.bar(x + offset, vals, 0.35, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Performance (%)", fontsize=12)
    ax.set_title("Global Model Comparison (Validation, N = 1,440)", fontsize=14,
                 fontweight="bold", pad=15)
    ax.legend(fontsize=11, loc="upper left")
    save(fig, "global_comparison.png")


def stratified_bars(models, n_finetuned, title, name):
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(14 if n_finetuned else 13, 6.5))
    for i, s in enumerate(STRATA):
        label, color = STRATUM_STYLE[s]
        vals = [100 * r["stratified"][s]["f1"] for _, r in models]
        bars = ax.bar(x + (i - 1) * 0.25, vals, 0.25, label=label, color=color)
        for bar in bars[len(models) - n_finetuned:]:
            bar.set_hatch("//")
            bar.set_edgecolor("black")
    ax.set_xticks(x)
    ax.set_xticklabels([m for m, _ in models], rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_ylabel("F1 (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=11, loc="upper left")
    save(fig, name)


def forest_plot():
    sig = load("significance_results.json")
    names = list(sig["bootstrap_ci"])
    boot = [sig["bootstrap_ci"][m] for m in names]
    means = np.array([100 * b["mean"] for b in boot])
    lo = np.array([100 * b["lower"] for b in boot])
    hi = np.array([100 * b["upper"] for b in boot])
    y = np.arange(len(names))[::-1]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.errorbar(means, y, xerr=[means - lo, hi - means], fmt="o", color="#1f77b4",
                markersize=8, elinewidth=2.5, capsize=8, label="Mean dF1 with 95% bootstrap CI")
    ax.axvline(0, color="grey", linestyle="--", lw=1.5)
    for yi, m, h in zip(y, names, hi):
        t = sig["mcnemar"][m]
        p = "p < 0.0001" if t["pvalue"] < 1e-4 else f"p = {t['pvalue']:.3f}"
        ax.text(h + 0.5, yi, f"{100 * sig['bootstrap_ci'][m]['mean']:+.2f} pp\nMcNemar {p}",
                va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlim(min(lo.min(), 0) - 2, hi.max() + 7)
    ax.set_xlabel("dF1 = F1(Combined TF-IDF) - F1(model), percentage points\n"
                  "(> 0 favours Combined TF-IDF)", fontsize=11)
    ax.set_title("Paired Comparison against Combined TF-IDF (Validation)", fontsize=14,
                 fontweight="bold", pad=15)
    ax.legend(loc="upper right", fontsize=10)
    save(fig, "significance_forest_plot.png")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    models, finetuned = collect_models()
    global_comparison(models + finetuned)
    stratified_bars(models, 0, "F1 by Publisher Stratum (Validation)", "stratified_comparison.png")
    stratified_bars(models + finetuned, len(finetuned),
                    "F1 by Publisher Stratum incl. Fine-tuned Transformers (hatched, Validation)",
                    "finetuned_models.png")
    forest_plot()


if __name__ == "__main__":
    main()
