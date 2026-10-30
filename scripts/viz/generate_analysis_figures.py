"""
Hindawi analysis figures (thesis §5.1-5.3, §5.8).

Inputs: models/tfidf_combined.joblib (for the score histograms) and
results/hindawi_analysis_results.json (scripts/eval/hindawi_analysis.py).

Outputs (docs/figures/):
    probability_distribution_hindawi_spandidos.png
    retraction_causes_by_publisher.png
    dose_response_ci_overlap.png
    permutation_test_distribution.png
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    FIGURES_DIR, MODELS_DIR, RESULTS_DIR, PublisherLookup, labels, load_split, texts,
)

POS_COLOR, NEG_COLOR = "#E06666", "#6FA8DC"
BOX = dict(boxstyle="round", facecolor="white", alpha=0.85)

plt.style.use("seaborn-v0_8-whitegrid")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=300)
    plt.close(fig)
    print(f"Saved docs/figures/{name}")


def probability_distribution(res):
    model = joblib.load(MODELS_DIR / "tfidf_combined.joblib")
    val = load_split("val")
    y, strata = labels(val), PublisherLookup().strata(val)
    probs = model["classifier"].predict_proba(model["vectorizer"].transform(texts(val)))[:, 1]
    t = model["threshold"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, s, title in zip(axes, ("hindawi", "spandidos"), ("Hindawi", "Spandidos")):
        st = res["probability_stats"][s]
        for lab, color, name in ((0, NEG_COLOR, "Negatives"), (1, POS_COLOR, "Positives")):
            p = probs[(strata == s) & (y == lab)]
            ax.hist(p, bins=25, range=(0, 1), alpha=0.6, density=True, color=color,
                    label=f"{name} (N={len(p)})")
        ax.axvline(t, color="black", linestyle="--", label=f"Threshold t={t:.2f}")
        ax.text(0.98, 0.74,
                f"Pos: mean={st['positives']['mean']:.3f}, median={st['positives']['median']:.3f}\n"
                f"Neg: mean={st['negatives']['mean']:.3f}, median={st['negatives']['median']:.3f}\n"
                f"AUC={100 * res['stratum_ci'][s]['auc']:.1f}%",
                transform=ax.transAxes, fontsize=10, va="top", ha="right", bbox=BOX)
        ax.set_title(f"{title} - predicted fraud probability", fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted probability", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.legend(loc="upper right")
    fig.suptitle("Predicted Probability by Publisher (Combined TF-IDF, validation)",
                 fontsize=15, fontweight="bold")
    save(fig, "probability_distribution_hindawi_spandidos.png")


def retraction_causes(res):
    causes = [("image_duplication", "Image\nduplication"),
              ("peer_review_manipulation", "Compromised\npeer review"),
              ("computer_generated_content", "Computer-generated\ncontent")]
    rc = res["retraction_causes"]
    x = np.arange(len(causes))
    fig, ax = plt.subplots(figsize=(12, 6))
    for offset, pub, color in ((-0.175, "spandidos", NEG_COLOR), (0.175, "hindawi", POS_COLOR)):
        vals = [100 * rc[pub][c]["share"] for c, _ in causes]
        bars = ax.bar(x + offset, vals, 0.35, color=color,
                      label=f"{pub.capitalize()} (N={rc[pub]['n_matched']} positives)")
        ax.bar_label(bars, fmt="%.1f%%", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in causes], fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Retracted articles citing the reason (%)", fontsize=12)
    ax.set_title("Retraction Reasons by Publisher (train + validation positives, RWDB by DOI)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=12, loc="upper left")
    save(fig, "retraction_causes_by_publisher.png")


def dose_response(res):
    dr = res["dose_response"]
    rows = [("combined_tfidf", "Combined TF-IDF"), ("pubmedbert_frozen", "PubMedBERT\n(frozen)")]
    groups = [("A_computer_generated", "Subgroup A - computer-generated content", POS_COLOR, 0.2),
              ("B_other_process_fraud", "Subgroup B - other process fraud", NEG_COLOR, -0.2)]
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (m, _) in enumerate(rows):
        y0 = len(rows) - 1 - i
        lo, hi = dr[m]["ci_overlap"]
        ax.fill_betweenx([y0 - 0.35, y0 + 0.35], lo, hi, color="grey", alpha=0.12)
        for g, label, color, dy in groups:
            d = dr[m][g]
            auc, (a, b) = 100 * d["auc"], (100 * v for v in d["auc_ci"])
            ax.errorbar(auc, y0 + dy, xerr=[[auc - a], [b - auc]], fmt="o", color=color,
                        markersize=9, elinewidth=3, capsize=7,
                        label=f"{label} (N={d['n_pos']})" if i == 0 else None)
            ax.text(b + 0.5, y0 + dy, f"{auc:.1f}% [{a:.1f}, {b:.1f}]", va="center", fontsize=10)
    ax.axvline(50, color="grey", linestyle=":", lw=1)
    ax.set_yticks(range(len(rows))[::-1])
    ax.set_yticklabels([label for _, label in rows], fontsize=12, fontweight="bold")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(45, 82)
    ax.set_xlabel("ROC-AUC vs all Hindawi negatives (%), 95% bootstrap CI (2,000 resamples)",
                  fontsize=11)
    ax.set_title("Dose-Response Check: Hindawi Validation Positives by Retraction Reason",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=10)
    save(fig, "dose_response_ci_overlap.png")


def permutation_test(res):
    pt = res["permutation_test"]
    f1 = 100 * np.array(pt["f1"])
    obs = 100 * pt["observed"]["f1"]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(f1, bins=15, alpha=0.7, color="#3182bd", edgecolor="white",
            label=f"F1 with permuted labels ({len(f1)} seeds): "
                  f"mean {f1.mean():.2f}%, sd {f1.std():.2f}%")
    ax.axvline(obs, color="#de2d26", lw=2.5,
               label=f"Observed F1 = {obs:.2f}% ({pt['z_f1']:.1f} sd above the null mean)")
    ax.axvline(f1.mean(), color="#3182bd", linestyle="--", label=f"Null mean = {f1.mean():.2f}%")
    n_ge = int(round(pt["empirical_p_f1"] * len(f1)))
    ax.text(0.02, 0.95, f"Empirical p = {pt['empirical_p_f1']:.3f}\n"
                        f"({n_ge}/{len(f1)} permutations >= observed)",
            transform=ax.transAxes, fontsize=11, va="top", bbox=BOX)
    ax.set_xlim(min(10, f1.min() - 2), obs + 4)
    ax.set_xlabel("Validation F1 (%)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Label-Permutation Test (Combined TF-IDF, validation N = 1,440)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="upper right", fontsize=10)
    save(fig, "permutation_test_distribution.png")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "hindawi_analysis_results.json", encoding="utf-8") as f:
        res = json.load(f)
    probability_distribution(res)
    retraction_causes(res)
    dose_response(res)
    permutation_test(res)


if __name__ == "__main__":
    main()
