"""
Plot best validation F1 per regularisation strength C (Combined TF-IDF sweep).

Input:  results/hyperparameter_sweep_full.json (scripts/train/exhaustive_search.py)
Output: docs/figures/hyperparameter_c_selection.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FIGURES_DIR, RESULTS_DIR  # noqa: E402


def main():
    with open(RESULTS_DIR / "hyperparameter_sweep_full.json", encoding="utf-8") as f:
        sweep = json.load(f)

    c_vals = [r["C"] for r in sweep["per_C"]]
    f1_vals = [100 * r["overall"]["f1"] for r in sweep["per_C"]]
    best = sweep["best"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(c_vals, f1_vals, marker="o", color="#1f77b4", label="Best validation F1 per C")
    ax.scatter([best["C"]], [100 * best["F1"]], color="red", s=150, zorder=5,
               label=f"Selected: C={best['C']:g}, t={best['threshold']:.2f}, "
                     f"F1={100 * best['F1']:.2f}%")
    ax.set_xscale("log")
    ax.set_xticks(c_vals)
    ax.set_xticklabels([f"{c:g}" for c in c_vals])
    ax.set_title("Validation F1 vs Regularisation Strength C (Combined TF-IDF, min_df=5)",
                 fontsize=14, pad=15)
    ax.set_xlabel("C (inverse regularisation strength, log scale)", fontsize=12)
    ax.set_ylabel("Best validation F1 (%)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="lower right")
    fig.text(0.5, 0.01, "Source: results/hyperparameter_sweep_full.json "
             "(scripts/train/exhaustive_search.py, validation split only)",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    out = FIGURES_DIR / "hyperparameter_c_selection.png"
    fig.savefig(out, dpi=300)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
