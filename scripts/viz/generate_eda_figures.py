"""
Exploratory data analysis figures for CPM-11K (thesis §1-3).

Outputs (docs/figures/):
    eda_annual_distribution.png   positives vs negatives per publication year
    eda_publisher_share.png       share of each class by publisher
    eda_purge_cascade.png         fate of the initial candidate pool
    eda_split_volumes.png         train / validation / test / holdout sizes
    eda_class_imbalance.png       global class ratio
    eda_abstract_lengths.png      abstract length (words) by class
"""
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FIGURES_DIR, PublisherLookup, load_split  # noqa: E402

POS_COLOR, NEG_COLOR = "#E06666", "#6FA8DC"
POS_LABEL, NEG_LABEL = "Positives (paper mills)", "Negatives (controls)"
TOP_PUBLISHERS = ["Hindawi", "Spandidos", "Wiley", "Verduci Editore",
                  "Taylor and Francis - Dove Press", "Portland Press",
                  "Royal Society of Chemistry (RSC)", "Elsevier", "Taylor and Francis"]

# Candidate-pool accounting from the dataset build (fetch_abstracts.py run log):
# candidates -> no recoverable abstract -> not oncology -> published 2024-2025 -> kept.
PURGE = {
    "positives": {"pool": 2551, "no_abstract": 322, "not_oncology": 112, "recent": 6, "kept": 2111},
    "negatives": {"pool": 9273, "no_abstract": 148, "not_oncology": 95, "recent": 35, "kept": 8995},
}

plt.style.use("seaborn-v0_8-whitegrid")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=300)
    plt.close(fig)
    print(f"Saved docs/figures/{name}")


def annual_distribution(dataset):
    pos = Counter(r["year"] for r in dataset if r["label"] == 1)
    neg = Counter(r["year"] for r in dataset if r["label"] == 0)
    years = sorted(set(pos) | set(neg))
    x = np.arange(len(years))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 0.175, [pos[y] for y in years], 0.35, label=POS_LABEL, color=POS_COLOR)
    ax.bar(x + 0.175, [neg[y] for y in years], 0.35, label=NEG_LABEL, color=NEG_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_title("Annual Article Distribution (CPM-11K)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Publication year", fontsize=12)
    ax.set_ylabel("Number of articles", fontsize=12)
    ax.legend(fontsize=11)
    save(fig, "eda_annual_distribution.png")


def publisher_share(dataset):
    lookup = PublisherLookup()
    pubs = {lab: Counter(lookup.publisher(r) for r in dataset if r["label"] == lab) for lab in (0, 1)}
    totals = {lab: sum(c.values()) for lab, c in pubs.items()}
    n_other = len((set(pubs[0]) | set(pubs[1])) - set(TOP_PUBLISHERS))
    labels = TOP_PUBLISHERS + [f"Other ({n_other} publishers)"]

    def shares(lab):
        top = [pubs[lab][p] for p in TOP_PUBLISHERS]
        return [100 * v / totals[lab] for v in top + [totals[lab] - sum(top)]]

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(y - 0.175, shares(1), 0.35, label=POS_LABEL, color=POS_COLOR)
    ax.barh(y + 0.175, shares(0), 0.35, label=NEG_LABEL, color=NEG_COLOR)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Share of class (%)", fontsize=12)
    ax.set_title("Publisher Share by Class (CPM-11K)", fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=11)
    save(fig, "eda_publisher_share.png")


def purge_cascade():
    steps = ["no_abstract", "not_oncology", "recent", "kept"]
    labels = ["No recoverable\nabstract", "Not oncology", "Published\n2024-2025", "Kept"]
    x = np.arange(len(steps))
    fig, ax = plt.subplots(figsize=(10, 6))
    for offset, cls, color, label in ((-0.175, "positives", POS_COLOR, POS_LABEL),
                                      (0.175, "negatives", NEG_COLOR, NEG_LABEL)):
        p = PURGE[cls]
        rates = [100 * p[s] / p["pool"] for s in steps]
        bars = ax.bar(x + offset, rates, 0.35, label=f"{label}, pool N={p['pool']:,}", color=color)
        ax.bar_label(bars, fmt="%.1f%%", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Share of candidate pool (%)", fontsize=12)
    ax.set_title("Fate of the Candidate Pool", fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=11)
    save(fig, "eda_purge_cascade.png")


def split_volumes(splits):
    names = ["Train", "Validation", "Test", "Hindawi holdout"]
    pos = [sum(r["label"] for r in s) for s in splits]
    neg = [len(s) - p for s, p in zip(splits, pos)]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(names, pos, label="Positives", color=POS_COLOR)
    ax.bar(names, neg, bottom=pos, label="Negatives", color=NEG_COLOR)
    for i, (p, n) in enumerate(zip(pos, neg)):
        ax.text(i, p + n, f"{p + n:,}\n({p:,} / {n:,})", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.15 * max(p + n for p, n in zip(pos, neg)))
    ax.set_title("Split Volumes (positives / negatives)", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Number of articles", fontsize=12)
    ax.legend(fontsize=11)
    save(fig, "eda_split_volumes.png")


def class_imbalance(dataset):
    n_pos = sum(r["label"] for r in dataset)
    n_neg = len(dataset) - n_pos
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie([n_pos, n_neg], colors=[POS_COLOR, NEG_COLOR], startangle=140,
           labels=[f"{POS_LABEL}\n{n_pos:,} ({100 * n_pos / len(dataset):.1f}%)",
                   f"{NEG_LABEL}\n{n_neg:,} ({100 * n_neg / len(dataset):.1f}%)"],
           textprops={"fontsize": 11, "weight": "bold"})
    ax.set_title(f"Class Ratio 1:{n_neg / n_pos:.2f} (CPM-11K)", fontsize=14, fontweight="bold", pad=15)
    save(fig, "eda_class_imbalance.png")


def abstract_lengths(dataset):
    fig, ax = plt.subplots(figsize=(10, 6))
    for lab, color, label in ((1, POS_COLOR, POS_LABEL), (0, NEG_COLOR, NEG_LABEL)):
        words = [len(r["abstract"].split()) for r in dataset if r["label"] == lab]
        ax.hist(words, bins=50, alpha=0.6, density=True, color=color,
                label=f"{label} (median {np.median(words):.0f} words)")
    ax.set_title("Abstract Length Distribution (CPM-11K)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Words per abstract", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(fontsize=11)
    save(fig, "eda_abstract_lengths.png")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_split("dataset")
    splits = [load_split(s) for s in ("train", "val", "test", "holdout")]
    annual_distribution(dataset)
    publisher_share(dataset)
    purge_cascade()
    split_volumes(splits)
    class_imbalance(dataset)
    abstract_lengths(dataset)


if __name__ == "__main__":
    main()
