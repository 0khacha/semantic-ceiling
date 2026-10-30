"""
Shared helpers for the CPM-11K training, evaluation and figure scripts.

Every path is resolved from the repository root, so scripts can be launched
from any working directory. Scripts import this module with:

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import ...
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score, roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RWDB_CSV = ROOT / "data" / "raw" / "rwdb" / "retraction_watch.csv"
NLM_JSON = DATA_DIR / "journal_to_nlm.json"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "docs" / "figures"

SPLITS = ("train", "val", "test", "holdout")
STRATA = ("hindawi", "spandidos", "others")
THRESHOLD_GRID = np.arange(0.1, 0.9, 0.01)

PUBMEDBERT_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
SCIBERT_NAME = "allenai/scibert_scivocab_uncased"
FROZEN_PUBMEDBERT_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
MINILM_NAME = "sentence-transformers/all-MiniLM-L6-v2"

csv.field_size_limit(2**31 - 1)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ── Data ──────────────────────────────────────────────────────────────────────

def load_split(name):
    """Load one canonical (v2) split: train, val, test or holdout."""
    with open(DATA_DIR / f"cancer_pm_{name}.json", encoding="utf-8") as f:
        return json.load(f)


def texts(records):
    return [(r.get("title") or "") + " " + (r.get("abstract") or "") for r in records]


def labels(records):
    return np.array([int(r["label"]) for r in records])


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"Saved {path.relative_to(ROOT)}")


# ── Publisher strata ──────────────────────────────────────────────────────────

_HINDAWI_JOURNALS = (
    ("computational and mathematical methods in medicine", "comput math methods med"),
    ("biomed research international", "biomed res int"),
)


class PublisherLookup:
    """Maps a record's NLM journal abbreviation to its RWDB publisher name."""

    def __init__(self):
        self.journal_to_publisher = {}
        with open(RWDB_CSV, encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                j, p = row.get("Journal", "").strip(), row.get("Publisher", "").strip()
                if j and p:
                    self.journal_to_publisher[j] = p
        with open(NLM_JSON, encoding="utf-8") as f:
            self.nlm_to_raw = {v: k for k, v in json.load(f).items()}

    def publisher(self, rec):
        nlm = rec["journal"]
        raw = self.nlm_to_raw.get(nlm, nlm)
        for full, abbrev in _HINDAWI_JOURNALS:
            if full in raw.lower() or abbrev in nlm.lower():
                return "Hindawi"
        return self.journal_to_publisher.get(raw, "Direct PubMed")

    def stratum(self, rec):
        pub = self.publisher(rec)
        return pub.lower() if pub in ("Hindawi", "Spandidos") else "others"

    def strata(self, records):
        return np.array([self.stratum(r) for r in records])


# ── Metrics ───────────────────────────────────────────────────────────────────

def best_threshold(probs, y, grid=THRESHOLD_GRID):
    """Threshold on `grid` that maximises F1 (first maximum wins)."""
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = round(float(t), 2), f1
    return best_t


def metrics_at(probs, y, threshold):
    preds = (probs >= threshold).astype(int)
    out = {
        "threshold": float(threshold),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "n_pos": int(y.sum()),
        "n_neg": int(len(y) - y.sum()),
    }
    if len(np.unique(y)) > 1:
        out["auc"] = float(roc_auc_score(y, probs))
        out["pr_auc"] = float(average_precision_score(y, probs))
    return out


def stratified_metrics(probs, y, strata, threshold):
    return {s: metrics_at(probs[strata == s], y[strata == s], threshold) for s in STRATA}


def trivial_f1(y):
    """F1 of the constant 'always positive' classifier."""
    p = y.mean()
    return float(2 * p / (1 + p))


def mcnemar(y, preds_model, preds_ref):
    """McNemar with continuity correction; b = model right & reference wrong, c = reverse."""
    from scipy.stats import chi2
    b = int(((preds_model == y) & (preds_ref != y)).sum())
    c = int(((preds_model != y) & (preds_ref == y)).sum())
    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "pvalue": 1.0}
    stat = (abs(b - c) - 1.0) ** 2 / (b + c)
    return {"b": b, "c": c, "chi2": float(stat), "pvalue": float(chi2.sf(stat, 1))}


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values for a {name: p} mapping."""
    names = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    for rank, name in enumerate(names):
        running = max(running, min(1.0, (len(names) - rank) * pvalues[name]))
        adjusted[name] = running
    return adjusted


# ── Frozen sentence embeddings ────────────────────────────────────────────────

EMBEDDINGS_DIR = MODELS_DIR / "embeddings"


def frozen_embeddings(model_name, split):
    """Sentence-transformer embeddings of one split, cached under models/embeddings/."""
    cache = EMBEDDINGS_DIR / f"{model_name.replace('/', '__')}__{split}.npy"
    if cache.exists():
        return np.load(cache)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, local_files_only=True)
    emb = model.encode(texts(load_split(split)), batch_size=32, show_progress_bar=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache, emb)
    return emb


def frozen_embedding_probs(model_name, eval_split="val"):
    """Scores for `eval_split` from an LR (C=1, balanced) fitted on frozen train-split embeddings."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=42)
    clf.fit(frozen_embeddings(model_name, "train"), labels(load_split("train")))
    return clf.predict_proba(frozen_embeddings(model_name, eval_split))[:, 1]


# ── Fine-tuned transformers ───────────────────────────────────────────────────

def bert_probabilities(checkpoint, model_name, records, batch_size=16, max_length=512):
    """Sigmoid scores of a fine-tuned BERTClassifier checkpoint on `records`."""
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

    class BERTClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
            self.classifier = nn.Sequential(
                nn.Dropout(0.1), nn.Linear(self.encoder.config.hidden_size, 1))

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            return self.classifier(out.last_hidden_state[:, 0, :])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = BERTClassifier()
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()

    all_texts, probs = texts(records), []
    with torch.no_grad():
        for i in range(0, len(all_texts), batch_size):
            enc = tokenizer(all_texts[i:i + batch_size], truncation=True, max_length=max_length,
                            padding="max_length", return_tensors="pt").to(device)
            logits = model(enc["input_ids"], enc["attention_mask"])
            probs.extend(torch.sigmoid(logits.squeeze(-1)).cpu().numpy().tolist())
    return np.array(probs)


FINETUNED = {"pubmedbert": PUBMEDBERT_NAME, "scibert": SCIBERT_NAME}
PREDICTIONS_DIR = MODELS_DIR / "predictions"


def finetuned_probabilities(key, split):
    """Scores of the epoch-3 fine-tuned `key` model on a split, cached under models/predictions/."""
    cache = PREDICTIONS_DIR / f"{key}_epoch_3__{split}.npy"
    if cache.exists():
        return np.load(cache)
    probs = bert_probabilities(MODELS_DIR / f"{key}_epoch_3.pt", FINETUNED[key], load_split(split))
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache, probs)
    return probs
