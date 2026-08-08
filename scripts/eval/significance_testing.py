import json
import os
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.stats import chi2


# ── Fine-tuned BERT helpers ───────────────────────────────────────────────────

class BERTClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.encoder.config.hidden_size, 1),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.last_hidden_state[:, 0, :])


class AbstractDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=512):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        text = (rec.get("title", "") or "") + " " + (rec.get("abstract", "") or "")
        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(int(rec["label"]), dtype=torch.float32),
        }


def get_finetuned_probs(model_path, model_name, records, device):
    """Load a fine-tuned BERT checkpoint and return sigmoid probabilities on records."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = BERTClassifier(model_name)
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    loader = DataLoader(AbstractDataset(records, tokenizer), batch_size=16, shuffle=False)
    probs = []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(ids, mask)
            probs.extend(torch.sigmoid(logits.squeeze(-1)).cpu().numpy())
    return np.array(probs)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
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

    # 2. Combined TF-IDF — load canonical C=15.0 model from joblib (do not re-train)
    print("Loading Combined TF-IDF C=15.0 from models/tfidf_combined.joblib...")
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

    # 4. PubMedBERT (frozen embeddings + LR)
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

    # 5. PubMedBERT fine-tuned (epoch 3)
    print("\nLoading PubMedBERT fine-tuned (epoch 3) from models/pubmedbert_epoch_3.pt...")
    with open("results/pubmedbert_finetuned_results.json", "r") as f:
        _pm_ft_meta = json.load(f)
    _pm_ft_thresh = _pm_ft_meta["best_threshold"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probs_pm_ft = get_finetuned_probs(
        "models/pubmedbert_epoch_3.pt",
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        val, device,
    )
    preds_pm_ft = (probs_pm_ft >= _pm_ft_thresh).astype(int)
    print(f"  PubMedBERT fine-tuned threshold (from training JSON): {_pm_ft_thresh:.4f}")

    # 6. SciBERT fine-tuned (epoch 3)
    print("\nLoading SciBERT fine-tuned (epoch 3) from models/scibert_epoch_3.pt...")
    with open("results/scibert_finetuned_results.json", "r") as f:
        _sci_ft_meta = json.load(f)
    _sci_ft_thresh = _sci_ft_meta["best_threshold"]
    probs_sci_ft = get_finetuned_probs(
        "models/scibert_epoch_3.pt",
        "allenai/scibert_scivocab_uncased",
        val, device,
    )
    preds_sci_ft = (probs_sci_ft >= _sci_ft_thresh).astype(int)
    print(f"  SciBERT fine-tuned threshold (from training JSON): {_sci_ft_thresh:.4f}")

    print("\n" + "=" * 50)
    print("REPLICATED METRICS (Validation)")
    print("=" * 50)
    print(f"TF-IDF Baseline:          F1 = {f1_score(val_labels, preds_base):.4%}")
    print(f"Combined TF-IDF:           F1 = {f1_score(val_labels, preds_comb):.4%}")
    print(f"Char N-grams:             F1 = {f1_score(val_labels, preds_char):.4%}")
    print(f"PubMedBERT (frozen):      F1 = {f1_score(val_labels, preds_pm):.4%}")
    print(f"PubMedBERT (fine-tuned):  F1 = {f1_score(val_labels, preds_pm_ft):.4%}")
    print(f"SciBERT (fine-tuned):     F1 = {f1_score(val_labels, preds_sci_ft):.4%}")

    # =========================================================================
    # McNemar's Test (paired contingency tables)
    # =========================================================================
    print("\n" + "=" * 50)
    print("MCNEMAR'S TEST RESULTS (vs. Combined TF-IDF)")
    print("=" * 50)

    def run_mcnemar(preds_a, preds_b, name_a, name_b):
        b = int(((preds_a == val_labels) & (preds_b != val_labels)).sum())
        c = int(((preds_a != val_labels) & (preds_b == val_labels)).sum())
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

    mcnemar_base   = run_mcnemar(preds_base,   preds_comb, "TF-IDF Baseline",        "Combined TF-IDF")
    mcnemar_char   = run_mcnemar(preds_char,   preds_comb, "Char N-grams",           "Combined TF-IDF")
    mcnemar_pm     = run_mcnemar(preds_pm,     preds_comb, "PubMedBERT (frozen)",    "Combined TF-IDF")
    mcnemar_pm_ft  = run_mcnemar(preds_pm_ft,  preds_comb, "PubMedBERT (fine-tuned)","Combined TF-IDF")
    mcnemar_sci_ft = run_mcnemar(preds_sci_ft, preds_comb, "SciBERT (fine-tuned)",   "Combined TF-IDF")

    # =========================================================================
    # Bootstrapping F1 Difference (2,000 iterations)
    # =========================================================================
    print("\n" + "=" * 50)
    print("BOOTSTRAP F1 DIFFERENCE (95% Confidence Intervals, 2000 iterations)")
    print("=" * 50)

    np.random.seed(42)
    n_iterations = 2000
    n_samples = len(val_labels)

    diff_base_list   = []
    diff_char_list   = []
    diff_pm_list     = []
    diff_pm_ft_list  = []
    diff_sci_ft_list = []

    for i in range(n_iterations):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_y = val_labels[idx]
        if len(np.unique(boot_y)) < 2:
            continue
        f1_comb   = f1_score(boot_y, preds_comb[idx])
        f1_base   = f1_score(boot_y, preds_base[idx])
        f1_char   = f1_score(boot_y, preds_char[idx])
        f1_pm     = f1_score(boot_y, preds_pm[idx])
        f1_pm_ft  = f1_score(boot_y, preds_pm_ft[idx])
        f1_sci_ft = f1_score(boot_y, preds_sci_ft[idx])
        diff_base_list.append(f1_comb   - f1_base)
        diff_char_list.append(f1_comb   - f1_char)
        diff_pm_list.append(f1_comb     - f1_pm)
        diff_pm_ft_list.append(f1_comb  - f1_pm_ft)
        diff_sci_ft_list.append(f1_comb - f1_sci_ft)

    def get_ci(diff_list):
        lo = np.percentile(diff_list, 2.5)
        hi = np.percentile(diff_list, 97.5)
        mean_diff = np.mean(diff_list)
        return mean_diff, lo, hi

    mean_b, lo_b, hi_b = get_ci(diff_base_list)
    print(f"Combined TF-IDF vs TF-IDF Baseline dF1:")
    print(f"  Mean Difference: {mean_b*100:+.2f}%")
    print(f"  95% CI: [{lo_b*100:.2f}%, {hi_b*100:.2f}%]")
    print(f"  Distinguishable from 0: {not (lo_b <= 0 <= hi_b)}")

    mean_c, lo_c, hi_c = get_ci(diff_char_list)
    print(f"Combined TF-IDF vs Char N-grams dF1:")
    print(f"  Mean Difference: {mean_c*100:+.2f}%")
    print(f"  95% CI: [{lo_c*100:.2f}%, {hi_c*100:.2f}%]")
    print(f"  Distinguishable from 0: {not (lo_c <= 0 <= hi_c)}")

    mean_p, lo_p, hi_p = get_ci(diff_pm_list)
    print(f"Combined TF-IDF vs PubMedBERT (frozen) dF1:")
    print(f"  Mean Difference: {mean_p*100:+.2f}%")
    print(f"  95% CI: [{lo_p*100:.2f}%, {hi_p*100:.2f}%]")
    print(f"  Distinguishable from 0: {not (lo_p <= 0 <= hi_p)}")

    mean_pmft, lo_pmft, hi_pmft = get_ci(diff_pm_ft_list)
    print(f"Combined TF-IDF vs PubMedBERT (fine-tuned) dF1:")
    print(f"  Mean Difference: {mean_pmft*100:+.2f}%")
    print(f"  95% CI: [{lo_pmft*100:.2f}%, {hi_pmft*100:.2f}%]")
    print(f"  Distinguishable from 0: {not (lo_pmft <= 0 <= hi_pmft)}")

    mean_scift, lo_scift, hi_scift = get_ci(diff_sci_ft_list)
    print(f"Combined TF-IDF vs SciBERT (fine-tuned) dF1:")
    print(f"  Mean Difference: {mean_scift*100:+.2f}%")
    print(f"  95% CI: [{lo_scift*100:.2f}%, {hi_scift*100:.2f}%]")
    print(f"  Distinguishable from 0: {not (lo_scift <= 0 <= hi_scift)}")

    # Save results
    results = {
        "mcnemar": {
            "TF-IDF Baseline":        mcnemar_base,
            "Char N-grams":           mcnemar_char,
            "PubMedBERT (frozen)":    mcnemar_pm,
            "PubMedBERT (fine-tuned)": mcnemar_pm_ft,
            "SciBERT (fine-tuned)":   mcnemar_sci_ft,
        },
        "bootstrap_ci": {
            "TF-IDF Baseline":   {"mean": float(mean_b),     "lower": float(lo_b),     "upper": float(hi_b)},
            "Char N-grams":      {"mean": float(mean_c),     "lower": float(lo_c),     "upper": float(hi_c)},
            "PubMedBERT (frozen)": {"mean": float(mean_p),   "lower": float(lo_p),     "upper": float(hi_p)},
            "PubMedBERT (fine-tuned)": {"mean": float(mean_pmft),  "lower": float(lo_pmft),  "upper": float(hi_pmft)},
            "SciBERT (fine-tuned)":    {"mean": float(mean_scift), "lower": float(lo_scift), "upper": float(hi_scift)},
        },
        "thresholds_used": {
            "TF-IDF Baseline":   0.37,
            "Combined TF-IDF":    float(_frozen_thresh),
            "Char N-grams":      float(_char_best_thresh),
            "PubMedBERT (frozen)":    float(_pm_best_thresh),
            "PubMedBERT (fine-tuned)": float(_pm_ft_thresh),
            "SciBERT (fine-tuned)":    float(_sci_ft_thresh),
        },
    }

    os.makedirs("results", exist_ok=True)
    with open("results/significance_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSignificance testing results saved to results/significance_results.json")


if __name__ == "__main__":
    main()
