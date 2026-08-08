import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
import numpy as np

class BERTClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)  # OI-4(a)
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(self.encoder.config.hidden_size, 1))
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.last_hidden_state[:, 0, :])

class AbstractDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=512):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        rec = self.records[idx]
        text = (rec.get("title", "") or "") + " " + (rec.get("abstract", "") or "")
        encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        return {"input_ids": encoding["input_ids"].squeeze(0), "attention_mask": encoding["attention_mask"].squeeze(0), "label": torch.tensor(int(rec["label"]), dtype=torch.float32)}

def eval_model(model_path, model_name, records, device, thresh=0.36):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)  # OI-4(a)
    model = BERTClassifier(model_name)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False)["model_state_dict"])
    model.to(device)
    model.eval()
    
    loader = DataLoader(AbstractDataset(records, tokenizer), batch_size=16, shuffle=False)
    probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(input_ids, mask)
            probs.extend(torch.sigmoid(logits.squeeze(-1)).cpu().numpy())
    
    y = np.array([r["label"] for r in records])
    p = np.array(probs)
    preds = (p >= thresh).astype(int)
    
    auc = roc_auc_score(y, p)
    f1 = f1_score(y, preds, zero_division=0)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    pr_auc = average_precision_score(y, p)
    
    return {"prec": prec, "rec": rec, "f1": f1, "auc": auc, "pr_auc": pr_auc}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # OI-4(b): canonical data paths
    with open("data/processed/cancer_pm_test.json", "r", encoding="utf-8") as f: test_records = json.load(f)
    with open("data/processed/cancer_pm_holdout.json", "r", encoding="utf-8") as f: holdout_records = json.load(f)

    # For stratified test
    with open("data/processed/journal_to_nlm.json", "r", encoding="utf-8") as f: nlm2raw = {v: k for k, v in json.load(f).items()}
    import csv
    j2p = {}
    with open("data/raw/rwdb/retraction_watch.csv", "r", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            j, p = row.get("Journal", "").strip(), row.get("Publisher", "").strip()
            if j and p: j2p[j] = p
            
    def get_publisher(rec):
        j = rec['journal']
        raw_j = nlm2raw.get(j, j)
        pub = j2p.get(raw_j, 'Direct PubMed')
        if 'computational and mathematical methods in medicine' in raw_j.lower() or 'comput math methods med' in j.lower(): pub = 'Hindawi'
        elif 'biomed research international' in raw_j.lower() or 'biomed res int' in j.lower(): pub = 'Hindawi'
        return pub

    test_h = [r for r in test_records if get_publisher(r) == 'Hindawi']
    test_s = [r for r in test_records if get_publisher(r) == 'Spandidos']
    test_o = [r for r in test_records if get_publisher(r) not in ['Hindawi', 'Spandidos']]

    # OI-4(c): load val-optimal thresholds from training JSONs; do NOT use hardcoded 0.36
    import os as _os
    with open("results/pubmedbert_finetuned_results.json") as _f:
        _pm_thresh = json.load(_f)["best_threshold"]   # 0.28
    with open("results/scibert_finetuned_results.json") as _f:
        _sci_thresh = json.load(_f)["best_threshold"]  # 0.23

    print("--- PubMedBERT (Epoch 3) ---")
    print(f"    Using val-optimal threshold: {_pm_thresh:.4f}")
    pm_path = "models/pubmedbert_epoch_3.pt"
    pm_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    
    res_pm_global    = eval_model(pm_path, pm_name, test_records,  device, thresh=_pm_thresh)
    print(f"Test Global: Prec={res_pm_global['prec']:.4f}, Rec={res_pm_global['rec']:.4f}, F1={res_pm_global['f1']:.4f}, AUC={res_pm_global['auc']:.4f}, PR-AUC={res_pm_global['pr_auc']:.4f}")

    res_pm_hindawi   = eval_model(pm_path, pm_name, test_h,        device, thresh=_pm_thresh)
    print(f"Test Hindawi: Prec={res_pm_hindawi['prec']:.4f}, Rec={res_pm_hindawi['rec']:.4f}, F1={res_pm_hindawi['f1']:.4f}, AUC={res_pm_hindawi['auc']:.4f}")

    res_pm_spandidos = eval_model(pm_path, pm_name, test_s,        device, thresh=_pm_thresh)
    print(f"Test Spandidos: Prec={res_pm_spandidos['prec']:.4f}, Rec={res_pm_spandidos['rec']:.4f}, F1={res_pm_spandidos['f1']:.4f}, AUC={res_pm_spandidos['auc']:.4f}")

    res_pm_others    = eval_model(pm_path, pm_name, test_o,        device, thresh=_pm_thresh)
    print(f"Test Others: Prec={res_pm_others['prec']:.4f}, Rec={res_pm_others['rec']:.4f}, F1={res_pm_others['f1']:.4f}, AUC={res_pm_others['auc']:.4f}")

    res_pm_holdout   = eval_model(pm_path, pm_name, holdout_records, device, thresh=_pm_thresh)
    print(f"Holdout Hindawi: Prec={res_pm_holdout['prec']:.4f}, Rec={res_pm_holdout['rec']:.4f}, F1={res_pm_holdout['f1']:.4f}, AUC={res_pm_holdout['auc']:.4f}, PR-AUC={res_pm_holdout['pr_auc']:.4f}")
    
    print("\n--- SciBERT (Epoch 3) ---")
    print(f"    Using val-optimal threshold: {_sci_thresh:.4f}")
    sci_path = "models/scibert_epoch_3.pt"
    sci_name = "allenai/scibert_scivocab_uncased"

    res_sci_global    = eval_model(sci_path, sci_name, test_records,   device, thresh=_sci_thresh)
    print(f"Test Global: Prec={res_sci_global['prec']:.4f}, Rec={res_sci_global['rec']:.4f}, F1={res_sci_global['f1']:.4f}, AUC={res_sci_global['auc']:.4f}, PR-AUC={res_sci_global['pr_auc']:.4f}")

    res_sci_hindawi   = eval_model(sci_path, sci_name, test_h,         device, thresh=_sci_thresh)
    print(f"Test Hindawi: Prec={res_sci_hindawi['prec']:.4f}, Rec={res_sci_hindawi['rec']:.4f}, F1={res_sci_hindawi['f1']:.4f}, AUC={res_sci_hindawi['auc']:.4f}")

    res_sci_spandidos = eval_model(sci_path, sci_name, test_s,         device, thresh=_sci_thresh)
    print(f"Test Spandidos: Prec={res_sci_spandidos['prec']:.4f}, Rec={res_sci_spandidos['rec']:.4f}, F1={res_sci_spandidos['f1']:.4f}, AUC={res_sci_spandidos['auc']:.4f}")

    res_sci_others    = eval_model(sci_path, sci_name, test_o,         device, thresh=_sci_thresh)
    print(f"Test Others: Prec={res_sci_others['prec']:.4f}, Rec={res_sci_others['rec']:.4f}, F1={res_sci_others['f1']:.4f}, AUC={res_sci_others['auc']:.4f}")

    res_sci_holdout   = eval_model(sci_path, sci_name, holdout_records, device, thresh=_sci_thresh)
    print(f"Holdout Hindawi: Prec={res_sci_holdout['prec']:.4f}, Rec={res_sci_holdout['rec']:.4f}, F1={res_sci_holdout['f1']:.4f}, AUC={res_sci_holdout['auc']:.4f}, PR-AUC={res_sci_holdout['pr_auc']:.4f}")

    # OI-4(d): write JSON — reuse first-pass named variables, no duplicate inference
    _os.makedirs("results", exist_ok=True)
    output = {
        "thresholds_used": {"pubmedbert": float(_pm_thresh), "scibert": float(_sci_thresh)},
        "pubmedbert": {
            "test_global":    res_pm_global,
            "test_hindawi":   res_pm_hindawi,
            "test_spandidos": res_pm_spandidos,
            "test_others":    res_pm_others,
            "holdout":        res_pm_holdout,
        },
        "scibert": {
            "test_global":    res_sci_global,
            "test_hindawi":   res_sci_hindawi,
            "test_spandidos": res_sci_spandidos,
            "test_others":    res_sci_others,
            "holdout":        res_sci_holdout,
        },
    }
    with open("results/finetuned_test_results.json", "w", encoding="utf-8") as _jf:
        json.dump(output, _jf, indent=2)
    print("\nResults written to results/finetuned_test_results.json")

if __name__ == "__main__": main()
