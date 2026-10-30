"""
End-to-end fine-tuning of a BERT-family encoder on CPM-11K (title + abstract).

    python scripts/train/finetune_transformer.py --model pubmedbert
    python scripts/train/finetune_transformer.py --model scibert

Protocol (identical for both models): [CLS] -> dropout(0.1) -> linear head,
BCE loss, AdamW (lr 2e-5, weight decay 0.01), linear schedule with 10% warmup,
batch 4 x 2 gradient-accumulation steps (effective batch 8), FP16 autocast on CUDA,
3 epochs, seed 42, max length 512. The epoch with the best validation F1 is
kept, and its F1-optimal validation threshold is frozen for test/holdout use.

Outputs:
    models/{model}_epoch_{1,2,3}.pt, models/{model}_finetuned_best.pt
    results/{model}_finetuned_results.json
"""
import argparse
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    MODELS_DIR, PUBMEDBERT_NAME, RESULTS_DIR, SCIBERT_NAME, PublisherLookup,
    best_threshold, labels, load_split, metrics_at, save_json, stratified_metrics, texts,
)

MODEL_NAMES = {"pubmedbert": PUBMEDBERT_NAME, "scibert": SCIBERT_NAME}
MAX_LENGTH = 512
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 2  # 4 x 2 = 8 sequences per optimizer step
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.10
EPOCHS = 3
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AbstractDataset(Dataset):
    def __init__(self, records, tokenizer):
        self.texts, self.labels, self.tokenizer = texts(records), labels(records), tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, max_length=MAX_LENGTH,
                             padding="max_length", return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class BERTClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(self.encoder.config.hidden_size, 1))

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.last_hidden_state[:, 0, :])


def predict(model, loader, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            with autocast(device.type, enabled=device.type == "cuda"):
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            probs.extend(torch.sigmoid(logits.float().squeeze(-1)).cpu().numpy().tolist())
    return np.array(probs)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--model", choices=sorted(MODEL_NAMES), required=True)
    key = parser.parse_args().model
    model_name = MODEL_NAMES[key]

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fine-tuning {model_name} on {device}")

    train, val = load_split("train"), load_split("val")
    y_val = labels(val)
    val_strata = PublisherLookup().strata(val)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_loader = DataLoader(AbstractDataset(train, tokenizer), batch_size=BATCH_SIZE,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(AbstractDataset(val, tokenizer), batch_size=BATCH_SIZE,
                            shuffle=False, pin_memory=True)

    model = BERTClassifier(model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * WARMUP_FRACTION), total_steps)
    criterion = nn.BCEWithLogitsLoss()
    use_amp = device.type == "cuda"
    scaler = GradScaler(device.type, enabled=use_amp)
    MODELS_DIR.mkdir(exist_ok=True)

    history = []
    t_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader, 1):
            with autocast(device.type, enabled=use_amp):
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                loss = criterion(logits.squeeze(-1), batch["label"].to(device)) / GRAD_ACCUM_STEPS
            scaler.scale(loss).backward()
            if step % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            epoch_loss += loss.item() * GRAD_ACCUM_STEPS
            n_batches += 1

        probs = predict(model, val_loader, device)
        t = best_threshold(probs, y_val)
        m = metrics_at(probs, y_val, t)
        history.append({
            "epoch": epoch,
            "loss": epoch_loss / max(n_batches, 1),
            "auc": m["auc"],
            "best_f1": m["f1"],
            "best_thresh": t,
            "f1_05": metrics_at(probs, y_val, 0.5)["f1"],
        })
        print(f"Epoch {epoch}/{EPOCHS}: loss={history[-1]['loss']:.4f} "
              f"val AUC={m['auc']:.4f} best F1={m['f1']:.4%} (t={t:.2f})")
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict()},
                   MODELS_DIR / f"{key}_epoch_{epoch}.pt")
    print(f"Training completed in {(time.time() - t_start) / 60:.1f} min")

    best = max(history, key=lambda h: h["best_f1"])
    best_ckpt = MODELS_DIR / f"{key}_epoch_{best['epoch']}.pt"
    model.load_state_dict(torch.load(best_ckpt, weights_only=False)["model_state_dict"])
    probs = predict(model, val_loader, device)
    t = best["best_thresh"]
    overall = metrics_at(probs, y_val, t)

    save_json({
        "epoch_history": history,
        "best_epoch": best["epoch"],
        "best_threshold": t,
        "overall": {k: overall[k] for k in ("precision", "recall", "f1", "auc", "pr_auc")},
        "stratified": stratified_metrics(probs, y_val, val_strata, t),
    }, RESULTS_DIR / f"{key}_finetuned_results.json")
    shutil.copyfile(best_ckpt, MODELS_DIR / f"{key}_finetuned_best.pt")


if __name__ == "__main__":
    main()
