#!/usr/bin/env python3
"""
train_distilbert.py - DistilBERT sentiment classifier with early stopping, metric saving, and resume training
"""

import torch
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

# ------------------------
# Config
# ------------------------
ROOT = Path.cwd()
DATA_DIR = ROOT / "data/sentiment"
OUTPUT_DIR = ROOT / "submissions"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_PATH = ROOT / "sentiment_model_distilbert.pt"
METRICS_PATH = OUTPUT_DIR / "dev_metrics.csv"

EPOCHS = 10
PATIENCE = 2
BATCH_SIZE = 16
LR = 2e-5
MAX_LEN = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# Dataset class
# ------------------------
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ------------------------
# Load data
# ------------------------
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

X = train_df["text"].astype(str).tolist()
y = train_df["label"].tolist()

X_train, X_dev, y_train, y_dev = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
train_dataset = SentimentDataset(X_train, y_train, tokenizer, MAX_LEN)
dev_dataset = SentimentDataset(X_dev, y_dev, tokenizer, MAX_LEN)
test_dataset = SentimentDataset(test_df["text"].astype(str).tolist(), [0]*len(test_df), tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ------------------------
# Model (resume if exists)
# ------------------------
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=len(set(y))
)
if MODEL_PATH.exists():
    print("♻️ Loading previously trained model to resume training...")
    model.load_state_dict(torch.load(MODEL_PATH))

model.to(device)

optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

# ------------------------
# Training & Evaluation functions
# ------------------------
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_model(model, loader):
    model.eval()
    preds, labels_list = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
    acc = accuracy_score(labels_list, preds)
    f1 = f1_score(labels_list, preds, average="weighted")
    return acc, f1

# ------------------------
# Training loop with early stopping
# ------------------------
best_acc = 0
patience_counter = 0

for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, scheduler)
    acc, f1 = eval_model(model, dev_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Dev Acc: {acc:.4f} | F1: {f1:.4f}")

    if acc > best_acc:
        torch.save(model.state_dict(), MODEL_PATH)
        best_acc = acc
        best_f1 = f1
        patience_counter = 0
        print(f"💾 Best model updated! Dev Acc: {best_acc:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("⏹ Early stopping triggered.")
            break

# ------------------------
# Save best dev metrics
# ------------------------
import csv
if METRICS_PATH.exists():
    with open(METRICS_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([best_acc, best_f1])
else:
    with open(METRICS_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["accuracy", "f1_score"])
        writer.writerow([best_acc, best_f1])

print(f"📊 Best Dev metrics saved to {METRICS_PATH}")

# ------------------------
# Predictions on test set
# ------------------------
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        test_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())

out_path = OUTPUT_DIR / "sentiment_test_predictions.csv"
pd.DataFrame({"text": test_df["text"], "label": test_preds}).to_csv(out_path, index=False)
print(f"✅ Test predictions saved to {out_path}")
