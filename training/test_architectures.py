import os, json, torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW

BASE = r"C:\Users\sivak\Downloads\crime data\TFIF"
meta_path   = os.path.join(BASE, "data", "metadata.json")
splits_path = os.path.join(BASE, "data", "splits.json")

with open(meta_path) as f: records = json.load(f)
with open(splits_path) as f: splits = json.load(f)

# Load all features into memory
train_feats, train_labels = [], []
val_feats, val_labels = [], []
test_feats, test_labels = [], []

for idx in splits["train"]:
    rec = records[idx]
    train_feats.append(torch.load(rec["feature_path"], map_location="cpu"))
    train_labels.append(rec["label"])
    
for idx in splits["val"]:
    rec = records[idx]
    val_feats.append(torch.load(rec["feature_path"], map_location="cpu"))
    val_labels.append(rec["label"])

for idx in splits["test"]:
    rec = records[idx]
    test_feats.append(torch.load(rec["feature_path"], map_location="cpu"))
    test_labels.append(rec["label"])

train_X = torch.stack(train_feats).float()  # (203, 16, 960)
train_Y = torch.tensor(train_labels, dtype=torch.long)
val_X = torch.stack(val_feats).float()      # (43, 16, 960)
val_Y = torch.tensor(val_labels, dtype=torch.long)
test_X = torch.stack(test_feats).float()    # (44, 16, 960)
test_Y = torch.tensor(test_labels, dtype=torch.long)

print(f"Loaded train: {train_X.shape}, val: {val_X.shape}, test: {test_X.shape}")

# Define a few architectures
class AvgPoolMLP(nn.Module):
    def __init__(self, hidden=64, dropout=0.5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(960, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 7)
        )
    def forward(self, x):
        # x: (B, T, 960)
        mean_feat = x.mean(dim=1)  # (B, 960)
        return self.fc(mean_feat)

class MaxPoolMLP(nn.Module):
    def __init__(self, hidden=64, dropout=0.5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(960, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 7)
        )
    def forward(self, x):
        max_feat, _ = x.max(dim=1)  # (B, 960)
        return self.fc(max_feat)

class SimpleLinear(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(960, 7)
        )
    def forward(self, x):
        return self.fc(x.mean(dim=1))

class AttentionModel(nn.Module):
    def __init__(self, hidden=32, dropout=0.4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(960, hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.attn = nn.Linear(hidden, 1)
        self.fc = nn.Linear(hidden, 7)
    def forward(self, x):
        # x: (B, 16, 960)
        h = self.proj(x)  # (B, 16, H)
        scores = self.attn(h).squeeze(-1) # (B, 16)
        weights = torch.softmax(scores, dim=-1) # (B, 16)
        out = (h * weights.unsqueeze(-1)).sum(dim=1) # (B, H)
        return self.fc(out)

# Evaluate each
models = {
    "AvgPoolMLP_64": lambda: AvgPoolMLP(64, 0.5),
    "AvgPoolMLP_32": lambda: AvgPoolMLP(32, 0.5),
    "MaxPoolMLP_64": lambda: MaxPoolMLP(64, 0.5),
    "SimpleLinear_0.3": lambda: SimpleLinear(0.3),
    "SimpleLinear_0.5": lambda: SimpleLinear(0.5),
    "AttentionModel_32": lambda: AttentionModel(32, 0.4),
    "AttentionModel_64": lambda: AttentionModel(64, 0.5)
}

results = {}
for name, model_fn in models.items():
    # Train 5 times and take average validation/test accuracy
    val_accs, test_accs = [], []
    for seed in range(3):
        torch.manual_seed(seed)
        model = model_fn()
        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss()
        
        best_val = 0.0
        best_test = 0.0
        
        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            logits = model(train_X)
            loss = criterion(logits, train_Y)
            loss.backward()
            optimizer.step()
            
            # eval
            model.eval()
            with torch.no_grad():
                val_logits = model(val_X)
                val_acc = (val_logits.argmax(1) == val_Y).float().mean().item()
                
                test_logits = model(test_X)
                test_acc = (test_logits.argmax(1) == test_Y).float().mean().item()
                
                if val_acc > best_val:
                    best_val = val_acc
                    best_test = test_acc
                    
        val_accs.append(best_val)
        test_accs.append(best_test)
        
    print(f"Model: {name:20s} | Avg Best Val Acc: {np.mean(val_accs):.4f} | Avg Test Acc at Best Val: {np.mean(test_accs):.4f}")
