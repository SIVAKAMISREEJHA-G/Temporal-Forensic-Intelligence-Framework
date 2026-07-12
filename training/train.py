import os, json, sys, csv, torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold

# Ensure training dir is on path
sys.path.insert(0, os.path.dirname(__file__))
from model import TFIFClassifier

BASE = r"C:\Users\sivak\Downloads\crime data\TFIF"
SAVE_DIR = os.path.join(BASE, "model", "saved_model")
os.makedirs(SAVE_DIR, exist_ok=True)

CLASSES = ["Abuse","Car Accident","Explosion","Fighting","Normal","Riot","Shooting"]
NUM_CLASSES = len(CLASSES)

def load_data():
    meta_path = os.path.join(BASE, "data", "metadata.json")
    splits_path = os.path.join(BASE, "data", "splits.json")
    with open(meta_path) as f: records = json.load(f)
    with open(splits_path) as f: splits = json.load(f)

    # We combine train and val indices for cross validation
    cv_indices = splits["train"] + splits["val"]
    test_indices = splits["test"]

    cv_feats = [torch.load(records[i]["feature_path"], map_location="cpu") for i in cv_indices]
    cv_labels = [records[i]["label"] for i in cv_indices]
    
    test_feats = [torch.load(records[i]["feature_path"], map_location="cpu") for i in test_indices]
    test_labels = [records[i]["label"] for i in test_indices]

    return (torch.stack(cv_feats).float(), torch.tensor(cv_labels, dtype=torch.long),
            torch.stack(test_feats).float(), torch.tensor(test_labels, dtype=torch.long))

def train_fold(fold, train_idx, val_idx, cv_X, cv_Y, epochs=60, lr=5e-4):
    device = torch.device("cpu")
    
    train_X, train_Y = cv_X[train_idx], cv_Y[train_idx]
    val_X, val_Y = cv_X[val_idx], cv_Y[val_idx]
    
    # Simple data augmentation on train features
    def augment(x):
        # Feature dropout (zeroes out 5% of features)
        mask = (torch.rand(x.shape) > 0.05).float()
        return x * mask

    model = TFIFClassifier(num_classes=NUM_CLASSES).to(device)
    
    # Compute class weights for this fold
    counts = np.bincount(train_Y.numpy(), minlength=NUM_CLASSES).astype(float)
    weights = counts.sum() / (NUM_CLASSES * (counts + 1e-6))
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        # Shuffle train data
        perm = torch.randperm(train_X.size(0))
        tr_X_shuf, tr_Y_shuf = train_X[perm], train_Y[perm]
        
        # Batch loop
        batch_size = 16
        t_loss, t_correct = 0.0, 0
        for i in range(0, tr_X_shuf.size(0), batch_size):
            bx = augment(tr_X_shuf[i:i+batch_size]).to(device)
            by = tr_Y_shuf[i:i+batch_size].to(device)
            
            optimizer.zero_grad()
            logits, _ = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            t_loss += loss.item() * bx.size(0)
            t_correct += (logits.argmax(1) == by).sum().item()
            
        scheduler.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_logits, _ = model(val_X.to(device))
            val_loss = criterion(val_logits, val_Y.to(device)).item()
            val_acc = (val_logits.argmax(1) == val_Y.to(device)).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    print(f"Fold {fold} finished. Best Val Acc: {best_val_acc:.4f}")
    save_path = os.path.join(SAVE_DIR, f"best_model_fold_{fold}.pt")
    torch.save({"model_state_dict": best_state, "val_acc": best_val_acc}, save_path)
    return best_val_acc

def main():
    print("Loading preprocessed features...")
    cv_X, cv_Y, test_X, test_Y = load_data()
    print(f"CV Data: {cv_X.shape}, Test Data: {test_X.shape}")

    # Set up 5-fold stratified cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs = []
    
    log_path = os.path.join(BASE, "model", "training_log.csv")
    with open(log_path, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["fold", "best_val_acc"])
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(cv_X, cv_Y)):
            print(f"\n--- Training Fold {fold} ---")
            val_acc = train_fold(fold, train_idx, val_idx, cv_X, cv_Y, epochs=80, lr=5e-4)
            fold_accs.append(val_acc)
            writer.writerow([fold, round(val_acc, 4)])
            csvf.flush()

    print(f"\nAll folds completed. Mean CV Accuracy: {np.mean(fold_accs):.4f}")
    
    # Save a flag file class_weights.json so database knows weights
    cw_path = os.path.join(SAVE_DIR, "..", "class_weights.json")
    with open(cw_path, "w") as f:
        json.dump({"mean_cv_acc": float(np.mean(fold_accs))}, f, indent=2)

if __name__ == "__main__":
    main()
