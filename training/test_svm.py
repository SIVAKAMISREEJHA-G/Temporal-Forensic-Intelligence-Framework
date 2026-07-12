import os, json, torch
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

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
    train_feats.append(torch.load(rec["feature_path"], map_location="cpu").mean(dim=0).numpy())
    train_labels.append(rec["label"])
    
for idx in splits["val"]:
    rec = records[idx]
    val_feats.append(torch.load(rec["feature_path"], map_location="cpu").mean(dim=0).numpy())
    val_labels.append(rec["label"])

for idx in splits["test"]:
    rec = records[idx]
    test_feats.append(torch.load(rec["feature_path"], map_location="cpu").mean(dim=0).numpy())
    test_labels.append(rec["label"])

train_X, train_Y = np.stack(train_feats), np.array(train_labels)
val_X, val_Y = np.stack(val_feats), np.array(val_labels)
test_X, test_Y = np.stack(test_feats), np.array(test_labels)

print(f"Mean pooled features train: {train_X.shape}, val: {val_X.shape}, test: {test_X.shape}")

classifiers = {
    "Linear SVM": SVC(C=1.0, kernel="linear", probability=True),
    "RBF SVM (C=10)": SVC(C=10.0, kernel="rbf", probability=True),
    "RBF SVM (C=1)": SVC(C=1.0, kernel="rbf", probability=True),
    "Logistic Regression (L2)": LogisticRegression(C=1.0, max_iter=1000),
    "Extra Trees (100)": ExtraTreesClassifier(n_estimators=100, random_state=42),
    "Random Forest (100)": RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, clf in classifiers.items():
    pipe = make_pipeline(StandardScaler(), clf)
    pipe.fit(train_X, train_Y)
    val_acc = pipe.score(val_X, val_Y)
    test_acc = pipe.score(test_X, test_Y)
    print(f"Classifier: {name:25s} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")
