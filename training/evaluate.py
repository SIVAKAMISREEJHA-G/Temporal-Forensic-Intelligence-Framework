import os, sys, json, csv
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              confusion_matrix, roc_auc_score)

sys.path.insert(0, os.path.dirname(__file__))
from model import TFIFClassifier
from dataset import get_loaders

BASE      = r"C:\Users\sivak\Downloads\crime data\TFIF"
SAVE_DIR  = os.path.join(BASE, "model", "saved_model")
REPORT_DIR = os.path.join(BASE, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

CLASSES = ["Abuse","Car Accident","Explosion","Fighting","Normal","Riot","Shooting"]

def load_ensemble():
    models = []
    for fold in range(5):
        path = os.path.join(SAVE_DIR, f"best_model_fold_{fold}.pt")
        if not os.path.exists(path):
            print(f"Warning: Fold {fold} model not found at {path}")
            continue
        ckpt = torch.load(path, map_location="cpu")
        model = TFIFClassifier(num_classes=len(CLASSES))
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models.append(model)
    return models

def evaluate():
    models = load_ensemble()
    if not models:
        print("Error: No models found for evaluation!")
        return

    _, _, test_loader = get_loaders(batch_size=16)

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for feats, labels in test_loader:
            # Average predictions across ensemble
            batch_probs = torch.zeros(feats.size(0), len(CLASSES))
            for model in models:
                logits, _ = model(feats)
                probs = torch.softmax(logits, dim=1)
                batch_probs += probs
            batch_probs /= len(models)
            
            preds = batch_probs.argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(batch_probs.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, sup = precision_recall_fscore_support(all_labels, all_preds,
                                                          average=None, labels=range(len(CLASSES)),
                                                          zero_division=0)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(all_labels, all_preds,
                                                               average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(CLASSES)))

    # ROC-AUC (one-vs-rest)
    try:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(all_labels, classes=range(len(CLASSES)))
        roc_auc = roc_auc_score(y_bin, all_probs, average="macro", multi_class="ovr")
    except Exception:
        roc_auc = 0.0

    print(f"\n{'='*60}")
    print("ENSEMBLE TEST SET EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {f1_m:.4f}")
    print(f"Macro P  : {prec_m:.4f}")
    print(f"Macro R  : {rec_m:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"\nPer-class metrics:")
    for i,cls in enumerate(CLASSES):
        print(f"  {cls:15s}: P={prec[i]:.3f} R={rec[i]:.3f} F1={f1[i]:.3f} (n={sup[i]})")

    # ── Load CV metrics from log ─────────────────────────────────────────
    log_path = os.path.join(BASE, "model", "training_log.csv")
    cv_rows_html = ""
    if os.path.exists(log_path):
        with open(log_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                cv_rows_html += f"<tr><td>Fold {row['fold']}</td><td>{float(row['best_val_acc']):.4f}</td></tr>"

    # ── Build HTML report with base64 embedded figures ───────────────────
    figs_b64 = {}

    def fig_to_b64(fig):
        import io, base64
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(8,7), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES))); ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", color="white", fontsize=9)
    ax.set_yticklabels(CLASSES, color="white", fontsize=9)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j] < cm.max()/2 else "black", fontsize=8)
    ax.set_xlabel("Predicted", color="white"); ax.set_ylabel("True", color="white")
    ax.set_title("Confusion Matrix (Ensemble)", color="white", fontsize=13)
    plt.colorbar(im, ax=ax)
    figs_b64["cm"] = fig_to_b64(fig); plt.close(fig)

    # Per-class bar chart
    fig, ax = plt.subplots(figsize=(10,4), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")
    x = np.arange(len(CLASSES)); width = 0.25
    ax.bar(x-width, prec,  width, label="Precision", color="#e94560", alpha=0.9)
    ax.bar(x,       rec,   width, label="Recall",    color="#0f3460", alpha=0.9)
    ax.bar(x+width, f1,    width, label="F1",        color="#533483", alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(CLASSES, rotation=30, ha="right", color="white", fontsize=9)
    ax.tick_params(colors="white"); ax.set_ylim(0,1.05)
    ax.set_title("Ensemble Per-Class Precision / Recall / F1", color="white")
    ax.legend(labelcolor="white", facecolor="#1a1a2e")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for sp in ("bottom","left"): ax.spines[sp].set_color("#555")
    figs_b64["perclass"] = fig_to_b64(fig); plt.close(fig)

    # Build HTML
    rows = "".join(f"<tr><td>{CLASSES[i]}</td><td>{prec[i]:.3f}</td><td>{rec[i]:.3f}</td><td>{f1[i]:.3f}</td><td>{int(sup[i])}</td></tr>" for i in range(len(CLASSES)))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>TFIF Model Performance Report</title>
<style>
  body{{margin:0;padding:32px;background:#0d1117;color:#e6edf3;font-family:'Segoe UI',sans-serif}}
  h1{{color:#58a6ff;border-bottom:2px solid #21262d;padding-bottom:12px}}
  h2{{color:#79c0ff;margin-top:32px}}
  .metric-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:20px 0}}
  .card{{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:16px;text-align:center}}
  .card .val{{font-size:2em;font-weight:700;color:#58a6ff}}
  .card .label{{font-size:.85em;color:#8b949e;margin-top:4px}}
  table{{width:100%;border-collapse:collapse;margin-top:12px}}
  th{{background:#21262d;padding:10px;text-align:left;color:#8b949e;font-size:.9em}}
  td{{padding:9px;border-bottom:1px solid #21262d}}
  tr:hover td{{background:#1c2128}}
  img{{border-radius:8px;margin:12px 0}}
</style></head>
<body>
<h1>🔬 TFIF — Ensemble Performance Report</h1>
<p>5-Fold Stratified Cross-Validation Ensemble</p>
<div class="metric-grid">
  <div class="card"><div class="val">{acc:.1%}</div><div class="label">Test Accuracy</div></div>
  <div class="card"><div class="val">{f1_m:.3f}</div><div class="label">Macro F1</div></div>
  <div class="card"><div class="val">{prec_m:.3f}</div><div class="label">Macro Precision</div></div>
  <div class="card"><div class="val">{roc_auc:.3f}</div><div class="label">ROC-AUC (macro)</div></div>
</div>
<h2>5-Fold CV Validation Results</h2>
<table><thead><tr><th>Fold</th><th>Best Val Accuracy</th></tr></thead>
<tbody>{cv_rows_html}</tbody></table>
<h2>Confusion Matrix</h2>
<img src="data:image/png;base64,{figs_b64['cm']}" style="max-width:700px">
<h2>Per-Class Metrics</h2>
<img src="data:image/png;base64,{figs_b64['perclass']}" style="max-width:100%">
<table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
<tbody>{rows}</tbody></table>
</body></html>"""

    report_path = os.path.join(REPORT_DIR, "model_performance_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nReport saved -> {report_path}")
    
    # Save a JSON file with stats for the readme/verification
    stats_path = os.path.join(SAVE_DIR, "ensemble_stats.json")
    with open(stats_path, "w") as f:
        json.dump({"test_accuracy": acc, "macro_f1": f1_m, "roc_auc": roc_auc}, f, indent=2)
        
    return {"accuracy": acc, "macro_f1": f1_m, "roc_auc": roc_auc}


if __name__ == "__main__":
    evaluate()
