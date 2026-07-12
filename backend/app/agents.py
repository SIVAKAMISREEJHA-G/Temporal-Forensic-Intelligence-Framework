"""
agents.py — Preprocessing, Classification, TemporalReasoning,
            ReportGeneration, and Orchestrator agents.
"""
import os, sys, cv2, json, math, shutil, base64
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from datetime import datetime
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

# add training dir to path so model.py is importable
TRAINING_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "training")
sys.path.insert(0, TRAINING_DIR)
from model import TFIFClassifier

BASE         = os.path.join(os.path.dirname(__file__), "..", "..")
MODEL_DIR    = os.path.join(BASE, "model", "saved_model")
UPLOAD_DIR   = os.path.join(BASE, "backend", "uploads")
KEYFRAME_DIR = os.path.join(BASE, "backend", "keyframes")
os.makedirs(UPLOAD_DIR,   exist_ok=True)
os.makedirs(KEYFRAME_DIR, exist_ok=True)

CLASSES = ["Abuse","Car Accident","Explosion","Fighting","Normal","Riot","Shooting"]
SEVERITY = {
    "Normal": "low", "Car Accident": "medium", "Abuse": "high",
    "Fighting": "high", "Riot": "high", "Shooting": "critical", "Explosion": "critical"
}
NUM_FRAMES = 16
IMG_SIZE   = 224


# ─── Singleton backbone ───────────────────────────────────────────────────────
_backbone = None
_transform = None

def _get_backbone():
    global _backbone, _transform
    if _backbone is None:
        weights   = MobileNet_V3_Large_Weights.IMAGENET1K_V2
        backbone  = mobilenet_v3_large(weights=weights)
        backbone.classifier = torch.nn.Identity()
        backbone.eval()
        _backbone = backbone
        _transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
    return _backbone, _transform


# ─── Singleton classifier ─────────────────────────────────────────────────────
_classifiers = []

def _get_classifiers():
    global _classifiers
    if not _classifiers:
        for fold in range(5):
            path = os.path.join(MODEL_DIR, f"best_model_fold_{fold}.pt")
            if not os.path.exists(path):
                # Fallback to look for best_model.pt
                fallback = os.path.join(MODEL_DIR, "best_model.pt")
                if fold == 0 and os.path.exists(fallback):
                    print(f"Loading single model fallback: {fallback}")
                    ckpt = torch.load(fallback, map_location="cpu")
                    model = TFIFClassifier(num_classes=len(CLASSES))
                    model.load_state_dict(ckpt["model_state_dict"])
                    model.eval()
                    _classifiers.append(model)
                    break
                raise FileNotFoundError(f"Fold model not found: {path}")
            ckpt = torch.load(path, map_location="cpu")
            model = TFIFClassifier(num_classes=len(CLASSES))
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            _classifiers.append(model)
    return _classifiers


# ═══════════════════════════════════════════════════════════════════════════════
class PreprocessingAgent:
    """Extracts metadata and frame embeddings from an uploaded video."""

    def process(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        fps        = cap.get(cv2.CAP_PROP_FPS) or 24
        fc         = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration   = fc / fps
        resolution = f"{width}x{height}"
        cap.release()

        frames = self._extract_frames(video_path, fc)
        feats  = self._embed_frames(frames)          # (16, 960) tensor
        return {
            "fps": fps, "duration": round(duration, 2),
            "resolution": resolution, "total_frames": fc,
            "raw_frames": frames,    "features": feats,
        }

    def _extract_frames(self, path, total):
        backbone, transform = _get_backbone()
        cap = cv2.VideoCapture(path)
        indices = np.linspace(0, total-1, NUM_FRAMES, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frm = cap.read()
            if not ret:
                frm = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            else:
                frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
            frames.append(frm)
        cap.release()
        return frames

    def _embed_frames(self, frames):
        backbone, transform = _get_backbone()
        tensors = []
        with torch.no_grad():
            for frm in frames:
                img = Image.fromarray(frm)
                t   = transform(img).unsqueeze(0)
                feat = backbone(t).squeeze(0)
                tensors.append(feat)
        return torch.stack(tensors)   # (16, 960)


# ═══════════════════════════════════════════════════════════════════════════════
class ClassificationAgent:
    """Runs the trained BiLSTM model on pre-extracted features."""

    def classify(self, features: torch.Tensor):
        models = _get_classifiers()
        all_logits = []
        all_attns = []
        with torch.no_grad():
            for model in models:
                logits, attn_w = model(features.unsqueeze(0))   # (1,7), (1,16)
                all_logits.append(torch.softmax(logits, dim=1).squeeze(0))
                all_attns.append(attn_w.squeeze(0))
                
        probs = torch.stack(all_logits).mean(dim=0)
        attn_np = torch.stack(all_attns).mean(dim=0).numpy()
        
        pred_idx   = probs.argmax().item()
        conf       = probs[pred_idx].item()
        per_class  = {CLASSES[i]: round(probs[i].item(), 4) for i in range(len(CLASSES))}
        return {
            "predicted_class": CLASSES[pred_idx],
            "confidence":      round(conf, 4),
            "per_class_scores": per_class,
            "attention_weights": attn_np.tolist(),
            "severity":         SEVERITY[CLASSES[pred_idx]],
        }


# ═══════════════════════════════════════════════════════════════════════════════
class TemporalReasoningAgent:
    """Builds a timeline and selects key evidence frames."""

    def analyse(self, video_path: str, raw_frames, attn_weights, duration: float, video_id: int):
        attn = np.array(attn_weights)
        n    = len(raw_frames)
        sec_per_frame = duration / n

        # ── Key-frame selection (top-4 by attention) ──────────────────────
        top_k  = min(4, n)
        top_idx = np.argsort(attn)[-top_k:][::-1]
        kf_dir  = os.path.join(KEYFRAME_DIR, str(video_id))
        os.makedirs(kf_dir, exist_ok=True)
        keyframes = []
        for rank, fi in enumerate(sorted(top_idx)):
            ts     = round(fi * sec_per_frame, 1)
            fname  = f"kf_{rank:02d}_t{ts:.1f}s.jpg"
            fpath  = os.path.join(kf_dir, fname)
            img_bgr = cv2.cvtColor(raw_frames[fi], cv2.COLOR_RGB2BGR)
            cv2.imwrite(fpath, img_bgr)
            keyframes.append({
                "frame_idx": int(fi),
                "timestamp_sec": ts,
                "attention": round(float(attn[fi]), 4),
                "path": fpath,
                "url":  f"/api/videos/{video_id}/keyframe/{fname}",
            })

        # ── Temporal segmentation (4 coarse segments) ─────────────────────
        seg_len = n // 4
        segments = []
        labels   = ["Pre-incident / Setup", "Escalation", "Peak Incident", "Aftermath"]
        for s in range(4):
            s_start = s * seg_len
            s_end   = (s+1)*seg_len if s < 3 else n
            avg_attn = float(np.mean(attn[s_start:s_end]))
            t_start  = round(s_start * sec_per_frame, 1)
            t_end    = round(s_end   * sec_per_frame, 1)
            segments.append({
                "segment": s+1,
                "label":   labels[s],
                "time_start": t_start,
                "time_end":   t_end,
                "avg_attention": round(avg_attn, 4),
                "activity_level": "High" if avg_attn > 0.07 else ("Medium" if avg_attn > 0.04 else "Low"),
            })

        return {"keyframes": keyframes, "segments": segments}


# ═══════════════════════════════════════════════════════════════════════════════
class OrchestratorAgent:
    """Drives the full analysis pipeline and updates the database."""

    def __init__(self, db_conn_factory):
        self.get_conn = db_conn_factory
        self.prep_agent   = PreprocessingAgent()
        self.clf_agent    = ClassificationAgent()
        self.temp_agent   = TemporalReasoningAgent()

    def _update_job(self, video_id, status, stage, message=""):
        conn = self.get_conn()
        now  = datetime.utcnow().isoformat()
        conn.execute(
            "INSERT INTO job_logs (video_id,status,stage,message,updated_at) VALUES (?,?,?,?,?)",
            (video_id, status, stage, message, now)
        )
        conn.commit()

    def run(self, video_id: int, video_path: str):
        conn = self.get_conn()
        try:
            # 1. Preprocessing
            self._update_job(video_id, "processing", "preprocessing")
            meta = self.prep_agent.process(video_path)
            conn.execute(
                "UPDATE videos SET duration=?, resolution=?, fps=? WHERE id=?",
                (meta["duration"], meta["resolution"], meta["fps"], video_id)
            )
            conn.commit()

            # 2. Classification
            self._update_job(video_id, "processing", "classifying")
            clf  = self.clf_agent.classify(meta["features"])

            # 3. Temporal reasoning
            self._update_job(video_id, "processing", "temporal_reasoning")
            temporal = self.temp_agent.analyse(
                video_path, meta["raw_frames"],
                clf["attention_weights"], meta["duration"], video_id
            )

            # 4. Persist prediction
            now = datetime.utcnow().isoformat()
            conn.execute(
                """INSERT INTO predictions
                   (video_id,predicted_class,confidence,per_class_json,timeline_json,keyframes_json,created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (video_id, clf["predicted_class"], clf["confidence"],
                 json.dumps(clf["per_class_scores"]),
                 json.dumps(temporal["segments"]),
                 json.dumps(temporal["keyframes"]),
                 now)
            )
            conn.commit()

            # 5. Report generation
            self._update_job(video_id, "processing", "generating_report")
            from report_generator import ReportGenerationAgent
            rga = ReportGenerationAgent()
            report_data = rga.generate(video_id, video_path, clf, temporal, meta)

            pdf_path    = report_data.get("pdf_path", "")
            conn.execute(
                "INSERT INTO reports (video_id,report_json,pdf_path,created_at) VALUES (?,?,?,?)",
                (video_id, json.dumps(report_data["report_json"]), pdf_path, now)
            )
            conn.commit()

            self._update_job(video_id, "done", "complete")

        except Exception as e:
            self._update_job(video_id, "failed", "error", str(e))
            raise
