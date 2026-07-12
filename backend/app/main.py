"""
main.py — FastAPI application entry point.
Run: uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
"""
import os, sys, json, shutil, uuid, threading
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ── Make sure report_generator.py is importable ──────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.database import init_db, get_conn
from app.agents   import OrchestratorAgent

UPLOAD_DIR   = os.path.join(os.path.dirname(__file__), "..", "uploads")
KEYFRAME_DIR = os.path.join(os.path.dirname(__file__), "..", "keyframes")
REPORT_DIR   = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(UPLOAD_DIR,   exist_ok=True)
os.makedirs(KEYFRAME_DIR, exist_ok=True)
os.makedirs(REPORT_DIR,   exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="TFIF API", version="1.0.0",
              description="Temporal Forensic Intelligence Framework")

app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"])

# Serve uploaded videos & keyframes statically
app.mount("/uploads",   StaticFiles(directory=UPLOAD_DIR),   name="uploads")
app.mount("/keyframes", StaticFiles(directory=KEYFRAME_DIR), name="keyframes")


@app.on_event("startup")
def startup():
    init_db()
    # Pre-warm the classifier model
    try:
        from app.agents import _get_classifier
        _get_classifier()
        print("Classifier model loaded.")
    except Exception as e:
        print(f"Classifier not yet available: {e}")


# ─── background processing ────────────────────────────────────────────────────
def _process_video(video_id: int, file_path: str):
    conn  = get_conn()
    orch  = OrchestratorAgent(get_conn)
    orch.run(video_id, file_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/videos/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4",".avi",".mkv",".mov")):
        raise HTTPException(400, "Only video files are accepted.")
    uid      = uuid.uuid4().hex
    safe     = f"{uid}_{file.filename.replace(' ','_')}"
    out_path = os.path.join(UPLOAD_DIR, safe)
    with open(out_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    conn = get_conn()
    now  = datetime.utcnow().isoformat()
    cur  = conn.execute(
        "INSERT INTO videos (filename,orig_name,upload_time,file_path) VALUES (?,?,?,?)",
        (safe, file.filename, now, out_path)
    )
    video_id = cur.lastrowid
    conn.execute(
        "INSERT INTO job_logs (video_id,status,stage,message,updated_at) VALUES (?,?,?,?,?)",
        (video_id, "queued", "upload", "", now)
    )
    conn.commit()

    t = threading.Thread(target=_process_video, args=(video_id, out_path), daemon=True)
    t.start()

    return {"video_id": video_id, "status": "queued", "filename": file.filename}


@app.get("/api/videos/{video_id}/status")
def get_status(video_id: int):
    conn = get_conn()
    row  = conn.execute(
        "SELECT status, stage, message, updated_at FROM job_logs WHERE video_id=? ORDER BY id DESC LIMIT 1",
        (video_id,)
    ).fetchone()
    if not row:
        raise HTTPException(404, "Video not found")
    return dict(row)


@app.get("/api/videos/{video_id}/result")
def get_result(video_id: int):
    conn = get_conn()
    pred = conn.execute(
        "SELECT * FROM predictions WHERE video_id=? ORDER BY id DESC LIMIT 1", (video_id,)
    ).fetchone()
    if not pred:
        raise HTTPException(404, "Result not ready yet")
    return {
        "predicted_class":  pred["predicted_class"],
        "confidence":       pred["confidence"],
        "per_class_scores": json.loads(pred["per_class_json"]),
        "timeline":         json.loads(pred["timeline_json"]),
        "keyframes":        json.loads(pred["keyframes_json"]),
    }


@app.get("/api/videos/{video_id}/report")
def get_report(video_id: int):
    conn = get_conn()
    rep  = conn.execute(
        "SELECT * FROM reports WHERE video_id=? ORDER BY id DESC LIMIT 1", (video_id,)
    ).fetchone()
    if not rep:
        raise HTTPException(404, "Report not ready yet")
    return {"report": json.loads(rep["report_json"]), "created_at": rep["created_at"]}


@app.get("/api/videos/{video_id}/report/download")
def download_report(video_id: int):
    conn = get_conn()
    rep  = conn.execute(
        "SELECT pdf_path FROM reports WHERE video_id=? ORDER BY id DESC LIMIT 1", (video_id,)
    ).fetchone()
    if not rep or not rep["pdf_path"] or not os.path.exists(rep["pdf_path"]):
        raise HTTPException(404, "PDF not available")
    return FileResponse(rep["pdf_path"], media_type="application/pdf",
                        filename=f"TFIF_Report_{video_id:06d}.pdf")


@app.get("/api/videos")
def list_videos():
    conn = get_conn()
    rows = conn.execute("""
        SELECT v.id, v.orig_name, v.upload_time, v.duration, v.resolution,
               p.predicted_class, p.confidence,
               j.status, j.stage
        FROM videos v
        LEFT JOIN predictions p ON p.video_id = v.id
        LEFT JOIN (SELECT video_id, status, stage FROM job_logs
                   WHERE id IN (SELECT MAX(id) FROM job_logs GROUP BY video_id)) j
                  ON j.video_id = v.id
        ORDER BY v.id DESC LIMIT 100
    """).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/dashboard/stats")
def dashboard_stats():
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    done  = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    avg_conf = conn.execute("SELECT AVG(confidence) FROM predictions").fetchone()[0] or 0
    class_counts_raw = conn.execute(
        "SELECT predicted_class, COUNT(*) as cnt FROM predictions GROUP BY predicted_class"
    ).fetchall()
    class_counts = {r["predicted_class"]: r["cnt"] for r in class_counts_raw}
    recent = conn.execute("""
        SELECT v.id, v.orig_name, v.upload_time, p.predicted_class, p.confidence
        FROM videos v LEFT JOIN predictions p ON p.video_id=v.id
        ORDER BY v.id DESC LIMIT 5
    """).fetchall()
    return {
        "total_videos": total,
        "processed_videos": done,
        "avg_confidence": round(avg_conf, 4),
        "class_distribution": class_counts,
        "recent_uploads": [dict(r) for r in recent],
    }


@app.get("/api/videos/{video_id}/keyframe/{filename}")
def serve_keyframe(video_id: int, filename: str):
    path = os.path.join(KEYFRAME_DIR, str(video_id), filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Keyframe not found")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}
