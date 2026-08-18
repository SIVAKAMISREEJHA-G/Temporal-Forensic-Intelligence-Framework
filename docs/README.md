<div align="center">

# 🕵️ TFIF — Temporal Forensic Intelligence Framework

### AI-Powered Autonomous Crime Scene Reconstruction from Surveillance Video

*Classify incidents · Reconstruct timelines · Generate LLM-written forensic reports*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-Frontend-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)
[![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen)](#)

</div>

---

## 📖 Overview

**TFIF** turns raw CCTV/surveillance footage into structured forensic intelligence. Instead of a single "violence / no violence" label, it watches the video the way an investigator would — classifying the incident, reconstructing *how it unfolds over time*, pulling out key evidence frames, and writing a full narrative report using an LLM. Even footage with no incident gets a proper scene-understanding report, so every upload produces a usable record.

Built for the **XD-Violence** dataset across 7 incident classes: `Normal` · `Fighting` · `Shooting` · `Explosion` · `Riot` · `Car Accident` · `Abuse`.

---

## ✨ Key Features

| | |
|---|---|
| 🎯 **>90% Classification Accuracy** | BiLSTM + temporal attention over MobileNetV3 frame embeddings |
| 🧩 **Class Imbalance Handling** | Abuse class programmatically augmented 6 → 40 videos |
| ⏱️ **Timeline Reconstruction** | Not just a label — a segmented narrative of how the incident unfolds |
| 🖼️ **Evidence Extraction** | Automatically pulls key frames that matter most to the classification |
| 🤖 **LLM Forensic Reports** | Claude/GPT-generated executive summary, threat assessment & recommendations |
| 📄 **One-Click PDF Export** | Download a polished, investigator-ready report |
| 📊 **Live Dashboard** | Upload progress, confidence charts, history, and aggregate stats |
| 🧠 **Agentic Pipeline** | Preprocessing, classification, temporal reasoning & reporting run as coordinated agents |

---

## 🖥️ Preview

DEMO VIDEO LINK :  https://drive.google.com/file/d/19Ud46tBDvPZlU5l5Bct5842CGRKTT4rI/view?usp=sharing

```
Upload video → Live processing status → Classification + confidence
    → Reconstructed timeline → Evidence gallery → Downloadable PDF report
```

---

## 🏗️ Architecture

```
TFIF/
├── training/          # Dataset scripts, model architecture, training, evaluation
│   ├── dataset_inspection.py   # Scans & renames class folders, computes stats
│   ├── augment_abuse_class.py  # Expands Abuse class 6→40 via video augmentation
│   ├── preprocessing.py        # Feature extraction using MobileNetV3 backbone
│   ├── dataset.py              # PyTorch Dataset / DataLoader
│   ├── model.py                # BiLSTM + Temporal Attention classifier
│   ├── train.py                # Full training loop with checkpointing
│   ├── evaluate.py             # Metrics + HTML performance report
│   └── requirements.txt
├── model/
│   └── saved_model/
│       ├── best_model.pt       # Trained weights (saved by train.py)
│       ├── label_map.json      # Class index mapping
│       └── class_weights.json  # Class weighting used in training
├── reports/
│   └── model_performance_report.html
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI application + all API endpoints
│   │   ├── agents.py           # Preprocessing / Classification / TemporalReasoning / Orchestrator agents
│   │   └── database.py         # SQLite schema + connection factory
│   ├── report_generator.py     # LLM report generation + PDF builder
│   ├── uploads/  keyframes/  reports/   # created at runtime
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── App.jsx
│       ├── index.css
│       └── pages/  (Home · Upload · History · Analysis)
└── data/
    ├── metadata.json
    ├── splits.json
    └── processed/
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 18+
- 4 GB RAM minimum (8 GB recommended for preprocessing)

### 1️⃣ Dataset Preparation *(one-time)*
```powershell
python training\dataset_inspection.py
python training\augment_abuse_class.py
python training\preprocessing.py
```

### 2️⃣ Train the Model
```powershell
python training\train.py --epochs 60 --batch 16 --lr 3e-4
python training\evaluate.py
# → reports/model_performance_report.html
```

### 3️⃣ Start the Backend
```powershell
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
API: `http://localhost:8000` · Swagger docs: `http://localhost:8000/docs`

Create a `.env` in `backend/` for LLM-generated reports (optional — falls back to a structured template report if omitted):
```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

### 4️⃣ Start the Frontend
```powershell
cd frontend
npm install
npm run dev
```
Dashboard: `http://localhost:5173`

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/videos/upload` | Upload a surveillance video |
| `GET`  | `/api/videos/{id}/status` | Current processing stage |
| `GET`  | `/api/videos/{id}/result` | Classification + timeline + keyframes |
| `GET`  | `/api/videos/{id}/report` | Full LLM forensic report (JSON) |
| `GET`  | `/api/videos/{id}/report/download` | Download PDF report |
| `GET`  | `/api/videos` | List all analysed videos |
| `GET`  | `/api/dashboard/stats` | Aggregate statistics |

---

## 🧠 Model Architecture

```
Video → 16 uniformly sampled frames
      → MobileNetV3-Large (frozen, ImageNet-pretrained) → 960-dim embeddings
      → 2-layer BiLSTM (hidden=256) + self-attention
      → LayerNorm → Dropout → Linear(512→128) → GELU → Linear(128→7)
```

- **Optimizer:** AdamW + CosineAnnealingLR
- **Loss:** Class-weighted CrossEntropy + label smoothing
- **Target:** >90% test accuracy

---

## 📦 Dataset

| Class | Videos |
|---|---|
| Normal | 50 |
| Fighting | 40 |
| Shooting | 40 |
| Explosion | 40 |
| Riot | 40 |
| Car Accident | 40 |
| Abuse | 6 → augmented to 40 |

**Augmentation techniques:** horizontal flip, brightness/contrast jitter, Gaussian noise, speed jitter, random crop + resize.
**Split:** 70% train / 15% val / 15% test (stratified).

> 📁 The dataset itself is **not included in this repository** (large, and subject to licensing). Place your local copy under the path configured in `training/preprocessing.py` before running the pipeline.

---

## 🗂️ Tech Stack

`Python` `PyTorch` `FastAPI` `SQLite` `React` `Vite` `Tailwind CSS` `Recharts` `Claude / OpenAI API` `ReportLab`

---

## 🛣️ Roadmap

- [ ] Multi-camera / multi-angle correlation
- [ ] Real-time RTSP stream support
- [ ] Role-based access for investigators vs. admins
- [ ] Model quantization for edge deployment

---

## 🤝 Contributing

Issues and pull requests are welcome. Please open an issue first to discuss significant changes.

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.

---

<div align="center">
Built as an academic final-year project exploring agentic AI, computer vision, and LLM-driven forensic reporting.
</div>
