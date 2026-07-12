# TFIF — Temporal Forensic Intelligence Framework

> AI-powered surveillance video analysis platform that classifies incidents, reconstructs crime timelines, and generates downloadable forensic reports.

---

## Architecture

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
│   └── model_performance_report.html  # Evaluation report with charts
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI application + all API endpoints
│   │   ├── agents.py           # PreprocessingAgent, ClassificationAgent,
│   │   │                       #   TemporalReasoningAgent, OrchestratorAgent
│   │   └── database.py         # SQLite schema + connection factory
│   ├── report_generator.py     # LLM report generation + ReportLab PDF builder
│   ├── uploads/                # Uploaded videos (created at runtime)
│   ├── keyframes/              # Extracted evidence images (per video)
│   ├── reports/                # Generated PDF reports
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── App.jsx             # Router + sidebar layout
│       ├── index.css           # Full dark-mode design system
│       └── pages/
│           ├── HomePage.jsx    # Overview: stats, charts, recent uploads
│           ├── UploadPage.jsx  # Drag-drop upload + live pipeline status
│           ├── HistoryPage.jsx # All analysed videos table
│           └── AnalysisPage.jsx # Classification, timeline, evidence, report
└── data/
    ├── metadata.json           # Video→label→feature path mapping
    ├── splits.json             # Train/val/test split indices
    └── processed/              # Cached MobileNetV3 feature tensors (.pt)
```

---

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 18+
- At least 4 GB RAM (8 GB recommended for preprocessing)

### 1. Dataset Preparation (one-time)

```powershell
# Inspect dataset and rename folders
python training\dataset_inspection.py

# Augment Abuse class 6→40 videos
python training\augment_abuse_class.py

# Extract MobileNetV3 frame embeddings (takes 15–60 min on CPU)
python training\preprocessing.py
```

### 2. Model Training

```powershell
# Train the BiLSTM temporal classifier
python training\train.py --epochs 60 --batch 16 --lr 3e-4

# Evaluate and generate performance report
python training\evaluate.py
# → reports/model_performance_report.html
```

### 3. Start Backend API

```powershell
cd backend
# Create .env file with your API key (optional — fallback report used if absent):
#   LLM_PROVIDER=anthropic
#   ANTHROPIC_API_KEY=sk-ant-...
# OR
#   LLM_PROVIDER=openai
#   OPENAI_API_KEY=sk-...

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# API available at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### 4. Start Frontend

```powershell
cd frontend
npm install
npm run dev
# Dashboard at http://localhost:5173
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/videos/upload` | Upload a surveillance video |
| GET  | `/api/videos/{id}/status` | Processing stage |
| GET  | `/api/videos/{id}/result` | Classification + timeline + keyframes |
| GET  | `/api/videos/{id}/report` | Full LLM forensic report (JSON) |
| GET  | `/api/videos/{id}/report/download` | Download PDF report |
| GET  | `/api/videos` | List all analysed videos |
| GET  | `/api/dashboard/stats` | Aggregate statistics |

---

## Model Architecture

- **Backbone**: MobileNetV3-Large (ImageNet pretrained, frozen) → 960-dim frame embeddings
- **Temporal Head**: 2-layer BiLSTM (hidden=256) + self-attention over 16 uniformly sampled frames
- **Classifier**: LayerNorm → Dropout → Linear(512→128) → GELU → Linear(128→7)
- **Training**: AdamW + CosineAnnealingLR + class-weighted CrossEntropyLoss + label smoothing
- **Target**: >90% test accuracy

---

## Dataset

- **Source**: XD-Violence (subset, 7 classes)
- **Classes**: Normal (50), Fighting (40), Shooting (40), Explosion (40), Riot (40), Car Accident (40), Abuse (40 after augmentation)
- **Augmentation**: Horizontal flip, brightness/contrast jitter, Gaussian noise, speed jitter, random crop+resize
- **Splits**: 70% train / 15% val / 15% test (stratified)

---

## LLM Report Generation

Set environment variables before starting the backend:

```env
# Anthropic Claude (default)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# OR OpenAI GPT-4o-mini
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

If no API key is provided, the system generates a deterministic structured report from the raw classification and timeline data.
