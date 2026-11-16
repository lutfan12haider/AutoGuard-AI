# AutoGuard AI — Engine Diagnostic System

A full-stack application that classifies car engine audio recordings into:
- **Healthy Engine**
- **Engine Knock**
- **Engine Misfire**

---

## Project Structure

```
AutoGuard/
├── data/kwerimoto/prepared/     ← Dataset (do not modify)
│   ├── train/
│   ├── validation/
│   ├── test/
│   └── real_test/
├── ml/
│   ├── features.py              ← MFCC feature extraction
│   ├── train.py                 ← Train SVM + Random Forest
│   └── test_inference.py        ← Inference smoke test
├── models/
│   ├── best_model.pkl           ← Trained Random Forest
│   ├── label_encoder.pkl
│   └── metadata.json            ← Evaluation metrics
├── backend/
│   ├── main.py                  ← FastAPI application
│   ├── model.py                 ← Model loading + inference
│   ├── database.py              ← SQLite helpers
│   ├── schemas.py               ← Pydantic schemas
│   └── autoguard.db             ← Created automatically at startup
├── venv/                        ← Python virtual environment
├── download_dataset.py
├── inspect_dataset.py
└── README.md

C:\Users\lutfa\Downloads\AutoGuard AI Diagnostic Dashboard\
├── src/
│   ├── App.tsx                  ← React application (all 4 pages)
│   ├── api.ts                   ← API service layer
│   ├── main.tsx                 ← React entrypoint
│   └── index.css                ← All styles
├── package.json
└── vite.config.ts
```

---

## Requirements

- **Python 3.13** (already installed)
- **Node.js 22+** (already installed)
- **pnpm 10** (install once: `npm install -g pnpm`)

---

## Running the Application

### Step 1 — Start the Backend

Open a terminal in `C:\Users\lutfa\AutoGuard\` and run:

```powershell
.\venv\Scripts\python.exe -m uvicorn backend.main:app --port 8000
```

You should see:
```
[DB] Database ready at: ...autoguard.db
[Model] Loading trained model ...
[Model] Loaded: Random Forest
[Model] Warming up (JIT compilation) ...
[Model] Warmup complete.
[Startup] Ready.
INFO: Uvicorn running on http://0.0.0.0:8000
```

### Step 2 — Start the Frontend

Open a **second terminal** in `C:\Users\lutfa\Downloads\AutoGuard AI Diagnostic Dashboard\` and run:

```powershell
pnpm dev
```

You should see:
```
  VITE v8.0.5  ready in XXX ms
  ➜  Local:   http://localhost:8443/
```

Open **http://localhost:8443** in your browser.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Backend + model status |
| POST | `/api/analyze` | Upload WAV → get prediction |
| GET | `/api/history` | All diagnostic records |
| GET | `/api/history/{id}` | Single record by ID |
| GET | `/api/stats` | Dashboard statistics |

---

## ML Model

- **Algorithm:** Random Forest (200 trees, balanced class weights)
- **Features:** 40 MFCC coefficients × (mean + std) = 80-dimensional vector
- **Sample rate:** 22050 Hz, max 10s per clip
- **Test accuracy:** 87.7% | Macro F1: 0.887

### Retrain from scratch

```powershell
.\venv\Scripts\python.exe ml\train.py
```

---

## Troubleshooting

**Frontend shows "Backend Offline" in sidebar**
- Make sure the backend server is running on port 8000
- Check: `http://localhost:8000/health`

**`ModuleNotFoundError: No module named 'fastapi'`**
- Activate the venv first, or use the full path: `.\venv\Scripts\python.exe`

**Port 8000 already in use**
- Change the port: `--port 8001` and update `BASE_URL` in `src/api.ts`

**Audio file rejected**
- Only `.wav` files are accepted, max 25 MB, minimum 0.5 seconds
