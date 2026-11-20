# AutoGuard AI — Acoustic Engine Diagnostic & 3D Failure Lab

A full-stack automotive acoustic screening and diagnostic assistant that analyzes engine audio recordings, classifies supported sound patterns (**Healthy**, **Engine Knocking**, **Misfire**, or **Inconclusive / Low Confidence**), visualizes inspection zones on an interactive 3D cutaway engine, logs diagnostic history to SQLite, and provides structured mechanical inspection recommendations.

> **Important Clinical/Mechanical Limitation:** Acoustic ML classification is a screening indicator only and never serves as physical proof of a specific mechanical fault. Always confirm findings with a qualified mechanic and OBD-II / physical testing before replacing parts.

---

## Technology Stack

- **Backend:** Python 3.13, FastAPI, Uvicorn, Pydantic v2, SQLite (`backend/autoguard.db`)
- **Audio & ML Runtime (CPU):** Librosa, SoundFile, NumPy, SciPy, scikit-learn (SVM), PyTorch CPU (`AutoGuardCNN`)
- **Frontend:** React 19, TypeScript, Vite 6, Tailwind CSS v4, Three.js / React Three Fiber (`@react-three/fiber`, `@react-three/drei`), Lucide Icons

---

## Project Structure (Key Directories)

```text
AutoGuard/
├── backend/
│   ├── main.py                     # FastAPI endpoints (/health, /api/consensus, /api/analyze, /api/diagnose, /api/history, /api/stats, /api/samples)
│   ├── model.py                    # SVM loading, Numba JIT warmup, windowed inference & abstain checks
│   ├── cnn_model.py                # AutoGuardCNN loading & batched spectrogram inference
│   ├── repair_recommendations.py   # Structured safety warnings & confidence-tiered repair guidance
│   ├── database.py                 # SQLite persistence layer (autoguard.db)
│   ├── schemas.py                  # Pydantic response models
│   └── test_backend.py             # Automated FastAPI endpoint suite
├── frontend/
│   ├── src/
│   │   ├── App.tsx                 # Main router, theme toggle (Luxury Dark / Light), health polling
│   │   ├── pages/                  # DashboardPage, EngineLabPage, DiagnosisPage, ResultsPage, HistoryPage
│   │   ├── components/             # EngineLab3D (GLTF interactive 3D engine), SpectrogramWorkbench, StatusBadge
│   │   └── services/api.ts         # FastAPI client & fallback mappings
│   └── package.json
├── models/
│   ├── best_model.pkl              # Selected V1 RBF-SVM (288-dim MFCC + spectral features, Platt calibrated)
│   ├── label_encoder.pkl           # Class encoder (engine_knock, healthy, misfire)
│   ├── metadata.json               # V1 evaluation summary
│   ├── thresholds.json             # Validation-selected thresholds (abstain=0.75, disagree=0.20)
│   └── v2/track1/                  # Experimental V2 9-class models (preserved separately)
├── ml/
│   ├── autoguard_model.pt          # Trained AutoGuardCNN weights
│   ├── features.py / train.py      # Original V1 pipeline
│   └── v2/                         # Experimental V2 pipeline (isolated)
└── data/                           # Datasets (read-only) & prepared caches
```

---

## Installation Instructions

1. **Python virtual environment (`venv` already configured at project root):**
   ```powershell
   # Activate or verify existing venv
   .\venv\Scripts\python.exe -m pip install -r requirements.txt  # if installing fresh packages
   ```
   *(Required backend packages: `fastapi`, `uvicorn`, `python-multipart`, `scikit-learn`, `torch`, `librosa`, `soundfile`, `numpy`, `joblib`, `pydantic`.)*

2. **Frontend dependencies (already installed in `frontend/node_modules/`):**
   ```powershell
   cd frontend
   npm install
   ```

---

## Exact Commands to Start the Application (Submission Day)

Open **two separate PowerShell terminals** in `C:\Users\lutfa\AutoGuard`:

### Terminal 1 — Start FastAPI Backend (Port `8000`)
```powershell
cd C:\Users\lutfa\AutoGuard
.\venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
Wait until you see:
```text
[Model] Warmup complete.
[CNN] Loaded AutoGuardCNN  classes=['engine_knock', 'healthy', 'misfire']
[Startup] Ready.
```

### Terminal 2 — Start React/Vite Frontend (Port `8443`)
```powershell
cd C:\Users\lutfa\AutoGuard\frontend
npm run dev
```
Then open **[http://localhost:8443](http://localhost:8443)** in your browser.

---

## Selected Model & Performance Summary

- **Production Pipeline (`/api/consensus`, default):** **V1 Consensus Ensemble (`0.45 AutoGuardCNN + 0.55 V1 Acoustic SVM`)**
  - **Input Features:**
    - SVM branch: `288`-dim MFCC (`40`) + $\Delta$-MFCC + $\Delta^2$-MFCC + spectral contrast + chroma + ZCR/RMS/centroid/rolloff/bandwidth (`22,050 Hz`, up to `10s` overlapping windows)
    - CNN branch: `128 × 128` log-Mel spectrogram (`AutoGuardCNN`, ~321k parameters, ~1.3 MB checkpoint)
  - **Verified Holdout Metrics (KweriMoto 3-Class Benchmark):**
    - **Validation (`N=153`):** `82.35%` Accuracy | `82.05%` Balanced Accuracy | `0.8106` Macro-F1 | **`9.90%` False-Healthy Rate**
    - **Test (`N=150`, excluding 4 clips `<0.5s`):** `88.00%` Accuracy | `90.81%` Balanced Accuracy | `0.8981` Macro-F1 | **`10.20%` False-Healthy Rate**
    - **Real-Test (`N=38` YouTube Knock Clips):** `97.37%` (`37/38`)
  - **Runtime & Memory (CPU / 8 GB RAM):** `+9.4 MB` model memory overhead (`385.5 MB` peak RSS across 346 consecutive clips); `~315–355 ms` per clip on CPU.
  - **Calibration Note:** While individual SVM predictions use Platt scaling (`probability=True`), the fused consensus and CNN softmax outputs are explicitly labeled as **uncalibrated model scores**.
  - *(Note: Experimental V2 9-class models remain isolated offline in `ml/v2/` and `models/v2/track1/`, while the live production application exclusively uses the reliable 3-class V1 Consensus pipeline.)*

---

## Supported Formats & Classes

- **Audio Formats:** `.wav`, `.mp3` (`0.5s – 180.0s` duration, non-silent, up to `25 MB`).
- **Supported Classes:**
  1. `healthy` — Normal idle/running engine signature
  2. `engine_knock` — Metallic tapping / detonation-like acoustic pattern
  3. `misfire` — Irregular firing / rough combustion cadence
  4. `uncertain` — Returned automatically when score `< 0.55` (`low` confidence) or window predictions disagree (`std > 0.20`)

---

## Known Limitations

1. **Scope limited to 3 core categories:** Recordings with unrelated issues (e.g., worn brakes, alternator whine, dead battery) or heavy background wind/traffic noise are forced into the nearest of the 3 classes or flagged `uncertain`.
2. **Uncalibrated consensus scores:** Reported percentages reflect combined uncalibrated consensus scores (`CNN softmax + SVM probability`), not calibrated physical fault probabilities.
3. **Minimum duration & signal quality:** Recordings shorter than `0.5s` (`1.0s` in browser UI preview) or silent/corrupted files are rejected (`HTTP 400 / 422`).
4. **No standalone fault proof:** Acoustic classification must be verified with OBD-II trouble codes and physical inspection before mechanical repairs.

---

## Running Verification Tests

```powershell
# Automated backend API & error-handling tests (with backend stopped on port 8000):
cd C:\Users\lutfa\AutoGuard
.\venv\Scripts\python.exe backend\test_backend.py

# Production frontend TypeScript & Vite bundle check:
cd C:\Users\lutfa\AutoGuard\frontend
npm run build
```
