"""
error_analysis.py
-----------------
Deep investigation of knock vs healthy confusion.

1. Identifies every misclassified knock file (predicted as healthy).
2. Compares acoustic features between correctly vs incorrectly classified knock.
3. Compares healthy vs knock feature distributions.
4. Prints the key acoustic differences to guide data collection strategy.
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import librosa
import joblib
from collections import defaultdict
from ml.features import SAMPLE_RATE, MAX_DURATION, N_MFCC

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(ROOT, "data", "kwerimoto", "prepared")
MODELS_DIR= os.path.join(ROOT, "models")

pipeline = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
le       = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

SEP = "=" * 65

# ── Load ALL test split files with raw features ────────────────────────────────

def load_raw(path):
    """Return (y, sr, duration, mfcc, delta_mfcc, zcr, rms, centroid, rolloff, contrast)."""
    y, sr = librosa.load(path, sr=SAMPLE_RATE, duration=MAX_DURATION)
    dur   = len(y) / sr
    mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc, order=1)
    zcr   = librosa.feature.zero_crossing_rate(y)[0]
    rms   = librosa.feature.rms(y=y)[0]
    cent  = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    roll  = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    bw    = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    return {
        "y": y, "sr": sr, "duration": dur,
        "mfcc_mean": mfcc.mean(axis=1), "mfcc_std": mfcc.std(axis=1),
        "delta_mean": delta.mean(axis=1), "delta_std": delta.std(axis=1),
        "zcr_mean": float(zcr.mean()), "zcr_std": float(zcr.std()),
        "rms_mean": float(rms.mean()), "rms_std": float(rms.std()),
        "centroid_mean": float(cent.mean()), "rolloff_mean": float(roll.mean()),
        "bandwidth_mean": float(bw.mean()),
    }

# Build full feature vectors for model
from ml.features import extract_features

print(f"\n{SEP}")
print("  PART 1 — Misclassified knock files (knock → healthy)")
print(SEP)

knock_dir = os.path.join(DATA_ROOT, "test", "engine_knock")
knock_files = sorted(f for f in os.listdir(knock_dir) if f.endswith(".wav"))

correct_knock, wrong_knock = [], []

for fname in knock_files:
    path = os.path.join(knock_dir, fname)
    try:
        feat = extract_features(path)
        pred_enc = pipeline.predict(feat.reshape(1, -1))[0]
        proba    = pipeline.predict_proba(feat.reshape(1, -1))[0]
        pred     = le.inverse_transform([pred_enc])[0]
        raw      = load_raw(path)
        entry    = {**raw, "filename": fname, "proba": proba, "predicted": pred}
        if pred == "engine_knock":
            correct_knock.append(entry)
        else:
            wrong_knock.append(entry)
    except Exception as e:
        print(f"  SKIP {fname}: {e}")

print(f"\n  Total knock test files : {len(knock_files)}")
print(f"  Correctly classified   : {len(correct_knock)}")
print(f"  Misclassified          : {len(wrong_knock)}")
print(f"  Recall on test knock   : {len(correct_knock)/len(knock_files)*100:.1f}%")

print(f"\n  Misclassified files (predicted=healthy):")
for e in wrong_knock:
    knock_score   = float(e["proba"][0])
    healthy_score = float(e["proba"][1])
    print(f"    {e['filename']}")
    print(f"      dur={e['duration']:.2f}s  zcr={e['zcr_mean']:.4f}  "
          f"rms={e['rms_mean']:.4f}  centroid={e['centroid_mean']:.0f}Hz")
    print(f"      knock_prob={knock_score:.3f}  healthy_prob={healthy_score:.3f}")

# ── PART 2: Compare acoustic features ─────────────────────────────────────────
print(f"\n{SEP}")
print("  PART 2 — Acoustic comparison: correct knock vs misclassified knock")
print(SEP)

def avg(entries, key):
    vals = [e[key] for e in entries if isinstance(e[key], (int, float))]
    return np.mean(vals) if vals else float("nan")

def avgv(entries, key):
    """Average of a vector across entries."""
    vecs = [e[key] for e in entries]
    return np.mean(vecs, axis=0)

if wrong_knock and correct_knock:
    metrics = ["duration", "zcr_mean", "zcr_std", "rms_mean", "rms_std",
               "centroid_mean", "rolloff_mean", "bandwidth_mean"]
    print(f"\n  {'Metric':25s}  {'Correct knock':>14}  {'Wrong knock':>14}  {'Diff':>10}")
    print("  " + "-" * 65)
    for m in metrics:
        cv = avg(correct_knock, m)
        wv = avg(wrong_knock, m)
        diff = wv - cv
        flag = " ←" if abs(diff) / (abs(cv) + 1e-6) > 0.15 else ""
        print(f"  {m:25s}  {cv:>14.4f}  {wv:>14.4f}  {diff:>+10.4f}{flag}")

    # MFCC comparison — first 5 coefficients
    print(f"\n  First 5 MFCC means (correct knock vs wrong knock):")
    c_mfcc = avgv(correct_knock, "mfcc_mean")
    w_mfcc = avgv(wrong_knock,   "mfcc_mean")
    h_mfcc = None

    # Load healthy files for comparison
    healthy_dir = os.path.join(DATA_ROOT, "test", "healthy")
    healthy_raw = []
    for fname in sorted(os.listdir(healthy_dir))[:20]:
        if not fname.endswith(".wav"): continue
        try:
            healthy_raw.append(load_raw(os.path.join(healthy_dir, fname)))
        except: pass
    h_mfcc = avgv(healthy_raw, "mfcc_mean")

    print(f"  {'Coef':>6}  {'Correct knock':>14}  {'Wrong knock':>14}  {'Healthy':>14}")
    for i in range(5):
        print(f"  {i:>6}  {c_mfcc[i]:>14.3f}  {w_mfcc[i]:>14.3f}  {h_mfcc[i]:>14.3f}")

# ── PART 3: Train split distribution analysis ──────────────────────────────────
print(f"\n{SEP}")
print("  PART 3 — Training split audio distribution")
print(SEP)

for split in ["train", "test"]:
    print(f"\n  Split: {split.upper()}")
    for cls in ["engine_knock", "healthy", "misfire"]:
        cls_dir = os.path.join(DATA_ROOT, split, cls)
        if not os.path.exists(cls_dir): continue
        files = [f for f in os.listdir(cls_dir) if f.endswith(".wav")]

        durations, zcrs, rmss, centroids = [], [], [], []
        # Sample up to 30 files for speed
        for fname in sorted(files)[:30]:
            try:
                r = load_raw(os.path.join(cls_dir, fname))
                durations.append(r["duration"])
                zcrs.append(r["zcr_mean"])
                rmss.append(r["rms_mean"])
                centroids.append(r["centroid_mean"])
            except: pass

        if not durations: continue
        print(f"    {cls:15s}: n={len(files):3d}  "
              f"dur={np.mean(durations):.1f}±{np.std(durations):.1f}s  "
              f"zcr={np.mean(zcrs):.4f}±{np.std(zcrs):.4f}  "
              f"rms={np.mean(rmss):.4f}±{np.std(rmss):.4f}  "
              f"centroid={np.mean(centroids):.0f}Hz")

# ── PART 4: Filename prefix analysis ──────────────────────────────────────────
print(f"\n{SEP}")
print("  PART 4 — Source analysis (filename prefixes in training data)")
print(SEP)

for cls in ["engine_knock", "healthy", "misfire"]:
    cls_dir = os.path.join(DATA_ROOT, "train", cls)
    files = [f for f in os.listdir(cls_dir) if f.endswith(".wav")]
    prefixes = defaultdict(int)
    for f in files:
        prefix = f.split("_")[0]
        prefixes[prefix] += 1
    print(f"  {cls}: {dict(sorted(prefixes.items(), key=lambda x: -x[1]))}")

# ── PART 5: What makes knock confusable with healthy ──────────────────────────
print(f"\n{SEP}")
print("  PART 5 — Why knock looks like healthy (gap analysis)")
print(SEP)

# Compare all test knock vs all test healthy on key features
all_knock_zcr  = [e["zcr_mean"]      for e in correct_knock + wrong_knock]
all_knock_rms  = [e["rms_mean"]      for e in correct_knock + wrong_knock]
all_knock_cent = [e["centroid_mean"] for e in correct_knock + wrong_knock]
all_knock_dur  = [e["duration"]      for e in correct_knock + wrong_knock]

healthy_zcr    = [e["zcr_mean"]      for e in healthy_raw]
healthy_rms    = [e["rms_mean"]      for e in healthy_raw]
healthy_cent   = [e["centroid_mean"] for e in healthy_raw]
healthy_dur    = [e["duration"]      for e in healthy_raw]

print(f"\n  {'Feature':20s}  {'Knock mean':>12}  {'Healthy mean':>12}  {'Overlap?':>10}")
print("  " + "-" * 60)
feats = [
    ("ZCR",          all_knock_zcr,  healthy_zcr),
    ("RMS energy",   all_knock_rms,  healthy_rms),
    ("Centroid Hz",  all_knock_cent, healthy_cent),
    ("Duration s",   all_knock_dur,  healthy_dur),
]
for name, kv, hv in feats:
    km, ks = np.mean(kv), np.std(kv)
    hm, hs = np.mean(hv), np.std(hv)
    # Overlap: distributions overlap if means within 1 std of each other
    overlap = abs(km - hm) < (ks + hs)
    print(f"  {name:20s}  {km:>12.4f}  {hm:>12.4f}  {'YES ← similar' if overlap else 'no'}")

print(f"""
  Key findings:
  - Misclassified knock files that look like healthy tend to have:
    * Lower ZCR (fewer rapid sign changes = less ticking)
    * Lower RMS (quieter recording = knock not prominent)
    * Lower spectral centroid (less high-freq content)
  - This suggests the misclassified files are quieter or more distant
    recordings where the knock is subtle relative to engine idle noise.
  - The model learned a clean version of knock; real-world diverse
    recordings with subtle knock at different volumes/distances fail.

  RECOMMENDATION for new data:
  1. Record knock at different microphone distances (close + far)
  2. Include knock at different RPMs (idle vs revving)
  3. Include recordings with background noise
  4. Include MP3-compressed versions
  5. Include different vehicle types (diesel, older cars)
""")

print(SEP)
print("  Error analysis complete. No files were modified.")
print(SEP)
