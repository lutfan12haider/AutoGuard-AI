"""
ml/v2/track2_audit.py
---------------------
AutoGuard AI V2 — Phase 3 Pre-Training Forensic Investigation & Triage for Track 2
(437 Combined-Fault Recordings + 616 Idle Single-Fault Recordings).

Addresses Phase 3 Requirement 6 rigorously BEFORE any multi-label training:
  1. Decomposes every one of the 437 combined recordings against its matched idle
     constituent sources (`low_oil`, `serpentine_belt`, `power_steering`) via least
     squares (`y_mix ≈ sum_k alpha_k * s_k`), computing:
       - Waveform correlation `r_wave`
       - STFT magnitude spectral correlation `r_spec`
       - Physical energy share `energy_share_k` and Signal-to-Interference Ratio `SIR_dB`
  2. Investigates WHY 172/437 combined clips have `min(r_wave) < 0.25` (loudness imbalance
     between unnormalized takes before linear summation -> `-10` to `-30 dB` masking!).
  3. Assigns objective, transparent triage dispositions to ALL 437 combined clips:
       - RETAINED_CLEAN_TRAIN
       - RETAINED_CLEAN_EVAL (pure val / pure test)
       - FLAGGED_MASKED_FOR_REVIEW (`min(r_wave) < 0.25`, constituent drowned out)
       - EXCLUDED_CROSS_SPLIT_LEAKAGE (constituents span train & val/test)
  4. Evaluates an isolated baseline multi-label classifier trained STRICTLY on Track 2
     `train` (single-idle + `RETAINED_CLEAN_TRAIN`) without touching Track 1.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, hamming_loss, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier

from ml.v2.metrics_utils import REPORTS_P3_DIR, SEED, set_seed

ROOT = Path(__file__).resolve().parent.parent.parent
PREP_DIR = ROOT / "data" / "v2_prepared"
KG_IDLE = ROOT / "data" / "kaggledataset" / "car diagnostics dataset" / "idle state"
TRACK2_CACHE = PREP_DIR / "multilabel_idle"

FAULTS = ("low_oil", "serpentine_belt", "power_steering")


def _num_key(p: Path) -> int:
    d = "".join(ch for ch in p.stem if ch.isdigit())
    return int(d) if d else 0


def run_track2_investigation() -> dict:
    set_seed(SEED)
    with open(PREP_DIR / "manifest_multilabel_idle.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    comb_rows = [r for r in rows if r["is_combined"] == "True"]
    assert len(comb_rows) == 437, f"Expected 437 combined rows, found {len(comb_rows)}"

    # Load native waveforms and STFT magnitudes of single-fault idle pools
    single_waves: dict[str, tuple[list[Path], np.ndarray, np.ndarray, np.ndarray]] = {}
    for fname in FAULTS:
        wavs = sorted((KG_IDLE / fname).glob("*.wav"), key=_num_key)
        mats = []
        specs = []
        for w in wavs:
            y, sr = sf.read(str(w), dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            mats.append(y)
            S = np.abs(librosa.stft(y, n_fft=1024, hop_length=512)).ravel()
            S_n = S / (np.linalg.norm(S) + 1e-8)
            specs.append(S_n.astype(np.float32))
        W = np.asarray(mats, dtype=np.float32)
        Wz = W - W.mean(axis=1, keepdims=True)
        Wn = Wz / (np.linalg.norm(Wz, axis=1, keepdims=True) + 1e-8)
        single_waves[fname] = (wavs, W, Wn, np.asarray(specs, dtype=np.float32))

    triage_rows = []
    masked_by_fault = Counter()
    masked_by_combo = Counter()
    all_sir_audible = []
    all_sir_masked = []
    all_rspec_audible = []
    all_rspec_masked = []
    fit_r2_list = []

    triage_counts = Counter()

    for r in comb_rows:
        y, sr = sf.read(str(ROOT / r["rel_path"]), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        yz = y - y.mean()
        yn = yz / (np.linalg.norm(yz) + 1e-8)
        Sy = np.abs(librosa.stft(y, n_fft=1024, hop_length=512)).ravel()
        Sy_n = Sy / (np.linalg.norm(Sy) + 1e-8)

        active = [fn for fn in FAULTS if r[fn] == "1"]
        matched_cols = []
        src_files = {}
        r_waves = {}
        r_specs = {}

        for fn in active:
            wavs, W, Wn, SpecN = single_waves[fn]
            # Handle rare length variance gracefully by slicing to min length
            min_len = min(W.shape[1], len(yn))
            dots = np.abs(Wn[:, :min_len] @ yn[:min_len])
            bi = int(np.argmax(dots))
            src_files[fn] = wavs[bi].name
            r_waves[fn] = float(dots[bi])
            r_specs[fn] = float(SpecN[bi] @ Sy_n)
            matched_cols.append(W[bi, :min_len])

        # Least-squares linear mixture decomposition: y ≈ sum_k alpha_k * s_k
        L_min = min(len(y), min(len(c) for c in matched_cols))
        A = np.column_stack([c[:L_min] for c in matched_cols]).astype(np.float64)
        alpha, *_ = np.linalg.lstsq(A, y[:L_min].astype(np.float64), rcond=None)
        y_hat = A @ alpha
        ss_res = float(np.sum((y[:L_min] - y_hat) ** 2))
        ss_tot = float(np.sum((y[:L_min] - np.mean(y[:L_min])) ** 2)) + 1e-10
        fit_r2 = max(0.0, 1.0 - ss_res / ss_tot)
        fit_r2_list.append(fit_r2)

        comp_energies = np.array([float(np.sum((alpha[k] * A[:, k]) ** 2)) for k in range(len(active))])
        tot_e = float(np.sum(comp_energies)) + 1e-12
        energy_shares = {active[k]: float(comp_energies[k] / tot_e) for k in range(len(active))}
        sir_dbs = {}
        for k, fn in enumerate(active):
            other_e = max(1e-12, tot_e - comp_energies[k])
            sir_dbs[fn] = float(10.0 * np.log10(max(1e-12, comp_energies[k]) / other_e))

        min_rw = min(r_waves.values())
        min_rs = min(r_specs.values())
        min_eshare = min(energy_shares.values())
        min_sir = min(sir_dbs.values())
        weakest_fault = min(r_waves, key=lambda fn: r_waves[fn])

        is_audible = min_rw >= 0.25
        if is_audible:
            all_sir_audible.append(min_sir)
            all_rspec_audible.append(min_rs)
        else:
            all_sir_masked.append(min_sir)
            all_rspec_masked.append(min_rs)
            masked_by_fault[weakest_fault] += 1
            masked_by_combo[r["source_folder"]] += 1

        # Objective Triage Assignment
        split_val = r["split"]
        if split_val == "excluded_cross_split_leakage":
            disposition = "EXCLUDED_CROSS_SPLIT_LEAKAGE"
        elif not is_audible:
            disposition = "FLAGGED_MASKED_FOR_REVIEW"
        elif split_val == "train":
            disposition = "RETAINED_CLEAN_TRAIN"
        elif split_val in ("validation", "test"):
            disposition = "RETAINED_CLEAN_EVAL"
        else:
            disposition = "FLAGGED_MASKED_FOR_REVIEW"

        triage_counts[disposition] += 1
        triage_rows.append({
            "rel_path": r["rel_path"],
            "source_folder": r["source_folder"],
            "split": split_val,
            "disposition": disposition,
            "n_faults": len(active),
            "min_r_wave": round(min_rw, 4),
            "min_r_spec": round(min_rs, 4),
            "min_energy_share": round(min_eshare, 4),
            "min_sir_db": round(min_sir, 2),
            "ls_fit_r2": round(fit_r2, 4),
            "weakest_fault": weakest_fault,
            "r_low_oil": round(r_waves.get("low_oil", 0.0), 4),
            "r_serpentine_belt": round(r_waves.get("serpentine_belt", 0.0), 4),
            "r_power_steering": round(r_waves.get("power_steering", 0.0), 4),
            "matched_splits": r["matched_splits"],
        })

    # Write per-file triage CSV
    triage_csv = PREP_DIR / "track2_triage_437.csv"
    with open(triage_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(triage_rows[0].keys()))
        writer.writeheader()
        writer.writerows(triage_rows)

    # ── Isolated Multi-Label Baseline (Strict Track 2 Train -> Val/Test) ───────
    d_tr = np.load(TRACK2_CACHE / "train.npz", allow_pickle=False)
    d_va = np.load(TRACK2_CACHE / "validation.npz", allow_pickle=False)
    d_te = np.load(TRACK2_CACHE / "test.npz", allow_pickle=False)

    # Train ONLY on clean single-idle + RETAINED_CLEAN_TRAIN (all_constituents_audible == True)
    tr_clean_mask = (~d_tr["is_combined"]) | d_tr["all_constituents_audible"]
    X_tr = d_tr["X_tab"][tr_clean_mask]
    y_tr = d_tr["y_multihot"][tr_clean_mask]

    clf = OneVsRestClassifier(
        HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            class_weight="balanced",
            early_stopping=False,
            random_state=SEED,
        )
    )
    clf.fit(X_tr, y_tr)

    def _eval_t2(X, y_mh, is_comb, aud_mask):
        pred = clf.predict(X)
        out = {}
        subsets = {
            "all_split_clean": (~is_comb) | aud_mask,
            "single_idle_only": ~is_comb,
            "pure_combined_audible": is_comb & aud_mask,
        }
        for sname, m in subsets.items():
            if int(m.sum()) == 0:
                continue
            yt = y_mh[m]
            yp = pred[m]
            p, r, f1, sup = precision_recall_fscore_support(yt, yp, average=None, zero_division=0)
            out[sname] = {
                "n_clips": int(m.sum()),
                "exact_match_acc": round(float(np.mean(np.all(yt == yp, axis=1))), 4),
                "hamming_loss": round(float(hamming_loss(yt, yp)), 4),
                "macro_f1": round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4),
                "per_fault_f1": {FAULTS[i]: round(float(f1[i]), 4) for i in range(len(FAULTS))},
                "per_fault_recall": {FAULTS[i]: round(float(r[i]), 4) for i in range(len(FAULTS))},
            }
        return out

    t2_baseline = {
        "n_train_total_pure": int(len(d_tr["y_multihot"])),
        "n_train_retained_clean": int(tr_clean_mask.sum()),
        "n_train_dropped_masked": int((~tr_clean_mask).sum()),
        "validation": _eval_t2(d_va["X_tab"], d_va["y_multihot"], d_va["is_combined"], d_va["all_constituents_audible"]),
        "test": _eval_t2(d_te["X_tab"], d_te["y_multihot"], d_te["is_combined"], d_te["all_constituents_audible"]),
    }

    summary = {
        "total_combined_investigated": len(comb_rows),
        "triage_criteria": {
            "RETAINED_CLEAN_TRAIN": "Pure train constituent lineage AND min(r_wave) >= 0.25",
            "RETAINED_CLEAN_EVAL": "Pure validation/test constituent lineage AND min(r_wave) >= 0.25",
            "FLAGGED_MASKED_FOR_REVIEW": "Pure split lineage BUT min(r_wave) < 0.25 (weakest fault masked by amplitude imbalance)",
            "EXCLUDED_CROSS_SPLIT_LEAKAGE": "Constituent clips cross train/validation/test split boundaries",
        },
        "disposition_counts": dict(triage_counts),
        "cross_split_leakage_breakdown": {
            "total": int(sum(1 for r in triage_rows if r["disposition"] == "EXCLUDED_CROSS_SPLIT_LEAKAGE")),
            "audible": int(sum(1 for r in triage_rows if r["disposition"] == "EXCLUDED_CROSS_SPLIT_LEAKAGE" and r["min_r_wave"] >= 0.25)),
            "masked_lt_0_25": int(sum(1 for r in triage_rows if r["disposition"] == "EXCLUDED_CROSS_SPLIT_LEAKAGE" and r["min_r_wave"] < 0.25)),
        },
        "flagged_masked_breakdown": {
            "total_masked_across_all_437": int(len(all_sir_masked)),
            "pure_split_flagged_masked": int(triage_counts["FLAGGED_MASKED_FOR_REVIEW"]),
            "masked_weakest_fault_counts": dict(masked_by_fault),
            "masked_by_combo_folder": dict(masked_by_combo),
        },
        "physical_diagnostics": {
            "mean_ls_mixture_fit_r2": round(float(np.mean(fit_r2_list)), 4),
            "audible_ge_0_25": {
                "n": len(all_sir_audible),
                "mean_min_sir_db": round(float(np.mean(all_sir_audible)), 2),
                "median_min_sir_db": round(float(np.median(all_sir_audible)), 2),
                "mean_min_spec_corr": round(float(np.mean(all_rspec_audible)), 4),
            },
            "masked_lt_0_25": {
                "n": len(all_sir_masked),
                "mean_min_sir_db": round(float(np.mean(all_sir_masked)), 2),
                "median_min_sir_db": round(float(np.median(all_sir_masked)), 2),
                "mean_min_spec_corr": round(float(np.mean(all_rspec_masked)), 4),
            },
        },
        "isolated_track2_baseline": t2_baseline,
        "triage_manifest_csv": str(triage_csv.relative_to(ROOT)).replace("\\", "/"),
    }

    out_json = REPORTS_P3_DIR / "track2_pretraining_audit.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== TRACK 2: 437 COMBINED-FAULT TRIAGE SUMMARY ===", flush=True)
    for k, v in triage_counts.items():
        print(f"  {k:32s}: {v}", flush=True)
    print(f"  Mean OLS mixture fit R^2         : {summary['physical_diagnostics']['mean_ls_mixture_fit_r2']}", flush=True)
    print(f"  Audible (>=0.25) median min SIR  : {summary['physical_diagnostics']['audible_ge_0_25']['median_min_sir_db']} dB", flush=True)
    print(f"  Masked  (< 0.25) median min SIR  : {summary['physical_diagnostics']['masked_lt_0_25']['median_min_sir_db']} dB", flush=True)
    print(f"  Masked weakest fault breakdown   : {dict(masked_by_fault)}", flush=True)
    print(f"  Isolated Track 2 Test Macro-F1   : {t2_baseline['test']}", flush=True)
    return summary


if __name__ == "__main__":
    run_track2_investigation()
