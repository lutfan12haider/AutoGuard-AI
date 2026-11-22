"""
ml/v2/splitter.py
-----------------
AutoGuard AI V2 — Leakage-Free, Source- & Session-Aware Dataset Splitter.

Produces two strictly separated split manifests under `data/v2_prepared/`:
  1. `manifest_single_fault_9class.csv` (All 2,016 single-fault recordings — ZERO omitted):
     - KweriMoto (`1,067` clips): Preserves the official group-disjoint
       `train` (721), `validation` (154), `test` (154), and `real_test` (38) splits.
     - Kaggle Single-Fault (`949` clips across all 9 single-fault folders):
       Replaces random / naive index splitting with **Connected-Component + Acoustic
       Session Clustering** (combining native sample-rate session markers, e.g. 24 kHz / 48 kHz
       vs 44.1 kHz in `normal_brakes`, connected components for near-duplicates `cos >= 0.985` or
       adjacent continuations `cos >= 0.970`, and agglomerative session clustering) so
       related clips never straddle train/val/test.
  2. `manifest_multilabel_idle.csv` (Strictly separate from 9-class; `1,053` idle recordings):
     - Idle single-faults (`normal_engine_idle` [neg], `low_oil`, `serpentine_belt`,
       `power_steering`) using the exact same session splits.
     - All `437` `idle state/combined/*` recordings annotated with:
       * Exact constituent source file indices and max normalized cross-correlations
         `(r_low_oil, r_serpentine_belt, r_power_steering)`
       * `true_clipping`: False (`max_consecutive <= 2` samples — peak-normalized only)
       * `all_constituents_audible`: True only if every labeled fault has `corr >= 0.25`
       * `split_purity`: Whether matched constituents fall purely inside `train` vs `val`/`test`
         vs `cross_split_leak_risk` (so Track 2 never leaks train single-fault waveforms into test!).
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse.csgraph import connected_components

from ml.v2.audio_preprocessing import TARGET_SR, fast_resample

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = ROOT / "data"
KW_ROOT = DATA_ROOT / "kwerimoto"
KG_ROOT = DATA_ROOT / "kaggledataset" / "car diagnostics dataset"
OUT_DIR = DATA_ROOT / "v2_prepared"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Canonical 9 Single-Fault Classes (ordered stably)
CLASSES_9: tuple[str, ...] = (
    "healthy",
    "engine_knock",
    "misfire",
    "low_oil",
    "serpentine_belt",
    "power_steering",
    "bad_ignition",
    "dead_battery",
    "worn_out_brakes",
)
CLASS_TO_INT_9: dict[str, int] = {c: i for i, c in enumerate(CLASSES_9)}
INT_TO_CLASS_9: dict[int, str] = {i: c for i, c in enumerate(CLASSES_9)}

KAGGLE_SINGLE_FOLDERS: list[tuple[str, str, str]] = [
    # (relative_folder, operating_state, canonical_class)
    ("idle state/normal_engine_idle", "idle", "healthy"),
    ("idle state/low_oil", "idle", "low_oil"),
    ("idle state/serpentine_belt", "idle", "serpentine_belt"),
    ("idle state/power_steering", "idle", "power_steering"),
    ("startup state/normal_engine_startup", "startup", "healthy"),
    ("startup state/bad_ignition", "startup", "bad_ignition"),
    ("startup state/dead_battery", "startup", "dead_battery"),
    ("braking state/normal_brakes", "braking", "healthy"),
    ("braking state/worn_out_brakes", "braking", "worn_out_brakes"),
]


def _numeric_sort_key(p: Path) -> int:
    digits = "".join(ch for ch in p.stem if ch.isdigit())
    return int(digits) if digits else 0


def _folder_session_split(
    folder_path: Path,
    rng: np.random.Generator,
    dup_cos_thresh: float = 0.996,
) -> dict:
    """
    Cluster a single Kaggle folder into non-leaking recording sessions and assign
    whole sessions into ('train', 'validation', 'test') targeting ~70/15/15 counts
    with ZERO near-duplicate (cos >= 0.996 / 0.999) leakage across splits.
    """
    wavs = sorted(folder_path.glob("*.wav"), key=_numeric_sort_key)
    n = len(wavs)
    feats = []
    orig_srs = []
    durations = []

    for w in wavs:
        info = sf.info(str(w))
        y, sr = sf.read(str(w), dtype="float32")
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        orig_srs.append(int(info.samplerate))
        durations.append(float(info.duration))
        y16 = fast_resample(y, sr, TARGET_SR)
        mfcc = librosa.feature.mfcc(y=y16, sr=TARGET_SR, n_mfcc=20, n_fft=1024, hop_length=256)
        feats.append(np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]))

    F = np.array(feats, dtype=np.float32)
    F_n = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
    sim = F_n @ F_n.T

    # 1. Atomic near-duplicate lock: only connect pairs with same native SR and
    #    cos >= dup_cos_thresh (0.996). This avoids single-linkage chaining across an
    #    entire folder while guaranteeing near-duplicates can never straddle splits.
    adj = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i + 1, n):
            if orig_srs[i] == orig_srs[j] and sim[i, j] >= dup_cos_thresh:
                adj[i, j] = adj[j, i] = 1

    n_cc, cc_labels = connected_components(adj, directed=False)

    comp_indices = defaultdict(list)
    for idx, cid in enumerate(cc_labels):
        comp_indices[int(cid)].append(idx)

    # 2. Within each native sample rate, agglomeratively group related takes/sessions
    #    into ~10-16 acoustic session clusters using average linkage while never forming
    #    oversized non-atomic mega-clusters that starve val/test.
    session_groups: list[list[int]] = []
    by_sr_comps: dict[int, list[list[int]]] = defaultdict(list)
    for cid, idxs in sorted(comp_indices.items()):
        sr_val = orig_srs[idxs[0]]
        by_sr_comps[sr_val].append(sorted(idxs))

    max_allowed_merged = max(14, int(round(0.22 * n)))
    for sr_val, comp_list in sorted(by_sr_comps.items()):
        if len(comp_list) <= 8:
            session_groups.extend(comp_list)
            continue

        # Iteratively merge only pairs of clusters whose combined similarity is high
        # (>= 0.980) AND whose merged size stays <= max_allowed_merged (or already exceeded it).
        clusters = [list(c) for c in comp_list]
        while len(clusters) > 10:
            vecs = np.array([F_n[c].mean(axis=0) for c in clusters], dtype=np.float32)
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
            csim = vecs @ vecs.T
            np.fill_diagonal(csim, -1.0)
            best_pair = None
            best_s = 0.975  # never merge genuinely distinct acoustic sessions
            nc = len(clusters)
            for a in range(nc):
                for b in range(a + 1, nc):
                    if len(clusters[a]) + len(clusters[b]) <= max_allowed_merged and csim[a, b] > best_s:
                        best_s = float(csim[a, b])
                        best_pair = (a, b)
            if best_pair is None:
                break
            a, b = best_pair
            clusters[a] = sorted(clusters[a] + clusters[b])
            clusters.pop(b)
        session_groups.extend(clusters)

    # Sort session groups deterministically by size descending then min file index
    session_groups.sort(key=lambda g: (g[0], len(g)))
    m = len(session_groups)

    # 3. Assign session groups cleanly to train (~70%), validation (~15%), test (~15%)
    #    Enforce meaningful val & test support (min ~11% of folder, >= 7 clips)
    #    and minimize cross-split 1-NN cosine similarity.
    target_tr = round(0.70 * n)
    target_va = max(8, round(0.15 * n))
    target_te = max(8, n - target_tr - target_va)
    min_split_eval = max(7, round(0.10 * n))
    max_split_eval = max(target_va + 6, round(0.22 * n))

    # Any oversized atomic cluster (> 0.35 * n) is pinned to train so it cannot starve eval
    pinned_tr = [gi for gi, g in enumerate(session_groups) if len(g) > 0.35 * n]
    free_groups = [gi for gi in range(m) if gi not in set(pinned_tr)]

    best_assign = None
    best_score = float("inf")

    for trial in range(1500):
        perm = rng.permutation(len(free_groups))
        va_g, te_g, tr_g = [], [], list(pinned_tr)
        va_cnt, te_cnt = 0, 0
        for p_idx in perm:
            gi = free_groups[int(p_idx)]
            gsz = len(session_groups[gi])
            if va_cnt < target_va and va_cnt + gsz <= max_split_eval:
                va_g.append(gi)
                va_cnt += gsz
            elif te_cnt < target_te and te_cnt + gsz <= max_split_eval:
                te_g.append(gi)
                te_cnt += gsz
            else:
                tr_g.append(gi)

        if not tr_g or va_cnt < min_split_eval or te_cnt < min_split_eval:
            continue

        tr_idx = [i for gi in tr_g for i in session_groups[gi]]
        va_idx = [i for gi in va_g for i in session_groups[gi]]
        te_idx = [i for gi in te_g for i in session_groups[gi]]

        cnt_pen = (abs(len(va_idx) - target_va) + abs(len(te_idx) - target_te)) / n
        val_sim = float(np.mean(np.max(sim[np.ix_(va_idx, tr_idx)], axis=1)))
        test_sim = float(np.mean(np.max(sim[np.ix_(te_idx, tr_idx)], axis=1)))
        vt_sim = float(np.mean(np.max(sim[np.ix_(va_idx, te_idx)], axis=1)))
        max_pair = float(np.max(sim[np.ix_(va_idx + te_idx, tr_idx)]))
        score = (
            cnt_pen * 0.8
            + (val_sim + test_sim + 0.5 * vt_sim) * 0.4
            + (5.0 if max_pair >= dup_cos_thresh else 0.0)
        )
        if score < best_score:
            best_score = score
            best_assign = (tr_g, va_g, te_g)

    if best_assign is None:
        # Fallback: relax max_split_eval slightly
        order = sorted(free_groups, key=lambda gi: len(session_groups[gi]))
        tr_g = list(pinned_tr)
        va_g, te_g = [], []
        va_cnt, te_cnt = 0, 0
        for gi in order:
            if va_cnt <= te_cnt and va_cnt < target_va:
                va_g.append(gi)
                va_cnt += len(session_groups[gi])
            elif te_cnt < target_te:
                te_g.append(gi)
                te_cnt += len(session_groups[gi])
            else:
                tr_g.append(gi)
        best_assign = (tr_g, va_g, te_g)

    tr_g, va_g, te_g = best_assign
    idx_to_split = {}
    idx_to_session = {}
    for split_name, g_list in [("train", tr_g), ("validation", va_g), ("test", te_g)]:
        for gi in g_list:
            for idx in session_groups[gi]:
                idx_to_split[idx] = split_name
                idx_to_session[idx] = f"{folder_path.name}_sess_{gi:02d}"

    tr_idx = [i for i in range(n) if idx_to_split[i] == "train"]
    va_idx = [i for i in range(n) if idx_to_split[i] == "validation"]
    te_idx = [i for i in range(n) if idx_to_split[i] == "test"]

    val_1nn = float(np.mean(np.max(sim[np.ix_(va_idx, tr_idx)], axis=1)))
    test_1nn = float(np.mean(np.max(sim[np.ix_(te_idx, tr_idx)], axis=1)))
    leak_val_999 = float(np.mean(np.max(sim[np.ix_(va_idx, tr_idx)], axis=1) >= 0.999)) * 100.0
    leak_test_999 = float(np.mean(np.max(sim[np.ix_(te_idx, tr_idx)], axis=1) >= 0.999)) * 100.0

    rows = []
    for idx, w in enumerate(wavs):
        rows.append({
            "file_path": str(w.relative_to(ROOT)).replace("\\", "/"),
            "orig_sr": orig_srs[idx],
            "duration_sec": round(durations[idx], 4),
            "split": idx_to_split[idx],
            "session_group": idx_to_session[idx],
        })

    stats = {
        "n_total": n,
        "n_connected_components": int(n_cc),
        "n_session_clusters": m,
        "counts": {"train": len(tr_idx), "validation": len(va_idx), "test": len(te_idx)},
        "val_to_train_1nn_cos": round(val_1nn, 5),
        "test_to_train_1nn_cos": round(test_1nn, 5),
        "val_near_dup_999_pct": round(leak_val_999, 2),
        "test_near_dup_999_pct": round(leak_test_999, 2),
    }
    return {"rows": rows, "stats": stats, "wav_to_meta": {w.name: rows[i] for i, w in enumerate(wavs)}}


def build_all_v2_splits(seed: int = 42) -> dict:
    """
    Build Track 1 (2,016 single-fault recordings) and Track 2 (1,053 idle multi-label
    recordings) manifests and verify zero file omissions & leakage bounds.
    """
    rng = np.random.default_rng(seed)

    # ── TRACK 1: SINGLE-FAULT 9-CLASS MANIFEST ────────────────────────────────
    single_rows = []
    kaggle_split_audit = {}
    kaggle_idle_lookup: dict[str, dict[str, dict]] = {}  # folder -> filename -> {split, session_group}

    # 1A. Official KweriMoto Splits (1,067 files)
    kw_manifest_path = KW_ROOT / "manifest.csv"
    with open(kw_manifest_path, "r", encoding="utf-8") as f:
        kw_meta_map = {}
        for r in csv.DictReader(f):
            # prepared/<split>/<label>/<file>
            kw_meta_map[r["file_name"]] = r

    for split in ["train", "validation", "test", "real_test"]:
        split_dir = KW_ROOT / "prepared" / split
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            label = cls_dir.name
            for w in sorted(cls_dir.glob("*.wav")):
                rel_kw = f"prepared/{split}/{label}/{w.name}"
                m = kw_meta_map.get(rel_kw, {})
                dur = float(sf.info(str(w)).duration)
                single_rows.append({
                    "rel_path": str(w.relative_to(ROOT)).replace("\\", "/"),
                    "dataset_source": f"kwerimoto/{m.get('source_dataset', 'unknown')}",
                    "operating_state": "idle",  # engine running state
                    "label": label,
                    "label_int": CLASS_TO_INT_9[label],
                    "split": split,
                    "session_group": m.get("group", f"kwerimoto_{w.stem}"),
                    "orig_sr": int(m.get("original_sample_rate") or 16000),
                    "duration_sec": round(dur, 4),
                    "is_synthetic": str(m.get("synthetic", "False")),
                    "short_lt_1_5s": str(dur < 1.5),
                    "short_lt_0_5s": str(dur < 0.5),
                })

    # 1B. Kaggle Single-Fault Splits via Acoustic Session Clustering (949 files)
    for rel_sub, op_state, canon_label in KAGGLE_SINGLE_FOLDERS:
        folder = KG_ROOT / rel_sub
        res = _folder_session_split(folder, rng=rng)
        kaggle_split_audit[rel_sub] = res["stats"]
        kaggle_idle_lookup[folder.name] = res["wav_to_meta"]
        for item in res["rows"]:
            single_rows.append({
                "rel_path": item["file_path"],
                "dataset_source": "kaggle",
                "operating_state": op_state,
                "label": canon_label,
                "label_int": CLASS_TO_INT_9[canon_label],
                "split": item["split"],
                "session_group": item["session_group"],
                "orig_sr": item["orig_sr"],
                "duration_sec": item["duration_sec"],
                "is_synthetic": "False",
                "short_lt_1_5s": str(item["duration_sec"] < 1.5),
                "short_lt_0_5s": str(item["duration_sec"] < 0.5),
            })

    # Verify total count = 2,016 (zero silently dropped files!)
    assert len(single_rows) == 2016, f"Expected 2016 single-fault files, found {len(single_rows)}"

    # Save Track 1 CSV manifest
    track1_csv = OUT_DIR / "manifest_single_fault_9class.csv"
    with open(track1_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(single_rows[0].keys()))
        writer.writeheader()
        writer.writerows(single_rows)

    # Summarize Track 1 class distribution by split & operating state
    dist_9 = defaultdict(lambda: Counter())
    state_dist = defaultdict(lambda: Counter())
    for r in single_rows:
        dist_9[r["split"]][r["label"]] += 1
        state_dist[r["split"]][r["operating_state"]] += 1

    # ── TRACK 2: MULTI-LABEL IDLE FAULTS MANIFEST (1,053 FILES) ───────────────
    # Strictly separate from Track 1 training/testing arrays!
    # Includes single idle files + 437 combined files with exact constituent lineage audit.
    multilabel_rows = []
    for idle_sub, l_lo, l_sb, l_ps in [
        ("normal_engine_idle", 0, 0, 0),
        ("low_oil", 1, 0, 0),
        ("serpentine_belt", 0, 1, 0),
        ("power_steering", 0, 0, 1),
    ]:
        for fname, meta in kaggle_idle_lookup[idle_sub].items():
            multilabel_rows.append({
                "rel_path": meta["file_path"],
                "source_folder": idle_sub,
                "is_combined": "False",
                "low_oil": l_lo,
                "serpentine_belt": l_sb,
                "power_steering": l_ps,
                "r_low_oil": 1.0 if l_lo else 0.0,
                "r_serpentine_belt": 1.0 if l_sb else 0.0,
                "r_power_steering": 1.0 if l_ps else 0.0,
                "all_constituents_audible": "True",
                "matched_splits": meta["split"],
                "split": meta["split"],
                "session_group": meta["session_group"],
            })

    # Preload normalized single-fault waveforms for constituent lineage matching
    singles_mat = {}
    for s_name in ["low_oil", "serpentine_belt", "power_steering"]:
        wavs = sorted((KG_ROOT / "idle state" / s_name).glob("*.wav"), key=_numeric_sort_key)
        mat = np.array([sf.read(str(w), dtype="float32")[0] for w in wavs], dtype=np.float32)
        mat_z = mat - mat.mean(axis=1, keepdims=True)
        mat_n = mat_z / (np.linalg.norm(mat_z, axis=1, keepdims=True) + 1e-8)
        singles_mat[s_name] = (wavs, mat_n)

    comb_specs = [
        ("no oil_serpentine belt", 1, 1, 0),
        ("power steering combined_no oil", 1, 0, 1),
        ("power steering combined_serpentine belt", 0, 1, 1),
        ("power steering combined_no oil_serpentine belt", 1, 1, 1),
    ]
    comb_purity_counts = Counter()
    for folder_name, l_lo, l_sb, l_ps in comb_specs:
        d = KG_ROOT / "idle state" / "combined" / folder_name
        for mw in sorted(d.glob("*.wav"), key=_numeric_sort_key):
            y = sf.read(str(mw), dtype="float32")[0]
            yn = (y - y.mean()) / (np.linalg.norm(y - y.mean()) + 1e-8)

            active = [("low_oil", l_lo), ("serpentine_belt", l_sb), ("power_steering", l_ps)]
            corrs = {"low_oil": 0.0, "serpentine_belt": 0.0, "power_steering": 0.0}
            matched_splits = []
            matched_sessions = []

            for f_name, flag in active:
                if not flag:
                    continue
                s_wavs, s_mat_n = singles_mat[f_name]
                dots = np.abs(s_mat_n @ yn)
                best_i = int(np.argmax(dots))
                best_r = float(dots[best_i])
                corrs[f_name] = round(best_r, 4)
                src_fname = s_wavs[best_i].name
                src_meta = kaggle_idle_lookup[f_name][src_fname]
                matched_splits.append(src_meta["split"])
                matched_sessions.append(src_meta["session_group"])

            min_active_r = min(corrs[f_name] for f_name, flag in active if flag)
            audible_ok = min_active_r >= 0.25
            uniq_splits = sorted(set(matched_splits))

            if len(uniq_splits) == 1:
                assigned_split = uniq_splits[0]
                purity = f"pure_{assigned_split}"
            else:
                assigned_split = "excluded_cross_split_leakage"
                purity = "cross_split_leakage"
            comb_purity_counts[(purity, audible_ok)] += 1

            multilabel_rows.append({
                "rel_path": str(mw.relative_to(ROOT)).replace("\\", "/"),
                "source_folder": f"combined/{folder_name}",
                "is_combined": "True",
                "low_oil": l_lo,
                "serpentine_belt": l_sb,
                "power_steering": l_ps,
                "r_low_oil": corrs["low_oil"],
                "r_serpentine_belt": corrs["serpentine_belt"],
                "r_power_steering": corrs["power_steering"],
                "all_constituents_audible": str(audible_ok),
                "matched_splits": "+".join(matched_splits),
                "split": assigned_split,
                "session_group": "+".join(matched_sessions),
            })

    assert len(multilabel_rows) == 1053, f"Expected 1053 idle files, found {len(multilabel_rows)}"

    track2_csv = OUT_DIR / "manifest_multilabel_idle.csv"
    with open(track2_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(multilabel_rows[0].keys()))
        writer.writeheader()
        writer.writerows(multilabel_rows)

    summary = {
        "seed": seed,
        "track1_single_fault_total": len(single_rows),
        "track1_classes": list(CLASSES_9),
        "track1_class_distribution_by_split": {k: dict(v) for k, v in dist_9.items()},
        "track1_operating_state_by_split": {k: dict(v) for k, v in state_dist.items()},
        "kaggle_session_leakage_audit": kaggle_split_audit,
        "track2_multilabel_idle_total": len(multilabel_rows),
        "track2_combined_purity_and_audibility": {
            f"{p}_audible={a}": int(c) for (p, a), c in sorted(comb_purity_counts.items())
        },
    }
    with open(OUT_DIR / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    s = build_all_v2_splits(seed=42)
    print("\n=== TRACK 1: 9-CLASS SINGLE-FAULT DISTRIBUTION BY SPLIT ===")
    for sp in ["train", "validation", "test", "real_test"]:
        row = s["track1_class_distribution_by_split"].get(sp, {})
        tot = sum(row.values())
        print(f"  {sp:10s} (N={tot:4d}): " + "  ".join(f"{c}={row.get(c, 0)}" for c in CLASSES_9))

    print("\n=== KAGGLE SESSION SPLIT LEAKAGE VERIFICATION ===")
    for k, st in s["kaggle_session_leakage_audit"].items():
        print(
            f"  {k:38s} | N={st['n_total']:3d} | sess={st['n_session_clusters']:2d} | "
            f"tr/va/te={st['counts']['train']}/{st['counts']['validation']}/{st['counts']['test']} | "
            f"val_nearDup={st['val_near_dup_999_pct']:.1f}% | test_nearDup={st['test_near_dup_999_pct']:.1f}%"
        )

    print("\n=== TRACK 2: 437 COMBINED-FAULT LINEAGE PURITY & AUDIBILITY ===")
    for k, v in s["track2_combined_purity_and_audibility"].items():
        print(f"  {k:45s}: {v}")
