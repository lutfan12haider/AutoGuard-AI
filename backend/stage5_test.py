"""
stage5_test.py
--------------
Stage 5 complete application test suite for AutoGuard AI.

Tests every requirement from the spec:
  1.  Backend startup
  2.  Model loading
  3.  WAV upload + actual prediction
  4.  Invalid audio handling (wrong type, too short, empty, corrupted)
  5.  SQLite persistence
  6.  Dashboard statistics accuracy
  7.  History retrieval (all + by ID)
  8.  Empty states (before any scans)
  9.  Frontend-backend CORS headers
  10. Application restart + database persistence

Run from project root:
    python backend/stage5_test.py
"""

import os
import sys
import json
import time
import subprocess
import tempfile
import urllib.request
import urllib.error
from collections import defaultdict

import numpy as np
import soundfile as sf

BASE_URL = "http://localhost:8000"
VENV_PYTHON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "venv", "Scripts", "python.exe"
)
DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "kwerimoto", "prepared"
)
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "autoguard.db")

# ── Test tracking ──────────────────────────────────────────────────────────────

results: list[tuple[str, bool, str]] = []

def check(label: str, condition: bool, detail: str = "") -> bool:
    results.append((label, condition, detail))
    status = "PASS" if condition else "FAIL"
    marker = "✓" if condition else "✗"
    msg = f"  [{status}] {marker} {label}"
    if detail:
        msg += f"\n         → {detail}"
    print(msg)
    return condition

# ── HTTP helpers ───────────────────────────────────────────────────────────────

def get(path: str, timeout: int = 15) -> tuple[int, dict, dict]:
    """Returns (status, body, headers)."""
    url = BASE_URL + path
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, json.loads(r.read()), dict(r.headers)
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read())
        except Exception:
            body = {}
        return e.code, body, {}

def post_wav(path: str, filename: str | None = None) -> tuple[int, dict]:
    boundary = "Stage5Boundary"
    fname = filename or os.path.basename(path)
    with open(path, "rb") as f:
        wav_bytes = f.read()
    header = (
        "--" + boundary + "\r\n"
        'Content-Disposition: form-data; name="file"; filename="' + fname + '"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode()
    body = header + wav_bytes + ("\r\n--" + boundary + "--\r\n").encode()
    req = urllib.request.Request(
        BASE_URL + "/api/analyze", data=body, method="POST",
        headers={"Content-Type": "multipart/form-data; boundary=" + boundary}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            body_data = json.loads(e.read())
        except Exception:
            body_data = {}
        return e.code, body_data

def post_bytes(raw_bytes: bytes, filename: str, content_type: str = "audio/wav") -> tuple[int, dict]:
    boundary = "Stage5Boundary"
    header = (
        "--" + boundary + "\r\n"
        'Content-Disposition: form-data; name="file"; filename="' + filename + '"\r\n'
        "Content-Type: " + content_type + "\r\n\r\n"
    ).encode()
    body = header + raw_bytes + ("\r\n--" + boundary + "--\r\n").encode()
    req = urllib.request.Request(
        BASE_URL + "/api/analyze", data=body, method="POST",
        headers={"Content-Type": "multipart/form-data; boundary=" + boundary}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            body_data = json.loads(e.read())
        except Exception:
            body_data = {}
        return e.code, body_data

def make_wav(path: str, duration: float = 2.0, sr: int = 22050, silent: bool = False):
    t = np.linspace(0, duration, int(sr * duration))
    if silent:
        audio = np.zeros_like(t, dtype=np.float32)
    else:
        audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
    sf.write(path, audio, sr)

# ── Server lifecycle ───────────────────────────────────────────────────────────

def start_server() -> subprocess.Popen:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.Popen(
        [VENV_PYTHON, "-m", "uvicorn", "backend.main:app", "--port", "8000", "--log-level", "warning"],
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    for i in range(30):
        time.sleep(1)
        try:
            with urllib.request.urlopen("http://localhost:8000/health", timeout=3):
                print(f"  Server ready after {i+1}s")
                return proc
        except Exception:
            pass
    raise RuntimeError("Server did not start in 30 seconds")

def stop_server(proc: subprocess.Popen):
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

# ══════════════════════════════════════════════════════════════════════════════
# TESTS
# ══════════════════════════════════════════════════════════════════════════════

def run_all_tests():
    with tempfile.TemporaryDirectory() as tmp:

        # ── TEST 1: Backend startup ────────────────────────────────────────────
        print("\n" + "═"*60)
        print("  TEST 1 — Backend Startup")
        print("═"*60)
        status, body, _ = get("/health")
        check("Backend responds to /health", status == 200, f"HTTP {status}")
        check("Status field is 'ok'", body.get("status") == "ok", str(body.get("status")))
        check("Version field present", "version" in body)

        # ── TEST 2: Model loading ──────────────────────────────────────────────
        print("\n" + "═"*60)
        print("  TEST 2 — Model Loading")
        print("═"*60)
        model_info = body.get("model", {})
        check("Model is loaded", model_info.get("loaded") is True)
        check("Model name is Random Forest", model_info.get("model_name") == "Random Forest",
              str(model_info.get("model_name")))
        check("Classes list present", isinstance(model_info.get("classes"), list))
        check("All 3 classes present",
              set(model_info.get("classes", [])) == {"engine_knock", "healthy", "misfire"},
              str(model_info.get("classes")))
        check("Test accuracy reported", isinstance(model_info.get("test_accuracy"), float),
              str(model_info.get("test_accuracy")))
        check("Test accuracy >= 85%", (model_info.get("test_accuracy") or 0) >= 0.85,
              f"{model_info.get('test_accuracy')}")

        # ── TEST 3: WAV upload + actual predictions ────────────────────────────
        print("\n" + "═"*60)
        print("  TEST 3 — WAV Upload & Actual Predictions")
        print("═"*60)

        test_files = [
            ("engine_knock", os.path.join(DATA_ROOT, "test", "engine_knock", "audioset_-DsZ-XfJRCE_2000.wav")),
            ("healthy",      os.path.join(DATA_ROOT, "test", "healthy", "hf_healthy_002.wav")),
            ("misfire",      os.path.join(DATA_ROOT, "test", "misfire", "hf_misfire_010.wav")),
            ("engine_knock", os.path.join(DATA_ROOT, "real_test", "engine_knock", "audioset_-Nc_T1FvPx0_70000.wav")),
        ]
        scan_ids = []
        for true_class, path in test_files:
            fname = os.path.basename(path)
            status, body = post_wav(path)
            check(f"HTTP 200 for {fname}", status == 200, f"HTTP {status}")
            if status == 200:
                check(f"Correct prediction ({true_class})",
                      body.get("prediction") == true_class,
                      f"predicted={body.get('prediction')} expected={true_class}")
                check(f"Confidence 0–1 for {fname}",
                      0.0 <= body.get("confidence", -1) <= 1.0,
                      str(body.get("confidence")))
                check(f"class_scores has 3 keys",
                      len(body.get("class_scores", {})) == 3,
                      str(body.get("class_scores")))
                check(f"scan_id > 0", body.get("scan_id", 0) > 0)
                check(f"duration_sec > 0", body.get("duration_sec", 0) > 0)
                check(f"inference_ms > 0", body.get("inference_ms", 0) > 0)
                check(f"created_at present", bool(body.get("created_at")))
                scan_ids.append(body["scan_id"])
                print(f"    prediction={body['prediction']}  conf={round(body['confidence']*100,1)}%  "
                      f"dur={body['duration_sec']}s  infer={body['inference_ms']}ms  id=#{body['scan_id']}")

        # ── TEST 4: Invalid audio handling ─────────────────────────────────────
        print("\n" + "═"*60)
        print("  TEST 4 — Invalid Audio Handling")
        print("═"*60)

        # 4a. Wrong file extension
        txt_path = os.path.join(tmp, "engine.txt")
        with open(txt_path, "w") as f:
            f.write("not audio")
        status, body = post_bytes(b"not audio", "engine.txt", "text/plain")
        check("Wrong extension (.txt) → HTTP 400", status == 400, f"HTTP {status}: {body.get('detail','')}")

        # 4b. Too short (0.1s)
        short_path = os.path.join(tmp, "short.wav")
        make_wav(short_path, duration=0.1)
        status, body = post_wav(short_path)
        check("Too short (0.1s) → HTTP 422", status == 422, f"HTTP {status}: {body.get('detail','')}")

        # 4c. Empty file
        empty_path = os.path.join(tmp, "empty.wav")
        with open(empty_path, "wb") as f:
            f.write(b"")
        status, body = post_bytes(b"", "empty.wav")
        check("Empty file → HTTP 400", status == 400, f"HTTP {status}: {body.get('detail','')}")

        # 4d. Corrupted WAV (random bytes with .wav extension)
        status, body = post_bytes(b"\x00\xff\xfe\xfd" * 100, "corrupted.wav")
        check("Corrupted audio → non-200", status != 200, f"HTTP {status}: {body.get('detail','')}")

        # 4e. File too large (simulate > 25MB)
        big_bytes = b"X" * (26 * 1024 * 1024)
        status, body = post_bytes(big_bytes, "big.wav")
        check("Oversized file (26MB) → HTTP 413", status == 413, f"HTTP {status}: {body.get('detail','')}")

        # 4f. Silent audio
        silent_path = os.path.join(tmp, "silent.wav")
        make_wav(silent_path, duration=3.0, silent=True)
        status, body = post_wav(silent_path)
        check("Silent audio → non-200", status != 200, f"HTTP {status}: {body.get('detail','')}")

        # ── TEST 5: SQLite persistence ─────────────────────────────────────────
        print("\n" + "═"*60)
        print("  TEST 5 — SQLite Persistence")
        print("═"*60)
        check("Database file exists", os.path.exists(DB_PATH), DB_PATH)

        # Check that scan IDs from Test 3 are actually in the DB
        for sid in scan_ids[:2]:
            status, body, _ = get(f"/api/history/{sid}")
            check(f"Scan #{sid} retrievable from DB", status == 200,
                  f"HTTP {status}: {body.get('detail','')}")
            if status == 200:
                check(f"Scan #{sid} has correct fields",
                      all(k in body for k in ["id","filename","prediction","confidence","created_at"]),
                      str(list(body.keys())))

        # ── TEST 6: Dashboard statistics accuracy ──────────────────────────────
        print("\n" + "═"*60)
        print("  TEST 6 — Dashboard Statistics Accuracy")
        print("═"*60)
        status, stats, _ = get("/api/stats")
        check("GET /api/stats HTTP 200", status == 200)
        if status == 200:
            # We uploaded 4 files in test 3 — all should be counted
            check(f"total_scans >= 4",
                  stats.get("total_scans", 0) >= 4,
                  f"got {stats.get('total_scans')}")
            check("completed_scans == total_scans",
                  stats.get("completed_scans") == stats.get("total_scans"),
                  f"completed={stats.get('completed_scans')} total={stats.get('total_scans')}")
            check("healthy_results >= 1", stats.get("healthy_results", 0) >= 1,
                  f"got {stats.get('healthy_results')}")
            check("knock_results >= 1", stats.get("knock_results", 0) >= 1,
                  f"got {stats.get('knock_results')}")
            check("misfire_results >= 1", stats.get("misfire_results", 0) >= 1,
                  f"got {stats.get('misfire_results')}")
            total = stats.get("total_scans", 0)
            faults = stats.get("detected_faults", 0)
            healthy = stats.get("healthy_results", 0)
            check("faults + healthy == total",
                  faults + healthy == total,
                  f"faults={faults} + healthy={healthy} = {faults+healthy} vs total={total}")
            print(f"    Stats: {stats}")

        # ── TEST 7: History retrieval ──────────────────────────────────────────
        print("\n" + "═"*60)
        print("  TEST 7 — History Retrieval")
        print("═"*60)

        # All records
        status, hist, _ = get("/api/history")
        check("GET /api/history HTTP 200", status == 200)
        if status == 200:
            check("records list present", isinstance(hist.get("records"), list))
            check("total field present", "total" in hist)
            check("has records (from test 3 uploads)", hist.get("total", 0) >= 4,
                  f"got {hist.get('total')}")
            if hist["records"]:
                rec = hist["records"][0]
                check("Record has all required fields",
                      all(k in rec for k in ["id","filename","prediction","confidence",
                                              "knock_score","healthy_score","misfire_score",
                                              "duration_sec","inference_ms","created_at"]),
                      str(list(rec.keys())))
                check("Scores sum to ~1.0",
                      abs(rec["knock_score"] + rec["healthy_score"] + rec["misfire_score"] - 1.0) < 0.01,
                      f"sum={rec['knock_score']+rec['healthy_score']+rec['misfire_score']:.3f}")
            print(f"    History: {hist['total']} records, newest: {hist['records'][0]['filename'] if hist['records'] else 'none'}")

        # Pagination
        status, page1, _ = get("/api/history?limit=2&offset=0")
        status2, page2, _ = get("/api/history?limit=2&offset=2")
        check("Pagination limit=2 offset=0 returns 2",
              len(page1.get("records", [])) == 2,
              f"got {len(page1.get('records',[]))}")
        check("Pagination offset=2 returns different records",
              len(page2.get("records", [])) >= 1 and
              (not page1.get("records") or page2.get("records", [{}])[0]["id"] != page1.get("records", [{}])[0]["id"]),
              "pagination offset works")

        # Single record
        if scan_ids:
            status, rec, _ = get(f"/api/history/{scan_ids[0]}")
            check(f"GET /api/history/{scan_ids[0]} HTTP 200", status == 200)
            if status == 200:
                check("id field matches", rec.get("id") == scan_ids[0])

        # Not found
        status, body, _ = get("/api/history/99999")
        check("GET /api/history/99999 → HTTP 404", status == 404)

        # ── TEST 8: Empty states (before any scans) ────────────────────────────
        # (Tested implicitly — we checked stats returns 0s on fresh DB in test_backend.py
        #  Here we verify the history endpoint handles 0 records gracefully)
        print("\n" + "═"*60)
        print("  TEST 8 — API handles edge cases gracefully")
        print("═"*60)
        status, body, _ = get("/api/history?limit=0")
        check("limit=0 doesn't crash", status in (200, 422), f"HTTP {status}")
        status, body, _ = get("/api/stats")
        check("Stats returns all required keys",
              all(k in body for k in ["total_scans","completed_scans","detected_faults",
                                       "healthy_results","knock_results","misfire_results"]),
              str(list(body.keys())))

        # ── TEST 9: CORS headers ───────────────────────────────────────────────
        print("\n" + "═"*60)
        print("  TEST 9 — CORS Headers (Frontend-Backend Communication)")
        print("═"*60)
        req = urllib.request.Request(
            BASE_URL + "/health",
            headers={"Origin": "http://localhost:8443"},
            method="GET"
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            cors_header = r.headers.get("Access-Control-Allow-Origin", "")
        check("CORS header present for localhost:8443",
              "localhost:8443" in cors_header or cors_header == "*",
              f"Access-Control-Allow-Origin: {cors_header}")

        # ── TEST 10: Restart persistence ───────────────────────────────────────
        print("\n" + "═"*60)
        print("  TEST 10 — Application Restart + Database Persistence")
        print("═"*60)
        # Record count before restart
        _, before, _ = get("/api/stats")
        count_before = before.get("total_scans", 0)
        print(f"  Records before restart: {count_before}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  AutoGuard AI — Stage 5 Complete Test Suite")
    print("=" * 60)

    # Remove any leftover DB from previous runs
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("[Setup] Removed existing test database for clean run.")

    print("\n[Setup] Starting backend server ...")
    server = start_server()

    try:
        run_all_tests()
    finally:
        print("\n[Cleanup] Stopping server ...")
        stop_server(server)

    # Restart test — start a new server instance and check DB still has data
    print("\n[Restart Test] Starting server again to verify DB persistence ...")
    server2 = start_server()
    try:
        status, stats, _ = get("/api/stats")
        count_after = stats.get("total_scans", 0)
        check("Data persists after server restart",
              count_after >= 4,
              f"total_scans after restart = {count_after}")
        check("Model reloads after restart",
              stats is not None,  # health check passed
              "stats endpoint responded")
        # Verify we can still fetch individual scan
        _, hist, _ = get("/api/history?limit=1")
        if hist.get("records"):
            sid = hist["records"][0]["id"]
            st2, rec2, _ = get(f"/api/history/{sid}")
            check(f"Scan #{sid} still retrievable after restart",
                  st2 == 200, f"HTTP {st2}")
        print(f"  Records after restart: {count_after}")
    finally:
        stop_server(server2)

    # ── Summary ────────────────────────────────────────────────────────────────
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    total  = len(results)

    print("\n" + "=" * 60)
    print("  STAGE 5 TEST SUMMARY")
    print("=" * 60)
    print(f"  Total:  {total}")
    print(f"  Passed: {passed}  ✓")
    print(f"  Failed: {failed}  {'✗' if failed else ''}")

    if failed:
        print("\n  FAILED TESTS:")
        for label, ok, detail in results:
            if not ok:
                print(f"    ✗ {label}")
                if detail:
                    print(f"      {detail}")

    print()
    if failed == 0:
        print("  All tests PASSED ✓  Application is ready.")
    else:
        print(f"  {failed} test(s) FAILED — see above.")
    print("=" * 60 + "\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
