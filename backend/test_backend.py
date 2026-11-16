"""
test_backend.py
---------------
Automated endpoint tests for the AutoGuard AI FastAPI backend.

Starts the server in a subprocess, runs all tests, then shuts it down.

Run from the project root:
    python backend/test_backend.py
"""

import os
import sys
import json
import time
import subprocess
import tempfile
import urllib.request
import urllib.error

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


# ── Helpers ────────────────────────────────────────────────────────────────────

def get(path: str) -> tuple[int, dict]:
    url = BASE_URL + path
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def post_file(path: str, file_path: str) -> tuple[int, dict]:
    import urllib.request
    boundary = "AutoGuardBoundary"
    with open(file_path, "rb") as f:
        file_data = f.read()

    filename = os.path.basename(file_path)
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + file_data + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        BASE_URL + path,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def make_test_wav(path: str, duration: float = 2.0, sr: int = 22050):
    """Create a simple sine-wave WAV file for testing."""
    t = np.linspace(0, duration, int(sr * duration))
    audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
    sf.write(path, audio, sr)


def check(label: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return condition


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_health():
    print("\n[Test] GET /health")
    status, body = get("/health")
    check("HTTP 200", status == 200, f"got {status}")
    check("status=ok", body.get("status") == "ok")
    check("model loaded", body.get("model", {}).get("loaded") is True)
    check("database connected", "connected" in body.get("database", ""))
    return status == 200


def test_analyze_valid(wav_path: str, label: str = "test file"):
    print(f"\n[Test] POST /api/analyze — {label}")
    status, body = post_file("/api/analyze", wav_path)
    check("HTTP 200", status == 200, f"got {status}")
    check("has prediction", "prediction" in body, str(body))
    if "prediction" in body:
        pred = body["prediction"]
        check(f"valid class ({pred})", pred in ("engine_knock", "healthy", "misfire"))
        check("confidence in 0-1", 0.0 <= body.get("confidence", -1) <= 1.0)
        check("has class_scores", isinstance(body.get("class_scores"), dict))
        check("has scan_id", body.get("scan_id", 0) > 0)
        check("has duration_sec", body.get("duration_sec", 0) > 0)
        check("has inference_ms", body.get("inference_ms", 0) > 0)
    return status == 200 and "prediction" in body


def test_analyze_invalid_type(tmp_dir: str):
    print("\n[Test] POST /api/analyze — invalid file type (.txt)")
    txt_path = os.path.join(tmp_dir, "test.txt")
    with open(txt_path, "w") as f:
        f.write("not audio")

    import urllib.request
    boundary = "AutoGuardBoundary"
    with open(txt_path, "rb") as f:
        file_data = f.read()
    body_bytes = (
        f"--{boundary}\r\nContent-Disposition: form-data; "
        f'name="file"; filename="test.txt"\r\nContent-Type: text/plain\r\n\r\n'
    ).encode() + file_data + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        BASE_URL + "/api/analyze",
        data=body_bytes,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            status = r.status
    except urllib.error.HTTPError as e:
        status = e.code
    check("HTTP 400", status == 400, f"got {status}")
    return status == 400


def test_analyze_too_short(tmp_dir: str):
    print("\n[Test] POST /api/analyze — audio too short (0.1s)")
    short_path = os.path.join(tmp_dir, "short.wav")
    make_test_wav(short_path, duration=0.1)
    status, body = post_file("/api/analyze", short_path)
    check("HTTP 422", status == 422, f"got {status} body={body}")
    return status == 422


def test_history_empty_or_populated():
    print("\n[Test] GET /api/history")
    status, body = get("/api/history")
    check("HTTP 200", status == 200, f"got {status}")
    check("has records key", "records" in body)
    check("has total key", "total" in body)
    return status == 200


def test_stats():
    print("\n[Test] GET /api/stats")
    status, body = get("/api/stats")
    check("HTTP 200", status == 200, f"got {status}")
    check("has total_scans", "total_scans" in body)
    check("total_scans >= 0", body.get("total_scans", -1) >= 0)
    check("has detected_faults", "detected_faults" in body)
    return status == 200


def test_history_scan(scan_id: int):
    print(f"\n[Test] GET /api/history/{scan_id}")
    status, body = get(f"/api/history/{scan_id}")
    check("HTTP 200", status == 200, f"got {status}")
    check("id matches", body.get("id") == scan_id)
    return status == 200


def test_history_not_found():
    print("\n[Test] GET /api/history/99999 — not found")
    status, body = get("/api/history/99999")
    check("HTTP 404", status == 404, f"got {status}")
    return status == 404


def test_analyze_real_knock():
    """Test with an actual engine_knock file from the real_test split."""
    knock_dir = os.path.join(DATA_ROOT, "real_test", "engine_knock")
    if not os.path.exists(knock_dir):
        print("\n[Test] POST /api/analyze — real_test knock: SKIP (dir not found)")
        return True
    files = [f for f in os.listdir(knock_dir) if f.endswith(".wav")]
    if not files:
        print("\n[Test] POST /api/analyze — real_test knock: SKIP (no files)")
        return True
    wav_path = os.path.join(knock_dir, files[0])
    return test_analyze_valid(wav_path, label=f"real knock: {files[0]}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  AutoGuard AI — Backend Endpoint Test Suite")
    print("=" * 60)

    # Start server
    print("\n[Setup] Starting FastAPI server ...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    server = subprocess.Popen(
        [VENV_PYTHON, "-m", "uvicorn", "backend.main:app", "--port", "8000", "--log-level", "warning"],
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for it to be ready
    ready = False
    for i in range(30):
        time.sleep(1)
        try:
            with urllib.request.urlopen("http://localhost:8000/health", timeout=3):
                ready = True
                break
        except Exception:
            pass
        print(f"  Waiting ... ({i+1}s)")

    if not ready:
        print("[FAIL] Server did not start in 30 seconds.")
        server.terminate()
        sys.exit(1)

    print("[Setup] Server is ready.\n")
    results = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a valid test WAV file
        test_wav = os.path.join(tmp_dir, "test_engine.wav")
        make_test_wav(test_wav, duration=3.0)

        results.append(test_health())
        results.append(test_history_empty_or_populated())
        results.append(test_stats())
        results.append(test_history_not_found())
        results.append(test_analyze_invalid_type(tmp_dir))
        results.append(test_analyze_too_short(tmp_dir))

        # Valid WAV analyze — do twice to test DB persistence
        r1 = test_analyze_valid(test_wav, label="sine wave (first)")
        r2 = test_analyze_valid(test_wav, label="sine wave (second)")
        results.extend([r1, r2])

        # After two scans, check history has records and stats updated
        print("\n[Test] GET /api/stats — after 2 scans")
        status, body = get("/api/stats")
        check("total_scans >= 2", body.get("total_scans", 0) >= 2, str(body))
        results.append(status == 200)

        # Scan by ID
        if r1:
            # Fetch the most recent record
            _, hist = get("/api/history?limit=1")
            if hist.get("records"):
                scan_id = hist["records"][0]["id"]
                results.append(test_history_scan(scan_id))

        # Real engine_knock file
        results.append(test_analyze_real_knock())

    # Shut down
    print("\n[Cleanup] Stopping server ...")
    server.terminate()
    try:
        server.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server.kill()

    # Summary
    passed = sum(results)
    total  = len(results)
    print("\n" + "=" * 60)
    print(f"  Results: {passed}/{total} tests passed")
    if passed == total:
        print("  All tests PASSED ✓")
    else:
        print("  Some tests FAILED ✗")
    print("=" * 60 + "\n")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
