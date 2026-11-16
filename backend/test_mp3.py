"""Quick test: create an MP3, send it to the backend, verify prediction."""
import os, sys, subprocess, time, urllib.request, json, tempfile
import numpy as np
import soundfile as sf

VENV_PYTHON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "venv", "Scripts", "python.exe")
BASE = "http://localhost:8000"

def post_audio(path, filename):
    boundary = "Mp3TestBoundary"
    with open(path, "rb") as f:
        data = f.read()
    ct = "audio/mpeg" if filename.endswith(".mp3") else "audio/wav"
    header = (
        "--" + boundary + "\r\n"
        'Content-Disposition: form-data; name="file"; filename="' + filename + '"\r\n'
        "Content-Type: " + ct + "\r\n\r\n"
    ).encode()
    body = header + data + ("\r\n--" + boundary + "--\r\n").encode()
    req = urllib.request.Request(BASE + "/api/analyze", data=body, method="POST",
        headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())

# Start server
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
server = subprocess.Popen(
    [VENV_PYTHON, "-m", "uvicorn", "backend.main:app", "--port", "8000", "--log-level", "warning"],
    cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

for i in range(20):
    time.sleep(1)
    try:
        urllib.request.urlopen(BASE + "/health", timeout=2)
        break
    except: pass

print("=== MP3 Support Test ===\n")

with tempfile.TemporaryDirectory() as tmp:
    # Create a test MP3 using soundfile (sine wave)
    mp3_path = os.path.join(tmp, "engine_test.mp3")
    t = np.linspace(0, 3.0, int(22050 * 3.0))
    audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
    sf.write(mp3_path, audio, 22050, format="MP3")
    print(f"Created test MP3: {os.path.getsize(mp3_path):,} bytes")

    # Test 1: Valid MP3
    status, body = post_audio(mp3_path, "engine_test.mp3")
    ok = status == 200 and "prediction" in body
    print(f"[{'PASS' if ok else 'FAIL'}] Valid MP3 upload: HTTP {status}")
    if status == 200:
        print(f"       prediction={body['prediction']}  conf={round(body['confidence']*100,1)}%  scan_id=#{body['scan_id']}")

    # Test 2: Wrong extension still rejected
    status2, body2 = post_audio(mp3_path, "engine_test.txt")
    ok2 = status2 == 400
    print(f"[{'PASS' if ok2 else 'FAIL'}] .txt extension still rejected: HTTP {status2}")

    # Test 3: WAV still works
    wav_path = os.path.join(tmp, "engine_test.wav")
    sf.write(wav_path, audio, 22050)
    status3, body3 = post_audio(wav_path, "engine_test.wav")
    ok3 = status3 == 200 and "prediction" in body3
    print(f"[{'PASS' if ok3 else 'FAIL'}] WAV still works: HTTP {status3}")

server.terminate()
server.wait(timeout=5)
all_ok = ok and ok2 and ok3
print(f"\n{'All tests passed ✓' if all_ok else 'Some tests failed ✗'}")
sys.exit(0 if all_ok else 1)
