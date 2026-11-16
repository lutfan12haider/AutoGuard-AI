"""Debug MP3 against the already-running server on port 8000."""
import os, tempfile, numpy as np, soundfile as sf, urllib.request, urllib.error, json

BASE = "http://localhost:8000"

with tempfile.TemporaryDirectory() as tmp:
    mp3_path = os.path.join(tmp, "engine_test.mp3")
    t = np.linspace(0, 3.0, int(22050 * 3.0))
    audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
    sf.write(mp3_path, audio, 22050, format="MP3")
    print("MP3 size:", os.path.getsize(mp3_path), "bytes")

    with open(mp3_path, "rb") as f:
        data = f.read()
    print("First 4 bytes:", data[:4])

    boundary = "DebugBoundary"
    part1 = b"--" + boundary.encode() + b"\r\n"
    part1 += b'Content-Disposition: form-data; name="file"; filename="engine_test.mp3"\r\n'
    part1 += b"Content-Type: audio/mpeg\r\n\r\n"
    part2 = b"\r\n--" + boundary.encode() + b"--\r\n"
    body = part1 + data + part2

    req = urllib.request.Request(
        BASE + "/api/analyze",
        data=body,
        method="POST",
        headers={"Content-Type": "multipart/form-data; boundary=" + boundary},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            result = json.loads(r.read())
            print("SUCCESS:", result["prediction"], round(result["confidence"]*100,1), "%")
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, ":", e.read().decode())
