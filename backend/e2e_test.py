"""Quick end-to-end test against the running backend."""
import urllib.request
import json
import os

BASE = "http://localhost:8000"
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "kwerimoto", "prepared")

def post_wav(path, label):
    boundary = "E2EBoundary"
    filename = os.path.basename(path)
    with open(path, "rb") as f:
        wav_bytes = f.read()
    header = (
        "--" + boundary + "\r\n"
        'Content-Disposition: form-data; name="file"; filename="' + filename + '"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode()
    body = header + wav_bytes + ("\r\n--" + boundary + "--\r\n").encode()
    req = urllib.request.Request(
        BASE + "/api/analyze", data=body, method="POST",
        headers={"Content-Type": "multipart/form-data; boundary=" + boundary}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())

# Health
with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
    h = json.loads(r.read())
    print("Health:", h["status"], "| model:", h["model"]["loaded"])

# Upload a healthy file
p = os.path.join(DATA, "test", "healthy", "hf_healthy_002.wav")
res = post_wav(p, "healthy")
print("Healthy scan  -> prediction:", res["prediction"],
      "| conf:", round(res["confidence"]*100,1), "%",
      "| scan_id:", res["scan_id"])

# Upload an engine_knock file
p2 = os.path.join(DATA, "test", "engine_knock", "audioset_-DsZ-XfJRCE_2000.wav")
res2 = post_wav(p2, "engine_knock")
print("Knock scan    -> prediction:", res2["prediction"],
      "| conf:", round(res2["confidence"]*100,1), "%",
      "| scan_id:", res2["scan_id"])

# Upload a misfire file
p3 = os.path.join(DATA, "test", "misfire", "hf_misfire_010.wav")
res3 = post_wav(p3, "misfire")
print("Misfire scan  -> prediction:", res3["prediction"],
      "| conf:", round(res3["confidence"]*100,1), "%",
      "| scan_id:", res3["scan_id"])

# Stats
with urllib.request.urlopen(BASE + "/api/stats", timeout=5) as r:
    s = json.loads(r.read())
    print("Stats:", s)

# History
with urllib.request.urlopen(BASE + "/api/history?limit=3", timeout=5) as r:
    hist = json.loads(r.read())
    print("History total:", hist["total"], "| most recent:", hist["records"][0]["filename"])
