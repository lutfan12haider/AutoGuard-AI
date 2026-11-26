import { useEffect, useRef, useState } from "react";
import { AlertCircle, Mic, Square } from "lucide-react";
import { encodeWavFromPcm } from "../utils/audioDsp";
import { Badge, Button } from "./ui";

export function LiveMicRecorder({
  onCaptured,
}: {
  onCaptured: (file: File) => void;
}) {
  const [recording, setRecording] = useState(false);
  const [elapsedSec, setElapsedSec] = useState(0);
  const [levelPct, setLevelPct] = useState(0);
  const [error, setError] = useState("");

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const chunksRef = useRef<Float32Array[]>([]);
  const rafRef = useRef<number | null>(null);
  const timerRef = useRef<number | null>(null);

  const cleanup = () => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    if (timerRef.current) window.clearInterval(timerRef.current);
    processorRef.current?.disconnect();
    analyserRef.current?.disconnect();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    audioCtxRef.current?.close().catch(() => {});
    rafRef.current = null;
    timerRef.current = null;
  };

  useEffect(() => {
    return () => cleanup();
  }, []);

  const drawLive = () => {
    const canvas = canvasRef.current;
    const analyser = analyserRef.current;
    if (!canvas || !analyser) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const timeData = new Float32Array(analyser.fftSize);

    const renderFrame = () => {
      analyser.getFloatTimeDomainData(timeData);

      let sumSq = 0;
      for (let i = 0; i < timeData.length; i++) sumSq += timeData[i] * timeData[i];
      const rms = Math.sqrt(sumSq / timeData.length);
      setLevelPct(Math.min(100, Math.round(rms * 350)));

      const w = canvas.width;
      const h = canvas.height;
      ctx.fillStyle = "#0B140F";
      ctx.fillRect(0, 0, w, h);

      ctx.strokeStyle = "#1C2E24";
      ctx.lineWidth = 1;
      for (const f of [0.25, 0.5, 0.75]) {
        ctx.beginPath();
        ctx.moveTo(0, h * f);
        ctx.lineTo(w, h * f);
        ctx.stroke();
      }

      ctx.strokeStyle = "#D86C2B";
      ctx.lineWidth = 2;
      ctx.beginPath();
      const step = Math.max(1, Math.floor(timeData.length / w));
      for (let x = 0; x < w; x++) {
        const sample = timeData[Math.min(timeData.length - 1, x * step)];
        const y = h * 0.5 - sample * (h * 0.42);
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      rafRef.current = requestAnimationFrame(renderFrame);
    };
    rafRef.current = requestAnimationFrame(renderFrame);
  };

  const startCapture = async () => {
    setError("");
    chunksRef.current = [];
    setElapsedSec(0);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });
      streamRef.current = stream;
      const AudioCtx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const ctx = new AudioCtx();
      audioCtxRef.current = ctx;

      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      analyserRef.current = analyser;

      const processor = ctx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (ev) => {
        const input = ev.inputBuffer.getChannelData(0);
        chunksRef.current.push(new Float32Array(input));
      };

      source.connect(analyser);
      analyser.connect(processor);
      processor.connect(ctx.destination);

      setRecording(true);
      const startTs = Date.now();
      timerRef.current = window.setInterval(() => {
        setElapsedSec((Date.now() - startTs) / 1000);
      }, 100);

      drawLive();
    } catch (err) {
      setError((err as Error).message || "Microphone access is unavailable.");
      cleanup();
    }
  };

  const stopCapture = () => {
    const sampleRate = audioCtxRef.current?.sampleRate || 22050;
    const chunks = chunksRef.current;
    cleanup();
    setRecording(false);

    const totalLen = chunks.reduce((acc, c) => acc + c.length, 0);
    if (totalLen < sampleRate * 0.6) {
      setError("Recording was too short. Try capturing at least 2 seconds of engine audio.");
      return;
    }

    const merged = new Float32Array(totalLen);
    let offset = 0;
    for (const c of chunks) {
      merged.set(c, offset);
      offset += c.length;
    }

    const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    const wavFile = encodeWavFromPcm(merged, sampleRate, `engine-recording-${stamp}.wav`);
    onCaptured(wavFile);
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-2.5">
          <Badge tone={recording ? "fault" : "neutral"} dot>
            {recording ? "Recording" : "Microphone ready"}
          </Badge>
          {recording && (
            <div className="flex items-center gap-1.5">
              <span className="text-[12px] text-[#8FA196]">Level</span>
              <div className="w-20 h-1.5 rounded-full bg-[#1C2E24] overflow-hidden">
                <div
                  className="h-full rounded-full bg-[#D86C2B] transition-all duration-75"
                  style={{ width: `${levelPct}%` }}
                />
              </div>
            </div>
          )}
        </div>

        <span className="font-mono-data text-[13px] font-medium text-[#EAE6DF]">
          {elapsedSec.toFixed(1)}s
        </span>
      </div>

      <canvas
        ref={canvasRef}
        width={680}
        height={132}
        className="w-full h-[132px] rounded-[8px] bg-[#0B140F] border border-[#1C2E24]"
      />

      {error && (
        <div className="rounded-[8px] border border-[#8C3B19] bg-[#2C150E] px-3.5 py-2.5 flex items-center gap-2 text-[13px] text-[#FDBA74]">
          <AlertCircle size={16} strokeWidth={1.5} />
          <span>{error}</span>
        </div>
      )}

      <div className="flex items-center justify-between flex-wrap gap-3">
        <span className="text-[13px] text-[#8FA196]">
          Record 10–15 seconds while the engine is idling.
        </span>

        {!recording ? (
          <Button type="button" onClick={startCapture}>
            <Mic size={15} strokeWidth={1.5} />
            Start recording
          </Button>
        ) : (
          <Button
            type="button"
            onClick={stopCapture}
            className="bg-[#DC2626] hover:bg-[#B91C1C]"
          >
            <Square size={14} strokeWidth={1.5} />
            Stop recording
          </Button>
        )}
      </div>
    </div>
  );
}
