import { useEffect, useRef, useState } from "react";
import { Pause, Play, Volume2 } from "lucide-react";
import type { AudioAnalysisMetrics, SpectrogramColormap } from "../types/diagnostic";
import { formatDuration } from "../services/api";
import { Card } from "./ui";

function getDarkColormapRgb(val: number, map: SpectrogramColormap): [number, number, number] {
  const t = Math.max(0, Math.min(1, val));

  if (map === "cyan") {
    if (t < 0.4) {
      const k = t / 0.4;
      return [Math.round(9 + k * 5), Math.round(13 + k * 45), Math.round(22 + k * 80)];
    }
    if (t < 0.78) {
      const k = (t - 0.4) / 0.38;
      return [Math.round(14 + k * 42), Math.round(58 + k * 131), Math.round(102 + k * 146)];
    }
    const k = (t - 0.78) / 0.22;
    return [Math.round(56 + k * 160), Math.round(189 + k * 55), Math.round(248 + k * 7)];
  }

  if (map === "viridis") {
    if (t < 0.4) {
      const k = t / 0.4;
      return [Math.round(15 + k * 18), Math.round(23 + k * 75), Math.round(42 + k * 68)];
    }
    if (t < 0.75) {
      const k = (t - 0.4) / 0.35;
      return [Math.round(33 + k * 3), Math.round(98 + k * 99), Math.round(110 - k * 16)];
    }
    const k = (t - 0.75) / 0.25;
    return [Math.round(34 + k * 180), Math.round(197 + k * 40), Math.round(94 - k * 40)];
  }

  // Automotive Dark Thermal
  if (t < 0.32) {
    const k = t / 0.32;
    return [Math.round(9 + k * 22), Math.round(13 + k * 28), Math.round(22 + k * 68)];
  }
  if (t < 0.64) {
    const k = (t - 0.32) / 0.32;
    return [Math.round(31 + k * 25), Math.round(41 + k * 148), Math.round(90 + k * 158)];
  }
  if (t < 0.86) {
    const k = (t - 0.64) / 0.22;
    return [Math.round(56 + k * 189), Math.round(189 - k * 31), Math.round(248 - k * 237)];
  }
  const k = (t - 0.86) / 0.14;
  return [Math.round(245 + k * 10), Math.round(158 + k * 65), Math.round(11 + k * 120)];
}

export function SpectrogramWorkbench({
  file,
  metrics,
  colormap = "thermal",
  seekRequestSec,
  showSpectrogram = true,
  decoding = false,
}: {
  file: File | null;
  metrics: AudioAnalysisMetrics | null;
  colormap?: SpectrogramColormap;
  seekRequestSec?: number | null;
  showSpectrogram?: boolean;
  decoding?: boolean;
}) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const waveCanvasRef = useRef<HTMLCanvasElement>(null);
  const specCanvasRef = useRef<HTMLCanvasElement>(null);

  const [audioUrl, setAudioUrl] = useState("");
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(0.85);

  useEffect(() => {
    if (!file) {
      setAudioUrl("");
      setPlaying(false);
      setCurrentTime(0);
      setDuration(0);
      return;
    }
    const url = URL.createObjectURL(file);
    setAudioUrl(url);
    setPlaying(false);
    setCurrentTime(0);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => {
    if (seekRequestSec == null || !audioRef.current) return;
    const target = Math.max(0, Math.min(seekRequestSec, duration || metrics?.durationSec || 0));
    audioRef.current.currentTime = target;
    setCurrentTime(target);
  }, [seekRequestSec, duration, metrics?.durationSec]);

  const effectiveDuration = duration || metrics?.durationSec || 0;
  const progressPct = effectiveDuration > 0 ? Math.min(100, (currentTime / effectiveDuration) * 100) : 0;

  useEffect(() => {
    const canvas = waveCanvasRef.current;
    if (!canvas || !metrics) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    ctx.strokeStyle = "#E2E8E4";
    ctx.lineWidth = 1;
    for (const frac of [0.25, 0.5, 0.75]) {
      ctx.beginPath();
      ctx.moveTo(0, h * frac);
      ctx.lineTo(w, h * frac);
      ctx.stroke();
    }

    const peaks = metrics.waveformPeaks;
    const barWidth = w / peaks.length;
    const midY = h / 2;

    peaks.forEach((amp, i) => {
      const x = i * barWidth;
      const frac = i / peaks.length;
      const played = frac * 100 <= progressPct;
      const barH = Math.max(3, amp * (h - 16));
      ctx.fillStyle = played ? "#16A34A" : amp > 0.78 ? "#EA580C" : "#94A39D";
      ctx.fillRect(x + 0.5, midY - barH / 2, Math.max(1.5, barWidth - 1.5), barH);
    });

    if (effectiveDuration > 0) {
      const px = (progressPct / 100) * w;
      ctx.strokeStyle = "#17211B";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(px, 0);
      ctx.lineTo(px, h);
      ctx.stroke();
    }
  }, [metrics, progressPct, effectiveDuration]);

  useEffect(() => {
    if (!showSpectrogram) return;
    const canvas = specCanvasRef.current;
    if (!canvas || !metrics || metrics.spectrogram.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const spec = metrics.spectrogram;
    const cols = spec.length;
    const rows = spec[0].length;
    const offscreen = document.createElement("canvas");
    offscreen.width = cols;
    offscreen.height = rows;
    const offCtx = offscreen.getContext("2d");
    if (!offCtx) return;

    const imgData = offCtx.createImageData(cols, rows);
    for (let x = 0; x < cols; x++) {
      for (let y = 0; y < rows; y++) {
        const val = spec[x][rows - 1 - y];
        const [r, g, b] = getDarkColormapRgb(val, colormap);
        const idx = (y * cols + x) * 4;
        imgData.data[idx] = r;
        imgData.data[idx + 1] = g;
        imgData.data[idx + 2] = b;
        imgData.data[idx + 3] = 255;
      }
    }
    offCtx.putImageData(imgData, 0, 0);

    ctx.imageSmoothingEnabled = true;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);

    ctx.strokeStyle = "rgba(148, 163, 184, 0.12)";
    ctx.lineWidth = 1;
    for (const frac of [0.25, 0.5, 0.75]) {
      ctx.beginPath();
      ctx.moveTo(0, canvas.height * frac);
      ctx.lineTo(canvas.width, canvas.height * frac);
      ctx.stroke();
    }

    if (effectiveDuration > 0) {
      const px = (progressPct / 100) * canvas.width;
      ctx.strokeStyle = "#F8FAFC";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(px, 0);
      ctx.lineTo(px, canvas.height);
      ctx.stroke();
    }
  }, [showSpectrogram, metrics, colormap, progressPct, effectiveDuration]);

  const seekByFraction = (fraction: number) => {
    const a = audioRef.current;
    if (!a || effectiveDuration <= 0) return;
    const clamped = Math.max(0, Math.min(1, fraction));
    const t = clamped * effectiveDuration;
    a.currentTime = t;
    setCurrentTime(t);
  };

  const togglePlay = () => {
    const a = audioRef.current;
    if (!a) return;
    if (a.paused) {
      a.volume = volume;
      a.play().catch(() => {});
      setPlaying(true);
    } else {
      a.pause();
      setPlaying(false);
    }
  };

  if (!metrics) {
    return (
      <Card flat className="p-6 text-center text-[13px] text-[#5C6B62]">
        {decoding
          ? "Loading audio preview..."
          : "Audio preview unavailable in browser; click Check engine to validate on server."}
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {audioUrl && (
        <audio
          ref={audioRef}
          src={audioUrl}
          onLoadedMetadata={(e) => setDuration(e.currentTarget.duration || metrics.durationSec)}
          onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
          onEnded={() => setPlaying(false)}
        />
      )}

      <Card flat className="p-4 space-y-3">
        <div className="flex items-center justify-between text-[13px]">
          <span className="font-medium text-[#17211B]">Acoustic waveform playback</span>
          <span className="font-mono-data text-[#5C6B62]">{formatDuration(effectiveDuration)}</span>
        </div>

        <canvas
          ref={waveCanvasRef}
          width={760}
          height={80}
          onClick={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            seekByFraction((e.clientX - rect.left) / rect.width);
          }}
          className="w-full h-[80px] rounded-[8px] bg-[#F8FAF9] border border-[#E2E8E4] cursor-pointer"
        />

        <div className="pt-1 flex items-center gap-3">
          <button
            type="button"
            onClick={togglePlay}
            disabled={!audioUrl}
            className="h-9 w-9 rounded-full bg-[#16A34A] hover:bg-[#15803D] text-white flex items-center justify-center transition-all active:scale-[0.96] disabled:opacity-40"
            aria-label={playing ? "Pause audio" : "Play audio"}
          >
            {playing ? <Pause size={15} strokeWidth={1.5} /> : <Play size={15} strokeWidth={1.5} className="ml-0.5" />}
          </button>

          <span className="font-mono-data text-[13px] text-[#17211B] min-w-[84px]">
            {formatDuration(currentTime)} <span className="text-[#94A39D]">/</span>{" "}
            <span className="text-[#5C6B62]">{formatDuration(effectiveDuration)}</span>
          </span>

          <input
            type="range"
            min={0}
            max={100}
            step={0.2}
            value={progressPct}
            onChange={(e) => seekByFraction(Number(e.target.value) / 100)}
            className="studio-slider flex-1 min-w-[100px]"
            aria-label="Audio playback timeline"
          />

          <div className="hidden sm:flex items-center gap-2 text-[#5C6B62]">
            <Volume2 size={15} strokeWidth={1.5} />
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={volume}
              onChange={(e) => {
                const v = Number(e.target.value);
                setVolume(v);
                if (audioRef.current) audioRef.current.volume = v;
              }}
              className="studio-slider w-16"
              aria-label="Audio volume"
            />
          </div>
        </div>
      </Card>

      {showSpectrogram && (
        <Card flat className="p-4 space-y-3">
          <div className="flex items-center justify-between text-[13px]">
            <span className="font-medium text-[#17211B]">Sound frequency map</span>
            <span className="text-[12px] text-[#5C6B62]">Brighter regions show higher acoustic energy</span>
          </div>

          <div className="flex gap-2.5">
            <div className="flex flex-col justify-between text-[11px] text-[#5C6B62] py-0.5 select-none w-9 text-right shrink-0">
              <span>High</span>
              <span>Mid</span>
              <span>Low</span>
            </div>
            <canvas
              ref={specCanvasRef}
              width={760}
              height={132}
              onClick={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                seekByFraction((e.clientX - rect.left) / rect.width);
              }}
              className="flex-1 w-full h-[132px] rounded-[8px] border border-[#E2E8E4] cursor-pointer"
            />
          </div>
        </Card>
      )}
    </div>
  );
}
