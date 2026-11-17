import { AlertTriangle, CheckCircle2, HelpCircle, ShieldAlert } from "lucide-react";
import { predictionToLabel, predictionToTone } from "../services/api";
import { Badge } from "./ui";

export function StatusBadge({
  prediction,
  shortLabel = false,
}: {
  prediction: string;
  shortLabel?: boolean;
}) {
  const tone = predictionToTone(prediction);

  const badgeTone: Record<typeof tone, "healthy" | "warning" | "fault" | "accent"> = {
    healthy: "healthy",
    knock: "warning",
    misfire: "fault",
    uncertain: "accent",
  };

  const text = shortLabel
    ? tone === "healthy"
      ? "Healthy"
      : tone === "knock"
        ? "Engine Knocking"
        : tone === "misfire"
          ? "Misfire"
          : "Unclear"
    : predictionToLabel(prediction);

  return (
    <Badge tone={badgeTone[tone]} dot>
      {text}
    </Badge>
  );
}

export function ConditionIcon({
  prediction,
  size = 22,
}: {
  prediction: string;
  size?: number;
}) {
  const tone = predictionToTone(prediction);
  if (tone === "healthy") return <CheckCircle2 size={size} strokeWidth={1.5} className="text-[#4ADE80]" />;
  if (tone === "knock") return <AlertTriangle size={size} strokeWidth={1.5} className="text-[#FBBF24]" />;
  if (tone === "uncertain") return <HelpCircle size={size} strokeWidth={1.5} className="text-[#38BDF8]" />;
  return <ShieldAlert size={size} strokeWidth={1.5} className="text-[#F87171]" />;
}

export function ConfidenceGauge({
  valuePct,
  tone = "accent",
  label = "Confidence",
  sublabel,
  size = 152,
}: {
  valuePct: number;
  tone?: "healthy" | "knock" | "misfire" | "uncertain" | "accent";
  label?: string;
  sublabel?: string;
  size?: number;
}) {
  const clamped = Math.max(0, Math.min(100, valuePct));
  const strokeWidth = 10;
  const center = size / 2;
  const radius = center - strokeWidth - 6;
  const circumference = 2 * Math.PI * radius;
  const filledLen = (clamped / 100) * circumference;

  const strokeColor = {
    healthy: "#22C55E",
    knock: "#F59E0B",
    misfire: "#EF4444",
    uncertain: "#38BDF8",
    accent: "#38BDF8",
  }[tone];

  return (
    <div
      className="relative inline-flex flex-col items-center justify-center select-none"
      style={{ width: size, height: size }}
    >
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke="#1E293B"
          strokeWidth={strokeWidth}
        />
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke={strokeColor}
          strokeWidth={strokeWidth}
          strokeDasharray={`${filledLen} ${circumference}`}
          strokeLinecap="round"
          transform={`rotate(-90 ${center} ${center})`}
          style={{ transition: "stroke-dasharray 0.6s cubic-bezier(0.22, 1, 0.36, 1)" }}
        />
      </svg>

      <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
        <span className="font-mono-data text-[28px] leading-none font-semibold text-[#F8FAFC]">
          {Math.round(clamped)}
          <span className="text-[16px] text-[#94A3B8] font-normal">%</span>
        </span>
        <span className="text-[12px] font-medium text-[#94A3B8] mt-1">{label}</span>
        {sublabel && <span className="text-[11px] text-[#64748B] mt-0.5">{sublabel}</span>}
      </div>
    </div>
  );
}
