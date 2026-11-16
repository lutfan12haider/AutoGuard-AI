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
        ? "Warning · Knock"
        : tone === "misfire"
          ? "Fault · Misfire"
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
  if (tone === "healthy") return <CheckCircle2 size={size} strokeWidth={1.5} className="text-[#16A34A]" />;
  if (tone === "knock") return <AlertTriangle size={size} strokeWidth={1.5} className="text-[#D97706]" />;
  if (tone === "uncertain") return <HelpCircle size={size} strokeWidth={1.5} className="text-[#4F46E5]" />;
  return <ShieldAlert size={size} strokeWidth={1.5} className="text-[#DC2626]" />;
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
    healthy: "#16A34A",
    knock: "#D97706",
    misfire: "#DC2626",
    uncertain: "#4F46E5",
    accent: "#4F46E5",
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
          stroke="#F1F5F9"
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
        <span className="font-mono-data text-[28px] leading-none font-semibold text-[#0F172A]">
          {Math.round(clamped)}
          <span className="text-[16px] text-[#64748B] font-normal">%</span>
        </span>
        <span className="text-[12px] font-medium text-[#64748B] mt-1">{label}</span>
        {sublabel && <span className="text-[11px] text-[#94A3B8] mt-0.5">{sublabel}</span>}
      </div>
    </div>
  );
}
