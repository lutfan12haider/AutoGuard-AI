import type { ButtonHTMLAttributes, DragEvent, ReactNode } from "react";
import { useState } from "react";
import { motion } from "framer-motion";
import { Check, ChevronRight } from "lucide-react";

/* ── Button ─────────────────────────────────────────────────────────────────── */
export function Button({
  variant = "primary",
  size = "md",
  className = "",
  children,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost" | "white" | "emeraldOutline";
  size?: "sm" | "md" | "lg";
}) {
  const base =
    "inline-flex items-center justify-center gap-2 font-medium rounded-[8px] transition-all duration-150 ease-out active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#22C55E] focus-visible:ring-offset-2 focus-visible:ring-offset-transparent disabled:opacity-45 disabled:active:scale-100 whitespace-nowrap";

  const variants = {
    primary:
      "bg-[#16A34A] hover:bg-[#15803D] text-white border border-[#15803D] shadow-[0_1px_4px_rgba(22,163,74,0.28)]",
    secondary:
      "bg-white hover:bg-[#F8FAF9] text-[#14221B] border border-[#D4DDD7] shadow-[0_1px_2px_rgba(23,33,27,0.04)]",
    ghost:
      "bg-transparent hover:bg-[#DCFCE7]/60 text-[#5C6B62] hover:text-[#17211B] border border-transparent",
    white:
      "bg-white hover:bg-[#F0FDF4] text-[#072B18] border border-white/80 font-semibold shadow-[0_2px_10px_rgba(0,0,0,0.18)]",
    emeraldOutline:
      "bg-[#0B3B23]/85 hover:bg-[#135232] text-white border border-[#2D6E49] shadow-[0_1px_4px_rgba(0,0,0,0.16)]",
  }[variant];

  const sizes = {
    sm: "h-8 px-3 text-[12px]",
    md: "h-9 px-4 text-[14px]",
    lg: "h-11 px-5 text-[14px]",
  }[size];

  return (
    <button className={`${base} ${variants} ${sizes} ${className}`} {...props}>
      {children}
    </button>
  );
}

/* ── Card ───────────────────────────────────────────────────────────────────── */
export function Card({
  children,
  className = "",
  flat = false,
}: {
  children: ReactNode;
  className?: string;
  flat?: boolean;
}) {
  return (
    <div className={`${flat ? "studio-card-flat" : "studio-card"} ${className}`}>
      {children}
    </div>
  );
}

/* ── Badge ──────────────────────────────────────────────────────────────────── */
export function Badge({
  tone = "neutral",
  dot = false,
  children,
  className = "",
}: {
  tone?: "neutral" | "accent" | "healthy" | "warning" | "fault";
  dot?: boolean;
  children: ReactNode;
  className?: string;
}) {
  const toneStyles = {
    neutral: "bg-[#F1F5F3] text-[#334239] border-[#D4DDD7]",
    accent: "bg-[#DCFCE7] text-[#15803D] border-[#86EFAC]",
    healthy: "bg-[#DCFCE7] text-[#15803D] border-[#86EFAC]",
    warning: "bg-[#FFEDD5] text-[#C2410C] border-[#FED7AA]",
    fault: "bg-[#FEE2E2] text-[#B91C1C] border-[#FECACA]",
  }[tone];

  const dotStyles = {
    neutral: "bg-[#64748B]",
    accent: "bg-[#16A34A]",
    healthy: "bg-[#16A34A]",
    warning: "bg-[#EA580C]",
    fault: "bg-[#DC2626]",
  }[tone];

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[12px] font-medium whitespace-nowrap ${toneStyles} ${className}`}
    >
      {dot && <span className={`h-1.5 w-1.5 rounded-full ${dotStyles}`} />}
      {children}
    </span>
  );
}

/* ── Tabs (Segmented Control) ───────────────────────────────────────────────── */
export function Tabs<T extends string>({
  value,
  onChange,
  items,
  layoutId = "segmented-tab",
  surface = "light",
}: {
  value: T;
  onChange: (val: T) => void;
  items: Array<{ id: T; label: string; icon?: ReactNode }>;
  layoutId?: string;
  surface?: "light" | "emerald";
}) {
  const isEmerald = surface === "emerald";
  return (
    <div
      role="tablist"
      className={`inline-flex items-center p-1 rounded-[8px] border gap-0.5 ${
        isEmerald
          ? "bg-[#093620] border-[#1F5E3D]"
          : "bg-[#F1F6F3] border-[#E2E8E4]"
      }`}
    >
      {items.map((item) => {
        const active = value === item.id;
        return (
          <button
            key={item.id}
            type="button"
            role="tab"
            aria-selected={active}
            onClick={() => onChange(item.id)}
            className={`relative px-3 py-1.5 rounded-[6px] text-[13px] font-medium transition-colors duration-150 inline-flex items-center gap-1.5 ${
              isEmerald
                ? active
                  ? "text-white"
                  : "text-[#CBE2D4] hover:text-white"
                : active
                  ? "text-[#17211B]"
                  : "text-[#5C6B62] hover:text-[#17211B]"
            }`}
          >
            {active && (
              <motion.span
                layoutId={layoutId}
                transition={{ type: "spring", bounce: 0.16, duration: 0.3 }}
                className={
                  isEmerald
                    ? "absolute inset-0 bg-[#16A34A] border border-[#22C55E]/50 rounded-[6px] shadow-[0_1px_6px_rgba(22,163,74,0.35)]"
                    : "absolute inset-0 bg-white border border-[#CBD5CF] rounded-[6px] shadow-[0_1px_2px_rgba(23,33,27,0.06)]"
                }
              />
            )}
            <span className="relative z-10 inline-flex items-center gap-1.5">
              {item.icon}
              {item.label}
            </span>
          </button>
        );
      })}
    </div>
  );
}

/* ── Stat Tile ──────────────────────────────────────────────────────────────── */
export function Stat({
  label,
  value,
  subtext,
  icon,
  trend,
  className = "",
}: {
  label: string;
  value: ReactNode;
  subtext?: string;
  icon?: ReactNode;
  trend?: ReactNode;
  className?: string;
}) {
  return (
    <Card flat className={`p-5 flex flex-col justify-between gap-3 ${className}`}>
      <div className="flex items-center justify-between gap-2">
        <span className="text-[12px] font-medium text-[#5C6B62]">{label}</span>
        {icon && (
          <span className="h-8 w-8 rounded-[8px] bg-[#DCFCE7]/60 border border-[#BBF7D0] flex items-center justify-center text-[#16A34A]">
            {icon}
          </span>
        )}
      </div>
      <div className="flex items-end justify-between gap-3">
        <div>
          <div className="font-mono-data text-[28px] leading-none font-semibold text-[#17211B]">
            {value}
          </div>
          {subtext && <p className="text-[12px] text-[#5C6B62] mt-2">{subtext}</p>}
        </div>
        {trend}
      </div>
    </Card>
  );
}

/* ── Stepper ────────────────────────────────────────────────────────────────── */
export function Stepper({
  steps,
  activeStep,
}: {
  steps: string[];
  activeStep: number; // 0-indexed
}) {
  return (
    <nav aria-label="Diagnostic steps" className="flex items-center flex-wrap gap-2">
      {steps.map((step, idx) => {
        const done = idx < activeStep;
        const current = idx === activeStep;
        return (
          <div key={step} className="flex items-center gap-2">
            <div
              className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-[13px] font-medium border transition-colors ${
                current
                  ? "bg-[#16A34A] text-white border-[#4ADE80]/60 shadow-[0_1px_6px_rgba(22,163,74,0.35)]"
                  : done
                    ? "bg-[#0D4A2B] text-[#86EFAC] border-[#22C55E]/40"
                    : "bg-[#093620] text-[#CBE2D4] border-[#1F5E3D]"
              }`}
            >
              <span
                className={`h-5 w-5 rounded-full text-[11px] font-mono-data font-semibold inline-flex items-center justify-center ${
                  current
                    ? "bg-white/25 text-white"
                    : done
                      ? "bg-[#16A34A] text-white"
                      : "bg-[#0E462A] text-[#A7C7B4]"
                }`}
              >
                {done ? <Check size={12} strokeWidth={2} /> : idx + 1}
              </span>
              <span>{step}</span>
            </div>
            {idx < steps.length - 1 && (
              <ChevronRight size={15} strokeWidth={1.5} className="text-[#5EA87D]" />
            )}
          </div>
        );
      })}
    </nav>
  );
}

/* ── Dropzone ───────────────────────────────────────────────────────────────── */
export function Dropzone({
  dragging,
  onDragOver,
  onDragLeave,
  onDrop,
  onClick,
  title,
  subtitle,
  hint,
}: {
  dragging: boolean;
  onDragOver: (e: DragEvent) => void;
  onDragLeave: () => void;
  onDrop: (e: DragEvent) => void;
  onClick: () => void;
  title: string;
  subtitle: string;
  hint: string;
}) {
  return (
    <div
      role="button"
      tabIndex={0}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      onClick={onClick}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onClick();
        }
      }}
      className={`rounded-[12px] border-2 border-dashed px-6 py-12 text-center transition-all duration-150 ease-out cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#16A34A] ${
        dragging
          ? "border-[#16A34A] bg-[#DCFCE7]/70"
          : "border-[#CBD5CF] bg-[#F8FAF9] hover:border-[#16A34A] hover:bg-[#DCFCE7]/35"
      }`}
    >
      {/* Emerald Automotive Waveform Illustration */}
      <div className="mx-auto mb-5 w-20 h-14 rounded-[12px] bg-white border border-[#E2E8E4] shadow-[0_2px_8px_rgba(23,33,27,0.06)] flex items-center justify-center">
        <svg width="48" height="28" viewBox="0 0 48 28" fill="none" aria-hidden="true">
          {[8, 16, 24, 14, 26, 18, 12, 20, 10].map((h, i) => (
            <rect
              key={i}
              x={3 + i * 5}
              y={(28 - h) / 2}
              width="2.5"
              height={h}
              rx="1.25"
              fill={i === 4 || i === 2 || i === 7 ? "#16A34A" : "#94A39D"}
            />
          ))}
        </svg>
      </div>

      <p className="text-[16px] font-semibold text-[#17211B]">{title}</p>
      <p className="text-[14px] text-[#5C6B62] mt-1">
        {subtitle} <span className="text-[#16A34A] font-semibold">Browse files</span>
      </p>
      <p className="text-[12px] text-[#7A8A80] mt-3 font-mono-data">{hint}</p>
    </div>
  );
}

/* ── Tooltip ────────────────────────────────────────────────────────────────── */
export function Tooltip({
  content,
  children,
}: {
  content: string;
  children: ReactNode;
}) {
  const [show, setShow] = useState(false);
  return (
    <span
      className="relative inline-flex items-center"
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
      onFocus={() => setShow(true)}
      onBlur={() => setShow(false)}
    >
      {children}
      {show && (
        <span
          role="tooltip"
          className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2.5 py-1 rounded-[6px] bg-[#17211B] border border-[#334239] text-white text-[12px] whitespace-nowrap shadow-lg z-30 pointer-events-none"
        >
          {content}
        </span>
      )}
    </span>
  );
}

/* ── Table Primitives ───────────────────────────────────────────────────────── */
export function Table({ children }: { children: ReactNode }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left border-collapse">{children}</table>
    </div>
  );
}

export function TableHeader({ children }: { children: ReactNode }) {
  return (
    <thead>
      <tr className="bg-[#F8FAF9] border-b border-[#E2E8E4] text-[12px] font-medium text-[#5C6B62]">
        {children}
      </tr>
    </thead>
  );
}

export function TableRow({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <tr className={`border-b border-[#E2E8E4] last:border-b-0 hover:bg-[#DCFCE7]/30 transition-colors ${className}`}>
      {children}
    </tr>
  );
}

/* ── Skeleton Loader & Empty State ──────────────────────────────────────────── */
export function Skeleton({ className = "h-4 w-full" }: { className?: string }) {
  return <div className={`animate-pulse rounded-[8px] bg-[#E2E8E4] ${className}`} />;
}

export function EmptyState({
  icon,
  title,
  description,
  action,
}: {
  icon: ReactNode;
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <div className="py-12 px-6 text-center flex flex-col items-center justify-center">
      <div className="h-12 w-12 rounded-full bg-[#DCFCE7] border border-[#86EFAC] flex items-center justify-center text-[#16A34A] mb-3">
        {icon}
      </div>
      <h3 className="text-[16px] font-semibold text-[#17211B]">{title}</h3>
      <p className="text-[14px] text-[#5C6B62] mt-1 max-w-sm">{description}</p>
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}
