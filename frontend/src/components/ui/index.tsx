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
  variant?: "primary" | "secondary" | "ghost";
  size?: "sm" | "md" | "lg";
}) {
  const base =
    "inline-flex items-center justify-center gap-2 font-medium rounded-[8px] transition-all duration-150 ease-out active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#38BDF8] focus-visible:ring-offset-2 focus-visible:ring-offset-[#090D16] disabled:opacity-45 disabled:active:scale-100 whitespace-nowrap";

  const variants = {
    primary:
      "bg-[#0284C7] hover:bg-[#0369A1] text-white border border-[#38BDF8]/30 shadow-[0_1px_10px_rgba(14,165,233,0.28)]",
    secondary:
      "bg-[#1E293B] hover:bg-[#334155] text-[#F8FAFC] border border-[#334155] shadow-[0_1px_2px_rgba(0,0,0,0.3)]",
    ghost:
      "bg-transparent hover:bg-[#1E293B] text-[#94A3B8] hover:text-[#F8FAFC] border border-transparent",
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
    neutral: "bg-[#1E293B] text-[#CBD5E1] border-[#334155]",
    accent: "bg-[#0EA5E9]/15 text-[#38BDF8] border-[#38BDF8]/35",
    healthy: "bg-[#22C55E]/15 text-[#4ADE80] border-[#22C55E]/35",
    warning: "bg-[#F59E0B]/15 text-[#FBBF24] border-[#F59E0B]/35",
    fault: "bg-[#EF4444]/15 text-[#F87171] border-[#EF4444]/35",
  }[tone];

  const dotStyles = {
    neutral: "bg-[#94A3B8]",
    accent: "bg-[#38BDF8]",
    healthy: "bg-[#22C55E]",
    warning: "bg-[#F59E0B]",
    fault: "bg-[#EF4444]",
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
}: {
  value: T;
  onChange: (val: T) => void;
  items: Array<{ id: T; label: string; icon?: ReactNode }>;
  layoutId?: string;
}) {
  return (
    <div
      role="tablist"
      className="inline-flex items-center p-1 rounded-[8px] bg-[#090D16] border border-[#1E293B] gap-0.5"
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
              active ? "text-[#F8FAFC]" : "text-[#94A3B8] hover:text-[#F8FAFC]"
            }`}
          >
            {active && (
              <motion.span
                layoutId={layoutId}
                transition={{ type: "spring", bounce: 0.16, duration: 0.3 }}
                className="absolute inset-0 bg-[#1E293B] border border-[#334155] rounded-[6px] shadow-[0_1px_3px_rgba(0,0,0,0.4)]"
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
        <span className="text-[12px] font-medium text-[#94A3B8]">{label}</span>
        {icon && (
          <span className="h-8 w-8 rounded-[8px] bg-[#090D16] border border-[#1E293B] flex items-center justify-center text-[#94A3B8]">
            {icon}
          </span>
        )}
      </div>
      <div className="flex items-end justify-between gap-3">
        <div>
          <div className="font-mono-data text-[28px] leading-none font-semibold text-[#F8FAFC]">
            {value}
          </div>
          {subtext && <p className="text-[12px] text-[#94A3B8] mt-2">{subtext}</p>}
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
                  ? "bg-[#0EA5E9]/15 text-[#38BDF8] border-[#38BDF8]/40"
                  : done
                    ? "bg-[#22C55E]/15 text-[#4ADE80] border-[#22C55E]/35"
                    : "bg-[#111827] text-[#94A3B8] border-[#1E293B]"
              }`}
            >
              <span
                className={`h-5 w-5 rounded-full text-[11px] font-mono-data font-semibold inline-flex items-center justify-center ${
                  current
                    ? "bg-[#0284C7] text-white"
                    : done
                      ? "bg-[#16A34A] text-white"
                      : "bg-[#1E293B] text-[#94A3B8]"
                }`}
              >
                {done ? <Check size={12} strokeWidth={2} /> : idx + 1}
              </span>
              <span>{step}</span>
            </div>
            {idx < steps.length - 1 && (
              <ChevronRight size={15} strokeWidth={1.5} className="text-[#475569]" />
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
      className={`rounded-[12px] border border-dashed px-6 py-12 text-center transition-all duration-150 ease-out cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#38BDF8] ${
        dragging
          ? "border-[#38BDF8] bg-[#0EA5E9]/10"
          : "border-[#334155] bg-[#090D16]/70 hover:border-[#38BDF8]/60 hover:bg-[#0EA5E9]/5"
      }`}
    >
      {/* Dark Automotive Waveform Illustration */}
      <div className="mx-auto mb-5 w-20 h-14 rounded-[12px] bg-[#111827] border border-[#1E293B] shadow-[0_2px_10px_rgba(0,0,0,0.4)] flex items-center justify-center">
        <svg width="48" height="28" viewBox="0 0 48 28" fill="none" aria-hidden="true">
          {[8, 16, 24, 14, 26, 18, 12, 20, 10].map((h, i) => (
            <rect
              key={i}
              x={3 + i * 5}
              y={(28 - h) / 2}
              width="2.5"
              height={h}
              rx="1.25"
              fill={i === 4 || i === 2 ? "#38BDF8" : "#475569"}
            />
          ))}
        </svg>
      </div>

      <p className="text-[16px] font-semibold text-[#F8FAFC]">{title}</p>
      <p className="text-[14px] text-[#94A3B8] mt-1">
        {subtitle} <span className="text-[#38BDF8] font-medium">Browse files</span>
      </p>
      <p className="text-[12px] text-[#64748B] mt-3 font-mono-data">{hint}</p>
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
          className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2.5 py-1 rounded-[6px] bg-[#1E293B] border border-[#334155] text-[#F8FAFC] text-[12px] whitespace-nowrap shadow-lg z-30 pointer-events-none"
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
      <tr className="bg-[#090D16] border-b border-[#1E293B] text-[12px] font-medium text-[#94A3B8]">
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
    <tr className={`border-b border-[#1E293B] last:border-b-0 hover:bg-[#1E293B]/45 transition-colors ${className}`}>
      {children}
    </tr>
  );
}

/* ── Skeleton Loader & Empty State ──────────────────────────────────────────── */
export function Skeleton({ className = "h-4 w-full" }: { className?: string }) {
  return <div className={`animate-pulse rounded-[8px] bg-[#1E293B] ${className}`} />;
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
      <div className="h-12 w-12 rounded-full bg-[#090D16] border border-[#1E293B] flex items-center justify-center text-[#94A3B8] mb-3">
        {icon}
      </div>
      <h3 className="text-[16px] font-semibold text-[#F8FAFC]">{title}</h3>
      <p className="text-[14px] text-[#94A3B8] mt-1 max-w-sm">{description}</p>
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}
