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
    "inline-flex items-center justify-center gap-2 font-medium rounded-[8px] transition-all duration-150 ease-out active:scale-[0.98] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C86B32] focus-visible:ring-offset-2 focus-visible:ring-offset-transparent disabled:opacity-45 disabled:active:scale-100 whitespace-nowrap";

  const variants = {
    primary:
      "bg-gradient-to-b from-[#9E5B36] via-[#7E4424] to-[#5E3017] hover:brightness-110 text-[#FBF3EB] border border-[#C97E53]/70 font-mono-data uppercase tracking-wider shadow-[0_6px_18px_rgba(168,85,40,0.34),inset_0_1px_0_rgba(255,220,190,0.25)]",
    secondary:
      "bg-[#122019] hover:bg-[#182B21] text-[#EAE6DF] border border-[#284033] shadow-[0_1px_3px_rgba(0,0,0,0.28)]",
    ghost:
      "bg-transparent hover:bg-[#15261D] text-[#8FA196] hover:text-[#EAE6DF] border border-transparent",
    white:
      "bg-gradient-to-b from-[#9E5B36] via-[#7E4424] to-[#5E3017] hover:brightness-110 text-[#FBF3EB] border border-[#C97E53]/70 font-mono-data font-semibold uppercase tracking-wider shadow-[0_6px_20px_rgba(168,85,40,0.38),inset_0_1px_0_rgba(255,220,190,0.28)]",
    emeraldOutline:
      "bg-gradient-to-b from-[#9E6E2B] via-[#7D541B] to-[#5C3C10] hover:brightness-110 text-[#FBF5E8] border border-[#CFA055]/70 font-mono-data font-semibold uppercase tracking-wider shadow-[0_6px_20px_rgba(175,120,35,0.34),inset_0_1px_0_rgba(255,235,190,0.28)]",
  }[variant];

  const sizes = {
    sm: "h-8 px-3 text-[11px]",
    md: "h-9 px-4 text-[12px]",
    lg: "h-10 px-5 text-[12px]",
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
    neutral: "bg-[#122019] text-[#B8C5BD] border-[#284033]",
    accent: "bg-[#152A1D] text-[#68B870] border-[#2F5938]",
    healthy: "bg-[#11291B] text-[#4ADE80] border-[#23633B]",
    warning: "bg-[#2A1B0E] text-[#F59E0B] border-[#784919]",
    fault: "bg-[#2C150E] text-[#F97316] border-[#8C3B19]",
  }[tone];

  const dotStyles = {
    neutral: "bg-[#7C8F83]",
    accent: "bg-[#5E9E62]",
    healthy: "bg-[#4ADE80]",
    warning: "bg-[#F59E0B]",
    fault: "bg-[#EA580C]",
  }[tone];

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[11px] font-mono-data uppercase tracking-wider font-semibold whitespace-nowrap ${toneStyles} ${className}`}
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
  surface?: "light" | "emerald";
}) {
  return (
    <div
      role="tablist"
      className="inline-flex items-center p-1 rounded-full border bg-[#0B140F] border-[#384B40] gap-0.5"
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
            className={`relative px-3.5 py-1 rounded-full text-[11px] font-mono-data uppercase tracking-wider font-semibold transition-colors duration-150 inline-flex items-center gap-1.5 ${
              active ? "text-[#F2E9D8]" : "text-[#8FA196] hover:text-[#EAE6DF]"
            }`}
          >
            {active && (
              <motion.span
                layoutId={layoutId}
                transition={{ type: "spring", bounce: 0.16, duration: 0.3 }}
                className="absolute inset-0 bg-[#182A20] border border-[#4D6657] rounded-full shadow-[0_1px_6px_rgba(0,0,0,0.4)]"
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
        <span className="text-[11px] font-mono-data uppercase tracking-wider font-semibold text-[#8FA196]">
          {label}
        </span>
        {icon && (
          <span className="h-8 w-8 rounded-[8px] bg-[#13231A] border border-[#274031] flex items-center justify-center text-[#D86C2B]">
            {icon}
          </span>
        )}
      </div>
      <div className="flex items-end justify-between gap-3">
        <div>
          <div className="font-mono-data text-[28px] leading-none font-bold text-[#EAE6DF]">
            {value}
          </div>
          {subtext && <p className="text-[12px] text-[#7C8F83] mt-2">{subtext}</p>}
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
              className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-[12px] font-mono-data uppercase tracking-wider font-medium border transition-colors ${
                current
                  ? "bg-gradient-to-b from-[#9E5B36] to-[#6E3B1E] text-[#FFF5EC] border-[#C97E53]/70 shadow-[0_2px_10px_rgba(200,107,50,0.3)]"
                  : done
                    ? "bg-[#13261B] text-[#68B870] border-[#2D5936]"
                    : "bg-[#0C1611] text-[#7C8F83] border-[#1E3327]"
              }`}
            >
              <span
                className={`h-5 w-5 rounded-full text-[11px] font-mono-data font-semibold inline-flex items-center justify-center ${
                  current
                    ? "bg-black/25 text-white"
                    : done
                      ? "bg-[#4E7A2E] text-white"
                      : "bg-[#13221A] text-[#7C8F83]"
                }`}
              >
                {done ? <Check size={12} strokeWidth={2} /> : idx + 1}
              </span>
              <span>{step}</span>
            </div>
            {idx < steps.length - 1 && (
              <ChevronRight size={15} strokeWidth={1.5} className="text-[#496153]" />
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
      className={`rounded-[12px] border-2 border-dashed px-6 py-12 text-center transition-all duration-150 ease-out cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C86B32] ${
        dragging
          ? "border-[#C86B32] bg-[#1B1812]"
          : "border-[#273E31] bg-[#0B140F] hover:border-[#C86B32]/80 hover:bg-[#111E17]"
      }`}
    >
      <div className="mx-auto mb-5 w-20 h-14 rounded-[10px] bg-[#122019] border border-[#263C30] shadow-[0_2px_8px_rgba(0,0,0,0.35)] flex items-center justify-center">
        <svg width="48" height="28" viewBox="0 0 48 28" fill="none" aria-hidden="true">
          {[8, 16, 24, 14, 26, 18, 12, 20, 10].map((h, i) => (
            <rect
              key={i}
              x={3 + i * 5}
              y={(28 - h) / 2}
              width="2.5"
              height={h}
              rx="1.25"
              fill={i === 4 || i === 2 || i === 7 ? "#D86C2B" : "#4E6A5A"}
            />
          ))}
        </svg>
      </div>

      <p className="text-[16px] font-semibold text-[#EAE6DF]">{title}</p>
      <p className="text-[14px] text-[#8FA196] mt-1">
        {subtitle} <span className="text-[#D86C2B] font-semibold">Browse files</span>
      </p>
      <p className="text-[12px] text-[#6A7F72] mt-3 font-mono-data">{hint}</p>
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
          className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2.5 py-1 rounded-[6px] bg-[#0B140F] border border-[#2B4336] text-[#EAE6DF] text-[12px] whitespace-nowrap shadow-lg z-30 pointer-events-none"
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
      <tr className="bg-[#111E17] border-b border-[#1C2E24] text-[11px] font-mono-data uppercase tracking-wider font-semibold text-[#8FA196]">
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
    <tr className={`border-b border-[#1C2E24] last:border-b-0 hover:bg-[#13221A] transition-colors ${className}`}>
      {children}
    </tr>
  );
}

/* ── Skeleton Loader & Empty State ──────────────────────────────────────────── */
export function Skeleton({ className = "h-4 w-full" }: { className?: string }) {
  return <div className={`animate-pulse rounded-[8px] bg-[#16271E] ${className}`} />;
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
      <div className="h-12 w-12 rounded-full bg-[#13231A] border border-[#294234] flex items-center justify-center text-[#D86C2B] mb-3">
        {icon}
      </div>
      <p className="text-[14px] font-semibold text-[#EAE6DF]">{title}</p>
      <p className="text-[13px] text-[#7C8F83] mt-1 max-w-sm">{description}</p>
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}
