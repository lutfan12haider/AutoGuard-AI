import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  Box,
  FileSearch,
  History,
  LayoutDashboard,
  Moon,
  Search,
  Sun,
  X,
} from "lucide-react";
import type { AppPage, DiagnosticRecord, ThemeMode } from "../types/diagnostic";
import { predictionToLabel } from "../services/api";

export function CommandPalette({
  open,
  onClose,
  onNavigate,
  recentScans,
  onInspectRecord,
  theme,
  onToggleTheme,
}: {
  open: boolean;
  onClose: () => void;
  onNavigate: (p: AppPage) => void;
  recentScans: DiagnosticRecord[];
  onInspectRecord: (rec: DiagnosticRecord) => void;
  theme: ThemeMode;
  onToggleTheme: () => void;
}) {
  const [query, setQuery] = useState("");

  useEffect(() => {
    if (!open) setQuery("");
  }, [open]);

  const actions = useMemo(() => {
    const base = [
      { id: "nav-dash", label: "Overview", hint: "Dashboard", icon: LayoutDashboard, run: () => onNavigate("dashboard") },
      { id: "nav-lab", label: "3D Engine Failure Lab", hint: "Interactive GLB Cutaway", icon: Box, run: () => onNavigate("engine_lab") },
      { id: "nav-diag", label: "Check engine", hint: "Upload or record", icon: Activity, run: () => onNavigate("diagnosis") },
      { id: "nav-res", label: "Latest results", hint: "Repair recommendations", icon: FileSearch, run: () => onNavigate("results") },
      { id: "nav-hist", label: "Scan history", hint: "Past checks", icon: History, run: () => onNavigate("history") },
      {
        id: "toggle-theme",
        label: theme === "dark" ? "Switch to Luxury Light Theme" : "Switch to Luxury Dark Theme",
        hint: "Appearance",
        icon: theme === "dark" ? Sun : Moon,
        run: onToggleTheme,
      },
    ];
    const scanItems = recentScans.slice(0, 8).map((r) => ({
      id: `scan-${r.id}`,
      label: r.filename,
      hint: `${predictionToLabel(r.prediction)} · ${Math.round(r.confidence * 100)}%`,
      icon: Activity,
      run: () => onInspectRecord(r),
    }));

    const all = [...base, ...scanItems];
    const q = query.trim().toLowerCase();
    if (!q) return all;
    return all.filter((a) => a.label.toLowerCase().includes(q) || a.hint.toLowerCase().includes(q));
  }, [query, recentScans, onNavigate, onInspectRecord, theme, onToggleTheme]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 bg-black/60 backdrop-blur-[2px] flex items-start justify-center pt-20 px-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-lg studio-card overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-3 px-4 py-3 border-b border-[var(--border)]">
          <Search size={16} strokeWidth={1.5} className="text-[var(--text-muted)]" />
          <input
            autoFocus
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search pages, 3D Engine Lab, theme, or past recordings..."
            className="flex-1 bg-transparent text-[14px] text-[var(--text)] placeholder-[var(--text-muted)] outline-none"
          />
          <button
            type="button"
            onClick={onClose}
            className="text-[var(--text-muted)] hover:text-[var(--text)] p-1 rounded-[6px]"
            aria-label="Close search"
          >
            <X size={16} strokeWidth={1.5} />
          </button>
        </div>

        <div className="max-h-80 overflow-y-auto divide-y divide-[var(--border)]">
          {actions.length === 0 ? (
            <div className="p-6 text-center text-[13px] text-[var(--text-muted)]">
              No matching pages or recordings.
            </div>
          ) : (
            actions.map((item) => {
              const Icon = item.icon;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => {
                    item.run();
                    onClose();
                  }}
                  className="w-full px-4 py-3 flex items-center gap-3 hover:bg-[var(--surface-subtle)] text-left transition-colors"
                >
                  <Icon size={16} strokeWidth={1.5} className="text-[#38BDF8] shrink-0" />
                  <span className="flex-1 text-[13px] font-medium text-[var(--text)] truncate">
                    {item.label}
                  </span>
                  <span className="text-[12px] text-[var(--text-muted)] shrink-0">{item.hint}</span>
                </button>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}
