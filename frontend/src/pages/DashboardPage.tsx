import { lazy, Suspense, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Box,
  CheckCircle2,
  Cpu,
  Database,
  Play,
  Plus,
  Sparkles,
  WifiOff,
} from "lucide-react";
import type {
  AppPage,
  DiagnosticRecord,
  HealthResponse,
  SampleAudioItem,
  StatsResponse,
} from "../types/diagnostic";
import { formatDuration, formatTimestamp, predictionToLabel } from "../services/api";
import { ConfidenceGauge, StatusBadge } from "../components/StatusBadge";
import { SparklineChart } from "../components/TelemetryCharts";
import { Badge, Button, Card, EmptyState, Skeleton, Stat } from "../components/ui";

const LazyEngineLab3D = lazy(() =>
  import("../components/EngineLab3D").then((m) => ({ default: m.EngineLab3D }))
);

const VEHICLE_HOTSPOTS = [
  {
    id: "bay",
    title: "Acoustic Powertrain Bay",
    x: "34%",
    y: "48%",
    detail: "2.0L DOHC Inline-4 · 1-3-4-2 firing order · Dual CNN + SVM acoustic array",
  },
  {
    id: "ignition",
    title: "Ignition & Fuel Injection",
    x: "26%",
    y: "44%",
    detail: "Coil-on-plug & high-pressure rail — primary inspection zone for Misfire",
  },
  {
    id: "combustion",
    title: "Combustion & Knock Sensor",
    x: "41%",
    y: "54%",
    detail: "Block knock sensor & rotating assembly — primary inspection zone for Engine Knocking",
  },
];

export function DashboardPage({
  stats,
  recentRecords,
  samples,
  loading,
  error,
  onNavigate,
  onInspectRecord,
  onQuickTestSample,
}: {
  stats: StatsResponse | null;
  recentRecords: DiagnosticRecord[];
  health: HealthResponse | null;
  samples: SampleAudioItem[];
  loading: boolean;
  error: string;
  onNavigate: (p: AppPage) => void;
  onInspectRecord: (r: DiagnosticRecord) => void;
  onQuickTestSample: (sample: SampleAudioItem) => void;
}) {
  const [selectedHotspot, setSelectedHotspot] = useState(VEHICLE_HOTSPOTS[0]);

  const total = stats?.total_scans ?? 0;
  const healthy = stats?.healthy_results ?? 0;
  const faults = stats?.detected_faults ?? 0;
  const knock = stats?.knock_results ?? 0;
  const misfire = stats?.misfire_results ?? 0;

  const healthyPct = total > 0 ? Math.round((healthy / total) * 100) : 100;
  const knockPct = total > 0 ? Math.round((knock / total) * 100) : 0;
  const misfirePct = total > 0 ? Math.round((misfire / total) * 100) : 0;

  const latestScan = recentRecords[0] ?? null;
  const confidenceTrend = recentRecords
    .slice(0, 10)
    .reverse()
    .map((r) => Math.round(r.confidence * 100));

  const featuredSamples = [
    samples.find((s) => s.expected_class === "healthy"),
    samples.find((s) => s.expected_class === "engine_knock"),
    samples.find((s) => s.expected_class === "misfire"),
  ].filter((s): s is SampleAudioItem => Boolean(s));

  return (
    <div className="space-y-6">
      {error && (
        <div className="rounded-[12px] border border-[#EF4444]/40 bg-[#EF4444]/10 px-4 py-3 flex items-center gap-3 text-[13px] text-[#F87171]">
          <WifiOff size={16} strokeWidth={1.5} className="shrink-0" />
          <span>Service offline — make sure the local diagnostic server is running.</span>
        </div>
      )}

      {/* ── Luxury Automotive Vehicle Hero Banner (Reference 3 Inspired) ─────── */}
      <Card className="overflow-hidden">
        <div className="grid grid-cols-1 xl:grid-cols-12">
          {/* Left: Luxury Diagnostic Command Copy & Telemetry Indicators */}
          <div className="xl:col-span-5 p-6 lg:p-8 flex flex-col justify-between gap-6 border-b xl:border-b-0 xl:border-r border-[var(--border)]">
            <div className="space-y-4">
              <div className="flex items-center gap-2 flex-wrap">
                <Badge tone="accent" dot>
                  AutoGuard AI · Acoustic Studio
                </Badge>
                {latestScan && <StatusBadge prediction={latestScan.prediction} shortLabel />}
              </div>

              <div>
                <h2 className="text-[28px] sm:text-[32px] font-semibold text-[var(--text)] leading-tight">
                  Acoustic Powertrain & 3D Engine Failure Lab
                </h2>
                <p className="text-[14px] text-[var(--text-muted)] mt-2 leading-relaxed">
                  Analyze engine sound across our three ML categories (<strong>Healthy</strong>,{" "}
                  <strong>Misfire</strong>, <strong>Engine Knocking</strong>) and inspect affected components in the interactive 3D GLB cutaway engine.
                </p>
              </div>

              {/* Selected Vehicle Hotspot Readout */}
              <div className="rounded-[10px] border border-[var(--border)] bg-[var(--surface-subtle)] p-3.5 space-y-1">
                <div className="flex items-center justify-between text-[12px]">
                  <span className="font-semibold text-[#38BDF8] flex items-center gap-1.5">
                    <Cpu size={14} />
                    {selectedHotspot.title}
                  </span>
                  <span className="font-mono-data text-[11px] text-[var(--text-muted)]">
                    ACTIVE ZONE
                  </span>
                </div>
                <p className="text-[12px] text-[var(--text-muted)]">{selectedHotspot.detail}</p>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <Button type="button" size="lg" onClick={() => onNavigate("diagnosis")}>
                <Plus size={15} strokeWidth={1.5} /> Check engine audio
              </Button>
              <Button
                type="button"
                variant="secondary"
                size="lg"
                onClick={() => onNavigate("engine_lab")}
              >
                <Box size={15} strokeWidth={1.5} /> Open 3D Engine Lab
              </Button>
            </div>
          </div>

          {/* Right: Realistic Luxury Sedan Photography with Interactive Telemetry Pins */}
          <div className="xl:col-span-7 relative min-h-[300px] sm:min-h-[360px] bg-[#070A12] overflow-hidden flex items-end">
            <img
              src="/images/hero-luxury-sedan.jpg"
              alt="Luxury performance sedan in acoustic diagnostic bay"
              className="absolute inset-0 w-full h-full object-cover object-center"
            />
            <div className="absolute inset-0 bg-gradient-to-r from-[#070A12]/80 via-transparent to-[#070A12]/70" />
            <div className="absolute inset-0 bg-gradient-to-t from-[#070A12]/90 via-transparent to-transparent" />

            {/* Interactive Telemetry Callout Hotspots on the Vehicle */}
            {VEHICLE_HOTSPOTS.map((spot) => {
              const active = selectedHotspot.id === spot.id;
              return (
                <button
                  key={spot.id}
                  type="button"
                  style={{ left: spot.x, top: spot.y }}
                  onClick={() => setSelectedHotspot(spot)}
                  className="absolute -translate-x-1/2 -translate-y-1/2 group z-10"
                  aria-label={`Inspect ${spot.title}`}
                >
                  <span
                    className={`h-7 w-7 rounded-full flex items-center justify-center border transition-transform ${
                      active
                        ? "bg-[#38BDF8] border-white text-[#070A12] scale-110 shadow-[0_0_20px_rgba(56,189,248,0.8)]"
                        : "bg-[#070A12]/80 border-[#38BDF8] text-[#38BDF8] hover:scale-110"
                    }`}
                  >
                    <span className="h-2 w-2 rounded-full bg-current" />
                  </span>
                </button>
              );
            })}

            {/* Bottom Overlay Telemetry Bar inside Vehicle Bay */}
            <div className="relative z-10 w-full p-4 sm:p-6 flex flex-col sm:flex-row sm:items-end justify-between gap-4">
              <div className="rounded-[10px] bg-[#070A12]/85 border border-[#1E293B] backdrop-blur-md px-4 py-3 space-y-1 max-w-sm">
                <div className="text-[11px] font-mono-data text-[#38BDF8] uppercase">
                  POWERTRAIN TELEMETRY · 2.0L DOHC I4
                </div>
                <div className="text-[14px] font-semibold text-white">
                  {latestScan
                    ? `Latest Scan: ${predictionToLabel(latestScan.prediction)} (${Math.round(
                        latestScan.confidence * 100
                      )}%)`
                    : "Ready for Acoustic Engine Screening"}
                </div>
                <div className="text-[12px] text-[#94A3B8]">
                  Click the glowing pins on the vehicle or open the 3D Engine Lab below.
                </div>
              </div>

              <div className="rounded-[10px] bg-[#070A12]/85 border border-[#1E293B] backdrop-blur-md px-4 py-3 flex items-center gap-3">
                <div>
                  <div className="font-mono-data text-[22px] font-semibold text-[#4ADE80] leading-none">
                    {healthyPct}%
                  </div>
                  <div className="text-[11px] text-[#94A3B8] mt-1">Healthy Rate</div>
                </div>
                <div className="h-8 w-px bg-[#1E293B]" />
                <div>
                  <div className="font-mono-data text-[22px] font-semibold text-white leading-none">
                    {total}
                  </div>
                  <div className="text-[11px] text-[#94A3B8] mt-1">Total Scans</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </Card>

      {/* ── Interactive 3D Engine Cutaway Lab Section on Overview ───────────── */}
      <div className="space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h3 className="text-[18px] font-semibold text-[var(--text)] flex items-center gap-2">
              <Sparkles size={17} className="text-[#38BDF8]" />
              <span>Interactive 3D Engine Failure Lab</span>
            </h3>
            <p className="text-[13px] text-[var(--text-muted)]">
              Rotate, zoom, explode, and click any 3D component. Currently synced to{" "}
              <strong>{latestScan ? predictionToLabel(latestScan.prediction) : "Healthy"}</strong>.
            </p>
          </div>
          <Button
            type="button"
            variant="secondary"
            size="sm"
            onClick={() => onNavigate("engine_lab")}
          >
            Expand full 3D Engine Lab <ArrowRight size={14} strokeWidth={1.5} />
          </Button>
        </div>

        <Suspense fallback={<Skeleton className="h-[460px] w-full rounded-[12px]" />}>
          <LazyEngineLab3D
            condition={latestScan ? latestScan.prediction : "healthy"}
            confidence={latestScan ? latestScan.confidence : 0.85}
            confidenceTier={latestScan?.confidence_tier ?? "high"}
            compact
          />
        </Suspense>
      </div>

      {/* 3 Clean Summary KPIs */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <Stat
          label="Total checks"
          value={loading ? <Skeleton className="h-7 w-16" /> : total.toLocaleString()}
          subtext="Saved recordings checked"
          icon={<Database size={16} strokeWidth={1.5} />}
          trend={<SparklineChart values={confidenceTrend} />}
        />
        <Stat
          label="Healthy results"
          value={loading ? <Skeleton className="h-7 w-16" /> : `${healthyPct}%`}
          subtext={`${healthy} healthy recordings`}
          icon={<CheckCircle2 size={16} strokeWidth={1.5} className="text-[#4ADE80]" />}
        />
        <Stat
          label="Fault patterns found"
          value={loading ? <Skeleton className="h-7 w-16" /> : faults.toLocaleString()}
          subtext={`${knock} knocking · ${misfire} misfire`}
          icon={<AlertTriangle size={16} strokeWidth={1.5} className="text-[#FBBF24]" />}
        />
      </div>

      {/* Distribution + Recent Checks */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <Card className="lg:col-span-5 p-6 flex flex-col justify-between gap-5">
          <div>
            <h3 className="text-[16px] font-semibold text-[var(--text)]">Classification breakdown</h3>
            <p className="text-[13px] text-[var(--text-muted)] mt-0.5">
              Distribution across supported classes: Healthy, Engine Knocking, and Misfire.
            </p>
          </div>

          {loading ? (
            <div className="space-y-3 py-8">
              <Skeleton className="h-24 w-24 rounded-full mx-auto" />
              <Skeleton className="h-4 w-48 mx-auto" />
            </div>
          ) : total === 0 ? (
            <EmptyState
              icon={<Activity size={20} strokeWidth={1.5} />}
              title="No checks yet"
              description="Run your first engine audio check to see your summary."
              action={
                <Button size="sm" onClick={() => onNavigate("diagnosis")}>
                  Check engine
                </Button>
              }
            />
          ) : (
            <div className="flex flex-col sm:flex-row items-center justify-center gap-6 py-2">
              <ConfidenceGauge
                valuePct={healthyPct}
                tone={healthyPct >= 50 ? "healthy" : knock >= misfire ? "knock" : "misfire"}
                label="Healthy"
                sublabel={`${total} checks`}
                size={156}
              />

              <div className="w-full sm:w-48 space-y-3.5">
                {[
                  { label: "Healthy", pct: healthyPct, bar: "bg-[#22C55E]" },
                  { label: "Engine Knocking", pct: knockPct, bar: "bg-[#F59E0B]" },
                  { label: "Misfire", pct: misfirePct, bar: "bg-[#EF4444]" },
                ].map((item) => (
                  <div key={item.label} className="space-y-1">
                    <div className="flex items-center justify-between text-[13px]">
                      <span className="text-[var(--text-secondary)] font-medium">{item.label}</span>
                      <span className="font-mono-data text-[var(--text)] font-semibold">
                        {item.pct}%
                      </span>
                    </div>
                    <div className="h-1.5 rounded-full bg-[var(--border)] overflow-hidden">
                      <div
                        className={`h-full rounded-full ${item.bar}`}
                        style={{ width: `${item.pct}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </Card>

        <Card className="lg:col-span-7 flex flex-col overflow-hidden">
          <div className="px-6 py-4 border-b border-[var(--border)] flex items-center justify-between">
            <div>
              <h3 className="text-[16px] font-semibold text-[var(--text)]">Recent checks</h3>
              <p className="text-[13px] text-[var(--text-muted)] mt-0.5">
                Select any recording to view its 3D engine inspection and repair recommendations.
              </p>
            </div>
            <Button type="button" variant="ghost" size="sm" onClick={() => onNavigate("history")}>
              See all <ArrowRight size={14} strokeWidth={1.5} />
            </Button>
          </div>

          <div className="divide-y divide-[var(--border)] flex-1">
            {loading ? (
              <div className="p-6 space-y-3">
                <Skeleton className="h-10 w-full" />
                <Skeleton className="h-10 w-full" />
                <Skeleton className="h-10 w-full" />
              </div>
            ) : recentRecords.length === 0 ? (
              <EmptyState
                icon={<Database size={20} strokeWidth={1.5} />}
                title="No recent checks"
                description="Your analyzed engine recordings will appear here."
              />
            ) : (
              recentRecords.slice(0, 5).map((row) => {
                const { date, time } = formatTimestamp(row.created_at);
                return (
                  <button
                    key={row.id}
                    type="button"
                    onClick={() => onInspectRecord(row)}
                    className="w-full px-6 py-3.5 flex items-center justify-between gap-4 hover:bg-[var(--surface-subtle)] text-left transition-colors"
                  >
                    <div className="min-w-0 flex-1">
                      <div className="text-[14px] font-medium text-[var(--text)] truncate">
                        {row.filename}
                      </div>
                      <div className="text-[12px] text-[var(--text-muted)] mt-0.5">
                        {date} · {time} · {formatDuration(row.duration_sec)}
                      </div>
                    </div>

                    <StatusBadge prediction={row.prediction} shortLabel />

                    <span className="font-mono-data text-[13px] font-semibold text-[var(--text)] w-11 text-right">
                      {Math.round(row.confidence * 100)}%
                    </span>
                  </button>
                );
              })
            )}
          </div>
        </Card>
      </div>

      {/* Try a Sample Sound */}
      {featuredSamples.length > 0 && (
        <Card className="p-6 space-y-4">
          <div>
            <h3 className="text-[16px] font-semibold text-[var(--text)]">
              Try a sample engine recording
            </h3>
            <p className="text-[13px] text-[var(--text-muted)] mt-0.5">
              Load a verified sample from the dataset (Healthy, Engine Knocking, or Misfire) to test the 3D Engine Failure Lab and Repair Recommendation System.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {featuredSamples.map((s) => (
              <button
                key={s.id}
                type="button"
                onClick={() => onQuickTestSample(s)}
                className="rounded-[10px] border border-[var(--border)] hover:border-[#38BDF8] bg-[var(--surface-subtle)] p-4 text-left transition-all flex items-center justify-between gap-3"
              >
                <div className="min-w-0">
                  <div className="text-[14px] font-semibold text-[var(--text)]">
                    {predictionToLabel(s.expected_class)}
                  </div>
                  <div className="text-[12px] text-[var(--text-muted)] truncate mt-0.5">
                    Sample clip · {s.size_kb} KB
                  </div>
                </div>
                <span className="h-8 w-8 rounded-full bg-[var(--surface)] border border-[var(--border)] flex items-center justify-center text-[#38BDF8] shrink-0">
                  <Play size={13} strokeWidth={1.5} className="ml-0.5" />
                </span>
              </button>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
