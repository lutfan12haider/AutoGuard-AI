import { lazy, Suspense, useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Database,
  Pause,
  Play,
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
import { Button, Card, EmptyState, Skeleton, Stat } from "../components/ui";

const LazyVehicleDiagnosticScene3D = lazy(() =>
  import("../components/VehicleDiagnosticScene3D").then((m) => ({
    default: m.VehicleDiagnosticScene3D,
  }))
);

const LazyEngineLab3D = lazy(() =>
  import("../components/EngineLab3D").then((m) => ({ default: m.EngineLab3D }))
);

const VEHICLE_HOTSPOTS = [
  {
    id: "bay",
    title: "Acoustic Powertrain Bay",
    shortSpec: "2.0L DOHC Inline-4",
    x: "41%",
    y: "51%",
    detail: "2.0L DOHC Inline-4 · 1-3-4-2 firing order · Dual CNN + SVM acoustic array",
  },
  {
    id: "ignition",
    title: "Ignition & Fuel Injection",
    shortSpec: "COP & Direct Rail",
    x: "30%",
    y: "45%",
    detail: "Coil-on-plug & high-pressure rail · primary inspection zone for Misfire",
  },
  {
    id: "combustion",
    title: "Combustion & Knock Sensor",
    shortSpec: "Block Knock Array",
    x: "46%",
    y: "56%",
    detail: "Block knock sensor & rotating assembly · primary inspection zone for Engine Knocking",
  },
];

const FIRING_CYLINDERS = [
  { cyl: "CYL 1", start: 0, end: 180 },
  { cyl: "CYL 3", start: 180, end: 360 },
  { cyl: "CYL 4", start: 360, end: 540 },
  { cyl: "CYL 2", start: 540, end: 720 },
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
  const [bayView, setBayView] = useState<"vehicle" | "cutaway">("vehicle");

  // 4D Spatial-Temporal State (3D Perspective Tilt + 4th Dimension Time/Crank-Phase Wave Propagation)
  const [tilt, setTilt] = useState<{ rx: number; ry: number }>({ rx: 0, ry: 0 });
  const [temporalPlaying, setTemporalPlaying] = useState(true);
  const [crankPhaseDeg, setCrankPhaseDeg] = useState(210);

  useEffect(() => {
    if (!temporalPlaying) return;
    let rafId = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = (now - last) / 1000;
      last = now;
      setCrankPhaseDeg((prev) => (prev + dt * 115) % 720);
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [temporalPlaying]);

  const total = stats?.total_scans ?? 0;
  const healthy = stats?.healthy_results ?? 0;
  const faults = stats?.detected_faults ?? 0;
  const knock = stats?.knock_results ?? 0;
  const misfire = stats?.misfire_results ?? 0;

  const healthyPct = total > 0 ? Math.round((healthy / total) * 100) : 100;
  const knockPct = total > 0 ? Math.round((knock / total) * 100) : 0;
  const misfirePct = total > 0 ? Math.round((misfire / total) * 100) : 0;

  const latestScan = recentRecords[0] ?? null;
  const latestLabel = latestScan
    ? predictionToLabel(latestScan.prediction).toUpperCase()
    : "ENGINE KNOCKING";
  const latestScorePct = latestScan ? Math.round(latestScan.confidence * 100) : 60;

  const confidenceTrend = recentRecords
    .slice(0, 10)
    .reverse()
    .map((r) => Math.round(r.confidence * 100));

  const featuredSamples = [
    samples.find((s) => s.expected_class === "healthy"),
    samples.find((s) => s.expected_class === "engine_knock"),
    samples.find((s) => s.expected_class === "misfire"),
  ].filter((s): s is SampleAudioItem => Boolean(s));

  // Compute live 4D wave expansion from crankPhaseDeg (0..720)
  const waveProgress = (crankPhaseDeg % 180) / 180; // 0..1 per cylinder firing stroke
  const pulseRadius1 = 26 + waveProgress * 46;
  const pulseRadius2 = 26 + ((waveProgress + 0.5) % 1) * 46;
  const activeFiring =
    FIRING_CYLINDERS.find((c) => crankPhaseDeg >= c.start && crankPhaseDeg < c.end) ??
    FIRING_CYLINDERS[0];
  const temporalSec = ((crankPhaseDeg / 720) * 2.4).toFixed(2);

  return (
    <div className="space-y-6">
      {error && (
        <div className="rounded-[10px] border border-[#8C3B19] bg-[#2C150E] px-4 py-3 flex items-center gap-3 text-[13px] text-[#FDBA74]">
          <WifiOff size={16} strokeWidth={1.5} className="shrink-0" />
          <span>Service offline. Make sure the local diagnostic server is running on port 8000.</span>
        </div>
      )}

      {/* ── 4D Tactical Hero Stage (Unobstructed Heading + Interactive 4D Hologram) ── */}
      <section className="relative rounded-[10px] overflow-hidden border border-[#192B20] bg-[#08110C] shadow-[0_24px_60px_rgba(0,0,0,0.68)]">
        <div
          className="pointer-events-none absolute inset-0"
          style={{
            background:
              "radial-gradient(circle at 72% 46%, rgba(216, 122, 43, 0.14) 0%, rgba(18, 45, 30, 0.22) 42%, rgba(8, 17, 12, 0.98) 78%)",
          }}
        />

        <div className="relative px-6 sm:px-8 pt-6 pb-4 grid grid-cols-1 lg:grid-cols-12 items-center gap-4">
          {/* Left Column (z-30): Unobstructed Poster Heading, Copy, Spec Pill & CTAs */}
          <div className="lg:col-span-5 relative z-30 flex flex-col justify-between space-y-4">
            <div className="select-none">
              <div className="font-display-hero text-[58px] sm:text-[74px] lg:text-[76px] xl:text-[88px] leading-[0.88] uppercase text-[#548033] tracking-[0.015em] whitespace-nowrap">
                AUTOGUARD AI
              </div>
              <div className="font-display-hero text-[58px] sm:text-[74px] lg:text-[76px] xl:text-[88px] leading-[0.88] uppercase text-[#F2E8D5] tracking-[0.015em] whitespace-nowrap mt-1">
                DIAGNOSTIC LAB
              </div>
            </div>

            <p className="text-[13.5px] sm:text-[14px] text-[#C8D2CC] leading-[1.55] max-w-[430px]">
              Classify engine sound across Healthy, Engine Knocking, and Misfire signatures, then
              inspect affected fault zones and acoustic wave propagation inside a 2023 Toyota
              Corolla and cutaway powertrain.
            </p>

            <div>
              <div
                title={selectedHotspot.detail}
                className="inline-flex items-center gap-2 rounded-full border border-[#495C50] bg-[#0B1510]/90 px-4 py-1.5 text-[11px] font-mono-data text-[#DCE5DF] tracking-wider"
              >
                <span className="uppercase font-semibold">{selectedHotspot.title}</span>
                <span className="text-[#D86C2B]">•</span>
                <span>{selectedHotspot.shortSpec}</span>
              </div>
            </div>

            <div className="flex flex-wrap sm:flex-nowrap items-center gap-2.5 pt-1">
              <button
                type="button"
                onClick={() => onNavigate("diagnosis")}
                className="inline-flex items-center gap-1.5 rounded-[8px] px-3.5 py-2.5 text-[11px] font-mono-data font-semibold uppercase tracking-wider text-[#FBF3EB] bg-gradient-to-b from-[#9E5B36] via-[#7E4424] to-[#5E3017] border border-[#C97E53]/75 shadow-[0_6px_22px_rgba(168,85,40,0.4),inset_0_1px_0_rgba(255,220,190,0.3)] hover:brightness-110 active:scale-[0.98] transition-all whitespace-nowrap"
              >
                <span>+ CHECK ENGINE AUDIO</span>
                <ArrowRight size={12} strokeWidth={2} />
              </button>

              <button
                type="button"
                onClick={() => onNavigate("engine_lab")}
                className="inline-flex items-center gap-1.5 rounded-[8px] px-3.5 py-2.5 text-[11px] font-mono-data font-semibold uppercase tracking-wider text-[#FBF5E8] bg-gradient-to-b from-[#9E6E2B] via-[#7D541B] to-[#5C3C10] border border-[#CFA055]/75 shadow-[0_6px_22px_rgba(175,120,35,0.35),inset_0_1px_0_rgba(255,235,190,0.3)] hover:brightness-110 active:scale-[0.98] transition-all whitespace-nowrap"
              >
                <span>OPEN ENGINE LAB</span>
                <ArrowRight size={12} strokeWidth={2} />
              </button>
            </div>
          </div>

          {/* Right Column (z-20): Interactive 4D Spatial-Temporal Cutaway Corolla Stage */}
          <div
            className="lg:col-span-7 relative z-20 flex flex-col justify-between"
            onMouseMove={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              const nx = (e.clientX - rect.left) / rect.width - 0.5;
              const ny = (e.clientY - rect.top) / rect.height - 0.5;
              setTilt({ rx: Number((-ny * 7).toFixed(2)), ry: Number((nx * 9).toFixed(2)) });
            }}
            onMouseLeave={() => setTilt({ rx: 0, ry: 0 })}
          >
            <div
              className="relative w-full aspect-[16/9] max-h-[380px] transition-transform duration-150 ease-out"
              style={{
                transform: `perspective(1100px) rotateX(${tilt.rx}deg) rotateY(${tilt.ry}deg)`,
                transformStyle: "preserve-3d",
              }}
            >
              {/* SVG Silhouette-Masked Car + Live 4D Temporal Acoustic Wave Propagation */}
              <svg
                viewBox="0 0 800 450"
                className="w-full h-full select-none"
                aria-label="4D spatial-temporal Toyota Corolla cutaway diagnostic hologram"
              >
                <defs>
                  <filter id="silhouetteFeather" x="-10%" y="-10%" width="120%" height="120%">
                    <feGaussianBlur stdDeviation="11" />
                  </filter>
                  <mask id="corollaCutawayMask">
                    <rect width="800" height="450" fill="black" />
                    <g filter="url(#silhouetteFeather)" fill="white">
                      <ellipse cx="400" cy="312" rx="348" ry="102" />
                      <polygon points="170,295 176,214 265,176 376,112 528,112 630,170 640,268 580,315" />
                    </g>
                  </mask>

                  <radialGradient id="engineCoreGlow" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stopColor="#F97316" stopOpacity="0.62" />
                    <stop offset="55%" stopColor="#EA580C" stopOpacity="0.22" />
                    <stop offset="100%" stopColor="#EA580C" stopOpacity="0" />
                  </radialGradient>
                </defs>

                <image
                  href="/images/corolla-cutaway-hero.jpg"
                  width="800"
                  height="450"
                  preserveAspectRatio="xMidYMid slice"
                  mask="url(#corollaCutawayMask)"
                />

                {/* 4D Dynamic Temporal Shockwave Propagation Rings (radiating with crankPhaseDeg) */}
                <circle cx="328" cy="232" r="82" fill="url(#engineCoreGlow)" />
                <circle
                  cx="328"
                  cy="232"
                  r={pulseRadius1}
                  fill="none"
                  stroke="#F59E0B"
                  strokeWidth="1.8"
                  strokeOpacity={1 - waveProgress * 0.75}
                />
                <circle
                  cx="328"
                  cy="232"
                  r={pulseRadius2}
                  fill="none"
                  stroke="#EA580C"
                  strokeWidth="1.5"
                  strokeDasharray="8 5"
                  strokeOpacity={0.75}
                />
                <circle
                  cx="328"
                  cy="232"
                  r="68"
                  fill="none"
                  stroke="#EA580C"
                  strokeWidth="1.2"
                  strokeOpacity="0.45"
                  strokeDasharray="12 8"
                />

                {/* Live 4D Harmonic Pressure Bars along the Front Acoustic Turntable Arc */}
                {Array.from({ length: 18 }).map((_, i) => {
                  const bx = 220 + i * 16;
                  const by = 362 + Math.sin(i * 0.35) * 8;
                  const phaseRad = (crankPhaseDeg * Math.PI) / 180;
                  const barH =
                    8 +
                    Math.abs(Math.sin(phaseRad * 2 + i * 0.55)) * 18 +
                    (i % 3 === 0 ? 6 : 0);
                  return (
                    <line
                      key={i}
                      x1={bx}
                      y1={by}
                      x2={bx}
                      y2={by - barH}
                      stroke={i % 4 === 0 ? "#F97316" : "#F59E0B"}
                      strokeWidth="2.2"
                      strokeOpacity="0.82"
                      strokeLinecap="round"
                    />
                  );
                })}

                {/* Angled HUD leader line from engine core to callout badge */}
                <polyline
                  points="328,232 372,142 406,142"
                  fill="none"
                  stroke="#F59E0B"
                  strokeWidth="1.8"
                  strokeOpacity="0.95"
                />
                <circle cx="328" cy="232" r="4.5" fill="#FDBA74" />
              </svg>

              {/* Callout Pin on Hood */}
              <button
                type="button"
                onClick={() => setSelectedHotspot(VEHICLE_HOTSPOTS[1])}
                style={{ left: "29.5%", top: "45%" }}
                className="pointer-events-auto absolute -translate-x-1/2 -translate-y-1/2 flex flex-col items-center z-30 group"
                title={VEHICLE_HOTSPOTS[1].title}
              >
                <span className="h-5 w-5 rounded-full border border-[#FDBA74] bg-[#EA580C]/90 flex items-center justify-center shadow-[0_0_12px_#EA580C] group-hover:scale-110 transition-transform">
                  <span className="h-2 w-2 rounded-full bg-white" />
                </span>
              </button>

              {/* Interactive Engine Bay Core Hotspot Pin */}
              <button
                type="button"
                style={{ left: VEHICLE_HOTSPOTS[0].x, top: VEHICLE_HOTSPOTS[0].y }}
                onClick={() => setSelectedHotspot(VEHICLE_HOTSPOTS[0])}
                className="pointer-events-auto absolute -translate-x-1/2 -translate-y-1/2 group z-30"
                aria-label={`Inspect ${VEHICLE_HOTSPOTS[0].title}`}
                title={VEHICLE_HOTSPOTS[0].title}
              >
                <span className="h-6 w-6 rounded-full flex items-center justify-center border bg-[#EA580C] border-[#FDE68A] text-white scale-110 shadow-[0_0_16px_rgba(234,88,12,0.95)]">
                  <span className="h-2 w-2 rounded-full bg-current" />
                </span>
              </button>

              {/* Floating Copper-Orange Diagnostic Callout Badge (Fully Inside Container) */}
              <div
                style={{ left: "49%", top: "29%" }}
                className="pointer-events-auto absolute z-30 -translate-y-1/2"
              >
                <div className="rounded-[5px] bg-gradient-to-r from-[#C85A22] to-[#983E12] border border-[#F59E0B]/80 px-3 py-1 text-[11px] font-mono-data font-bold uppercase tracking-wider text-[#FFF7ED] shadow-[0_0_24px_rgba(234,88,12,0.65)] whitespace-nowrap">
                  {latestLabel} ({latestScorePct}%)
                </div>
              </div>
            </div>

            {/* Temporal Phase & Crank-Cycle Scrubber Bar under the Vehicle */}
            <div className="mt-1 flex flex-wrap items-center justify-between gap-3 px-2">
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setTemporalPlaying((p) => !p)}
                  className="inline-flex items-center gap-1.5 rounded-full border border-[#3B4E43] bg-[#0B1510] hover:border-[#D86C2B] px-2.5 py-1 text-[10px] font-mono-data font-semibold uppercase tracking-wider text-[#F2E8D5] transition-colors"
                  aria-label={temporalPlaying ? "Pause temporal wavefield" : "Play temporal wavefield"}
                >
                  {temporalPlaying ? <Pause size={10} /> : <Play size={10} />}
                  <span>CYCLE t={temporalSec}s</span>
                </button>

                <input
                  type="range"
                  min={0}
                  max={719}
                  step={1}
                  value={Math.round(crankPhaseDeg)}
                  onChange={(e) => {
                    setTemporalPlaying(false);
                    setCrankPhaseDeg(Number(e.target.value));
                  }}
                  className="studio-slider w-24 sm:w-28"
                  aria-label="Crank phase angle"
                />

                <span className="font-mono-data text-[10px] text-[#D86C2B] font-semibold">
                  {Math.round(crankPhaseDeg)}° · {activeFiring.cyl}
                </span>
              </div>

              <span className="font-mono-data text-[10.5px] font-semibold uppercase tracking-[0.12em] text-[#C8D3CD]">
                POWERTRAIN TELEMETRY · 2.0L DOHC I4
              </span>
            </div>
          </div>
        </div>

        {/* ── Hero Bottom Single-Row Telemetry & View Switcher Bar ────────────── */}
        <div className="relative z-30 border-t border-[#1C2E24] bg-[#07100B]/95 px-5 sm:px-7 py-3 flex flex-wrap lg:flex-nowrap items-center justify-between gap-3">
          {/* Left: Latest Scan Readout */}
          <div className="font-['Oswald',_'Inter',_sans-serif] text-[14px] sm:text-[15px] font-semibold uppercase tracking-[0.05em] text-[#EAE6DF] whitespace-nowrap">
            LATEST SCAN:{" "}
            <span className="text-[#F2E8D5]">
              {latestLabel} ({latestScorePct}%)
            </span>
          </div>

          {/* Center: Healthy Rate & Total Scans */}
          <div className="flex items-center gap-5 shrink-0">
            <div>
              <div className="font-mono-data text-[22px] sm:text-[24px] font-bold text-[#D86C2B] leading-none">
                {healthyPct}%
              </div>
              <div className="font-mono-data text-[9.5px] font-semibold uppercase tracking-[0.1em] text-[#8FA196] mt-1 whitespace-nowrap">
                HEALTHY RATE
              </div>
            </div>

            <div className="h-7 w-px bg-[#23382C]" />

            <div>
              <div className="font-mono-data text-[22px] sm:text-[24px] font-bold text-[#F2EFE9] leading-none">
                {total}
              </div>
              <div className="font-mono-data text-[9.5px] font-semibold uppercase tracking-[0.1em] text-[#8FA196] mt-1 whitespace-nowrap">
                TOTAL SCANS
              </div>
            </div>
          </div>

          {/* Right: Segmented Corolla Bay / Cutaway Engine + Expand Full 4D Lab Pill */}
          <div className="flex items-center gap-2 shrink-0">
            <div
              role="tablist"
              aria-label="4D bay view mode"
              className="inline-flex items-center rounded-full border border-[#3B4E43] bg-[#0B1510] p-0.5"
            >
              <button
                type="button"
                role="tab"
                aria-selected={bayView === "vehicle"}
                onClick={() => setBayView("vehicle")}
                className={`px-3 py-1 rounded-full text-[10px] font-mono-data font-semibold uppercase tracking-wider transition-all whitespace-nowrap ${
                  bayView === "vehicle"
                    ? "bg-[#192B21] text-[#F2E8D5] border border-[#4D6557] shadow-xs"
                    : "text-[#8FA196] hover:text-[#EAE6DF]"
                }`}
              >
                COROLLA BAY
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={bayView === "cutaway"}
                onClick={() => setBayView("cutaway")}
                className={`px-3 py-1 rounded-full text-[10px] font-mono-data font-semibold uppercase tracking-wider transition-all whitespace-nowrap ${
                  bayView === "cutaway"
                    ? "bg-[#192B21] text-[#F2E8D5] border border-[#4D6557] shadow-xs"
                    : "text-[#8FA196] hover:text-[#EAE6DF]"
                }`}
              >
                CUTAWAY ENGINE
              </button>
            </div>

            <button
              type="button"
              onClick={() => onNavigate("engine_lab")}
              className="inline-flex items-center gap-1.5 rounded-full border border-[#3B4E43] bg-[#0B1510] hover:border-[#C87D52] hover:text-[#F2E8D5] px-3.5 py-1.5 text-[10px] font-mono-data font-semibold uppercase tracking-wider text-[#D5DFD9] transition-colors whitespace-nowrap"
            >
              <span>EXPAND FULL ENGINE LAB</span>
              <ArrowRight size={11} strokeWidth={2} />
            </button>
          </div>
        </div>
      </section>

      {/* ── Live Interactive WebGL Vehicle & Engine Bay Viewport ─────────── */}
      <div className="space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h3 className="text-[16px] font-mono-data uppercase tracking-wider font-semibold text-[#EAE6DF] flex items-center gap-2">
              <Sparkles size={16} className="text-[#D86C2B]" />
              <span>Interactive Vehicle &amp; Engine Diagnostic Bay</span>
            </h3>
            <p className="text-[13px] text-[#8FA196]">
              Live WebGL 2023 Toyota Corolla bay and 34-node time-animated powertrain synced to{" "}
              <strong className="text-[#F2E8D5]">
                {latestScan ? predictionToLabel(latestScan.prediction) : "Healthy"}
              </strong>.
            </p>
          </div>
        </div>

        <Suspense fallback={<Skeleton className="h-[440px] w-full rounded-[10px]" />}>
          {bayView === "vehicle" ? (
            <LazyVehicleDiagnosticScene3D
              condition={latestScan ? latestScan.prediction : "healthy"}
              confidence={latestScan ? latestScan.confidence : 0.85}
              confidenceTier={latestScan?.confidence_tier ?? "high"}
              sequenceKey={latestScan?.id ?? "overview"}
              autoPlay={false}
              compact
              onInspectSubsystem={() => setBayView("cutaway")}
            />
          ) : (
            <LazyEngineLab3D
              condition={latestScan ? latestScan.prediction : "healthy"}
              confidence={latestScan ? latestScan.confidence : 0.85}
              confidenceTier={latestScan?.confidence_tier ?? "high"}
              compact
            />
          )}
        </Suspense>
      </div>

      {/* 3 Summary KPIs */}
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
          icon={<CheckCircle2 size={16} strokeWidth={1.5} className="text-[#5E9E62]" />}
        />
        <Stat
          label="Fault patterns found"
          value={loading ? <Skeleton className="h-7 w-16" /> : faults.toLocaleString()}
          subtext={`${knock} knocking · ${misfire} misfire`}
          icon={<AlertTriangle size={16} strokeWidth={1.5} className="text-[#D86C2B]" />}
        />
      </div>

      {/* Distribution + Recent Checks */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <Card className="lg:col-span-5 p-6 flex flex-col justify-between gap-5">
          <div>
            <h3 className="text-[15px] font-mono-data uppercase tracking-wider font-semibold text-[var(--text)]">
              Classification breakdown
            </h3>
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
                  { label: "Healthy", pct: healthyPct, bar: "bg-[#5E9E62]" },
                  { label: "Engine Knocking", pct: knockPct, bar: "bg-[#D86C2B]" },
                  { label: "Misfire", pct: misfirePct, bar: "bg-[#E09A2B]" },
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
              <h3 className="text-[15px] font-mono-data uppercase tracking-wider font-semibold text-[var(--text)]">
                Recent checks
              </h3>
              <p className="text-[13px] text-[var(--text-muted)] mt-0.5">
                Select any recording to view its engine inspection and repair recommendations.
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
            <h3 className="text-[15px] font-mono-data uppercase tracking-wider font-semibold text-[var(--text)]">
              Try a sample engine recording
            </h3>
            <p className="text-[13px] text-[var(--text-muted)] mt-0.5">
              Load a verified clip from the dataset (Healthy, Engine Knocking, or Misfire) to test the Engine Lab and Repair Recommendation System.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {featuredSamples.map((s) => (
              <button
                key={s.id}
                type="button"
                onClick={() => onQuickTestSample(s)}
                className="rounded-[10px] border border-[var(--border)] hover:border-[#D86C2B] bg-[var(--surface-subtle)] p-4 text-left transition-all flex items-center justify-between gap-3"
              >
                <div className="min-w-0">
                  <div className="text-[14px] font-semibold text-[var(--text)]">
                    {predictionToLabel(s.expected_class)}
                  </div>
                  <div className="text-[12px] text-[var(--text-muted)] truncate mt-0.5">
                    Sample clip · {s.size_kb} KB
                  </div>
                </div>
                <span className="h-8 w-8 rounded-full bg-[#0B1510] border border-[var(--border-strong)] flex items-center justify-center text-[#D86C2B] shrink-0">
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
