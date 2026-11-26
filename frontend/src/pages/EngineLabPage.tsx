import { lazy, Suspense, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Box,
  CheckCircle2,
  Cpu,
  Flame,
  Play,
  Plus,
  ShieldAlert,
  Sparkles,
  Wrench,
} from "lucide-react";
import type {
  AppPage,
  SampleAudioItem,
  UnifiedDiagnosticReport,
} from "../types/diagnostic";
import {
  buildClientRepairRecommendations,
  fetchSampleAsFile,
  predictionToLabel,
  runUnifiedDiagnosis,
} from "../services/api";
import { ENGINE_COMPONENTS } from "../components/EngineLab3D";
import { StatusBadge } from "../components/StatusBadge";
import { Badge, Button, Card, Skeleton, Tabs } from "../components/ui";

const LazyVehicleDiagnosticScene3D = lazy(() =>
  import("../components/VehicleDiagnosticScene3D").then((m) => ({
    default: m.VehicleDiagnosticScene3D,
  }))
);

const LazyEngineLab3D = lazy(() =>
  import("../components/EngineLab3D").then((m) => ({ default: m.EngineLab3D }))
);

export function EngineLabPage({
  activeReport,
  samples,
  onReportUpdated,
  onNavigate,
}: {
  activeReport: UnifiedDiagnosticReport | null;
  samples: SampleAudioItem[];
  onReportUpdated: (report: UnifiedDiagnosticReport, file: File) => void;
  onNavigate: (p: AppPage) => void;
}) {
  const [runningSampleId, setRunningSampleId] = useState<string | null>(null);
  const [labError, setLabError] = useState("");
  const [labView, setLabView] = useState<"vehicle" | "cutaway">("vehicle");

  const currentPrediction = activeReport?.prediction ?? "healthy";
  const currentConfidence = activeReport?.confidence ?? 0.85;
  const currentTier = activeReport?.confidence_tier ?? "high";

  const recs = buildClientRepairRecommendations(
    currentPrediction,
    currentConfidence,
    activeReport?.uncertain ?? false,
    activeReport?.uncertain_reason,
    activeReport?.disagree_std ?? 0
  );

  const featuredSamples = [
    samples.find((s) => s.expected_class === "healthy"),
    samples.find((s) => s.expected_class === "misfire"),
    samples.find((s) => s.expected_class === "engine_knock"),
  ].filter((s): s is SampleAudioItem => Boolean(s));

  const handleRunLiveSampleInLab = async (sample: SampleAudioItem) => {
    setRunningSampleId(sample.id);
    setLabError("");
    try {
      const f = await fetchSampleAsFile(sample);
      const report = await runUnifiedDiagnosis(f, "dual", "sample", sample.expected_class);
      onReportUpdated(report, f);
      setLabView("vehicle");
    } catch (err) {
      setLabError((err as Error).message || "Failed to analyze sample clip.");
    } finally {
      setRunningSampleId(null);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div>
          <div className="inline-flex items-center gap-2 text-[12px] font-mono-data uppercase tracking-wider text-[#D86C2B] font-semibold mb-1">
            <Cpu size={14} strokeWidth={1.5} />
            <span>Interactive Powertrain Failure Lab · 2023 Toyota Corolla GLTF 2.0</span>
          </div>
          <h2 className="text-[28px] sm:text-[32px] font-semibold text-[#F2E8D5] leading-tight">
            Corolla Bay &amp; Engine Component Inspection Lab
          </h2>
          <p className="text-[14px] text-[#8FA196] mt-1 max-w-3xl">
            Watch the 2023 Toyota Corolla bonnet open, inspect the 4-cylinder engine bay, and click individual time-animated engine assemblies synchronized with live ML predictions.
          </p>
        </div>

        <div className="flex items-center gap-2.5 flex-wrap">
          <Button type="button" variant="emeraldOutline" onClick={() => onNavigate("results")}>
            View full repair report
          </Button>
          <Button type="button" variant="white" onClick={() => onNavigate("diagnosis")}>
            <Plus size={15} strokeWidth={1.75} /> Upload custom audio
          </Button>
        </div>
      </div>

      {/* Live Sample Audio Trigger Strip — Run Real ML Inference & Watch 3D Engine Update */}
      <Card className="p-4 sm:p-5 flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div className="flex items-start sm:items-center gap-3.5">
          <div className="h-10 w-10 rounded-[10px] bg-[#13231A] border border-[#284233] text-[#D86C2B] flex items-center justify-center shrink-0">
            <Activity size={19} strokeWidth={1.75} />
          </div>
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-[14px] font-semibold text-[var(--text)]">
                Active ML Telemetry Link:
              </span>
              <StatusBadge prediction={currentPrediction} />
              <span className="font-mono-data text-[12px] text-[var(--text-muted)]">
                ({Math.round(currentConfidence * 100)}% confidence ·{" "}
                {activeReport ? activeReport.filename : "Default baseline"})
              </span>
            </div>
            <p className="text-[12px] text-[var(--text-muted)] mt-0.5">
              Run a real acoustic sample below to trigger backend ML inference, play the 3D vehicle sequence, and highlight affected components.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          {featuredSamples.map((s) => {
            const busy = runningSampleId === s.id;
            return (
              <button
                key={s.id}
                type="button"
                disabled={Boolean(runningSampleId)}
                onClick={() => handleRunLiveSampleInLab(s)}
                className="px-3 py-2 rounded-[8px] border border-[var(--border)] bg-[var(--surface-subtle)] hover:border-[#D86C2B] text-[12px] font-medium text-[var(--text)] flex items-center gap-2 transition-all disabled:opacity-50"
              >
                <Play size={12} className={busy ? "animate-spin text-[#D86C2B]" : "text-[#D86C2B]"} />
                <span>
                  {busy ? "Running ML..." : `Test ${predictionToLabel(s.expected_class)} Audio`}
                </span>
              </button>
            );
          })}
        </div>
      </Card>

      {labError && (
        <div className="rounded-[10px] border border-[#8C3B19] bg-[#2C150E] px-4 py-3 text-[13px] text-[#FDBA74]">
          {labError}
        </div>
      )}

      {/* ════════════════════════════════════════════════════════════════════════
          CENTERPIECE: 3D COROLLA BAY & INTERACTIVE 3D GLB ENGINE FAILURE LAB
         ════════════════════════════════════════════════════════════════════════ */}
      <div className="space-y-3">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <Tabs
            surface="emerald"
            value={labView}
            onChange={setLabView}
            layoutId="engine-lab-view-tab"
            items={[
              { id: "vehicle", label: "Corolla Diagnostic Bay" },
              { id: "cutaway", label: "34-Part Engine Cutaway" },
            ]}
          />
          <span className="text-[12px] text-[#8FA196]">
            {labView === "vehicle"
              ? "Automated camera approach · Opening bonnet · Live fault illumination"
              : "Click any component in the cutaway to inspect ML correlation"}
          </span>
        </div>

        <Suspense fallback={<Skeleton className="h-[540px] w-full rounded-[12px]" />}>
          {labView === "vehicle" ? (
            <LazyVehicleDiagnosticScene3D
              condition={currentPrediction}
              confidence={currentConfidence}
              confidenceTier={currentTier}
              sequenceKey={activeReport?.created_at ?? activeReport?.filename ?? "baseline"}
              autoPlay={true}
              onInspectSubsystem={() => setLabView("cutaway")}
            />
          ) : (
            <LazyEngineLab3D
              condition={currentPrediction}
              confidence={currentConfidence}
              confidenceTier={currentTier}
              allowSimulationOverride={true}
            />
          )}
        </Suspense>
      </div>

      {/* ── ML Class-to-3D Component Mapping Reference & Photography Cards ── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Luxury Metallic Engine Cutaway Photography + ML Mapping */}
        <Card className="lg:col-span-5 overflow-hidden flex flex-col justify-between">
          <div className="relative h-52 overflow-hidden bg-[#070A12]">
            <img
              src="/images/engine-cutaway-macro.jpg"
              alt="Metallic internal combustion cutaway reference"
              className="w-full h-full object-cover opacity-90 hover:scale-105 transition-transform duration-500"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-[#070A12] via-[#070A12]/35 to-transparent" />
            <div className="absolute bottom-3 left-4 right-4 flex items-end justify-between">
              <div>
                <Badge tone="warning" dot>
                  Combustion & Valvetrain Optical Reference
                </Badge>
                <h3 className="text-[16px] font-semibold text-white mt-1.5">
                  Internal Cutaway & Flame Front Visualization
                </h3>
              </div>
            </div>
          </div>

          <div className="p-5 space-y-3 flex-1 flex flex-col justify-between">
            <div className="space-y-2.5 text-[13px]">
              <div className="rounded-[8px] border border-[#8C3B19] bg-[#2C150E]/75 p-3">
                <div className="font-semibold text-[#FDBA74] flex items-center gap-1.5">
                  <AlertTriangle size={14} />
                  <span>Engine Knocking Targets (Orange-Red Warning)</span>
                </div>
                <p className="text-[12px] text-[#B8C5BD] mt-1">
                  Highlights <strong className="text-[#F2E8D5]">Combustion Chambers</strong>, <strong className="text-[#F2E8D5]">Forged Pistons</strong>, <strong className="text-[#F2E8D5]">Connecting Rods</strong>, <strong className="text-[#F2E8D5]">Crankshaft</strong>, and <strong className="text-[#F2E8D5]">Oil Pan</strong>.
                </p>
              </div>

              <div className="rounded-[8px] border border-[#784919] bg-[#2A1B0E]/80 p-3">
                <div className="font-semibold text-[#FCD34D] flex items-center gap-1.5">
                  <Flame size={14} />
                  <span>Misfire Highlight Targets (Amber Warning)</span>
                </div>
                <p className="text-[12px] text-[#B8C5BD] mt-1">
                  Highlights <strong className="text-[#F2E8D5]">Spark Plugs (1–4)</strong>, <strong className="text-[#F2E8D5]">Ignition Coils (COP 1–4)</strong>, and <strong className="text-[#F2E8D5]">Fuel Injectors &amp; Common Rail</strong>.
                </p>
              </div>

              <div className="rounded-[8px] border border-[#23633B] bg-[#11291B]/75 p-3">
                <div className="font-semibold text-[#86EFAC] flex items-center gap-1.5">
                  <CheckCircle2 size={14} />
                  <span>Healthy Engine State (Green)</span>
                </div>
                <p className="text-[12px] text-[#B8C5BD] mt-1">
                  Subtle green ambient illumination with smooth 1-3-4-2 firing cadence across all four cylinders.
                </p>
              </div>
            </div>

            <p className="text-[11px] text-[var(--text-muted)] pt-2 border-t border-[var(--border)]">
              Audio classification never confirms a specific component is faulty. Always verify with physical testing.
            </p>
          </div>
        </Card>

        {/* Right: Active Repair Recommendations Linked to the 3D Engine */}
        <Card className="lg:col-span-7 p-6 flex flex-col justify-between space-y-5">
          <div className="space-y-4">
            <div className="flex items-center justify-between flex-wrap gap-2 border-b border-[var(--border)] pb-3">
              <div className="flex items-center gap-2">
                <Wrench size={17} strokeWidth={1.5} className="text-[#16A34A]" />
                <h3 className="text-[16px] font-semibold text-[var(--text)]">
                  ML-Linked Inspection &amp; Repair Guidance ({predictionToLabel(currentPrediction)})
                </h3>
              </div>
              <Badge tone={currentPrediction === "healthy" ? "healthy" : currentPrediction === "engine_knock" ? "fault" : "warning"} dot>
                {recs.display_label}
              </Badge>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <div className="text-[12px] font-semibold uppercase tracking-wider font-mono-data text-[var(--text-muted)]">
                  Possible Causes
                </div>
                {(activeReport?.possible_causes ?? recs.possible_causes).length > 0 ? (
                  <ul className="space-y-2">
                    {(activeReport?.possible_causes ?? recs.possible_causes).map((c, i) => (
                      <li
                        key={i}
                        className="rounded-[8px] border border-[var(--border)] bg-[var(--surface-subtle)] px-3 py-2 text-[13px] text-[var(--text)] flex items-start gap-2"
                      >
                        <span
                          className={`font-mono-data font-semibold ${
                            currentPrediction === "engine_knock" ? "text-[#DC2626]" : "text-[#EA580C]"
                          }`}
                        >
                          {i + 1}.
                        </span>
                        <span>{c}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <div className="rounded-[8px] border border-[var(--border)] bg-[var(--surface-subtle)] p-3 text-[13px] text-[var(--text-muted)]">
                    No supported fault pattern was identified. Engine sound appears consistent with the healthy class.
                  </div>
                )}
              </div>

              <div className="space-y-2">
                <div className="text-[12px] font-semibold uppercase tracking-wider font-mono-data text-[var(--text-muted)]">
                  Recommended Inspections
                </div>
                <ul className="space-y-2">
                  {(activeReport?.recommended_inspections ?? recs.recommended_inspections).map(
                    (insp, i) => (
                      <li
                        key={i}
                        className="rounded-[8px] border border-[var(--border)] bg-[var(--surface-subtle)] px-3 py-2 text-[13px] text-[var(--text)] flex items-start gap-2"
                      >
                        <span className="font-mono-data font-semibold text-[#16A34A]">{i + 1}.</span>
                        <span>{insp}</span>
                      </li>
                    )
                  )}
                </ul>
              </div>
            </div>

            <div className="rounded-[10px] border border-[#8C3B19] bg-[#2C150E]/75 p-3.5 flex items-start gap-3">
              <ShieldAlert size={18} className="text-[#EA580C] shrink-0 mt-0.5" />
              <div className="text-[13px]">
                <span className="font-semibold text-[var(--text)]">Safety Advisory: </span>
                <span className="text-[var(--text-secondary)]">
                  {activeReport?.safety_warning ?? recs.safety_warning}
                </span>
              </div>
            </div>
          </div>

          <div className="flex items-center justify-between flex-wrap gap-3 pt-3 border-t border-[var(--border)]">
            <span className="text-[12px] text-[var(--text-muted)]">
              {ENGINE_COMPONENTS.length} interactive assemblies · 36 GLB meshes
            </span>
            <Button size="sm" onClick={() => onNavigate("results")}>
              Open complete 5-section diagnostic report
            </Button>
          </div>
        </Card>
      </div>
    </div>
  );
}
