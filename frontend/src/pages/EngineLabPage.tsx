import { useState } from "react";
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
import { EngineLab3D, ENGINE_COMPONENTS } from "../components/EngineLab3D";
import { StatusBadge } from "../components/StatusBadge";
import { Badge, Button, Card } from "../components/ui";

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
          <div className="inline-flex items-center gap-2 text-[12px] font-mono-data uppercase tracking-wider text-[#38BDF8] mb-1">
            <Cpu size={14} strokeWidth={1.5} />
            <span>Interactive 3D Powertrain Failure Lab · GLTF 2.0</span>
          </div>
          <h2 className="text-[28px] sm:text-[32px] font-semibold text-[var(--text)] leading-tight">
            3D Engine Cutaway & Component Inspection Lab
          </h2>
          <p className="text-[14px] text-[var(--text-muted)] mt-1 max-w-3xl">
            Rotate, zoom, explode, and click individual 3D engine assemblies. Highlighted inspection zones dynamically reflect the active ML acoustic classification.
          </p>
        </div>

        <div className="flex items-center gap-2.5 flex-wrap">
          <Button type="button" variant="secondary" onClick={() => onNavigate("results")}>
            View full repair report
          </Button>
          <Button type="button" onClick={() => onNavigate("diagnosis")}>
            <Plus size={15} strokeWidth={1.5} /> Upload custom audio
          </Button>
        </div>
      </div>

      {/* Live Sample Audio Trigger Strip — Run Real ML Inference & Watch 3D Engine Update */}
      <Card className="p-4 sm:p-5 flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div className="flex items-start sm:items-center gap-3.5">
          <div className="h-10 w-10 rounded-[10px] bg-[#0EA5E9]/15 border border-[#38BDF8]/35 text-[#38BDF8] flex items-center justify-center shrink-0">
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
              Run a real acoustic sample below to trigger backend ML inference and update the 3D engine's highlighted inspection components in real time.
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
                className="px-3 py-2 rounded-[8px] border border-[var(--border)] bg-[var(--surface-subtle)] hover:border-[#38BDF8] text-[12px] font-medium text-[var(--text)] flex items-center gap-2 transition-all disabled:opacity-50"
              >
                <Play size={12} className={busy ? "animate-spin text-[#38BDF8]" : "text-[#38BDF8]"} />
                <span>
                  {busy ? "Running ML..." : `Test ${predictionToLabel(s.expected_class)} Audio`}
                </span>
              </button>
            );
          })}
        </div>
      </Card>

      {labError && (
        <div className="rounded-[10px] border border-[#EF4444]/40 bg-[#EF4444]/10 px-4 py-3 text-[13px] text-[#F87171]">
          {labError}
        </div>
      )}

      {/* ════════════════════════════════════════════════════════════════════════
          CENTERPIECE: GENUINE INTERACTIVE 3D GLB ENGINE FAILURE LAB
         ════════════════════════════════════════════════════════════════════════ */}
      <EngineLab3D
        condition={currentPrediction}
        confidence={currentConfidence}
        confidenceTier={currentTier}
        allowSimulationOverride={true}
      />

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
              <div className="rounded-[8px] border border-[var(--border)] bg-[var(--surface-subtle)] p-3">
                <div className="font-semibold text-[#F87171] flex items-center gap-1.5">
                  <Flame size={14} />
                  <span>Misfire Highlight Targets</span>
                </div>
                <p className="text-[12px] text-[var(--text-muted)] mt-1">
                  Highlights <strong>Spark Plugs (1–4)</strong>, <strong>Ignition Coils (COP 1–4)</strong>, and <strong>Fuel Injectors & Common Rail</strong> as candidate inspection areas.
                </p>
              </div>

              <div className="rounded-[8px] border border-[var(--border)] bg-[var(--surface-subtle)] p-3">
                <div className="font-semibold text-[#FBBF24] flex items-center gap-1.5">
                  <AlertTriangle size={14} />
                  <span>Engine Knocking Highlight Targets</span>
                </div>
                <p className="text-[12px] text-[var(--text-muted)] mt-1">
                  Highlights <strong>Combustion Chambers</strong>, <strong>Forged Pistons</strong>, <strong>Connecting Rods</strong>, <strong>Crankshaft</strong>, and <strong>Oil Pan</strong>.
                </p>
              </div>

              <div className="rounded-[8px] border border-[var(--border)] bg-[var(--surface-subtle)] p-3">
                <div className="font-semibold text-[#4ADE80] flex items-center gap-1.5">
                  <CheckCircle2 size={14} />
                  <span>Healthy Engine State</span>
                </div>
                <p className="text-[12px] text-[var(--text-muted)] mt-1">
                  No fault highlights active. Smooth 1-3-4-2 firing sequence across all four cylinders.
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
                <Wrench size={17} strokeWidth={1.5} className="text-[#38BDF8]" />
                <h3 className="text-[16px] font-semibold text-[var(--text)]">
                  ML-Linked Inspection & Repair Guidance ({predictionToLabel(currentPrediction)})
                </h3>
              </div>
              <Badge tone={currentPrediction === "healthy" ? "healthy" : currentPrediction === "engine_knock" ? "warning" : "fault"} dot>
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
                        <span className="font-mono-data text-[#FBBF24]">{i + 1}.</span>
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
                        <span className="font-mono-data text-[#38BDF8]">{i + 1}.</span>
                        <span>{insp}</span>
                      </li>
                    )
                  )}
                </ul>
              </div>
            </div>

            <div className="rounded-[10px] border border-[#EF4444]/35 bg-[#EF4444]/10 p-3.5 flex items-start gap-3">
              <ShieldAlert size={18} className="text-[#F87171] shrink-0 mt-0.5" />
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
