import { lazy, Suspense, useState } from "react";
import {
  AlertTriangle,
  Box,
  Check,
  CheckCircle2,
  ClipboardCheck,
  FileAudio,
  Info,
  Plus,
  Printer,
  ShieldAlert,
  Stethoscope,
  Wrench,
} from "lucide-react";
import type {
  AppPage,
  AudioAnalysisMetrics,
  SpectrogramColormap,
  UnifiedDiagnosticReport,
} from "../types/diagnostic";
import {
  buildClientRepairRecommendations,
  formatDuration,
  formatTimestamp,
  predictionToLabel,
  predictionToTone,
} from "../services/api";
import { ConditionIcon, ConfidenceGauge, StatusBadge } from "../components/StatusBadge";
import { SpectrogramWorkbench } from "../components/SpectrogramWorkbench";
import { Badge, Button, Card, EmptyState, Skeleton, Stepper } from "../components/ui";

const LazyEngineLab3D = lazy(() =>
  import("../components/EngineLab3D").then((m) => ({ default: m.EngineLab3D }))
);

export function ResultsPage({
  report,
  audioFile,
  audioMetrics,
  colormap,
  onNavigate,
}: {
  report: UnifiedDiagnosticReport | null;
  audioFile: File | null;
  audioMetrics: AudioAnalysisMetrics | null;
  colormap: SpectrogramColormap;
  onNavigate: (p: AppPage) => void;
}) {
  const [checkedInspections, setCheckedInspections] = useState<Record<number, boolean>>({});

  if (!report) {
    return (
      <Card className="p-8">
        <EmptyState
          icon={<ShieldAlert size={22} strokeWidth={1.5} />}
          title="No result selected"
          description="Check an engine recording or open any past scan from History."
          action={
            <Button type="button" onClick={() => onNavigate("diagnosis")}>
              <Plus size={15} strokeWidth={1.5} /> Check engine
            </Button>
          }
        />
      </Card>
    );
  }

  // Ensure structured recommendations are always present
  const fallbackRecs = buildClientRepairRecommendations(
    report.prediction,
    report.confidence,
    report.uncertain,
    report.uncertain_reason,
    report.disagree_std
  );

  const confidenceTier = report.confidence_tier || fallbackRecs.confidence_tier;
  const isLowConfidence = confidenceTier === "low";
  const isModerateConfidence = confidenceTier === "moderate";
  const isUnsupported = confidenceTier === "unsupported";
  const isHealthy = report.prediction === "healthy" && !isLowConfidence && !isUnsupported;

  const uncertaintyWarning = report.uncertainty_warning ?? fallbackRecs.uncertainty_warning;
  const diagnosisOverview = report.diagnosis_overview?.length
    ? report.diagnosis_overview
    : fallbackRecs.diagnosis_overview;
  const possibleCauses = report.possible_causes ?? fallbackRecs.possible_causes;
  const recommendedInspections = report.recommended_inspections?.length
    ? report.recommended_inspections
    : fallbackRecs.recommended_inspections;
  const possibleRepairs = report.possible_repairs ?? fallbackRecs.possible_repairs;
  const safetyWarning = report.safety_warning ?? fallbackRecs.safety_warning;
  const maintenanceAdvice = report.maintenance_advice?.length
    ? report.maintenance_advice
    : fallbackRecs.maintenance_advice;
  const disclaimer =
    report.disclaimer ??
    fallbackRecs.disclaimer ??
    "ML acoustic confidence is a screening indicator and is never proof of a specific mechanical fault.";

  const tone = isLowConfidence ? "uncertain" : predictionToTone(report.prediction);
  const label = isLowConfidence
    ? "Inconclusive / Low Confidence"
    : isUnsupported
      ? "Unsupported Audio"
      : predictionToLabel(report.prediction);

  const confPct = Math.round(report.confidence * 100);
  const { date, time } = formatTimestamp(report.created_at);

  const tierBadgeTone: Record<string, "healthy" | "warning" | "fault" | "accent"> = {
    high: "healthy",
    moderate: "warning",
    low: "fault",
    unsupported: "fault",
  };
  const tierLabel: Record<string, string> = {
    high: "High Confidence",
    moderate: "Moderate Confidence",
    low: "Low Confidence",
    unsupported: "Unsupported Audio",
  };

  const sortedScores: Array<[string, string, number, string]> = [
    ["engine_knock", "Engine Knocking", report.class_scores.engine_knock ?? 0, "bg-[#F59E0B]"],
    ["healthy", "Healthy", report.class_scores.healthy ?? 0, "bg-[#22C55E]"],
    ["misfire", "Misfire", report.class_scores.misfire ?? 0, "bg-[#EF4444]"],
  ];
  sortedScores.sort((a, b) => b[2] - a[2]);

  return (
    <div className="space-y-6">
      {/* Header & Stepper */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 no-print">
        <div>
          <h2 className="text-[28px] font-semibold text-[var(--text)] leading-tight">
            Diagnostic & Repair Recommendation Report
          </h2>
          <p className="text-[14px] text-[var(--text-muted)] mt-1">{report.filename}</p>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          <Button
            type="button"
            variant="secondary"
            size="sm"
            onClick={() => onNavigate("engine_lab")}
          >
            <Box size={14} strokeWidth={1.5} /> Full 3D Lab
          </Button>
          <Button type="button" variant="secondary" size="sm" onClick={() => window.print()}>
            <Printer size={14} strokeWidth={1.5} /> Print report
          </Button>
          <Button type="button" onClick={() => onNavigate("diagnosis")}>
            <Plus size={15} strokeWidth={1.5} /> Check another
          </Button>
        </div>
      </div>

      <div className="no-print">
        <Stepper steps={["Add audio", "Review", "Results"]} activeStep={2} />
      </div>

      {/* ════════════════════════════════════════════════════════════════════════
          SECTION 1: DIAGNOSIS OVERVIEW
         ════════════════════════════════════════════════════════════════════════ */}
      <section aria-labelledby="section-diagnosis-overview">
        <Card className="p-6 lg:p-8 space-y-6">
          <div className="flex items-center justify-between flex-wrap gap-3 border-b border-[var(--border)] pb-4">
            <div className="flex items-center gap-2.5">
              <Stethoscope size={18} strokeWidth={1.5} className="text-[#38BDF8]" />
              <h3 id="section-diagnosis-overview" className="text-[18px] font-semibold text-[var(--text)]">
                Diagnosis Overview
              </h3>
            </div>

            <div className="flex items-center gap-2 flex-wrap">
              {!isLowConfidence && <StatusBadge prediction={report.prediction} />}
              <Badge tone={tierBadgeTone[confidenceTier] || "accent"} dot>
                {tierLabel[confidenceTier] || "Confidence"} ({confPct}%)
              </Badge>
            </div>
          </div>

          {/* Uncertainty Warning Banner (Moderate or Low Confidence) */}
          {uncertaintyWarning && (
            <div
              className={`rounded-[10px] border p-4 flex items-start gap-3 ${
                isLowConfidence || isUnsupported
                  ? "border-[#EF4444]/40 bg-[#EF4444]/10 text-[#FCA5A5]"
                  : "border-[#F59E0B]/40 bg-[#F59E0B]/10 text-[#FDE68A]"
              }`}
            >
              <AlertTriangle size={18} strokeWidth={1.5} className="shrink-0 mt-0.5" />
              <div className="space-y-1 text-[13px] leading-relaxed">
                <div className="font-semibold">
                  {isLowConfidence
                    ? "Low Confidence — No Definitive Fault-Specific Diagnosis"
                    : isModerateConfidence
                      ? "Moderate Confidence — Uncertainty Advisory"
                      : "Audio Notice"}
                </div>
                <p>{uncertaintyWarning}</p>
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
            <div className="lg:col-span-8 space-y-4">
              <div className="flex items-start gap-3.5">
                <div className="h-12 w-12 rounded-[10px] bg-[var(--surface-subtle)] border border-[var(--border)] flex items-center justify-center shrink-0">
                  <ConditionIcon
                    prediction={isLowConfidence ? "uncertain" : report.prediction}
                    size={24}
                  />
                </div>
                <div>
                  <div className="text-[12px] uppercase tracking-wider font-mono-data text-[var(--text-muted)]">
                    Prediction
                  </div>
                  <h4 className="text-[24px] font-semibold text-[var(--text)] leading-snug">
                    {label}
                  </h4>
                </div>
              </div>

              {/* Overview Bullet Points (Matches Healthy 4 points verbatim when Healthy) */}
              <ul className="space-y-2.5 pt-1">
                {diagnosisOverview.map((line, idx) => (
                  <li
                    key={idx}
                    className="flex items-start gap-2.5 text-[14px] text-[var(--text-secondary)]"
                  >
                    <CheckCircle2
                      size={16}
                      strokeWidth={1.5}
                      className={`shrink-0 mt-0.5 ${
                        isHealthy
                          ? "text-[#4ADE80]"
                          : isLowConfidence
                            ? "text-[#38BDF8]"
                            : "text-[#FBBF24]"
                      }`}
                    />
                    <span>{line}</span>
                  </li>
                ))}
              </ul>

              {/* Non-100%-healthy / ML non-proof note */}
              <div className="rounded-[8px] bg-[var(--surface-subtle)] border border-[var(--border)] px-3.5 py-2.5 flex items-start gap-2.5 text-[12px] text-[var(--text-muted)]">
                <Info size={15} strokeWidth={1.5} className="text-[#38BDF8] shrink-0 mt-0.5" />
                <span>
                  {isHealthy
                    ? "Note: Audio analysis evaluates only supported engine sound classes (healthy, misfire, engine_knock) and does not confirm that the entire vehicle is 100% healthy."
                    : disclaimer}
                </span>
              </div>

              <div className="pt-3 border-t border-[var(--border)] flex items-center flex-wrap gap-x-6 gap-y-2 text-[13px] text-[var(--text-muted)]">
                <span className="inline-flex items-center gap-1.5 text-[var(--text)] font-medium">
                  <FileAudio size={15} strokeWidth={1.5} className="text-[#38BDF8]" />
                  {report.filename}
                </span>
                <span className="font-mono-data">{formatDuration(report.duration_sec)}</span>
                <span>
                  {date} · {time}
                </span>
              </div>
            </div>

            <div className="lg:col-span-4 flex flex-col items-center justify-center lg:border-l border-[var(--border)] lg:pl-6 gap-4">
              <ConfidenceGauge
                valuePct={confPct}
                tone={tone}
                label="ML Confidence"
                sublabel={tierLabel[confidenceTier]}
                size={160}
              />

              <div className="w-full space-y-2.5 pt-2 border-t border-[var(--border)]">
                {sortedScores.map(([key, name, score, barCls]) => {
                  const pct = Math.round(score * 100);
                  return (
                    <div key={key} className="space-y-1">
                      <div className="flex items-center justify-between text-[12px]">
                        <span className="text-[var(--text-secondary)]">{name}</span>
                        <span className="font-mono-data text-[var(--text)] font-semibold">
                          {pct}%
                        </span>
                      </div>
                      <div className="h-1.5 rounded-full bg-[var(--border)] overflow-hidden">
                        <div
                          className={`h-full rounded-full ${barCls}`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </Card>
      </section>

      {/* ════════════════════════════════════════════════════════════════════════
          INTERACTIVE 3D ENGINE FAILURE LAB (BOUND TO ACTUAL ML PREDICTION)
         ════════════════════════════════════════════════════════════════════════ */}
      <section aria-label="3D Engine Component Inspection">
        <Suspense fallback={<Skeleton className="h-[520px] w-full rounded-[12px]" />}>
          <LazyEngineLab3D
            condition={isLowConfidence ? "uncertain" : report.prediction}
            confidence={report.confidence}
            confidenceTier={confidenceTier}
            allowSimulationOverride={true}
          />
        </Suspense>
      </section>

      {/* ════════════════════════════════════════════════════════════════════════
          SECTION 2 & 3: POSSIBLE CAUSES + RECOMMENDED INSPECTIONS
         ════════════════════════════════════════════════════════════════════════ */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* SECTION 2: POSSIBLE CAUSES */}
        <section aria-labelledby="section-possible-causes" className="lg:col-span-6 flex">
          <Card className="p-6 flex-1 flex flex-col justify-between space-y-4">
            <div className="space-y-4">
              <div className="flex items-center justify-between border-b border-[var(--border)] pb-3">
                <div className="flex items-center gap-2.5">
                  <AlertTriangle size={18} strokeWidth={1.5} className="text-[#FBBF24]" />
                  <h3 id="section-possible-causes" className="text-[16px] font-semibold text-[var(--text)]">
                    Possible Causes
                  </h3>
                </div>
                <span className="text-[12px] text-[var(--text-muted)]">
                  Acoustic screening candidates
                </span>
              </div>

              {possibleCauses.length > 0 ? (
                <ul className="space-y-3">
                  {possibleCauses.map((cause, idx) => (
                    <li
                      key={idx}
                      className="rounded-[10px] border border-[var(--border)] bg-[var(--surface-subtle)] px-4 py-3 flex items-start gap-3 text-[14px] text-[var(--text)]"
                    >
                      <span className="h-5 w-5 rounded-full bg-[#F59E0B]/15 border border-[#F59E0B]/35 text-[#FBBF24] font-mono-data text-[11px] font-semibold flex items-center justify-center shrink-0 mt-0.5">
                        {idx + 1}
                      </span>
                      <span>{cause}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="rounded-[10px] border border-[var(--border)] bg-[var(--surface-subtle)] p-5 space-y-2 text-[14px] text-[var(--text-secondary)]">
                  {isHealthy ? (
                    <>
                      <div className="font-semibold text-[#4ADE80]">
                        No supported fault pattern was identified.
                      </div>
                      <p className="text-[13px] text-[var(--text-muted)]">
                        The engine sound appears consistent with the healthy class. No misfire or engine knocking causes are flagged.
                      </p>
                    </>
                  ) : (
                    <>
                      <div className="font-semibold text-[var(--text)]">
                        Fault-specific causes withheld (low confidence)
                      </div>
                      <p className="text-[13px] text-[var(--text-muted)]">
                        Because prediction confidence is below the minimum threshold, AutoGuard AI does not assign fault-specific causes to this recording.
                      </p>
                    </>
                  )}
                </div>
              )}
            </div>

            <p className="text-[12px] text-[var(--text-muted)] pt-3 border-t border-[var(--border)]">
              Sound patterns alone cannot prove a specific part has failed. Physical testing is required.
            </p>
          </Card>
        </section>

        {/* SECTION 3: RECOMMENDED INSPECTIONS */}
        <section aria-labelledby="section-recommended-inspections" className="lg:col-span-6 flex">
          <Card className="p-6 flex-1 flex flex-col justify-between space-y-4">
            <div className="space-y-4">
              <div className="flex items-center justify-between border-b border-[var(--border)] pb-3">
                <div className="flex items-center gap-2.5">
                  <ClipboardCheck size={18} strokeWidth={1.5} className="text-[#38BDF8]" />
                  <h3
                    id="section-recommended-inspections"
                    className="text-[16px] font-semibold text-[var(--text)]"
                  >
                    Recommended Inspections
                  </h3>
                </div>
                <span className="text-[12px] text-[var(--text-muted)]">
                  Click items to mark checked
                </span>
              </div>

              <div className="space-y-2.5">
                {recommendedInspections.map((item, idx) => {
                  const done = Boolean(checkedInspections[idx]);
                  return (
                    <button
                      key={idx}
                      type="button"
                      onClick={() =>
                        setCheckedInspections((prev) => ({ ...prev, [idx]: !prev[idx] }))
                      }
                      className={`w-full rounded-[10px] border px-4 py-3 text-left transition-all flex items-center justify-between gap-3 ${
                        done
                          ? "border-[#22C55E]/40 bg-[#22C55E]/10 text-[var(--text-muted)]"
                          : "border-[var(--border)] bg-[var(--surface-subtle)] hover:border-[var(--border-strong)] text-[var(--text)]"
                      }`}
                    >
                      <span className={`text-[14px] ${done ? "line-through" : ""}`}>{item}</span>
                      <span
                        className={`h-5 w-5 rounded-[6px] border flex items-center justify-center shrink-0 transition-colors ${
                          done
                            ? "border-[#22C55E] bg-[#22C55E] text-[#090D16]"
                            : "border-[var(--border-strong)] bg-[var(--surface)]"
                        }`}
                      >
                        {done && <Check size={12} strokeWidth={2.5} />}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>

            <p className="text-[12px] text-[var(--text-muted)] pt-3 border-t border-[var(--border)]">
              Verify each item systematically before replacing any components.
            </p>
          </Card>
        </section>
      </div>

      {/* ════════════════════════════════════════════════════════════════════════
          SECTION 4: POSSIBLE REPAIRS (AND MAINTENANCE ADVICE)
         ════════════════════════════════════════════════════════════════════════ */}
      <section aria-labelledby="section-possible-repairs">
        <Card className="p-6 space-y-4">
          <div className="flex items-center justify-between flex-wrap gap-2 border-b border-[var(--border)] pb-3">
            <div className="flex items-center gap-2.5">
              <Wrench size={18} strokeWidth={1.5} className="text-[#38BDF8]" />
              <h3 id="section-possible-repairs" className="text-[16px] font-semibold text-[var(--text)]">
                Possible Repairs
              </h3>
            </div>
            <span className="text-[12px] text-[var(--text-muted)]">
              Perform repairs only after physical inspection confirms a fault
            </span>
          </div>

          {possibleRepairs.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
              {possibleRepairs.map((repair, idx) => (
                <div
                  key={idx}
                  className="rounded-[10px] border border-[var(--border)] bg-[var(--surface-subtle)] p-4 flex items-start gap-3"
                >
                  <span className="h-6 w-6 rounded-[6px] bg-[#0EA5E9]/15 border border-[#38BDF8]/30 text-[#38BDF8] font-mono-data text-[12px] font-semibold flex items-center justify-center shrink-0 mt-0.5">
                    {idx + 1}
                  </span>
                  <span className="text-[14px] text-[var(--text)] leading-relaxed">{repair}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="rounded-[10px] border border-[var(--border)] bg-[var(--surface-subtle)] p-5 space-y-3">
              <div className="text-[14px] font-semibold text-[var(--text)]">
                {isHealthy
                  ? "No fault-specific repairs required — Routine Maintenance Advice"
                  : "No definitive repair actions recommended from this recording"}
              </div>
              <ul className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                {maintenanceAdvice.map((advice, idx) => (
                  <li
                    key={idx}
                    className="flex items-center gap-2 text-[13px] text-[var(--text-secondary)]"
                  >
                    <CheckCircle2 size={15} strokeWidth={1.5} className="text-[#4ADE80] shrink-0" />
                    <span>{advice}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </Card>
      </section>

      {/* ════════════════════════════════════════════════════════════════════════
          SECTION 5: SAFETY ADVISORY
         ════════════════════════════════════════════════════════════════════════ */}
      <section aria-labelledby="section-safety-advisory">
        <Card
          className={`p-6 border ${
            report.prediction === "misfire" || report.prediction === "engine_knock"
              ? "border-[#EF4444]/45 bg-[#EF4444]/10"
              : "border-[#38BDF8]/35"
          }`}
        >
          <div className="flex items-start gap-3.5">
            <div
              className={`h-10 w-10 rounded-[10px] flex items-center justify-center shrink-0 ${
                report.prediction === "misfire" || report.prediction === "engine_knock"
                  ? "bg-[#EF4444]/20 text-[#F87171] border border-[#EF4444]/40"
                  : "bg-[#0EA5E9]/15 text-[#38BDF8] border border-[#38BDF8]/35"
              }`}
            >
              <ShieldAlert size={20} strokeWidth={1.75} />
            </div>

            <div className="space-y-1.5">
              <h3 id="section-safety-advisory" className="text-[16px] font-semibold text-[var(--text)]">
                Safety Advisory
              </h3>
              <p className="text-[14px] text-[var(--text)] font-medium leading-relaxed">
                {safetyWarning}
              </p>
              <p className="text-[12px] text-[var(--text-muted)]">{disclaimer}</p>
            </div>
          </div>
        </Card>
      </section>

      {/* Audio Replay & Spectrogram (when checked in-session) */}
      {audioFile && audioMetrics && (
        <SpectrogramWorkbench
          file={audioFile}
          metrics={audioMetrics}
          colormap={colormap}
          showSpectrogram={true}
        />
      )}
    </div>
  );
}
