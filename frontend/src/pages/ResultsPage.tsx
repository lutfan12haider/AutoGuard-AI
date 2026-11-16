import { useState } from "react";
import { Check, FileAudio, Plus, Printer, ShieldAlert } from "lucide-react";
import type {
  AppPage,
  AudioAnalysisMetrics,
  SpectrogramColormap,
  UnifiedDiagnosticReport,
} from "../types/diagnostic";
import {
  formatDuration,
  formatTimestamp,
  getConditionMeta,
  predictionToLabel,
  predictionToTone,
} from "../services/api";
import { ConditionIcon, ConfidenceGauge, StatusBadge } from "../components/StatusBadge";
import { EngineSchematic } from "../components/EngineSchematic";
import { SpectrogramWorkbench } from "../components/SpectrogramWorkbench";
import { Button, Card, EmptyState, Stepper } from "../components/ui";

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
  const [checkedSteps, setCheckedSteps] = useState<Record<number, boolean>>({});

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

  const tone = predictionToTone(report.prediction);
  const label = predictionToLabel(report.prediction);
  const meta = getConditionMeta(report.prediction);
  const confPct = Math.round(report.confidence * 100);
  const { date, time } = formatTimestamp(report.created_at);

  const sortedScores: Array<[string, string, number, string]> = [
    ["engine_knock", "Engine Knock", report.class_scores.engine_knock ?? 0, "bg-[#D97706]"],
    ["healthy", "Healthy Engine", report.class_scores.healthy ?? 0, "bg-[#16A34A]"],
    ["misfire", "Engine Misfire", report.class_scores.misfire ?? 0, "bg-[#DC2626]"],
  ];
  sortedScores.sort((a, b) => b[2] - a[2]);

  return (
    <div className="space-y-6">
      {/* Header & Stepper */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 no-print">
        <div>
          <h2 className="text-[28px] font-semibold text-[#0F172A] leading-tight">
            Engine check results
          </h2>
          <p className="text-[14px] text-[#64748B] mt-1">{report.filename}</p>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
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

      {/* Main Verdict Card */}
      <Card className="p-6 lg:p-8 grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
        <div className="lg:col-span-8 space-y-4">
          <StatusBadge prediction={report.prediction} shortLabel />

          <div className="flex items-start gap-3.5">
            <div className="h-11 w-11 rounded-[10px] bg-[#FAFAF9] border border-[#E7E5E4] flex items-center justify-center shrink-0">
              <ConditionIcon prediction={report.prediction} size={22} />
            </div>
            <div>
              <h3 className="text-[24px] font-semibold text-[#0F172A] leading-snug">{label}</h3>
              <p className="text-[14px] text-[#475569] mt-0.5">{meta.headline}</p>
            </div>
          </div>

          <p className="text-[14px] text-[#64748B] leading-relaxed max-w-2xl">{meta.description}</p>

          <div className="pt-3 border-t border-[#E7E5E4] flex items-center flex-wrap gap-x-6 gap-y-2 text-[13px] text-[#64748B]">
            <span className="inline-flex items-center gap-1.5 text-[#0F172A] font-medium">
              <FileAudio size={15} strokeWidth={1.5} className="text-[#4F46E5]" />
              {report.filename}
            </span>
            <span className="font-mono-data">{formatDuration(report.duration_sec)}</span>
            <span>{date} · {time}</span>
          </div>
        </div>

        <div className="lg:col-span-4 flex flex-col items-center justify-center lg:border-l border-[#E7E5E4] lg:pl-6">
          <ConfidenceGauge valuePct={confPct} tone={tone} label="Confidence" size={160} />
        </div>
      </Card>

      {/* Engine Diagram + Condition Likelihood */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <Card className="lg:col-span-7 p-6 space-y-4">
          <div>
            <h3 className="text-[16px] font-semibold text-[#0F172A]">Where to check</h3>
            <p className="text-[13px] text-[#64748B] mt-0.5">
              Click any engine area below to see what to look for.
            </p>
          </div>
          <EngineSchematic condition={report.prediction} />
        </Card>

        <Card className="lg:col-span-5 p-6 flex flex-col justify-between gap-6">
          <div className="space-y-4">
            <div>
              <h3 className="text-[16px] font-semibold text-[#0F172A]">Sound breakdown</h3>
              <p className="text-[13px] text-[#64748B] mt-0.5">
                How closely the clip matches each condition.
              </p>
            </div>

            <div className="space-y-4">
              {sortedScores.map(([key, name, score, barCls]) => {
                const pct = Math.round(score * 100);
                return (
                  <div key={key} className="space-y-1.5">
                    <div className="flex items-center justify-between text-[13px]">
                      <span className="font-medium text-[#0F172A]">{name}</span>
                      <span className="font-mono-data font-semibold text-[#0F172A]">{pct}%</span>
                    </div>
                    <div className="h-2 rounded-full bg-[#F5F5F4] overflow-hidden">
                      <div className={`h-full rounded-full ${barCls}`} style={{ width: `${pct}%` }} />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <p className="text-[12px] text-[#64748B] pt-4 border-t border-[#E7E5E4]">
            AutoGuard AI is a screening guide. Always confirm mechanical repairs with a trusted mechanic.
          </p>
        </Card>
      </div>

      {/* Audio Replay (when checked in-session) */}
      {audioFile && audioMetrics && (
        <SpectrogramWorkbench
          file={audioFile}
          metrics={audioMetrics}
          colormap={colormap}
          showSpectrogram={false}
        />
      )}

      {/* Recommended Next Steps Checklist */}
      <Card className="p-6 space-y-4">
        <div>
          <h3 className="text-[16px] font-semibold text-[#0F172A]">
            {report.prediction === "healthy"
              ? "How to keep your engine healthy"
              : "How to overcome this issue"}
          </h3>
          <p className="text-[13px] text-[#64748B] mt-0.5">
            Practical steps based on this {label.toLowerCase()} diagnosis.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {meta.steps.map((step, idx) => {
            const done = Boolean(checkedSteps[idx]);
            return (
              <button
                key={idx}
                type="button"
                onClick={() => setCheckedSteps((prev) => ({ ...prev, [idx]: !prev[idx] }))}
                className={`rounded-[10px] border p-4 text-left transition-all flex flex-col gap-2 ${
                  done
                    ? "border-[#BBF7D0] bg-[#F0FDF4]/60"
                    : "border-[#E7E5E4] bg-[#FAFAF9]/60 hover:border-[#D6D3D1]"
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono-data text-[12px] text-[#64748B]">Step {idx + 1}</span>
                  <span
                    className={`h-5 w-5 rounded-[6px] border flex items-center justify-center transition-colors ${
                      done ? "border-[#16A34A] bg-[#16A34A] text-white" : "border-[#D6D3D1] bg-white"
                    }`}
                  >
                    {done && <Check size={12} strokeWidth={2} />}
                  </span>
                </div>
                <div className={`text-[14px] font-semibold ${done ? "line-through text-[#64748B]" : "text-[#0F172A]"}`}>
                  {step.title}
                </div>
                <p className="text-[13px] text-[#64748B] leading-relaxed">{step.detail}</p>
              </button>
            );
          })}
        </div>
      </Card>
    </div>
  );
}
