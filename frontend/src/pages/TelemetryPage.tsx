import { useEffect, useState } from "react";
import { CheckCircle2, Cpu, XCircle } from "lucide-react";
import type { ModelTelemetryResponse } from "../types/diagnostic";
import { fetchTelemetry, predictionToLabel } from "../services/api";
import { CnnEpochChart, ConfusionMatrixCard } from "../components/TelemetryCharts";
import { Card, Skeleton, Stat, Table, TableHeader, TableRow, Tabs } from "../components/ui";

export function TelemetryPage() {
  const [data, setData] = useState<ModelTelemetryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [clipFilter, setClipFilter] = useState<"all" | "errors" | "engine_knock" | "healthy" | "misfire">("all");

  useEffect(() => {
    let cancelled = false;
    fetchTelemetry()
      .then((res) => {
        if (!cancelled) setData(res);
      })
      .catch((err) => {
        if (!cancelled) setError((err as Error).message || "Failed to load telemetry.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-12 w-72" />
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
          <Skeleton className="h-28" />
          <Skeleton className="h-28" />
          <Skeleton className="h-28" />
          <Skeleton className="h-28" />
        </div>
        <Skeleton className="h-64" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <Card className="p-8 text-center text-[14px] text-[#F87171]">
        {error || "Model telemetry is unavailable."}
      </Card>
    );
  }

  const svm = data.svm;
  const cnnLog = data.cnn_training_log;
  const thresholds = data.thresholds;
  const eval30 = data.eval_30clips;

  const filteredClips = (eval30?.per_clip || []).filter((c) => {
    if (clipFilter === "errors") return !c.correct;
    if (clipFilter === "all") return true;
    return c.true === clipFilter;
  });

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-[28px] font-semibold text-[#F8FAFC] leading-tight">Model telemetry</h2>
        <p className="text-[14px] text-[#94A3B8] mt-1">
          Evaluation metrics, training progression, and stratified validation benchmarks.
        </p>
      </div>

      {/* Benchmark KPIs */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <Stat
          label="CNN test accuracy"
          value={cnnLog ? `${(cnnLog.test_accuracy * 100).toFixed(1)}%` : "—"}
          subtext={`Best epoch ${cnnLog?.best_epoch ?? "—"}`}
        />
        <Stat
          label="SVM test accuracy"
          value={svm ? `${(svm.test_metrics.accuracy * 100).toFixed(1)}%` : "—"}
          subtext={`Macro F1 ${svm?.test_metrics.macro_f1 ?? "—"}`}
        />
        <Stat
          label="Abstain threshold"
          value={thresholds ? `${Math.round(thresholds.abstain_threshold * 100)}%` : "75%"}
          subtext={`Retains ${thresholds ? Math.round(thresholds.retained_correct_frac * 100) : 99}% correct`}
        />
        <Stat
          label="Training samples"
          value={data.prepared_splits?.train?.n_samples?.toLocaleString() ?? "—"}
          subtext="4× augmented audio clips"
        />
      </div>

      {/* Training Chart + Specs */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
        <Card className="xl:col-span-7 p-6 space-y-4">
          <div className="flex items-center gap-2">
            <Cpu size={17} strokeWidth={1.5} className="text-[#38BDF8]" />
            <h3 className="text-[16px] font-semibold text-[#F8FAFC]">
              AutoGuardCNN training curve
            </h3>
          </div>
          {cnnLog && <CnnEpochChart epochs={cnnLog.per_epoch} />}
        </Card>

        <Card className="xl:col-span-5 p-6 space-y-4">
          <div>
            <h3 className="text-[16px] font-semibold text-[#F8FAFC]">Pipeline specifications</h3>
            <p className="text-[13px] text-[#94A3B8] mt-0.5">
              Identical preprocessing shared between training and inference.
            </p>
          </div>

          <div className="space-y-3 text-[13px]">
            <div className="rounded-[8px] border border-[#1E293B] bg-[#090D16] p-4">
              <div className="font-semibold text-[#F8FAFC]">AutoGuardCNN (v3)</div>
              <div className="grid grid-cols-2 gap-y-1.5 font-mono-data text-[12px] text-[#94A3B8] mt-2">
                <span>Features: {data.feature_config?.feature_dim?.toLocaleString() ?? "16,384"}</span>
                <span>Shape: {data.feature_config?.n_mels}×{data.feature_config?.n_frames}</span>
                <span>Sample rate: {data.feature_config?.target_sr} Hz</span>
                <span>Max freq: {data.feature_config?.fmax} Hz</span>
              </div>
            </div>

            <div className="rounded-[8px] border border-[#1E293B] bg-[#090D16] p-4">
              <div className="font-semibold text-[#F8FAFC]">Acoustic SVM (v2)</div>
              <div className="grid grid-cols-2 gap-y-1.5 font-mono-data text-[12px] text-[#94A3B8] mt-2">
                <span>Features: {svm?.feature_config.feature_dim ?? 288}</span>
                <span>Pooling: mean + std</span>
                <span>MFCC + Δ + Δ²: 240</span>
                <span>Spectral/Chroma: 48</span>
              </div>
            </div>
          </div>
        </Card>
      </div>

      {/* Confusion Matrices */}
      {svm && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
          <ConfusionMatrixCard
            title="SVM test split (153 clips)"
            subtitle={`Accuracy ${(svm.test_metrics.accuracy * 100).toFixed(1)}% · Macro F1 ${svm.test_metrics.macro_f1}`}
            classes={svm.classes}
            matrix={svm.test_metrics.confusion_matrix}
          />
          <ConfusionMatrixCard
            title="SVM validation split (153 clips)"
            subtitle={`Accuracy ${(svm.validation_metrics.accuracy * 100).toFixed(1)}%`}
            classes={svm.classes}
            matrix={svm.validation_metrics.confusion_matrix}
          />
          {eval30 && (
            <ConfusionMatrixCard
              title="30-clip stratified eval"
              subtitle={`Accuracy ${(eval30.accuracy * 100).toFixed(1)}% · 100% misfire recall`}
              classes={eval30.classes}
              matrix={eval30.confusion_matrix}
            />
          )}
        </div>
      )}

      {/* 30-Clip Evaluation Table */}
      {eval30 && (
        <Card className="overflow-hidden">
          <div className="p-5 border-b border-[#1E293B] flex items-center justify-between flex-wrap gap-3">
            <div>
              <h3 className="text-[16px] font-semibold text-[#F8FAFC]">30-clip stratified audit</h3>
              <p className="text-[13px] text-[#94A3B8] mt-0.5">
                Per-clip prediction results from held-out evaluation.
              </p>
            </div>
            <Tabs
              value={clipFilter}
              onChange={setClipFilter}
              layoutId="eval-clip-tab"
              items={[
                { id: "all", label: "All" },
                { id: "errors", label: "Errors" },
                { id: "engine_knock", label: "Knock" },
                { id: "healthy", label: "Healthy" },
                { id: "misfire", label: "Misfire" },
              ]}
            />
          </div>

          <div className="max-h-96 overflow-y-auto">
            <Table>
              <TableHeader>
                <th className="py-3 px-4">Match</th>
                <th className="py-3 px-4">Filename</th>
                <th className="py-3 px-4">Actual</th>
                <th className="py-3 px-4">Predicted</th>
                <th className="py-3 px-4">Confidence</th>
              </TableHeader>
              <tbody>
                {filteredClips.map((c) => (
                  <TableRow key={c.filename}>
                    <td className="py-3 px-4">
                      {c.correct ? (
                        <CheckCircle2 size={16} strokeWidth={1.5} className="text-[#4ADE80]" />
                      ) : (
                        <XCircle size={16} strokeWidth={1.5} className="text-[#F87171]" />
                      )}
                    </td>
                    <td className="py-3 px-4 font-mono-data text-[13px] text-[#F8FAFC]">
                      {c.filename}
                    </td>
                    <td className="py-3 px-4 text-[13px] text-[#94A3B8]">
                      {predictionToLabel(c.true)}
                    </td>
                    <td
                      className={`py-3 px-4 text-[13px] font-medium ${
                        c.correct ? "text-[#4ADE80]" : "text-[#F87171]"
                      }`}
                    >
                      {predictionToLabel(c.predicted)}
                    </td>
                    <td className="py-3 px-4 font-mono-data text-[13px] text-[#F8FAFC]">
                      {(c.confidence * 100).toFixed(1)}%
                    </td>
                  </TableRow>
                ))}
              </tbody>
            </Table>
          </div>
        </Card>
      )}
    </div>
  );
}
