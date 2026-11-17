import {
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Database,
  Play,
  Plus,
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
import { EngineSchematic } from "../components/EngineSchematic";
import { ConfidenceGauge, StatusBadge } from "../components/StatusBadge";
import { SparklineChart } from "../components/TelemetryCharts";
import { Button, Card, EmptyState, Skeleton, Stat } from "../components/ui";

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
  const total = stats?.total_scans ?? 0;
  const healthy = stats?.healthy_results ?? 0;
  const faults = stats?.detected_faults ?? 0;
  const knock = stats?.knock_results ?? 0;
  const misfire = stats?.misfire_results ?? 0;

  const healthyPct = total > 0 ? Math.round((healthy / total) * 100) : 0;
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

      {/* Hero Section */}
      <Card className="p-6 lg:p-8 grid grid-cols-1 xl:grid-cols-12 gap-8 items-center">
        <div className="xl:col-span-5 space-y-5">
          <div>
            <h2 className="text-[28px] sm:text-[32px] font-semibold text-[#F8FAFC] leading-tight">
              Engine health overview
            </h2>
            <p className="text-[14px] text-[#94A3B8] mt-2 leading-relaxed">
              Screen engine audio across our three supported categories (Healthy, Misfire, Engine Knocking) and view structured repair recommendations.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <Button type="button" size="lg" onClick={() => onNavigate("diagnosis")}>
              <Plus size={15} strokeWidth={1.5} /> Check engine
            </Button>
            <Button type="button" variant="secondary" size="lg" onClick={() => onNavigate("history")}>
              View history
            </Button>
          </div>
        </div>

        <div className="xl:col-span-7">
          <EngineSchematic condition={latestScan ? latestScan.prediction : "idle"} compact />
        </div>
      </Card>

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
            <h3 className="text-[16px] font-semibold text-[#F8FAFC]">Classification breakdown</h3>
            <p className="text-[13px] text-[#94A3B8] mt-0.5">
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
                      <span className="text-[#CBD5E1] font-medium">{item.label}</span>
                      <span className="font-mono-data text-[#F8FAFC] font-semibold">{item.pct}%</span>
                    </div>
                    <div className="h-1.5 rounded-full bg-[#1E293B] overflow-hidden">
                      <div className={`h-full rounded-full ${item.bar}`} style={{ width: `${item.pct}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </Card>

        <Card className="lg:col-span-7 flex flex-col overflow-hidden">
          <div className="px-6 py-4 border-b border-[#1E293B] flex items-center justify-between">
            <div>
              <h3 className="text-[16px] font-semibold text-[#F8FAFC]">Recent checks</h3>
              <p className="text-[13px] text-[#94A3B8] mt-0.5">Select any recording to view its diagnosis and repair recommendations.</p>
            </div>
            <Button type="button" variant="ghost" size="sm" onClick={() => onNavigate("history")}>
              See all <ArrowRight size={14} strokeWidth={1.5} />
            </Button>
          </div>

          <div className="divide-y divide-[#1E293B] flex-1">
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
                    className="w-full px-6 py-3.5 flex items-center justify-between gap-4 hover:bg-[#1E293B]/50 text-left transition-colors"
                  >
                    <div className="min-w-0 flex-1">
                      <div className="text-[14px] font-medium text-[#F8FAFC] truncate">
                        {row.filename}
                      </div>
                      <div className="text-[12px] text-[#94A3B8] mt-0.5">
                        {date} · {time} · {formatDuration(row.duration_sec)}
                      </div>
                    </div>

                    <StatusBadge prediction={row.prediction} shortLabel />

                    <span className="font-mono-data text-[13px] font-semibold text-[#F8FAFC] w-11 text-right">
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
            <h3 className="text-[16px] font-semibold text-[#F8FAFC]">Try a sample engine recording</h3>
            <p className="text-[13px] text-[#94A3B8] mt-0.5">
              Load a verified sample from the dataset (Healthy, Engine Knocking, or Misfire) to test the Repair Recommendation System.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {featuredSamples.map((s) => (
              <button
                key={s.id}
                type="button"
                onClick={() => onQuickTestSample(s)}
                className="rounded-[10px] border border-[#1E293B] hover:border-[#38BDF8]/60 bg-[#090D16] hover:bg-[#0EA5E9]/10 p-4 text-left transition-all flex items-center justify-between gap-3"
              >
                <div className="min-w-0">
                  <div className="text-[14px] font-semibold text-[#F8FAFC]">
                    {predictionToLabel(s.expected_class)}
                  </div>
                  <div className="text-[12px] text-[#94A3B8] truncate mt-0.5">
                    Sample clip · {s.size_kb} KB
                  </div>
                </div>
                <span className="h-8 w-8 rounded-full bg-[#111827] border border-[#1E293B] flex items-center justify-center text-[#38BDF8] shrink-0">
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
