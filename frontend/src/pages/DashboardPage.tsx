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
        <div className="rounded-[12px] border border-[#FECACA] bg-[#FEF2F2] px-4 py-3 flex items-center gap-3 text-[13px] text-[#DC2626]">
          <WifiOff size={16} strokeWidth={1.5} className="shrink-0" />
          <span>Service offline — make sure the local diagnostic server is running.</span>
        </div>
      )}

      {/* Hero Section */}
      <Card className="p-6 lg:p-8 grid grid-cols-1 xl:grid-cols-12 gap-8 items-center">
        <div className="xl:col-span-5 space-y-5">
          <div>
            <h2 className="text-[28px] sm:text-[32px] font-semibold text-[#0F172A] leading-tight">
              Engine health overview
            </h2>
            <p className="text-[14px] text-[#64748B] mt-2 leading-relaxed">
              Listen for engine knock or misfire from a short recording and track your vehicle checks in one place.
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
          subtext={`${healthy} smooth engines`}
          icon={<CheckCircle2 size={16} strokeWidth={1.5} className="text-[#16A34A]" />}
        />
        <Stat
          label="Issues found"
          value={loading ? <Skeleton className="h-7 w-16" /> : faults.toLocaleString()}
          subtext={`${knock} knock · ${misfire} misfire`}
          icon={<AlertTriangle size={16} strokeWidth={1.5} className="text-[#D97706]" />}
        />
      </div>

      {/* Distribution + Recent Checks */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <Card className="lg:col-span-5 p-6 flex flex-col justify-between gap-5">
          <div>
            <h3 className="text-[16px] font-semibold text-[#0F172A]">Results summary</h3>
            <p className="text-[13px] text-[#64748B] mt-0.5">
              Share of healthy checks vs. warnings.
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
                  { label: "Healthy", pct: healthyPct, bar: "bg-[#16A34A]" },
                  { label: "Engine Knock", pct: knockPct, bar: "bg-[#D97706]" },
                  { label: "Engine Misfire", pct: misfirePct, bar: "bg-[#DC2626]" },
                ].map((item) => (
                  <div key={item.label} className="space-y-1">
                    <div className="flex items-center justify-between text-[13px]">
                      <span className="text-[#475569] font-medium">{item.label}</span>
                      <span className="font-mono-data text-[#0F172A] font-semibold">{item.pct}%</span>
                    </div>
                    <div className="h-1.5 rounded-full bg-[#F5F5F4] overflow-hidden">
                      <div className={`h-full rounded-full ${item.bar}`} style={{ width: `${item.pct}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </Card>

        <Card className="lg:col-span-7 flex flex-col overflow-hidden">
          <div className="px-6 py-4 border-b border-[#E7E5E4] flex items-center justify-between">
            <div>
              <h3 className="text-[16px] font-semibold text-[#0F172A]">Recent checks</h3>
              <p className="text-[13px] text-[#64748B] mt-0.5">Select any recording to view its report.</p>
            </div>
            <Button type="button" variant="ghost" size="sm" onClick={() => onNavigate("history")}>
              See all <ArrowRight size={14} strokeWidth={1.5} />
            </Button>
          </div>

          <div className="divide-y divide-[#E7E5E4] flex-1">
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
                    className="w-full px-6 py-3.5 flex items-center justify-between gap-4 hover:bg-[#FAFAF9] text-left transition-colors"
                  >
                    <div className="min-w-0 flex-1">
                      <div className="text-[14px] font-medium text-[#0F172A] truncate">
                        {row.filename}
                      </div>
                      <div className="text-[12px] text-[#64748B] mt-0.5">
                        {date} · {time} · {formatDuration(row.duration_sec)}
                      </div>
                    </div>

                    <StatusBadge prediction={row.prediction} shortLabel />

                    <span className="font-mono-data text-[13px] font-semibold text-[#0F172A] w-11 text-right">
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
            <h3 className="text-[16px] font-semibold text-[#0F172A]">Try a sample sound</h3>
            <p className="text-[13px] text-[#64748B] mt-0.5">
              Don't have a recording handy? Load a sample engine sound to test the app.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {featuredSamples.map((s) => (
              <button
                key={s.id}
                type="button"
                onClick={() => onQuickTestSample(s)}
                className="rounded-[10px] border border-[#E7E5E4] hover:border-[#4F46E5]/60 bg-[#FAFAF9] hover:bg-[#EEF2FF]/35 p-4 text-left transition-all flex items-center justify-between gap-3"
              >
                <div className="min-w-0">
                  <div className="text-[14px] font-semibold text-[#0F172A]">
                    {predictionToLabel(s.expected_class)}
                  </div>
                  <div className="text-[12px] text-[#64748B] truncate mt-0.5">
                    Sample clip · {s.size_kb} KB
                  </div>
                </div>
                <span className="h-8 w-8 rounded-full bg-white border border-[#E7E5E4] flex items-center justify-center text-[#4F46E5] shrink-0">
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
