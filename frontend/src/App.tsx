import { lazy, Suspense, useCallback, useEffect, useState } from "react";
import { MotionConfig } from "framer-motion";
import {
  Activity,
  Box,
  ChevronRight,
  FileSearch,
  History,
  LayoutDashboard,
  Menu,
  X,
} from "lucide-react";
import type {
  AppPage,
  AudioAnalysisMetrics,
  DiagnosticRecord,
  HealthResponse,
  InferenceMode,
  SampleAudioItem,
  StatsResponse,
  ThemeMode,
  UnifiedDiagnosticReport,
} from "./types/diagnostic";
import {
  buildClientRepairRecommendations,
  checkHealth,
  deleteScan,
  fetchHistory,
  fetchSampleAsFile,
  fetchSamples,
  fetchStats,
  loadSettings,
  runUnifiedDiagnosis,
} from "./services/api";
import { DashboardPage } from "./pages/DashboardPage";
import { DiagnosisPage } from "./pages/DiagnosisPage";
import { ResultsPage } from "./pages/ResultsPage";
import { HistoryPage } from "./pages/HistoryPage";
import { CommandPalette } from "./components/CommandPalette";
import { AnalyzingOverlay } from "./components/Modals";
import { Skeleton } from "./components/ui";

const LazyEngineLabPage = lazy(() =>
  import("./pages/EngineLabPage").then((m) => ({ default: m.EngineLabPage }))
);

export default function App() {
  const [page, setPage] = useState<AppPage>("dashboard");
  const [menuOpen, setMenuOpen] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [settings] = useState(() => loadSettings());
  const theme: ThemeMode = "dark";

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [records, setRecords] = useState<DiagnosticRecord[]>([]);
  const [samples, setSamples] = useState<SampleAudioItem[]>([]);
  const [loadingDash, setLoadingDash] = useState(true);
  const [backendError, setBackendError] = useState("");

  const [analyzingMode, setAnalyzingMode] = useState<InferenceMode | null>(null);
  const [activeReport, setActiveReport] = useState<UnifiedDiagnosticReport | null>(null);
  const [activeFile, setActiveFile] = useState<File | null>(null);
  const [activeMetrics, setActiveMetrics] = useState<AudioAnalysisMetrics | null>(null);

  const [preloadedFile, setPreloadedFile] = useState<File | null>(null);
  const [preloadedSource, setPreloadedSource] = useState<"upload" | "mic" | "sample">("sample");
  const [preloadedExpected, setPreloadedExpected] = useState<string | undefined>(undefined);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", "dark");
  }, []);

  const toggleTheme = useCallback(() => {
    document.documentElement.setAttribute("data-theme", "dark");
  }, []);

  const refreshBackendData = useCallback(async () => {
    try {
      const [hRes, sRes, histRes, sampRes] = await Promise.allSettled([
        checkHealth(),
        fetchStats(),
        fetchHistory(200, 0),
        fetchSamples(),
      ]);

      if (hRes.status === "fulfilled") {
        setHealth(hRes.value);
        setBackendError("");
      } else {
        setHealth(null);
        setBackendError(hRes.reason?.message || "Service offline");
      }
      if (sRes.status === "fulfilled") setStats(sRes.value);
      if (histRes.status === "fulfilled") {
        setRecords(histRes.value.records);
      }
      if (sampRes.status === "fulfilled") setSamples(sampRes.value.samples);
    } finally {
      setLoadingDash(false);
    }
  }, []);

  useEffect(() => {
    refreshBackendData();
  }, [refreshBackendData]);

  useEffect(() => {
    if (health?.status === "ok") return;
    const id = window.setInterval(() => {
      refreshBackendData();
    }, 4000);
    return () => window.clearInterval(id);
  }, [health?.status, refreshBackendData]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen((v) => !v);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const navigate = useCallback((p: AppPage) => {
    setPage(p);
    setMenuOpen(false);
    setBackendError("");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  const handleRunDiagnosis = async (
    file: File,
    mode: InferenceMode,
    metrics: AudioAnalysisMetrics | null,
    source: "upload" | "mic" | "sample",
    expectedClass?: string
  ) => {
    setAnalyzingMode(mode);
    setBackendError("");
    try {
      const report = await runUnifiedDiagnosis(file, mode, source, expectedClass);
      setActiveReport(report);
      setActiveFile(file);
      setActiveMetrics(metrics);
      navigate("results");
      refreshBackendData();
    } catch (err) {
      setBackendError((err as Error).message || "Could not analyze this recording.");
    } finally {
      setAnalyzingMode(null);
    }
  };

  const handleInspectRecord = (rec: DiagnosticRecord) => {
    setActiveFile(null);
    setActiveMetrics(null);
    const fallback = buildClientRepairRecommendations(
      rec.prediction,
      rec.confidence,
      rec.confidence < 0.75
    );
    setActiveReport({
      scan_id: rec.id,
      filename: rec.filename,
      mode: "dual",
      prediction: rec.prediction,
      predicted_class: rec.predicted_class ?? fallback.predicted_class,
      confidence: rec.confidence,
      class_scores: {
        engine_knock: rec.knock_score,
        healthy: rec.healthy_score,
        misfire: rec.misfire_score,
      },
      duration_sec: rec.duration_sec,
      inference_ms: rec.inference_ms,
      created_at: rec.created_at,
      uncertain: rec.prediction === "uncertain" || rec.confidence < 0.75,
      disagree_std: 0,
      score_calibration: "uncalibrated_consensus_score",
      window_count: 1,
      windows: [],
      source: "history",
      confidence_tier: rec.confidence_tier ?? fallback.confidence_tier,
      uncertainty_warning: rec.uncertainty_warning ?? fallback.uncertainty_warning,
      diagnosis_overview: rec.diagnosis_overview?.length
        ? rec.diagnosis_overview
        : fallback.diagnosis_overview,
      possible_causes: rec.possible_causes ?? fallback.possible_causes,
      recommended_inspections: rec.recommended_inspections?.length
        ? rec.recommended_inspections
        : fallback.recommended_inspections,
      possible_repairs: rec.possible_repairs ?? fallback.possible_repairs,
      safety_warning: rec.safety_warning ?? fallback.safety_warning,
      maintenance_advice: rec.maintenance_advice?.length
        ? rec.maintenance_advice
        : fallback.maintenance_advice,
      disclaimer: rec.disclaimer ?? fallback.disclaimer,
    });
    navigate("results");
  };

  const handleQuickTestSample = async (sample: SampleAudioItem) => {
    try {
      const f = await fetchSampleAsFile(sample);
      setPreloadedFile(f);
      setPreloadedSource("sample");
      setPreloadedExpected(sample.expected_class);
      navigate("diagnosis");
    } catch (err) {
      setBackendError((err as Error).message || "Failed to load sample recording.");
    }
  };

  const handleDeleteRecord = async (scanId: number) => {
    await deleteScan(scanId);
    await refreshBackendData();
  };

  const navItems: Array<{
    id: AppPage;
    num: string;
    label: string;
    crumb: string;
    icon: typeof LayoutDashboard;
  }> = [
    { id: "dashboard", num: "01", label: "OVERVIEW", crumb: "Diagnostics", icon: LayoutDashboard },
    { id: "engine_lab", num: "02", label: "ENGINE LAB", crumb: "Engine Lab", icon: Box },
    { id: "diagnosis", num: "03", label: "CHECK ENGINE", crumb: "Check Engine", icon: Activity },
    { id: "results", num: "04", label: "RESULTS", crumb: "Results", icon: FileSearch },
    { id: "history", num: "05", label: "HISTORY", crumb: "History", icon: History },
  ];

  const isOnline = health?.status === "ok";
  const currentNav = navItems.find((n) => n.id === page);

  return (
    <MotionConfig reducedMotion="user">
      <div
        className="min-h-screen bg-[var(--bg)] text-[var(--text)]"
        style={{ backgroundImage: "var(--bg-gradient)" }}
      >
        {menuOpen && (
          <button
            type="button"
            onClick={() => setMenuOpen(false)}
            className="fixed inset-0 z-30 bg-black/70 backdrop-blur-[2px] lg:hidden"
            aria-label="Close navigation"
          />
        )}

        {/* Tactical Obsidian-Emerald Automotive Sidebar */}
        <aside
          className={`fixed inset-y-0 left-0 z-40 w-[184px] bg-[#08110C] border-r border-[#17281E] flex flex-col justify-between transition-transform duration-200 ease-out ${
            menuOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"
          }`}
        >
          <div>
            {/* Top Brand Box */}
            <div className="p-2.5 border-b border-[#16271D] flex items-center justify-between gap-2">
              <button
                type="button"
                onClick={() => navigate("dashboard")}
                className="w-full text-left rounded-[6px] border border-[#1E3327] bg-[#0B1610] hover:border-[#2E4D3B] px-3 py-2.5 transition-colors"
              >
                <div className="text-[13px] font-bold text-[#5E9E62] tracking-tight leading-tight">
                  AutoGuard AI
                </div>
                <div className="text-[11px] font-mono-data text-[#5E7466] mt-0.5">
                  Diagnostic Lab
                </div>
              </button>

              <button
                type="button"
                onClick={() => setMenuOpen(false)}
                className="lg:hidden p-1.5 rounded-[6px] text-[#7C8F83] hover:text-white"
                aria-label="Close sidebar"
              >
                <X size={16} strokeWidth={1.5} />
              </button>
            </div>

            {/* Numbered Tactical Navigation List */}
            <nav className="p-2.5 space-y-1" aria-label="Primary navigation">
              {navItems.map((item) => {
                const active = page === item.id;
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => navigate(item.id)}
                    className={`group w-full text-left transition-all duration-150 ${
                      active
                        ? "rounded-[6px] border border-[#2C3E34] bg-gradient-to-b from-[#142019] to-[#0D1712] px-3.5 py-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.06),0_4px_14px_rgba(0,0,0,0.4)]"
                        : "border-b border-[#132219] px-3.5 py-3 hover:bg-[#0D1812]"
                    }`}
                  >
                    <div
                      className={`font-mono-data text-[11px] font-bold leading-tight ${
                        active
                          ? "text-[#D86C2B]"
                          : "text-[#3E6B50] group-hover:text-[#6B9F7E]"
                      }`}
                    >
                      {item.num}
                    </div>
                    <div
                      className={`font-mono-data text-[12px] font-bold tracking-[0.06em] uppercase mt-0.5 leading-tight ${
                        active
                          ? "text-[#D86C2B]"
                          : "text-[#3E6B50] group-hover:text-[#8AB99B]"
                      }`}
                    >
                      {item.label}
                    </div>
                  </button>
                );
              })}
            </nav>
          </div>

          {/* Bottom Telemetry Status Box */}
          <div className="p-2.5 border-t border-[#16271D]">
            <div className="rounded-[6px] border border-[#231F1A] bg-[#0B120E] px-3 py-2.5">
              <div className="flex items-center gap-2">
                <span
                  className={`h-1.5 w-1.5 rounded-full shrink-0 ${
                    isOnline
                      ? "bg-[#E05A2B] shadow-[0_0_8px_#E05A2B]"
                      : "bg-[#EF4444]"
                  }`}
                />
                <span className="font-mono-data text-[10px] font-bold tracking-wider uppercase text-[#D86C2B] truncate">
                  {isOnline ? "SYSTEM READY" : "SERVICE OFFLINE"}
                </span>
              </div>
              <div className="font-mono-data text-[10px] text-[#8C623E] tracking-tight mt-0.5 pl-3.5">
                SYS.V4.0 · 1.2ms
              </div>
            </div>
          </div>
        </aside>

        {/* Main Content Area */}
        <main className="min-h-screen flex flex-col lg:pl-[184px]">
          {/* Dark Tactical Top Header Bar */}
          <header
            className="sticky top-0 z-20 h-14 backdrop-blur-md border-b border-[#16261D] px-4 sm:px-8 flex items-center justify-between gap-4 no-print"
            style={{ backgroundColor: "var(--header-bg)" }}
          >
            <div className="flex items-center gap-3 min-w-0">
              <button
                type="button"
                onClick={() => setMenuOpen(true)}
                className="lg:hidden p-1.5 rounded-[6px] border border-[#23382C] text-[#8FA196] hover:text-white"
                aria-label="Open navigation menu"
              >
                <Menu size={17} strokeWidth={1.5} />
              </button>

              <div className="flex items-center gap-2 text-[13px] min-w-0">
                <button
                  type="button"
                  onClick={() => navigate("dashboard")}
                  className="text-[#8C9E92] hover:text-[#EAE6DF] transition-colors"
                >
                  Overview
                </button>
                <ChevronRight
                  size={13}
                  strokeWidth={1.5}
                  className="text-[#566A5E]"
                />
                <h1 className="text-[13px] font-medium text-[#EAE6DF] truncate">
                  {currentNav?.crumb ?? "Diagnostics"}
                </h1>
              </div>
            </div>

            <div className="flex items-center gap-2.5">
              <span className="inline-flex items-center justify-center rounded-full bg-gradient-to-b from-[#C86B32] to-[#9A4C1E] border border-[#E08A54]/60 px-3.5 py-1 text-[10px] font-mono-data font-bold uppercase tracking-widest text-[#FFF5EC] shadow-[0_2px_12px_rgba(200,107,50,0.32)]">
                ACTIVE ZONE
              </span>

              <button
                type="button"
                onClick={() => setPaletteOpen(true)}
                title="Open Command Palette (⌘K)"
                className="inline-flex items-center justify-center rounded-full bg-[#0C1611] border border-[#495C50] hover:border-[#8FA196] px-3.5 py-1 text-[10px] font-mono-data font-semibold uppercase tracking-widest text-[#D8E2DC] transition-colors"
              >
                LOCAL LAB
              </button>
            </div>
          </header>

          <div className="flex-1 max-w-[1380px] w-full mx-auto px-4 sm:px-7 py-5">
            {backendError && page !== "dashboard" && (
              <div className="mb-5 rounded-[10px] border border-[#8C3B19] bg-[#2C150E] px-4 py-3 text-[13px] text-[#FDBA74] flex items-center justify-between gap-3 shadow-sm">
                <span>{backendError}</span>
                <button
                  type="button"
                  onClick={() => setBackendError("")}
                  className="text-[#FDBA74] hover:opacity-75"
                  aria-label="Dismiss error"
                >
                  <X size={15} strokeWidth={1.5} />
                </button>
              </div>
            )}

            <div>
              {page === "dashboard" && (
                <DashboardPage
                  stats={stats}
                  recentRecords={records}
                  health={health}
                  samples={samples}
                  loading={loadingDash}
                  error={backendError}
                  onNavigate={navigate}
                  onInspectRecord={handleInspectRecord}
                  onQuickTestSample={handleQuickTestSample}
                />
              )}

              {page === "engine_lab" && (
                <Suspense fallback={<Skeleton className="h-[640px] w-full rounded-[12px]" />}>
                  <LazyEngineLabPage
                    activeReport={activeReport}
                    samples={samples}
                    onReportUpdated={(report, f) => {
                      setActiveReport(report);
                      setActiveFile(f);
                      refreshBackendData();
                    }}
                    onNavigate={navigate}
                  />
                </Suspense>
              )}

              {page === "diagnosis" && (
                <DiagnosisPage
                  defaultMode={settings.defaultMode}
                  colormap={settings.colormap}
                  samples={samples}
                  preloadedFile={preloadedFile}
                  preloadedSource={preloadedSource}
                  preloadedExpected={preloadedExpected}
                  onClearPreload={() => setPreloadedFile(null)}
                  onClearBackendError={() => setBackendError("")}
                  onRunDiagnosis={handleRunDiagnosis}
                />
              )}

              {page === "results" && (
                <ResultsPage
                  report={activeReport}
                  audioFile={activeFile}
                  audioMetrics={activeMetrics}
                  colormap={settings.colormap}
                  onNavigate={navigate}
                />
              )}

              {page === "history" && (
                <HistoryPage
                  records={records}
                  stats={stats}
                  loading={loadingDash}
                  onNavigate={navigate}
                  onInspectRecord={handleInspectRecord}
                  onDeleteRecord={handleDeleteRecord}
                />
              )}
            </div>
          </div>
        </main>

        {analyzingMode && <AnalyzingOverlay mode={analyzingMode} />}

        <CommandPalette
          open={paletteOpen}
          onClose={() => setPaletteOpen(false)}
          onNavigate={navigate}
          recentScans={records}
          onInspectRecord={handleInspectRecord}
          theme={theme}
          onToggleTheme={toggleTheme}
        />
      </div>
    </MotionConfig>
  );
}
