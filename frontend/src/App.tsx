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
  Moon,
  PanelLeftClose,
  PanelLeftOpen,
  Search,
  Sun,
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
  saveSettings,
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
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [settings] = useState(() => loadSettings());
  const theme: ThemeMode = "light";

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
    document.documentElement.setAttribute("data-theme", "light");
  }, []);

  const toggleTheme = useCallback(() => {
    document.documentElement.setAttribute("data-theme", "light");
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

  const navItems: Array<{ id: AppPage; label: string; icon: typeof LayoutDashboard }> = [
    { id: "dashboard", label: "Overview", icon: LayoutDashboard },
    { id: "engine_lab", label: "3D Engine Lab", icon: Box },
    { id: "diagnosis", label: "Check engine", icon: Activity },
    { id: "results", label: "Results", icon: FileSearch },
    { id: "history", label: "History", icon: History },
  ];

  const isOnline = health?.status === "ok";
  const currentNav = navItems.find((n) => n.id === page);

  return (
    <MotionConfig reducedMotion="user">
      <div className="min-h-screen bg-[var(--bg)] text-[var(--text)]" style={{ backgroundImage: "var(--bg-gradient)" }}>
        {menuOpen && (
          <button
            type="button"
            onClick={() => setMenuOpen(false)}
            className="fixed inset-0 z-30 bg-black/60 backdrop-blur-[1px] lg:hidden"
            aria-label="Close navigation"
          />
        )}

        {/* Collapsible Forest Emerald Automotive Sidebar */}
        <aside
          className={`fixed inset-y-0 left-0 z-40 bg-gradient-to-b from-[#062C19] to-[#041D11] border-r border-[#12492C] flex flex-col transition-all duration-200 ease-out ${
            sidebarCollapsed ? "lg:w-[72px]" : "lg:w-[240px]"
          } w-[240px] ${menuOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"}`}
        >
          <div className="h-16 px-4 border-b border-[#12492C] flex items-center justify-between gap-2">
            <div className="flex items-center gap-2.5 min-w-0">
              <div className="h-8 w-8 rounded-[8px] bg-[#16A34A] border border-[#22C55E]/50 text-white flex items-center justify-center shrink-0 shadow-[0_2px_10px_rgba(22,163,74,0.4)]">
                <Activity size={17} strokeWidth={1.75} />
              </div>
              {!sidebarCollapsed && (
                <div className="truncate">
                  <div className="text-[14px] font-semibold text-white leading-tight">
                    AutoGuard AI
                  </div>
                  <div className="text-[11px] text-[#8EC4A3]">3D Diagnostic Lab</div>
                </div>
              )}
            </div>

            <button
              type="button"
              onClick={() => setSidebarCollapsed((c) => !c)}
              className="hidden lg:inline-flex p-1.5 rounded-[8px] text-[#95C4A9] hover:text-white hover:bg-[#0D3B23] transition-colors"
              aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            >
              {sidebarCollapsed ? (
                <PanelLeftOpen size={16} strokeWidth={1.5} />
              ) : (
                <PanelLeftClose size={16} strokeWidth={1.5} />
              )}
            </button>

            <button
              type="button"
              onClick={() => setMenuOpen(false)}
              className="lg:hidden p-1.5 rounded-[8px] text-[#95C4A9] hover:text-white"
              aria-label="Close sidebar"
            >
              <X size={18} strokeWidth={1.5} />
            </button>
          </div>

          <nav className="flex-1 p-3 space-y-1 overflow-y-auto" aria-label="Primary navigation">
            {navItems.map((item) => {
              const Icon = item.icon;
              const active = page === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => navigate(item.id)}
                  title={sidebarCollapsed ? item.label : undefined}
                  className={`w-full flex items-center ${
                    sidebarCollapsed ? "justify-center px-2" : "px-3"
                  } py-2.5 rounded-[8px] gap-3 text-[13px] font-medium transition-colors duration-150 ${
                    active
                      ? "bg-[#124B2D] text-white border border-[#22C55E]/40 shadow-sm"
                      : "text-[#CBE2D4] hover:text-white hover:bg-[#0D3922] border border-transparent"
                  }`}
                >
                  <Icon
                    size={17}
                    strokeWidth={1.6}
                    className={`shrink-0 ${active ? "text-[#4ADE80]" : "text-[#94C5A8]"}`}
                  />
                  {!sidebarCollapsed && <span className="truncate">{item.label}</span>}
                </button>
              );
            })}
          </nav>

          {/* Minimal Service Status */}
          <div className="p-3 border-t border-[#12492C]">
            <div
              className={`flex items-center ${
                sidebarCollapsed ? "justify-center" : "px-3"
              } py-1.5 gap-2.5 text-[12px] text-[#CBE2D4]`}
            >
              <span
                className={`h-2 w-2 rounded-full shrink-0 ${
                  isOnline ? "bg-[#4ADE80] shadow-[0_0_8px_#4ADE80]" : "bg-[#EF4444]"
                }`}
              />
              {!sidebarCollapsed && (
                <span className="truncate">{isOnline ? "System ready" : "Service offline"}</span>
              )}
            </div>
          </div>
        </aside>

        {/* Main Content Area */}
        <main
          className={`min-h-screen flex flex-col transition-all duration-200 ${
            sidebarCollapsed ? "lg:pl-[72px]" : "lg:pl-[240px]"
          }`}
        >
          {/* Crisp White Top Header Bar matching Design System */}
          <header
            className="sticky top-0 z-20 h-16 backdrop-blur-md border-b border-[#D8E5DD] px-4 sm:px-8 flex items-center justify-between gap-4 no-print"
            style={{ backgroundColor: "var(--header-bg)" }}
          >
            <div className="flex items-center gap-3 min-w-0">
              <button
                type="button"
                onClick={() => setMenuOpen(true)}
                className="lg:hidden p-1.5 rounded-[8px] border border-[#D8E5DD] text-[#5C6B62] hover:text-[#14221B]"
                aria-label="Open navigation menu"
              >
                <Menu size={18} strokeWidth={1.5} />
              </button>

              <div className="flex items-center gap-2 text-[13px] min-w-0">
                <span className="hidden sm:inline text-[#5C7165]">AutoGuard AI</span>
                <ChevronRight
                  size={14}
                  strokeWidth={1.5}
                  className="hidden sm:inline text-[#8FA396]"
                />
                <h1 className="text-[14px] font-semibold text-[#0A2618] truncate">
                  {currentNav?.label}
                </h1>
              </div>
            </div>

            <div className="flex items-center gap-2.5">
              <div className="hidden sm:inline-flex items-center gap-2 h-9 px-3.5 rounded-[8px] border border-[#167A46] bg-[#0B532F] text-[12px] font-medium text-white shadow-xs">
                <span className="h-2 w-2 rounded-full bg-[#4ADE80]" />
                <span>Diagnostic Lab · Local</span>
              </div>

              <button
                type="button"
                onClick={() => setPaletteOpen(true)}
                className="inline-flex items-center gap-2.5 h-9 px-3 rounded-[8px] border border-[#D8E5DD] bg-[#F5F8F6] hover:border-[#B8CEC1] text-[13px] text-[#4E6156] transition-colors"
              >
                <Search size={14} strokeWidth={1.5} />
                <span className="hidden sm:inline">Search...</span>
                <kbd className="font-mono-data text-[11px] bg-white border border-[#D8E5DD] rounded-[4px] px-1.5 py-0.5 text-[#526358]">
                  ⌘K
                </kbd>
              </button>
            </div>
          </header>

          <div className="flex-1 max-w-[1320px] w-full mx-auto px-4 sm:px-8 py-8">
            {backendError && page !== "dashboard" && (
              <div className="mb-6 rounded-[12px] border border-[#FECACA] bg-[#FEE2E2] px-4 py-3 text-[13px] text-[#B91C1C] flex items-center justify-between gap-3 shadow-sm">
                <span>{backendError}</span>
                <button
                  type="button"
                  onClick={() => setBackendError("")}
                  className="text-[#B91C1C] hover:opacity-75"
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
