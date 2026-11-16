import { useCallback, useEffect, useState } from "react";
import { AnimatePresence, MotionConfig, motion } from "framer-motion";
import {
  Activity,
  ChevronRight,
  FileSearch,
  History,
  LayoutDashboard,
  Menu,
  PanelLeftClose,
  PanelLeftOpen,
  Search,
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
  UnifiedDiagnosticReport,
} from "./types/diagnostic";
import {
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

export default function App() {
  const [page, setPage] = useState<AppPage>("dashboard");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const settings = loadSettings();

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
      if (histRes.status === "fulfilled") setRecords(histRes.value.records);
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
    setActiveReport({
      scan_id: rec.id,
      filename: rec.filename,
      mode: "dual",
      prediction: rec.prediction,
      confidence: rec.confidence,
      class_scores: {
        engine_knock: rec.knock_score,
        healthy: rec.healthy_score,
        misfire: rec.misfire_score,
      },
      duration_sec: rec.duration_sec,
      inference_ms: rec.inference_ms,
      created_at: rec.created_at,
      uncertain: rec.prediction === "uncertain",
      disagree_std: 0,
      window_count: 1,
      windows: [],
      source: "history",
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
    { id: "diagnosis", label: "Check engine", icon: Activity },
    { id: "results", label: "Results", icon: FileSearch },
    { id: "history", label: "History", icon: History },
  ];

  const isOnline = health?.status === "ok";
  const currentNav = navItems.find((n) => n.id === page);

  return (
    <MotionConfig reducedMotion="user">
      <div className="min-h-screen bg-[#FAFAF9] text-[#0F172A]">
        {menuOpen && (
          <button
            type="button"
            onClick={() => setMenuOpen(false)}
            className="fixed inset-0 z-30 bg-[#0F172A]/25 backdrop-blur-[1px] lg:hidden"
            aria-label="Close navigation"
          />
        )}

        {/* Slim White Collapsible Sidebar */}
        <aside
          className={`fixed inset-y-0 left-0 z-40 bg-white border-r border-[#E7E5E4] flex flex-col transition-all duration-200 ease-out ${
            sidebarCollapsed ? "lg:w-[72px]" : "lg:w-[240px]"
          } w-[240px] ${menuOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"}`}
        >
          <div className="h-16 px-4 border-b border-[#E7E5E4] flex items-center justify-between gap-2">
            <div className="flex items-center gap-2.5 min-w-0">
              <div className="h-8 w-8 rounded-[8px] bg-[#4F46E5] text-white flex items-center justify-center shrink-0">
                <Activity size={17} strokeWidth={1.75} />
              </div>
              {!sidebarCollapsed && (
                <div className="truncate">
                  <div className="text-[14px] font-semibold text-[#0F172A] leading-tight">
                    AutoGuard AI
                  </div>
                  <div className="text-[11px] text-[#64748B]">Engine sound check</div>
                </div>
              )}
            </div>

            <button
              type="button"
              onClick={() => setSidebarCollapsed((c) => !c)}
              className="hidden lg:inline-flex p-1.5 rounded-[8px] text-[#64748B] hover:text-[#0F172A] hover:bg-[#F5F5F4] transition-colors"
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
              className="lg:hidden p-1.5 rounded-[8px] text-[#64748B] hover:text-[#0F172A]"
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
                  } py-2 rounded-[8px] gap-3 text-[13px] font-medium transition-colors duration-150 ${
                    active
                      ? "bg-[#EEF2FF] text-[#4F46E5]"
                      : "text-[#64748B] hover:text-[#0F172A] hover:bg-[#F5F5F4]"
                  }`}
                >
                  <Icon size={17} strokeWidth={1.5} className="shrink-0" />
                  {!sidebarCollapsed && <span className="truncate">{item.label}</span>}
                </button>
              );
            })}
          </nav>

          {/* Minimal Service Status */}
          <div className="p-3 border-t border-[#E7E5E4]">
            <div
              className={`flex items-center ${
                sidebarCollapsed ? "justify-center" : "px-3"
              } py-1.5 gap-2.5 text-[12px] text-[#64748B]`}
            >
              <span
                className={`h-2 w-2 rounded-full shrink-0 ${
                  isOnline ? "bg-[#16A34A]" : "bg-[#DC2626]"
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
          {/* Clean Top Bar */}
          <header className="sticky top-0 z-20 h-16 bg-white/95 backdrop-blur-sm border-b border-[#E7E5E4] px-4 sm:px-8 flex items-center justify-between gap-4 no-print">
            <div className="flex items-center gap-3 min-w-0">
              <button
                type="button"
                onClick={() => setMenuOpen(true)}
                className="lg:hidden p-1.5 rounded-[8px] border border-[#E7E5E4] text-[#64748B] hover:text-[#0F172A]"
                aria-label="Open navigation menu"
              >
                <Menu size={18} strokeWidth={1.5} />
              </button>

              <div className="flex items-center gap-2 text-[13px] min-w-0">
                <span className="hidden sm:inline text-[#64748B]">AutoGuard AI</span>
                <ChevronRight size={14} strokeWidth={1.5} className="hidden sm:inline text-[#CBD5E1]" />
                <h1 className="text-[14px] font-semibold text-[#0F172A] truncate">
                  {currentNav?.label}
                </h1>
              </div>
            </div>

            <button
              type="button"
              onClick={() => setPaletteOpen(true)}
              className="inline-flex items-center gap-2.5 h-9 px-3 rounded-[8px] border border-[#E7E5E4] bg-[#FAFAF9] hover:border-[#D6D3D1] text-[13px] text-[#64748B] transition-colors"
            >
              <Search size={14} strokeWidth={1.5} />
              <span className="hidden sm:inline">Search</span>
              <kbd className="font-mono-data text-[11px] bg-white border border-[#E7E5E4] rounded-[4px] px-1.5 py-0.5 text-[#64748B]">
                ⌘K
              </kbd>
            </button>
          </header>

          <div className="flex-1 max-w-[1280px] w-full mx-auto px-4 sm:px-8 py-8">
            {backendError && page !== "dashboard" && (
              <div className="mb-6 rounded-[12px] border border-[#FECACA] bg-[#FEF2F2] px-4 py-3 text-[13px] text-[#DC2626] flex items-center justify-between gap-3">
                <span>{backendError}</span>
                <button
                  type="button"
                  onClick={() => setBackendError("")}
                  className="text-[#DC2626] hover:opacity-75"
                  aria-label="Dismiss error"
                >
                  <X size={15} strokeWidth={1.5} />
                </button>
              </div>
            )}

            <AnimatePresence mode="wait">
              <motion.div
                key={page}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                transition={{ duration: 0.16, ease: "easeOut" }}
              >
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

                {page === "diagnosis" && (
                  <DiagnosisPage
                    defaultMode={settings.defaultMode}
                    colormap={settings.colormap}
                    samples={samples}
                    preloadedFile={preloadedFile}
                    preloadedSource={preloadedSource}
                    preloadedExpected={preloadedExpected}
                    onClearPreload={() => setPreloadedFile(null)}
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
              </motion.div>
            </AnimatePresence>
          </div>
        </main>

        {analyzingMode && <AnalyzingOverlay mode={analyzingMode} />}

        <CommandPalette
          open={paletteOpen}
          onClose={() => setPaletteOpen(false)}
          onNavigate={navigate}
          recentScans={records}
          onInspectRecord={handleInspectRecord}
        />
      </div>
    </MotionConfig>
  );
}
