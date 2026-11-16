import type {
  AnalysisResult,
  AppSettings,
  CoreCondition,
  DiagnoseResult,
  DiagnosticRecord,
  EngineCondition,
  HealthResponse,
  HistoryResponse,
  InferenceMode,
  ModelTelemetryResponse,
  SampleAudioItem,
  StatsResponse,
  UnifiedDiagnosticReport,
} from "../types/diagnostic";

const SETTINGS_KEY = "autoguard_settings_v2";

export const DEFAULT_SETTINGS: AppSettings = {
  apiBaseUrl: "http://localhost:8000",
  defaultMode: "dual",
  colormap: "thermal",
  playbackRate: 1.0,
  autoSaveCnn: true,
};

export function loadSettings(): AppSettings {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) return DEFAULT_SETTINGS;
    return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function saveSettings(settings: AppSettings): void {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
}

function getBaseUrl(): string {
  return loadSettings().apiBaseUrl.replace(/\/+$/, "");
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail || body.error || detail;
    } catch {
      // ignore non-json error bodies
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${getBaseUrl()}/health`, {
    signal: AbortSignal.timeout(5000),
  });
  return handleResponse<HealthResponse>(res);
}

export async function analyzeWithSVM(file: File, save = true): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${getBaseUrl()}/api/analyze?save=${save ? "true" : "false"}`, {
    method: "POST",
    body: formData,
  });
  return handleResponse<AnalysisResult>(res);
}

export async function diagnoseWithCNN(file: File, save = true): Promise<DiagnoseResult> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${getBaseUrl()}/api/diagnose?save=${save ? "true" : "false"}`, {
    method: "POST",
    body: formData,
  });
  return handleResponse<DiagnoseResult>(res);
}

export async function consensusDiagnose(file: File, save = true): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${getBaseUrl()}/api/consensus?save=${save ? "true" : "false"}`, {
    method: "POST",
    body: formData,
  });
  return handleResponse<AnalysisResult>(res);
}

/**
 * Execute diagnosis using the unified consensus endpoint (or CNN/SVM directly),
 * always returning a single specific engine condition (healthy, engine_knock, or misfire)
 * along with its confidence score.
 */
export async function runUnifiedDiagnosis(
  file: File,
  mode: InferenceMode,
  source: "upload" | "mic" | "sample" = "upload",
  expectedClass?: string
): Promise<UnifiedDiagnosticReport> {
  const settings = loadSettings();

  if (mode === "cnn") {
    const cnn = await diagnoseWithCNN(file, settings.autoSaveCnn);
    const scores = cnn.confidence_scores;
    const topConf = Math.max(scores.engine_knock ?? 0, scores.healthy ?? 0, scores.misfire ?? 0);
    return {
      scan_id: cnn.scan_id ?? -1,
      filename: file.name,
      mode: "cnn",
      prediction: cnn.prediction,
      confidence: topConf,
      class_scores: scores,
      duration_sec: cnn.duration_sec,
      inference_ms: cnn.inference_ms,
      created_at: cnn.created_at || new Date().toISOString(),
      uncertain: false,
      uncertain_reason: null,
      model_prediction: cnn.prediction,
      disagree_std: 0,
      window_count: 1,
      windows: [],
      cnn_result: cnn,
      svm_result: null,
      source,
      expected_class: expectedClass,
    };
  }

  if (mode === "svm") {
    const svm = await analyzeWithSVM(file, true);
    const specificPred =
      svm.prediction === "uncertain"
        ? svm.model_prediction ||
          ((Object.entries(svm.class_scores).sort((a, b) => b[1] - a[1])[0]?.[0] as CoreCondition) ??
            "healthy")
        : svm.prediction;
    return {
      scan_id: svm.scan_id,
      filename: svm.filename,
      mode: "svm",
      prediction: specificPred,
      confidence: svm.confidence,
      class_scores: svm.class_scores,
      duration_sec: svm.duration_sec,
      inference_ms: svm.inference_ms,
      created_at: svm.created_at || new Date().toISOString(),
      uncertain: false,
      uncertain_reason: null,
      model_prediction: specificPred,
      disagree_std: svm.disagree_std,
      window_count: svm.window_count,
      windows: svm.windows,
      cnn_result: null,
      svm_result: svm,
      source,
      expected_class: expectedClass,
    };
  }

  // Dual consensus: single upload runs both CNN & SVM on the backend and persists the exact fused result
  const consensus = await consensusDiagnose(file, true);
  const specificPred =
    consensus.prediction === "uncertain"
      ? consensus.model_prediction ||
        ((Object.entries(consensus.class_scores).sort((a, b) => b[1] - a[1])[0]?.[0] as CoreCondition) ??
          "healthy")
      : consensus.prediction;

  return {
    scan_id: consensus.scan_id,
    filename: consensus.filename,
    mode: "dual",
    prediction: specificPred,
    confidence: consensus.confidence,
    class_scores: consensus.class_scores,
    duration_sec: consensus.duration_sec,
    inference_ms: consensus.inference_ms,
    created_at: consensus.created_at || new Date().toISOString(),
    uncertain: false,
    uncertain_reason: null,
    model_prediction: specificPred,
    disagree_std: consensus.disagree_std,
    window_count: consensus.window_count,
    windows: consensus.windows,
    cnn_result: null,
    svm_result: null,
    source,
    expected_class: expectedClass,
  };
}

export async function fetchHistory(limit = 200, offset = 0): Promise<HistoryResponse> {
  const res = await fetch(`${getBaseUrl()}/api/history?limit=${limit}&offset=${offset}`);
  return handleResponse<HistoryResponse>(res);
}

export async function fetchScan(scanId: number): Promise<DiagnosticRecord> {
  const res = await fetch(`${getBaseUrl()}/api/history/${scanId}`);
  return handleResponse<DiagnosticRecord>(res);
}

export async function deleteScan(scanId: number): Promise<{ deleted: boolean; scan_id: number }> {
  const res = await fetch(`${getBaseUrl()}/api/history/${scanId}`, { method: "DELETE" });
  return handleResponse<{ deleted: boolean; scan_id: number }>(res);
}

export async function fetchStats(): Promise<StatsResponse> {
  const res = await fetch(`${getBaseUrl()}/api/stats`);
  return handleResponse<StatsResponse>(res);
}

export async function fetchTelemetry(): Promise<ModelTelemetryResponse> {
  const res = await fetch(`${getBaseUrl()}/api/model-telemetry`);
  return handleResponse<ModelTelemetryResponse>(res);
}

export async function fetchSamples(): Promise<{ samples: SampleAudioItem[] }> {
  const res = await fetch(`${getBaseUrl()}/api/samples`);
  return handleResponse<{ samples: SampleAudioItem[] }>(res);
}

export async function fetchSampleAsFile(sample: SampleAudioItem): Promise<File> {
  const res = await fetch(`${getBaseUrl()}/api/samples/audio?path=${encodeURIComponent(sample.path)}`);
  if (!res.ok) {
    throw new Error(`Unable to fetch sample audio (${res.status})`);
  }
  const blob = await res.blob();
  return new File([blob], sample.filename, { type: blob.type || "audio/wav" });
}

export function getSampleAudioUrl(path: string): string {
  return `${getBaseUrl()}/api/samples/audio?path=${encodeURIComponent(path)}`;
}

// ── Condition Display Helpers ─────────────────────────────────────────────────

export function predictionToLabel(prediction: string): string {
  switch (prediction) {
    case "engine_knock":
      return "Engine Knock";
    case "healthy":
      return "Healthy Engine";
    case "misfire":
      return "Engine Misfire";
    case "uncertain":
      return "Unclear Audio";
    default:
      return prediction;
  }
}

export function predictionToTone(prediction: string): "healthy" | "knock" | "misfire" | "uncertain" {
  if (prediction === "healthy") return "healthy";
  if (prediction === "engine_knock") return "knock";
  if (prediction === "uncertain") return "uncertain";
  return "misfire";
}

export function formatTimestamp(iso: string): { date: string; time: string } {
  if (!iso) return { date: "Just now", time: "—" };
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return { date: "—", time: "—" };
  return {
    date: d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" }),
    time: d.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", hour12: true }),
  };
}

export function formatDuration(seconds: number): string {
  const safe = Math.max(0, seconds || 0);
  const m = Math.floor(safe / 60);
  const s = Math.floor(safe % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export interface ConditionDiagnosticMeta {
  headline: string;
  code: string;
  severity: "NOMINAL" | "WARNING" | "CRITICAL" | "CAUTION";
  description: string;
  acousticSignature: string;
  affectedZones: string[];
  steps: Array<{ title: string; detail: string; priority: "HIGH" | "MED" | "LOW" }>;
}

export function getConditionMeta(prediction: EngineCondition): ConditionDiagnosticMeta {
  if (prediction === "healthy") {
    return {
      headline: "Engine sounds smooth and steady",
      code: "Healthy",
      severity: "NOMINAL",
      description:
        "No unusual knocking or misfire sounds were detected. Your engine idle and rhythm sound normal.",
      acousticSignature: "Smooth, steady engine rhythm with no sharp metallic tapping.",
      affectedZones: ["Cylinders", "Ignition", "Belt & oil"],
      steps: [
        {
          title: "Save for future reference",
          detail: "Keep this check in your history so you can compare if the engine ever sounds different.",
          priority: "LOW",
        },
        {
          title: "Follow routine maintenance",
          detail: "Stick to your regular oil changes and scheduled tune-ups.",
          priority: "LOW",
        },
        {
          title: "Recheck if symptoms appear",
          detail: "If you notice shaking or noise while driving, record a new clip after warming up.",
          priority: "MED",
        },
      ],
    };
  }

  if (prediction === "engine_knock") {
    return {
      headline: "Metallic tapping or pinging sound detected",
      code: "Warning",
      severity: "WARNING",
      description:
        "The recording has sharp metallic knocking noises that can happen from low-octane fuel, carbon buildup, or worn engine bearings.",
      acousticSignature: "Sharp repetitive tapping sounds during combustion.",
      affectedZones: ["Cylinders & pistons", "Knock sensor", "Fuel & timing"],
      steps: [
        {
          title: "Check your fuel grade",
          detail: "Make sure you are using the octane rating recommended in your owner's manual.",
          priority: "HIGH",
        },
        {
          title: "Have a mechanic check it soon",
          detail: "Ask a shop to inspect ignition timing, knock sensors, and oil condition.",
          priority: "HIGH",
        },
        {
          title: "Drive gently for now",
          detail: "Avoid hard acceleration or towing until the engine has been checked.",
          priority: "MED",
        },
      ],
    };
  }

  if (prediction === "misfire") {
    return {
      headline: "Uneven firing or rough running detected",
      code: "Fault",
      severity: "CRITICAL",
      description:
        "The engine rhythm sounds uneven, which often points to a worn spark plug, ignition coil, fuel injector, low oil, or worn belt.",
      acousticSignature: "Irregular pauses or rough mechanical vibration in the idle.",
      affectedZones: ["Spark plugs & coils", "Fuel injectors", "Belt & oil"],
      steps: [
        {
          title: "Scan for dashboard fault codes",
          detail: "Have an OBD-II scan run at a repair shop or auto parts store to pinpoint the cylinder.",
          priority: "HIGH",
        },
        {
          title: "Inspect spark plugs and coils",
          detail: "Worn spark plugs or weak ignition coils are the most common cause of misfires.",
          priority: "HIGH",
        },
        {
          title: "Check oil level and engine belt",
          detail: "Verify engine oil is full and listen for squeaks or drag from the drive belt.",
          priority: "MED",
        },
      ],
    };
  }

  // uncertain
  return {
    headline: "Recording was not clear enough",
    code: "Retake",
    severity: "CAUTION",
    description:
      "Background noise or changing engine sounds made this clip hard to read. Try recording again closer to the engine.",
    acousticSignature: "Mixed or quiet audio across the recording.",
    affectedZones: ["Audio clarity"],
    steps: [
      {
        title: "Move closer to the engine bay",
        detail: "Hold your microphone 1–2 feet above the engine with the hood open, away from wind.",
        priority: "HIGH",
      },
      {
        title: "Record at steady idle",
        detail: "Capture 10–15 seconds without talking, music, or passing traffic nearby.",
        priority: "MED",
      },
      {
        title: "Run the check again",
        detail: "Upload the clearer recording to get a confident reading.",
        priority: "LOW",
      },
    ],
  };
}

export function exportHistoryToCsv(records: DiagnosticRecord[]): void {
  const headers = [
    "Scan ID",
    "Filename",
    "Prediction",
    "Confidence (%)",
    "Knock Score",
    "Healthy Score",
    "Misfire Score",
    "Duration (s)",
    "Inference (ms)",
    "Created At",
  ];
  const rows = records.map((r) => [
    r.id,
    `"${r.filename.replace(/"/g, '""')}"`,
    predictionToLabel(r.prediction),
    (r.confidence * 100).toFixed(1),
    (r.knock_score * 100).toFixed(1),
    (r.healthy_score * 100).toFixed(1),
    (r.misfire_score * 100).toFixed(1),
    r.duration_sec.toFixed(2),
    r.inference_ms.toFixed(1),
    r.created_at,
  ]);
  const csv = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `autoguard-diagnostics-${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

export function exportReportToJson(report: UnifiedDiagnosticReport): void {
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `autoguard-scan-${report.scan_id > 0 ? report.scan_id : "report"}.json`;
  a.click();
  URL.revokeObjectURL(url);
}
