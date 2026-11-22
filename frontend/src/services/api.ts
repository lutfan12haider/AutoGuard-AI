import type {
  AnalysisResult,
  AppSettings,
  ConfidenceTier,
  CoreCondition,
  DiagnoseResult,
  DiagnosticRecord,
  EngineCondition,
  HealthResponse,
  HistoryResponse,
  InferenceMode,
  ModelTelemetryResponse,
  RepairRecommendationPayload,
  SampleAudioItem,
  StatsResponse,
  UnifiedDiagnosticReport,
} from "../types/diagnostic";

const SETTINGS_KEY = "autoguard_settings_v2";

export const DEFAULT_SETTINGS: AppSettings = {
apiBaseUrl: import.meta.env.VITE_API_URL || "http://localhost:8000",
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

const CLIENT_REPAIR_DATA: Record<
  CoreCondition,
  {
    display_label: string;
    diagnosis_overview: string[];
    possible_causes: string[];
    recommended_inspections: string[];
    possible_repairs: string[];
    safety_warning: string;
    maintenance_advice: string[];
  }
> = {
  healthy: {
    display_label: "Healthy",
    diagnosis_overview: [
      "Engine sound appears consistent with the healthy class.",
      "No supported fault pattern was identified.",
      "Continue routine vehicle maintenance.",
      "Monitor unusual sounds if they develop.",
    ],
    possible_causes: [],
    recommended_inspections: [
      "Continue routine scheduled inspections according to the vehicle service manual.",
      "Monitor engine sound and dashboard indicators if new symptoms develop.",
    ],
    possible_repairs: [],
    safety_warning:
      "Audio analysis covers only acoustic patterns for supported categories (healthy, misfire, engine_knock) and does not guarantee that the entire vehicle is 100% free of mechanical or electronic issues.",
    maintenance_advice: [
      "Engine sound appears consistent with the healthy class.",
      "No supported fault pattern was identified.",
      "Continue routine vehicle maintenance.",
      "Monitor unusual sounds if they develop.",
    ],
  },
  misfire: {
    display_label: "Misfire",
    diagnosis_overview: [
      "Acoustic pattern is consistent with the Engine Misfire classification category.",
      "Audio screening indicates irregular combustion cadence; physical inspection is required to confirm the underlying cause.",
    ],
    possible_causes: [
      "Worn or faulty spark plugs.",
      "Defective ignition coil.",
      "Fuel injector problems.",
      "Fuel delivery issues.",
    ],
    recommended_inspections: [
      "Inspect spark plugs.",
      "Check ignition coils.",
      "Inspect fuel injectors.",
      "Check for engine warning lights and diagnostic trouble codes.",
    ],
    possible_repairs: [
      "Clean or replace spark plugs if necessary.",
      "Replace a faulty ignition coil after confirmation.",
      "Clean or repair fuel injectors if testing confirms a problem.",
      "Repair fuel delivery issues if identified.",
    ],
    safety_warning:
      "If the engine is shaking severely, losing power, or the check-engine light is flashing, stop driving and arrange professional inspection.",
    maintenance_advice: [
      "Follow manufacturer spark plug and ignition service intervals.",
      "Use clean, manufacturer-recommended fuel and replace fuel filters on schedule.",
      "Address check-engine warning lights promptly before catalytic converter damage occurs.",
    ],
  },
  engine_knock: {
    display_label: "Engine Knocking",
    diagnosis_overview: [
      "Acoustic pattern is consistent with the Engine Knocking classification category.",
      "Audio screening indicates metallic knocking or pinging tendencies; a professional mechanical check is needed to verify the source.",
    ],
    possible_causes: [
      "Low engine oil level.",
      "Incorrect fuel quality or octane rating.",
      "Incorrect ignition timing.",
      "Internal engine component wear.",
      "Combustion-related problems.",
    ],
    recommended_inspections: [
      "Check engine oil level and condition.",
      "Verify fuel quality and manufacturer's requirements.",
      "Inspect ignition and combustion systems.",
      "Have internal engine components inspected if knocking persists.",
    ],
    possible_repairs: [
      "Top up or replace engine oil using the correct specification if required.",
      "Address fuel quality issues.",
      "Correct ignition-related problems if confirmed.",
      "Repair or replace damaged internal components if diagnosed by a mechanic.",
    ],
    safety_warning:
      "Loud or persistent knocking can indicate serious engine damage. Stop driving and arrange professional inspection.",
    maintenance_advice: [
      "Check engine oil level regularly and adhere to recommended oil change intervals.",
      "Always refuel with the octane rating specified in the owner's manual.",
      "Avoid heavy engine load or high RPM operation until knocking sounds are inspected.",
    ],
  },
};

export function buildClientRepairRecommendations(
  predictedClass: string,
  confidence: number,
  uncertain = false,
  uncertainReason?: string | null,
  disagreeStd = 0
): RepairRecommendationPayload {
  const conf = Math.max(0, Math.min(1, confidence || 0));
  const disclaimer =
    "ML acoustic confidence is a screening indicator and is never proof of a specific mechanical fault. Always confirm diagnoses with a qualified mechanic.";

  const validClass: CoreCondition | null =
    predictedClass === "healthy" || predictedClass === "misfire" || predictedClass === "engine_knock"
      ? predictedClass
      : null;

  if (!validClass) {
    return {
      predicted_class: "unsupported",
      display_label: "Unsupported Audio",
      confidence: conf,
      confidence_tier: "unsupported",
      uncertainty_warning:
        "Unsupported audio or unrecognized classification category. AutoGuard AI exclusively supports healthy, misfire, and engine_knock engine recordings.",
      diagnosis_overview: [
        "The uploaded audio does not match a supported classification category.",
        "AutoGuard AI only evaluates healthy, misfire, and engine_knock patterns.",
      ],
      possible_causes: [],
      recommended_inspections: [
        "Verify that the recording captures a running vehicle engine in WAV or MP3 format.",
        "Re-record 10–15 seconds of steady engine idle close to the hood.",
      ],
      possible_repairs: [],
      safety_warning:
        "If your vehicle exhibits warning lights, rough running, or unusual noises, arrange a professional mechanical inspection.",
      maintenance_advice: [
        "Continue routine vehicle maintenance according to manufacturer specifications.",
      ],
      disclaimer,
    };
  }

  const base = CLIENT_REPAIR_DATA[validClass];
  if (conf < 0.55 || disagreeStd > 0.2) {
    return {
      predicted_class: validClass,
      display_label: "Inconclusive / Low Confidence",
      confidence: conf,
      confidence_tier: "low",
      uncertainty_warning: `Low confidence reading (${
        uncertainReason || `${(conf * 100).toFixed(1)}% is below the 55% minimum threshold`
      }). Because the acoustic signal is ambiguous or inconsistent across time windows, a definitive fault-specific diagnosis cannot be provided from this recording.`,
      diagnosis_overview: [
        "Audio confidence is too low to provide a definitive fault-specific diagnosis.",
        "Background noise, microphone distance, or fluctuating engine speed may have obscured the sound pattern.",
        "Re-record a clean 10–15 second clip at steady engine idle, or have a technician inspect the vehicle directly.",
      ],
      possible_causes: [],
      recommended_inspections: [
        "Re-record engine audio 1–2 feet from the open hood in a quiet setting.",
        "Check the dashboard for any active check-engine or oil warning lights.",
        "Arrange a professional inspection if you notice roughness, power loss, or metallic noise.",
      ],
      possible_repairs: [],
      safety_warning: base.safety_warning,
      maintenance_advice: [
        "Continue routine vehicle maintenance.",
        "Monitor unusual sounds if they develop.",
      ],
      disclaimer,
    };
  }

  const tier: ConfidenceTier = conf < 0.75 || uncertain ? "moderate" : "high";
  return {
    predicted_class: validClass,
    display_label: base.display_label,
    confidence: conf,
    confidence_tier: tier,
    uncertainty_warning:
      tier === "moderate"
        ? `Moderate confidence (${(conf * 100).toFixed(1)}% — below the 75% high-confidence threshold). The recommendations below are preliminary possibilities based on acoustic similarity and must be verified through physical inspection before any repair.`
        : null,
    diagnosis_overview: [...base.diagnosis_overview],
    possible_causes: [...base.possible_causes],
    recommended_inspections: [...base.recommended_inspections],
    possible_repairs: [...base.possible_repairs],
    safety_warning: base.safety_warning,
    maintenance_advice: [...base.maintenance_advice],
    disclaimer,
  };
}

/**
 * Execute diagnosis using the unified consensus endpoint (or CNN/SVM directly),
 * returning the predicted engine condition, confidence score, and structured
 * Repair Recommendation System payload.
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
    const topConf =
      cnn.confidence ?? Math.max(scores.engine_knock ?? 0, scores.healthy ?? 0, scores.misfire ?? 0);
    const fallback = buildClientRepairRecommendations(
      cnn.prediction,
      topConf,
      Boolean(cnn.uncertain),
      cnn.uncertain_reason,
      0
    );
    return {
      scan_id: cnn.scan_id ?? -1,
      filename: file.name,
      mode: "cnn",
      prediction: cnn.prediction,
      predicted_class: cnn.predicted_class ?? fallback.predicted_class,
      confidence: topConf,
      class_scores: scores,
      duration_sec: cnn.duration_sec,
      inference_ms: cnn.inference_ms,
      created_at: cnn.created_at || new Date().toISOString(),
      uncertain: Boolean(cnn.uncertain),
      uncertain_reason: cnn.uncertain_reason ?? null,
      model_prediction: cnn.prediction,
      disagree_std: 0,
      window_count: 1,
      windows: [],
      cnn_result: cnn,
      svm_result: null,
      source,
      expected_class: expectedClass,
      confidence_tier: cnn.confidence_tier ?? fallback.confidence_tier,
      uncertainty_warning: cnn.uncertainty_warning ?? fallback.uncertainty_warning,
      diagnosis_overview: cnn.diagnosis_overview?.length
        ? cnn.diagnosis_overview
        : fallback.diagnosis_overview,
      possible_causes: cnn.possible_causes ?? fallback.possible_causes,
      recommended_inspections: cnn.recommended_inspections?.length
        ? cnn.recommended_inspections
        : fallback.recommended_inspections,
      possible_repairs: cnn.possible_repairs ?? fallback.possible_repairs,
      safety_warning: cnn.safety_warning ?? fallback.safety_warning,
      maintenance_advice: cnn.maintenance_advice?.length
        ? cnn.maintenance_advice
        : fallback.maintenance_advice,
      disclaimer: cnn.disclaimer ?? fallback.disclaimer,
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
    const fallback = buildClientRepairRecommendations(
      specificPred,
      svm.confidence,
      svm.uncertain,
      svm.uncertain_reason,
      svm.disagree_std
    );
    return {
      scan_id: svm.scan_id,
      filename: svm.filename,
      mode: "svm",
      prediction: specificPred,
      predicted_class: svm.predicted_class ?? fallback.predicted_class,
      confidence: svm.confidence,
      class_scores: svm.class_scores,
      duration_sec: svm.duration_sec,
      inference_ms: svm.inference_ms,
      created_at: svm.created_at || new Date().toISOString(),
      uncertain: svm.uncertain,
      uncertain_reason: svm.uncertain_reason ?? null,
      model_prediction: specificPred,
      disagree_std: svm.disagree_std,
      window_count: svm.window_count,
      windows: svm.windows,
      cnn_result: null,
      svm_result: svm,
      source,
      expected_class: expectedClass,
      confidence_tier: svm.confidence_tier ?? fallback.confidence_tier,
      uncertainty_warning: svm.uncertainty_warning ?? fallback.uncertainty_warning,
      diagnosis_overview: svm.diagnosis_overview?.length
        ? svm.diagnosis_overview
        : fallback.diagnosis_overview,
      possible_causes: svm.possible_causes ?? fallback.possible_causes,
      recommended_inspections: svm.recommended_inspections?.length
        ? svm.recommended_inspections
        : fallback.recommended_inspections,
      possible_repairs: svm.possible_repairs ?? fallback.possible_repairs,
      safety_warning: svm.safety_warning ?? fallback.safety_warning,
      maintenance_advice: svm.maintenance_advice?.length
        ? svm.maintenance_advice
        : fallback.maintenance_advice,
      disclaimer: svm.disclaimer ?? fallback.disclaimer,
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
  const fallback = buildClientRepairRecommendations(
    specificPred,
    consensus.confidence,
    consensus.uncertain,
    consensus.uncertain_reason,
    consensus.disagree_std
  );

  return {
    scan_id: consensus.scan_id,
    filename: consensus.filename,
    mode: "dual",
    prediction: specificPred,
    predicted_class: consensus.predicted_class ?? fallback.predicted_class,
    confidence: consensus.confidence,
    class_scores: consensus.class_scores,
    duration_sec: consensus.duration_sec,
    inference_ms: consensus.inference_ms,
    created_at: consensus.created_at || new Date().toISOString(),
    uncertain: consensus.uncertain,
    uncertain_reason: consensus.uncertain_reason ?? null,
    model_prediction: specificPred,
    disagree_std: consensus.disagree_std,
    status: consensus.status,
    recommendation: consensus.recommendation ?? null,
    score_calibration: consensus.score_calibration ?? "uncalibrated_consensus_score",
    model_used: consensus.model_used ?? "V1 Consensus (AutoGuardCNN + SVM)",
    window_count: consensus.window_count,
    windows: consensus.windows,
    cnn_result: null,
    svm_result: null,
    source,
    expected_class: expectedClass,
    confidence_tier: consensus.confidence_tier ?? fallback.confidence_tier,
    uncertainty_warning: consensus.uncertainty_warning ?? fallback.uncertainty_warning,
    diagnosis_overview: consensus.diagnosis_overview?.length
      ? consensus.diagnosis_overview
      : fallback.diagnosis_overview,
    possible_causes: consensus.possible_causes ?? fallback.possible_causes,
    recommended_inspections: consensus.recommended_inspections?.length
      ? consensus.recommended_inspections
      : fallback.recommended_inspections,
    possible_repairs: consensus.possible_repairs ?? fallback.possible_repairs,
    safety_warning: consensus.safety_warning ?? fallback.safety_warning,
    maintenance_advice: consensus.maintenance_advice?.length
      ? consensus.maintenance_advice
      : fallback.maintenance_advice,
    disclaimer: consensus.disclaimer ?? fallback.disclaimer,
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
      return "Engine Knocking";
    case "healthy":
      return "Healthy";
    case "misfire":
      return "Misfire";
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
