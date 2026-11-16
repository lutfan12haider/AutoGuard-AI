export type EngineCondition = "healthy" | "engine_knock" | "misfire" | "uncertain";
export type CoreCondition = "healthy" | "engine_knock" | "misfire";
export type InferenceMode = "dual" | "cnn" | "svm";
export type SpectrogramColormap = "thermal" | "cyan" | "viridis";
export type AppPage = "dashboard" | "diagnosis" | "results" | "history" | "telemetry";

export interface ClassScores {
  engine_knock: number;
  healthy: number;
  misfire: number;
  [key: string]: number;
}

export interface WindowResult {
  window: number;
  start_sec: number;
  end_sec: number;
  skipped: boolean;
  prediction?: CoreCondition | null;
  confidence?: number | null;
  scores?: ClassScores | null;
  reason?: string | null;
}

export interface AnalysisResult {
  scan_id: number;
  filename: string;
  prediction: EngineCondition;
  confidence: number;
  class_scores: ClassScores;
  duration_sec: number;
  inference_ms: number;
  created_at: string;
  window_count: number;
  windows: WindowResult[];
  uncertain: boolean;
  uncertain_reason?: string | null;
  model_prediction?: CoreCondition | null;
  disagree_std: number;
}

export interface DiagnoseResult {
  prediction: CoreCondition;
  confidence_scores: ClassScores;
  duration_sec: number;
  inference_ms: number;
  model: string;
  scan_id?: number | null;
  created_at?: string | null;
}

export interface UnifiedDiagnosticReport {
  scan_id: number;
  filename: string;
  mode: InferenceMode;
  prediction: EngineCondition;
  confidence: number;
  class_scores: ClassScores;
  duration_sec: number;
  inference_ms: number;
  created_at: string;
  uncertain: boolean;
  uncertain_reason?: string | null;
  model_prediction?: CoreCondition | null;
  disagree_std: number;
  window_count: number;
  windows: WindowResult[];
  models_agree?: boolean;
  cnn_result?: DiagnoseResult | null;
  svm_result?: AnalysisResult | null;
  source?: "upload" | "mic" | "sample" | "history";
  expected_class?: string;
}

export interface DiagnosticRecord {
  id: number;
  filename: string;
  prediction: EngineCondition;
  confidence: number;
  knock_score: number;
  healthy_score: number;
  misfire_score: number;
  duration_sec: number;
  inference_ms: number;
  created_at: string;
}

export interface HistoryResponse {
  records: DiagnosticRecord[];
  total: number;
  limit: number;
  offset: number;
}

export interface StatsResponse {
  total_scans: number;
  completed_scans: number;
  detected_faults: number;
  healthy_results: number;
  knock_results: number;
  misfire_results: number;
}

export interface HealthResponse {
  status: string;
  model: {
    loaded: boolean;
    model_name?: string;
    classes?: string[];
    test_accuracy?: number;
    test_macro_f1?: number;
    abstain_threshold?: number;
    disagree_threshold?: number;
  };
  cnn?: {
    loaded: boolean;
    model_name?: string;
    classes?: string[];
    test_accuracy?: number;
    best_epoch?: number;
    best_val_loss?: number;
    feature_dim?: number;
    n_mels?: number;
    n_frames?: number;
  };
  database: string;
  version: string;
}

export interface SampleAudioItem {
  id: string;
  filename: string;
  path: string;
  expected_class: CoreCondition;
  source: string;
  size_kb: number;
}

export interface ModelTelemetryResponse {
  svm: {
    model_name: string;
    feature_config: {
      sample_rate: number;
      max_duration_seconds: number;
      feature_dim: number;
      feature_groups: string[];
      version: string;
    };
    classes: string[];
    training: {
      train_samples: number;
      train_class_counts: Record<string, number>;
      train_time_sec: number;
      infer_time_ms_per_sample: number;
    };
    validation_metrics: {
      accuracy: number;
      macro_f1: number;
      confusion_matrix: number[][];
    };
    test_metrics: {
      accuracy: number;
      macro_f1: number;
      confusion_matrix: number[][];
    };
    real_test_metrics: {
      total_files: number;
      correctly_identified_as_knock: number;
      accuracy: number;
    };
    comparison: Record<
      string,
      {
        val_accuracy: number;
        val_macro_f1: number;
        train_time_sec: number;
      }
    >;
  } | null;
  thresholds: {
    abstain_threshold: number;
    disagree_threshold: number;
    chosen_from: string;
    retained_correct_frac: number;
    precision_above_T: number;
    note: string;
  } | null;
  cnn_info: HealthResponse["cnn"] | null;
  cnn_training_log: {
    best_epoch: number;
    best_val_loss: number;
    test_accuracy: number;
    per_epoch: Array<{
      epoch: number;
      train_loss: number;
      val_loss: number;
      val_acc: number;
      time_sec: number;
    }>;
  } | null;
  feature_config: {
    target_sr: number;
    max_duration_sec: number;
    n_mels: number;
    n_frames: number;
    fmax: number;
    hop_length: number;
    win_length: number;
    feature_dim: number;
    representation: string;
    normalisation: string;
    version: string;
  } | null;
  prepared_splits: Record<
    string,
    {
      split: string;
      n_samples: number;
      feature_dim: number;
      augmented: boolean;
      class_counts: Record<string, number>;
      source_files: number;
    } | null
  >;
  eval_30clips: {
    model: string;
    total_clips: number;
    accuracy: number;
    macro_f1: number;
    confusion_matrix: number[][];
    classes: string[];
    per_clip: Array<{
      filename: string;
      true: string;
      predicted: string;
      correct: boolean;
      confidence: number;
      scores: ClassScores;
    }>;
  } | null;
}

export interface AudioAnalysisMetrics {
  durationSec: number;
  sampleRate: number;
  channels: number;
  rmsDb: number;
  peakAmp: number;
  crestFactorDb: number;
  zcr: number;
  spectralCentroidHz: number;
  spectralRolloffHz: number;
  peakFreqHz: number;
  bandEnergy: {
    lowPct: number;   // 50-500 Hz (fundamental firing / idle thump)
    midPct: number;   // 500-2000 Hz (mechanical valvetrain / belt)
    highPct: number;  // 2000-8000 Hz (metallic detonation / knock ping)
  };
  waveformPeaks: number[];       // normalized 0..1 array (~160 bins)
  spectrogram: number[][];       // [timeFrames][freqBins] normalized 0..1 (up to 8000 Hz)
  maxFreqHz: number;
  isSilent: boolean;
  tooShort: boolean;
}

export interface AppSettings {
  apiBaseUrl: string;
  defaultMode: InferenceMode;
  colormap: SpectrogramColormap;
  playbackRate: number;
  autoSaveCnn: boolean;
}
