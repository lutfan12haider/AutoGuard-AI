import { useEffect, useRef, useState, type ChangeEvent, type DragEvent } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Activity,
  AlertCircle,
  Database,
  FileAudio,
  Mic,
  Play,
  Trash2,
  Upload,
} from "lucide-react";
import type {
  AudioAnalysisMetrics,
  InferenceMode,
  SampleAudioItem,
  SpectrogramColormap,
} from "../types/diagnostic";
import { decodeAndAnalyzeAudio } from "../utils/audioDsp";
import { fetchSampleAsFile, predictionToLabel } from "../services/api";
import { SpectrogramWorkbench } from "../components/SpectrogramWorkbench";
import { LiveMicRecorder } from "../components/LiveMicRecorder";
import { Badge, Button, Card, Dropzone, EmptyState, Stepper, Tabs } from "../components/ui";

type CaptureTab = "upload" | "mic" | "samples";

export function DiagnosisPage({
  defaultMode: _defaultMode,
  colormap,
  samples,
  preloadedFile,
  preloadedSource,
  preloadedExpected,
  onClearPreload,
  onRunDiagnosis,
}: {
  defaultMode: InferenceMode;
  colormap: SpectrogramColormap;
  samples: SampleAudioItem[];
  preloadedFile: File | null;
  preloadedSource?: "upload" | "mic" | "sample";
  preloadedExpected?: string;
  onClearPreload: () => void;
  onRunDiagnosis: (
    file: File,
    mode: InferenceMode,
    metrics: AudioAnalysisMetrics | null,
    source: "upload" | "mic" | "sample",
    expectedClass?: string
  ) => void;
}) {
  const [tab, setTab] = useState<CaptureTab>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [source, setSource] = useState<"upload" | "mic" | "sample">("upload");
  const [expectedClass, setExpectedClass] = useState<string | undefined>(undefined);
  const [metrics, setMetrics] = useState<AudioAnalysisMetrics | null>(null);
  const [decoding, setDecoding] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  const [loadingSampleId, setLoadingSampleId] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (preloadedFile) {
      loadAudioFile(preloadedFile, preloadedSource || "sample", preloadedExpected);
      onClearPreload();
    }
  }, [preloadedFile]);

  const loadAudioFile = async (
    candidate: File,
    src: "upload" | "mic" | "sample",
    expected?: string
  ) => {
    const lower = candidate.name.toLowerCase();
    if (!lower.endsWith(".wav") && !lower.endsWith(".mp3")) {
      setError("Please choose a WAV or MP3 audio file.");
      return;
    }
    if (candidate.size > 25 * 1024 * 1024) {
      setError("File is larger than 25 MB.");
      return;
    }

    setError("");
    setFile(candidate);
    setSource(src);
    setExpectedClass(expected);
    setDecoding(true);
    try {
      const m = await decodeAndAnalyzeAudio(candidate);
      setMetrics(m);
      if (m.tooShort) {
        setError("Recording is too short. Please use at least 1 second of engine sound.");
      } else if (m.isSilent) {
        setError("No sound detected in this file. Try recording closer to the engine.");
      }
    } catch {
      setMetrics(null);
      setError("Could not preview audio in browser, but you can still run the check.");
    } finally {
      setDecoding(false);
    }
  };

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) loadAudioFile(f, "upload");
  };

  const handleSelectSample = async (item: SampleAudioItem) => {
    setLoadingSampleId(item.id);
    setError("");
    try {
      const f = await fetchSampleAsFile(item);
      await loadAudioFile(f, "sample", item.expected_class);
    } catch (err) {
      setError((err as Error).message || "Could not load sample sound.");
    } finally {
      setLoadingSampleId(null);
    }
  };

  // Pick a clean curated set of 6 sample clips (2 per category) so the list stays simple
  const curatedSamples = [
    ...samples.filter((s) => s.expected_class === "healthy").slice(0, 2),
    ...samples.filter((s) => s.expected_class === "engine_knock").slice(0, 2),
    ...samples.filter((s) => s.expected_class === "misfire").slice(0, 2),
  ];

  const canAnalyze = Boolean(file) && !(metrics?.tooShort || metrics?.isSilent);
  const currentStep = file && canAnalyze ? 1 : 0;

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-[28px] font-semibold text-[#0F172A] leading-tight">
            Check your engine
          </h2>
          <p className="text-[14px] text-[#64748B] mt-1">
            Upload or record a short clip of your engine running to check for knock or misfire.
          </p>
        </div>

        <Badge tone="healthy" dot>
          Private · Runs locally
        </Badge>
      </div>

      <Stepper steps={["Add audio", "Review", "Results"]} activeStep={currentStep} />

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-start">
        {/* Main Upload / Record Card */}
        <div className="xl:col-span-8 space-y-6">
          <Card className="p-6 space-y-5">
            <div className="flex items-center justify-between flex-wrap gap-3">
              <Tabs
                value={tab}
                onChange={setTab}
                layoutId="capture-source-tab"
                items={[
                  { id: "upload", label: "Upload file", icon: <Upload size={14} strokeWidth={1.5} /> },
                  { id: "mic", label: "Record mic", icon: <Mic size={14} strokeWidth={1.5} /> },
                  { id: "samples", label: "Sample sounds", icon: <Database size={14} strokeWidth={1.5} /> },
                ]}
              />

              {file && (
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    setFile(null);
                    setMetrics(null);
                    setError("");
                    setExpectedClass(undefined);
                  }}
                >
                  <Trash2 size={14} strokeWidth={1.5} /> Remove
                </Button>
              )}
            </div>

            <AnimatePresence mode="wait">
              {!file ? (
                <motion.div
                  key={tab}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -4 }}
                  transition={{ duration: 0.16, ease: "easeOut" }}
                >
                  {tab === "upload" && (
                    <>
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept=".wav,.mp3,audio/wav,audio/mpeg"
                        hidden
                        onChange={(e: ChangeEvent<HTMLInputElement>) => {
                          const f = e.target.files?.[0];
                          if (f) loadAudioFile(f, "upload");
                        }}
                      />
                      <Dropzone
                        dragging={dragging}
                        onDragOver={(e) => {
                          e.preventDefault();
                          setDragging(true);
                        }}
                        onDragLeave={() => setDragging(false)}
                        onDrop={handleDrop}
                        onClick={() => fileInputRef.current?.click()}
                        title="Drop your engine recording here"
                        subtitle="Drag and drop a WAV or MP3 file, or"
                        hint="WAV or MP3 · Up to 25 MB"
                      />
                    </>
                  )}

                  {tab === "mic" && (
                    <LiveMicRecorder
                      onCaptured={(wavFile) => {
                        loadAudioFile(wavFile, "mic");
                      }}
                    />
                  )}

                  {tab === "samples" && (
                    <div className="space-y-4">
                      <p className="text-[13px] text-[#64748B]">
                        Try a real car engine recording to see how the check works.
                      </p>

                      {curatedSamples.length === 0 ? (
                        <EmptyState
                          icon={<Database size={20} strokeWidth={1.5} />}
                          title="No sample clips available"
                          description="Upload your own WAV or MP3 clip, or record directly with your microphone."
                        />
                      ) : (
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                          {curatedSamples.map((s, idx) => (
                            <button
                              key={s.id}
                              type="button"
                              disabled={loadingSampleId === s.id}
                              onClick={() => handleSelectSample(s)}
                              className="rounded-[10px] border border-[#E7E5E4] hover:border-[#4F46E5]/60 bg-[#FAFAF9] hover:bg-[#EEF2FF]/35 p-4 text-left transition-all flex items-center justify-between gap-3"
                            >
                              <div className="min-w-0">
                                <div className="text-[14px] font-semibold text-[#0F172A]">
                                  {predictionToLabel(s.expected_class)} #{(idx % 2) + 1}
                                </div>
                                <div className="text-[12px] text-[#64748B] truncate mt-0.5">
                                  {s.filename}
                                </div>
                              </div>
                              <span className="shrink-0 h-8 w-8 rounded-full bg-white border border-[#E7E5E4] flex items-center justify-center text-[#4F46E5]">
                                <Play size={13} strokeWidth={1.5} className="ml-0.5" />
                              </span>
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </motion.div>
              ) : (
                <motion.div
                  key="loaded-file"
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.18, ease: "easeOut" }}
                  className="space-y-4"
                >
                  <div className="flex items-center justify-between flex-wrap gap-3 pb-3 border-b border-[#E7E5E4]">
                    <div className="flex items-center gap-3 min-w-0">
                      <div className="h-10 w-10 rounded-[8px] bg-[#EEF2FF] text-[#4F46E5] flex items-center justify-center shrink-0">
                        <FileAudio size={18} strokeWidth={1.5} />
                      </div>
                      <div className="min-w-0">
                        <div className="text-[14px] font-semibold text-[#0F172A] truncate">
                          {file.name}
                        </div>
                        <div className="text-[12px] text-[#64748B] font-mono-data">
                          {(file.size / 1024).toFixed(0)} KB
                        </div>
                      </div>
                    </div>

                    <Badge tone={decoding ? "accent" : error ? "fault" : "healthy"} dot>
                      {decoding ? "Loading" : error ? "Check audio" : "Ready"}
                    </Badge>
                  </div>

                  <SpectrogramWorkbench
                    file={file}
                    metrics={metrics}
                    colormap={colormap}
                    showSpectrogram={false}
                  />
                </motion.div>
              )}
            </AnimatePresence>

            {error && (
              <div className="rounded-[8px] border border-[#FECACA] bg-[#FEF2F2] px-3.5 py-2.5 flex items-center gap-2 text-[13px] text-[#DC2626]">
                <AlertCircle size={16} strokeWidth={1.5} className="shrink-0" />
                <span>{error}</span>
              </div>
            )}

            <div className="flex items-center justify-end pt-2 border-t border-[#E7E5E4]">
              <Button
                type="button"
                size="lg"
                disabled={!canAnalyze}
                onClick={() => file && onRunDiagnosis(file, "dual", metrics, source, expectedClass)}
              >
                <Activity size={16} strokeWidth={1.5} />
                Check engine
              </Button>
            </div>
          </Card>
        </div>

        {/* Calm Recording Tips */}
        <aside className="xl:col-span-4">
          <Card className="p-6 space-y-5">
            <div>
              <h3 className="text-[16px] font-semibold text-[#0F172A]">Tips for a clear recording</h3>
              <p className="text-[13px] text-[#64748B] mt-1">
                Follow these simple steps for the most accurate result.
              </p>
            </div>

            <ol className="space-y-4">
              {[
                {
                  title: "Warm up your car",
                  desc: "Let the engine run for a few minutes before recording.",
                },
                {
                  title: "Hold 1–2 feet away",
                  desc: "Point your phone near the open hood, away from wind or traffic.",
                },
                {
                  title: "Record 10–15 seconds",
                  desc: "Keep the engine idling steadily while you record.",
                },
                {
                  title: "Keep your hand steady",
                  desc: "Don't rest your phone directly on vibrating engine parts.",
                },
              ].map((tip, idx) => (
                <li key={tip.title} className="flex items-start gap-3">
                  <span className="h-6 w-6 rounded-full bg-[#EEF2FF] text-[#4F46E5] font-mono-data text-[12px] font-semibold flex items-center justify-center shrink-0 mt-0.5">
                    {idx + 1}
                  </span>
                  <div>
                    <div className="text-[14px] font-medium text-[#0F172A]">{tip.title}</div>
                    <p className="text-[13px] text-[#64748B] mt-0.5">{tip.desc}</p>
                  </div>
                </li>
              ))}
            </ol>
          </Card>
        </aside>
      </div>
    </div>
  );
}
