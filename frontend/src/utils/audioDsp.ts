import type { AudioAnalysisMetrics } from "../types/diagnostic";

/**
 * Pure TypeScript in-place Radix-2 Cooley-Tukey FFT.
 * `re` and `im` lengths must be a power of 2.
 */
function fftRadix2(re: Float32Array, im: Float32Array): void {
  const n = re.length;
  // Bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      const tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      const ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wlenRe = Math.cos(ang);
    const wlenIm = Math.sin(ang);
    const half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let wRe = 1;
      let wIm = 0;
      for (let j = 0; j < half; j++) {
        const uRe = re[i + j];
        const uIm = im[i + j];
        const vRe = re[i + j + half] * wRe - im[i + j + half] * wIm;
        const vIm = re[i + j + half] * wIm + im[i + j + half] * wRe;
        re[i + j] = uRe + vRe;
        im[i + j] = uIm + vIm;
        re[i + j + half] = uRe - vRe;
        im[i + j + half] = uIm - vIm;
        const nextWRe = wRe * wlenRe - wIm * wlenIm;
        wIm = wRe * wlenIm + wIm * wlenRe;
        wRe = nextWRe;
      }
    }
  }
}

/**
 * Decode an uploaded/recorded audio Blob/File via Web Audio API and extract
 * genuine waveform peaks, STFT spectrogram (0-8000 Hz), and acoustic biomarkers.
 */
export async function decodeAndAnalyzeAudio(file: Blob): Promise<AudioAnalysisMetrics> {
  const arrayBuf = await file.arrayBuffer();
  const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  const ctx = new AudioCtx();

  let audioBuffer: AudioBuffer;
  try {
    audioBuffer = await ctx.decodeAudioData(arrayBuf.slice(0));
  } finally {
    await ctx.close().catch(() => {});
  }

  const durationSec = audioBuffer.duration;
  const sampleRate = audioBuffer.sampleRate;
  const channels = audioBuffer.numberOfChannels;
  const length = audioBuffer.length;

  // Mix down to mono Float32Array
  const mono = new Float32Array(length);
  for (let ch = 0; ch < channels; ch++) {
    const channelData = audioBuffer.getChannelData(ch);
    for (let i = 0; i < length; i++) {
      mono[i] += channelData[i] / channels;
    }
  }

  // Time-domain stats: Peak, RMS, ZCR
  let sumSq = 0;
  let peakAmp = 0;
  let zeroCrossings = 0;
  for (let i = 0; i < length; i++) {
    const s = mono[i];
    const abs = Math.abs(s);
    if (abs > peakAmp) peakAmp = abs;
    sumSq += s * s;
    if (i > 0 && ((mono[i - 1] >= 0 && s < 0) || (mono[i - 1] < 0 && s >= 0))) {
      zeroCrossings++;
    }
  }

  const rms = Math.sqrt(sumSq / Math.max(length, 1));
  const rmsDb = rms > 1e-8 ? 20 * Math.log10(rms) : -96;
  const crestFactorDb = rms > 1e-8 && peakAmp > 1e-8 ? 20 * Math.log10(peakAmp / rms) : 0;
  const zcr = zeroCrossings / Math.max(length - 1, 1);
  const isSilent = peakAmp < 1e-5;
  const tooShort = durationSec < 0.5;

  // Waveform peak envelope (160 bins)
  const WAVE_BINS = 160;
  const waveformPeaks: number[] = new Array(WAVE_BINS).fill(0);
  const samplesPerBin = Math.max(1, Math.floor(length / WAVE_BINS));
  let maxWavePeak = 1e-6;
  for (let b = 0; b < WAVE_BINS; b++) {
    const start = b * samplesPerBin;
    const end = Math.min(length, start + samplesPerBin);
    let binPeak = 0;
    for (let i = start; i < end; i++) {
      const a = Math.abs(mono[i]);
      if (a > binPeak) binPeak = a;
    }
    waveformPeaks[b] = binPeak;
    if (binPeak > maxWavePeak) maxWavePeak = binPeak;
  }
  for (let b = 0; b < WAVE_BINS; b++) {
    waveformPeaks[b] = Math.min(1, waveformPeaks[b] / maxWavePeak);
  }

  // STFT Spectrogram (up to FMAX = 8000 Hz, matching backend ml/features.py)
  const FFT_SIZE = 1024;
  const MAX_FREQ_HZ = Math.min(8000, sampleRate / 2);
  const TIME_COLS = 128;
  const FREQ_ROWS = 72;

  const hann = new Float32Array(FFT_SIZE);
  for (let i = 0; i < FFT_SIZE; i++) {
    hann[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (FFT_SIZE - 1)));
  }

  const nyquistBins = FFT_SIZE / 2;
  const freqPerBin = sampleRate / FFT_SIZE;
  const maxUsefulBin = Math.min(nyquistBins - 1, Math.floor(MAX_FREQ_HZ / freqPerBin));

  const rawSpec: number[][] = [];
  let globalMaxDb = -120;
  const re = new Float32Array(FFT_SIZE);
  const im = new Float32Array(FFT_SIZE);

  // Accumulators for average spectral biomarkers
  let totalEnergy = 0;
  let lowEnergy = 0;
  let midEnergy = 0;
  let highEnergy = 0;
  let centroidNum = 0;
  const avgMagBins = new Float64Array(maxUsefulBin + 1);

  const usableLen = Math.max(1, length - FFT_SIZE);
  for (let col = 0; col < TIME_COLS; col++) {
    const startSample = Math.floor((col / Math.max(TIME_COLS - 1, 1)) * usableLen);
    re.fill(0);
    im.fill(0);
    for (let i = 0; i < FFT_SIZE; i++) {
      const idx = startSample + i;
      re[i] = idx < length ? mono[idx] * hann[i] : 0;
    }
    fftRadix2(re, im);

    const frameDb = new Array<number>(FREQ_ROWS).fill(-90);
    for (let row = 0; row < FREQ_ROWS; row++) {
      // Quasi-log/mel warped frequency mapping for visual fidelity in engine band
      const t0 = row / FREQ_ROWS;
      const t1 = (row + 1) / FREQ_ROWS;
      const f0 = MAX_FREQ_HZ * Math.pow(t0, 1.45);
      const f1 = MAX_FREQ_HZ * Math.pow(t1, 1.45);
      const b0 = Math.max(1, Math.floor(f0 / freqPerBin));
      const b1 = Math.min(maxUsefulBin, Math.max(b0, Math.ceil(f1 / freqPerBin)));
      let powerSum = 0;
      for (let b = b0; b <= b1; b++) {
        powerSum += re[b] * re[b] + im[b] * im[b];
      }
      const avgPower = powerSum / (b1 - b0 + 1);
      const db = 10 * Math.log10(avgPower + 1e-10);
      frameDb[row] = db;
      if (db > globalMaxDb) globalMaxDb = db;
    }
    rawSpec.push(frameDb);

    // Accumulate spectral biomarkers
    for (let b = 1; b <= maxUsefulBin; b++) {
      const pwr = re[b] * re[b] + im[b] * im[b];
      const mag = Math.sqrt(pwr);
      const hz = b * freqPerBin;
      avgMagBins[b] += mag;
      totalEnergy += pwr;
      centroidNum += hz * mag;
      if (hz >= 50 && hz < 500) lowEnergy += pwr;
      else if (hz >= 500 && hz < 2000) midEnergy += pwr;
      else if (hz >= 2000 && hz <= MAX_FREQ_HZ) highEnergy += pwr;
    }
  }

  // Normalize spectrogram to [0..1] over a 65 dB dynamic range
  const minDbFloor = globalMaxDb - 65;
  const spectrogram = rawSpec.map((col) =>
    col.map((db) => Math.max(0, Math.min(1, (db - minDbFloor) / 65)))
  );

  // Peak frequency, centroid, 85% rolloff
  let sumAvgMag = 0;
  let peakBin = 1;
  let peakMag = 0;
  for (let b = 1; b <= maxUsefulBin; b++) {
    sumAvgMag += avgMagBins[b];
    if (avgMagBins[b] > peakMag) {
      peakMag = avgMagBins[b];
      peakBin = b;
    }
  }
  const peakFreqHz = Math.round(peakBin * freqPerBin);
  const spectralCentroidHz = sumAvgMag > 1e-8 ? Math.round(centroidNum / sumAvgMag) : 0;

  let rolloffTarget = sumAvgMag * 0.85;
  let rolloffAccum = 0;
  let rolloffHz = MAX_FREQ_HZ;
  for (let b = 1; b <= maxUsefulBin; b++) {
    rolloffAccum += avgMagBins[b];
    if (rolloffAccum >= rolloffTarget) {
      rolloffHz = Math.round(b * freqPerBin);
      break;
    }
  }

  const bandTotal = Math.max(lowEnergy + midEnergy + highEnergy, 1e-9);
  return {
    durationSec,
    sampleRate,
    channels,
    rmsDb: Number(rmsDb.toFixed(1)),
    peakAmp: Number(peakAmp.toFixed(3)),
    crestFactorDb: Number(crestFactorDb.toFixed(1)),
    zcr: Number(zcr.toFixed(4)),
    spectralCentroidHz,
    spectralRolloffHz: rolloffHz,
    peakFreqHz,
    bandEnergy: {
      lowPct: Math.round((lowEnergy / bandTotal) * 100),
      midPct: Math.round((midEnergy / bandTotal) * 100),
      highPct: Math.round((highEnergy / bandTotal) * 100),
    },
    waveformPeaks,
    spectrogram,
    maxFreqHz: MAX_FREQ_HZ,
    isSilent,
    tooShort,
  };
}

/**
 * Encode raw mono Float32 PCM samples into a standard 16-bit PCM WAV File.
 */
export function encodeWavFromPcm(samples: Float32Array, sampleRate: number, filename: string): File {
  const numSamples = samples.length;
  const buffer = new ArrayBuffer(44 + numSamples * 2);
  const view = new DataView(buffer);

  const writeStr = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + numSamples * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true);  // PCM format
  view.setUint16(22, 1, true);  // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, numSamples * 2, true);

  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += 2;
  }

  return new File([buffer], filename, { type: "audio/wav" });
}
