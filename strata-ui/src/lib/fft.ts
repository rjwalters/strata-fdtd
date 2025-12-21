/**
 * Pure TypeScript FFT implementation using Cooley-Tukey radix-2 algorithm.
 * Zero dependencies, properly bundleable.
 */

/**
 * Compute FFT of real-valued input data.
 * Returns complex output as interleaved [re0, im0, re1, im1, ...] array.
 */
export function realFFT(input: number[], n: number): Float32Array {
  // Pad or truncate to size n
  const padded = new Array(n).fill(0);
  for (let i = 0; i < Math.min(input.length, n); i++) {
    padded[i] = input[i];
  }

  // Bit-reversal permutation
  const bits = Math.log2(n);
  const re = new Float32Array(n);
  const im = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    const j = bitReverse(i, bits);
    re[j] = padded[i];
  }

  // Cooley-Tukey FFT
  for (let size = 2; size <= n; size *= 2) {
    const halfSize = size / 2;
    const angleStep = (-2 * Math.PI) / size;

    for (let i = 0; i < n; i += size) {
      for (let j = 0; j < halfSize; j++) {
        const angle = angleStep * j;
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);

        const evenIdx = i + j;
        const oddIdx = i + j + halfSize;

        const tRe = cos * re[oddIdx] - sin * im[oddIdx];
        const tIm = sin * re[oddIdx] + cos * im[oddIdx];

        re[oddIdx] = re[evenIdx] - tRe;
        im[oddIdx] = im[evenIdx] - tIm;
        re[evenIdx] = re[evenIdx] + tRe;
        im[evenIdx] = im[evenIdx] + tIm;
      }
    }
  }

  // Interleave real and imaginary parts
  const out = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    out[2 * i] = re[i];
    out[2 * i + 1] = im[i];
  }

  return out;
}

/**
 * Reverse bits of an integer.
 */
function bitReverse(x: number, bits: number): number {
  let result = 0;
  for (let i = 0; i < bits; i++) {
    result = (result << 1) | (x & 1);
    x >>= 1;
  }
  return result;
}

/**
 * Complex spectrum result including both magnitude and phase information.
 */
export interface ComplexSpectrum {
  frequencies: Float32Array;
  magnitude: Float32Array;
  real: Float32Array;
  imag: Float32Array;
}

/**
 * Compute complex spectrum from real input data.
 * Applies Hanning window and returns complex FFT result for positive frequencies.
 */
export function computeComplexSpectrum(
  data: Float32Array,
  sampleRate: number,
  nfft?: number
): ComplexSpectrum {
  const n = nfft ?? nextPowerOf2(data.length);

  // Apply Hanning window
  const windowed: number[] = new Array(n).fill(0);
  for (let i = 0; i < Math.min(data.length, n); i++) {
    const window = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (data.length - 1)));
    windowed[i] = data[i] * window;
  }

  // Compute FFT
  const fftOut = realFFT(windowed, n);

  // Extract complex spectrum (only positive frequencies up to Nyquist)
  const nyquist = n / 2;
  const frequencies = new Float32Array(nyquist);
  const magnitude = new Float32Array(nyquist);
  const real = new Float32Array(nyquist);
  const imag = new Float32Array(nyquist);

  for (let i = 0; i < nyquist; i++) {
    const re = fftOut[2 * i];
    const im = fftOut[2 * i + 1];
    frequencies[i] = (i * sampleRate) / n;
    real[i] = re / n;
    imag[i] = im / n;
    magnitude[i] = Math.sqrt(re * re + im * im) / n;
  }

  return { frequencies, magnitude, real, imag };
}

/**
 * Compute power spectrum from real input data.
 * Applies Hanning window and returns only positive frequencies.
 */
export function computeSpectrum(
  data: Float32Array,
  sampleRate: number,
  nfft?: number
): { frequencies: Float32Array; magnitude: Float32Array } {
  const result = computeComplexSpectrum(data, sampleRate, nfft);
  return { frequencies: result.frequencies, magnitude: result.magnitude };
}

/**
 * Extract phase from complex transfer function H = Y/X.
 * @param sourceReal - Real part of source spectrum X
 * @param sourceImag - Imaginary part of source spectrum X
 * @param probeReal - Real part of probe spectrum Y
 * @param probeImag - Imaginary part of probe spectrum Y
 * @returns Phase in radians [-π, π]
 */
export function extractTransferPhase(
  sourceReal: Float32Array,
  sourceImag: Float32Array,
  probeReal: Float32Array,
  probeImag: Float32Array
): Float32Array {
  const n = sourceReal.length;
  const phase = new Float32Array(n);
  const epsilon = 1e-10;

  for (let i = 0; i < n; i++) {
    // H = Y/X = (Yr + jYi) / (Xr + jXi)
    // H = (Yr*Xr + Yi*Xi + j(Yi*Xr - Yr*Xi)) / (Xr² + Xi²)
    const xr = sourceReal[i];
    const xi = sourceImag[i];
    const yr = probeReal[i];
    const yi = probeImag[i];

    const denom = xr * xr + xi * xi + epsilon;
    const hr = (yr * xr + yi * xi) / denom;
    const hi = (yi * xr - yr * xi) / denom;

    phase[i] = Math.atan2(hi, hr);
  }

  return phase;
}

/**
 * Unwrap phase to produce continuous phase response.
 * Removes 2π discontinuities.
 * @param phase - Wrapped phase in radians
 * @returns Unwrapped phase in radians
 */
export function unwrapPhase(phase: Float32Array): Float32Array {
  const n = phase.length;
  const unwrapped = new Float32Array(n);
  unwrapped[0] = phase[0];

  for (let i = 1; i < n; i++) {
    let diff = phase[i] - phase[i - 1];
    // Wrap difference to [-π, π]
    while (diff > Math.PI) diff -= 2 * Math.PI;
    while (diff < -Math.PI) diff += 2 * Math.PI;
    unwrapped[i] = unwrapped[i - 1] + diff;
  }

  return unwrapped;
}

/**
 * Compute group delay from phase response.
 * τ(f) = -dφ/dω = -dφ/(2π·df)
 * @param phase - Unwrapped phase in radians
 * @param frequencies - Frequency array in Hz
 * @returns Group delay in seconds
 */
export function computeGroupDelay(
  phase: Float32Array,
  frequencies: Float32Array
): Float32Array {
  const n = phase.length;
  const groupDelay = new Float32Array(n);

  // Central differences for interior points
  for (let i = 1; i < n - 1; i++) {
    const df = frequencies[i + 1] - frequencies[i - 1];
    const dPhase = phase[i + 1] - phase[i - 1];
    // τ = -dφ/(2π·df)
    groupDelay[i] = -dPhase / (2 * Math.PI * df);
  }

  // Handle endpoints with forward/backward differences
  if (n > 1) {
    const df0 = frequencies[1] - frequencies[0];
    const dPhase0 = phase[1] - phase[0];
    groupDelay[0] = df0 > 0 ? -dPhase0 / (2 * Math.PI * df0) : 0;

    const dfN = frequencies[n - 1] - frequencies[n - 2];
    const dPhaseN = phase[n - 1] - phase[n - 2];
    groupDelay[n - 1] = dfN > 0 ? -dPhaseN / (2 * Math.PI * dfN) : 0;
  }

  return groupDelay;
}

/**
 * Find the next power of 2 >= n.
 */
export function nextPowerOf2(n: number): number {
  return Math.pow(2, Math.ceil(Math.log2(n)));
}

// =============================================================================
// Welch's Method and Coherence Analysis
// =============================================================================

/**
 * Result of coherence analysis between two signals.
 */
export interface CoherenceResult {
  /** Frequency bins in Hz */
  frequencies: Float32Array;
  /** Coherence values (0-1) at each frequency */
  coherence: Float32Array;
  /** Transfer function magnitude at each frequency */
  transferMagnitude: Float32Array;
  /** Transfer function phase in radians at each frequency */
  transferPhase: Float32Array;
}

/**
 * Apply Hanning window to a segment of data.
 * @param data - Input data segment
 * @param output - Output array (same length as data)
 */
function applyHanningWindow(data: Float32Array, output: Float32Array): void {
  const n = data.length;
  for (let i = 0; i < n; i++) {
    const window = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
    output[i] = data[i] * window;
  }
}

/**
 * Compute FFT of real-valued Float32Array input.
 * Returns complex output as interleaved [re0, im0, re1, im1, ...] array.
 */
function realFFTFloat32(input: Float32Array, n: number): Float32Array {
  // Pad or truncate to size n
  const padded = new Float32Array(n);
  for (let i = 0; i < Math.min(input.length, n); i++) {
    padded[i] = input[i];
  }

  // Bit-reversal permutation
  const bits = Math.log2(n);
  const re = new Float32Array(n);
  const im = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    const j = bitReverse(i, bits);
    re[j] = padded[i];
  }

  // Cooley-Tukey FFT
  for (let size = 2; size <= n; size *= 2) {
    const halfSize = size / 2;
    const angleStep = (-2 * Math.PI) / size;

    for (let i = 0; i < n; i += size) {
      for (let j = 0; j < halfSize; j++) {
        const angle = angleStep * j;
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);

        const evenIdx = i + j;
        const oddIdx = i + j + halfSize;

        const tRe = cos * re[oddIdx] - sin * im[oddIdx];
        const tIm = sin * re[oddIdx] + cos * im[oddIdx];

        re[oddIdx] = re[evenIdx] - tRe;
        im[oddIdx] = im[evenIdx] - tIm;
        re[evenIdx] = re[evenIdx] + tRe;
        im[evenIdx] = im[evenIdx] + tIm;
      }
    }
  }

  // Interleave real and imaginary parts
  const out = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    out[2 * i] = re[i];
    out[2 * i + 1] = im[i];
  }

  return out;
}

/**
 * Compute coherence function between two signals using Welch's method.
 *
 * Coherence γ²(f) measures the linear relationship between input and output:
 * γ²(f) = |Sxy(f)|² / (Sxx(f) * Syy(f))
 *
 * Where:
 * - Sxy(f) = Cross-spectral density between x and y
 * - Sxx(f) = Power spectral density of x (reference/input)
 * - Syy(f) = Power spectral density of y (measurement/output)
 *
 * @param reference - Reference signal (input/source) as Float32Array
 * @param measurement - Measurement signal (output/probe) as Float32Array
 * @param sampleRate - Sample rate in Hz
 * @param segmentSize - Size of each segment for Welch's method (default: 4096)
 * @param overlap - Overlap ratio between segments (default: 0.5 = 50%)
 * @returns Coherence result with frequencies, coherence, and transfer function
 */
export function computeCoherence(
  reference: Float32Array,
  measurement: Float32Array,
  sampleRate: number,
  segmentSize: number = 4096,
  overlap: number = 0.5
): CoherenceResult {
  // Ensure segment size is power of 2
  const nfft = nextPowerOf2(segmentSize);
  const hopSize = Math.floor(nfft * (1 - overlap));

  // Determine number of segments (use shorter signal length)
  const signalLength = Math.min(reference.length, measurement.length);
  const numSegments = Math.max(1, Math.floor((signalLength - nfft) / hopSize) + 1);

  // Number of positive frequencies (up to Nyquist)
  const numFreqs = nfft / 2;

  // Accumulators for spectral estimates (complex for Sxy, real for Sxx/Syy)
  const SxxSum = new Float32Array(numFreqs);
  const SyySum = new Float32Array(numFreqs);
  const SxyRealSum = new Float32Array(numFreqs);
  const SxyImagSum = new Float32Array(numFreqs);

  // Temporary arrays for windowed segments
  const windowedRef = new Float32Array(nfft);
  const windowedMeas = new Float32Array(nfft);

  // Process each segment
  for (let seg = 0; seg < numSegments; seg++) {
    const start = seg * hopSize;

    // Extract and window segments
    const refSegment = reference.subarray(start, start + nfft);
    const measSegment = measurement.subarray(start, start + nfft);

    // Apply Hanning window
    applyHanningWindow(refSegment, windowedRef);
    applyHanningWindow(measSegment, windowedMeas);

    // Compute FFTs
    const refFFT = realFFTFloat32(windowedRef, nfft);
    const measFFT = realFFTFloat32(windowedMeas, nfft);

    // Accumulate spectral estimates for positive frequencies
    for (let i = 0; i < numFreqs; i++) {
      const refRe = refFFT[2 * i];
      const refIm = refFFT[2 * i + 1];
      const measRe = measFFT[2 * i];
      const measIm = measFFT[2 * i + 1];

      // Sxx = |X|² = X * conj(X)
      SxxSum[i] += refRe * refRe + refIm * refIm;

      // Syy = |Y|² = Y * conj(Y)
      SyySum[i] += measRe * measRe + measIm * measIm;

      // Sxy = Y * conj(X) = (measRe + j*measIm) * (refRe - j*refIm)
      // Real part: measRe*refRe + measIm*refIm
      // Imag part: measIm*refRe - measRe*refIm
      SxyRealSum[i] += measRe * refRe + measIm * refIm;
      SxyImagSum[i] += measIm * refRe - measRe * refIm;
    }
  }

  // Compute final results
  const frequencies = new Float32Array(numFreqs);
  const coherence = new Float32Array(numFreqs);
  const transferMagnitude = new Float32Array(numFreqs);
  const transferPhase = new Float32Array(numFreqs);

  const epsilon = 1e-10; // Small value to avoid division by zero

  for (let i = 0; i < numFreqs; i++) {
    frequencies[i] = (i * sampleRate) / nfft;

    const Sxx = SxxSum[i] / numSegments;
    const Syy = SyySum[i] / numSegments;
    const SxyReal = SxyRealSum[i] / numSegments;
    const SxyImag = SxyImagSum[i] / numSegments;

    // |Sxy|² = SxyReal² + SxyImag²
    const SxyMagSq = SxyReal * SxyReal + SxyImag * SxyImag;

    // γ²(f) = |Sxy|² / (Sxx * Syy)
    const denominator = Sxx * Syy;
    if (denominator > epsilon) {
      coherence[i] = SxyMagSq / denominator;
      // Clamp to [0, 1] to handle numerical errors
      coherence[i] = Math.max(0, Math.min(1, coherence[i]));
    } else {
      coherence[i] = 0;
    }

    // Transfer function H(f) = Sxy / Sxx
    if (Sxx > epsilon) {
      const HReal = SxyReal / Sxx;
      const HImag = SxyImag / Sxx;
      transferMagnitude[i] = Math.sqrt(HReal * HReal + HImag * HImag);
      transferPhase[i] = Math.atan2(HImag, HReal);
    } else {
      transferMagnitude[i] = 0;
      transferPhase[i] = 0;
    }
  }

  return {
    frequencies,
    coherence,
    transferMagnitude,
    transferPhase,
  };
}

/**
 * Compute transfer function magnitude in decibels.
 * @param transferMagnitude - Linear magnitude from computeCoherence
 * @returns Magnitude in dB (20 * log10(magnitude))
 */
export function transferMagnitudeToDb(
  transferMagnitude: Float32Array
): Float32Array {
  const db = new Float32Array(transferMagnitude.length);
  for (let i = 0; i < transferMagnitude.length; i++) {
    db[i] = 20 * Math.log10(Math.max(transferMagnitude[i], 1e-10));
  }
  return db;
}

// =============================================================================
// Web Worker Support
// =============================================================================

import type {
  FFTWorkerRequest,
  FFTWorkerMessage,
} from "../workers/fft.worker";

let fftWorker: Worker | null = null;
let requestId = 0;
const pendingRequests = new Map<
  number,
  {
    resolve: (value: { frequencies: Float32Array; magnitude: Float32Array }) => void;
    reject: (error: Error) => void;
  }
>();

/**
 * Check if Web Workers are available in the current environment.
 */
export function hasWorkerSupport(): boolean {
  return typeof Worker !== "undefined";
}

/**
 * Get or create the FFT worker instance.
 */
function getFFTWorker(): Worker | null {
  if (!hasWorkerSupport()) {
    return null;
  }

  if (!fftWorker) {
    try {
      fftWorker = new Worker(
        new URL("../workers/fft.worker.ts", import.meta.url),
        { type: "module" }
      );

      fftWorker.onmessage = (event: MessageEvent<FFTWorkerMessage>) => {
        const message = event.data;
        const pending = pendingRequests.get(message.id);

        if (pending) {
          pendingRequests.delete(message.id);

          if (message.type === "error") {
            pending.reject(new Error(message.error));
          } else {
            pending.resolve({
              frequencies: message.frequencies,
              magnitude: message.magnitude,
            });
          }
        }
      };

      fftWorker.onerror = (error) => {
        console.error("FFT Worker error:", error);
        // Reject all pending requests
        for (const [id, pending] of pendingRequests) {
          pending.reject(new Error("Worker error"));
          pendingRequests.delete(id);
        }
        // Reset worker so it can be recreated
        fftWorker = null;
      };
    } catch {
      console.warn("Failed to create FFT worker, falling back to main thread");
      return null;
    }
  }

  return fftWorker;
}

/**
 * Compute spectrum asynchronously using a Web Worker.
 * Falls back to synchronous computation if workers are unavailable.
 *
 * @param data - Input audio samples
 * @param sampleRate - Sample rate in Hz
 * @param nfft - Optional FFT size (defaults to next power of 2)
 * @returns Promise resolving to { frequencies, magnitude }
 */
export async function computeSpectrumAsync(
  data: Float32Array,
  sampleRate: number,
  nfft?: number
): Promise<{ frequencies: Float32Array; magnitude: Float32Array }> {
  const worker = getFFTWorker();

  // Fallback to synchronous computation
  if (!worker) {
    return computeSpectrum(data, sampleRate, nfft);
  }

  const id = requestId++;

  return new Promise((resolve, reject) => {
    pendingRequests.set(id, { resolve, reject });

    const request: FFTWorkerRequest = {
      type: "computeSpectrum",
      id,
      data,
      sampleRate,
      nfft,
    };

    // Transfer the data buffer if possible (for large datasets)
    // Note: This transfers ownership, so the caller shouldn't use data after this
    if (data.length > 100000) {
      // Make a copy since we're transferring
      const dataCopy = new Float32Array(data);
      worker.postMessage(
        { ...request, data: dataCopy },
        { transfer: [dataCopy.buffer] }
      );
    } else {
      worker.postMessage(request);
    }
  });
}

/**
 * Terminate the FFT worker and clean up resources.
 * Call this when the worker is no longer needed.
 */
export function terminateFFTWorker(): void {
  if (fftWorker) {
    fftWorker.terminate();
    fftWorker = null;
    pendingRequests.clear();
  }
}
