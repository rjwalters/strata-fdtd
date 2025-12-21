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
