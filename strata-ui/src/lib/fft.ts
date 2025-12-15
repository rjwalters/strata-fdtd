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
 * Compute power spectrum from real input data.
 * Applies Hanning window and returns only positive frequencies.
 */
export function computeSpectrum(
  data: Float32Array,
  sampleRate: number,
  nfft?: number
): { frequencies: Float32Array; magnitude: Float32Array } {
  const n = nfft ?? nextPowerOf2(data.length);

  // Apply Hanning window
  const windowed: number[] = new Array(n).fill(0);
  for (let i = 0; i < Math.min(data.length, n); i++) {
    const window = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (data.length - 1)));
    windowed[i] = data[i] * window;
  }

  // Compute FFT
  const fftOut = realFFT(windowed, n);

  // Compute magnitude (only positive frequencies up to Nyquist)
  const nyquist = n / 2;
  const frequencies = new Float32Array(nyquist);
  const magnitude = new Float32Array(nyquist);

  for (let i = 0; i < nyquist; i++) {
    const re = fftOut[2 * i];
    const im = fftOut[2 * i + 1];
    frequencies[i] = (i * sampleRate) / n;
    magnitude[i] = Math.sqrt(re * re + im * im) / n;
  }

  return { frequencies, magnitude };
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
