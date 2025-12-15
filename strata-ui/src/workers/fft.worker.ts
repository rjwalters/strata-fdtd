/**
 * Web Worker for FFT computation.
 *
 * Offloads computationally intensive spectrum analysis to a background thread
 * to keep the main thread responsive during large dataset processing.
 */

import { realFFT, nextPowerOf2 } from "@/lib/fft";

// =============================================================================
// Message Types
// =============================================================================

export interface FFTWorkerRequest {
  type: "computeSpectrum";
  id: number;
  data: Float32Array;
  sampleRate: number;
  nfft?: number;
}

export interface FFTWorkerResponse {
  type: "computeSpectrum";
  id: number;
  frequencies: Float32Array;
  magnitude: Float32Array;
}

export interface FFTWorkerError {
  type: "error";
  id: number;
  error: string;
}

export type FFTWorkerMessage = FFTWorkerResponse | FFTWorkerError;

// =============================================================================
// Worker Logic
// =============================================================================

/**
 * Compute power spectrum from real input data.
 * Applies Hanning window and returns only positive frequencies.
 */
function computeSpectrumWorker(
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

self.onmessage = (event: MessageEvent<FFTWorkerRequest>) => {
  const request = event.data;

  try {
    switch (request.type) {
      case "computeSpectrum": {
        const result = computeSpectrumWorker(
          request.data,
          request.sampleRate,
          request.nfft
        );

        const response: FFTWorkerResponse = {
          type: "computeSpectrum",
          id: request.id,
          frequencies: result.frequencies,
          magnitude: result.magnitude,
        };

        // Transfer ownership of buffers for zero-copy
        self.postMessage(response, {
          transfer: [result.frequencies.buffer, result.magnitude.buffer],
        });
        break;
      }
    }
  } catch (error) {
    const errorResponse: FFTWorkerError = {
      type: "error",
      id: request.id,
      error: error instanceof Error ? error.message : "Unknown error",
    };
    self.postMessage(errorResponse);
  }
};
