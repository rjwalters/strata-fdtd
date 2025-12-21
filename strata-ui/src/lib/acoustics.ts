/**
 * Acoustic analysis utilities for room impulse response and acoustic metrics.
 *
 * Computes impulse response from transfer function via inverse FFT,
 * and derives standard acoustic metrics like RT60, clarity, and definition.
 *
 * References:
 * - ISO 3382-1:2009 - Measurement of room acoustic parameters
 * - https://en.wikipedia.org/wiki/Reverberation_time
 */

import { nextPowerOf2 } from "./fft";

// =============================================================================
// Types
// =============================================================================

/**
 * Result of impulse response analysis.
 */
export interface ImpulseResponseResult {
  /** Time-domain impulse response samples */
  impulseResponse: Float32Array;
  /** Time axis in seconds */
  timeAxis: Float32Array;
  /** Sample rate in Hz */
  sampleRate: number;
}

/**
 * Result of Schroeder backward integration (energy decay curve).
 */
export interface EnergyDecayResult {
  /** Energy decay curve in dB (Schroeder curve) */
  decayCurve: Float32Array;
  /** Time axis in seconds */
  timeAxis: Float32Array;
}

/**
 * Standard room acoustic metrics per ISO 3382-1.
 */
export interface AcousticMetrics {
  /** RT60 using T20 method (extrapolated from -5 to -25 dB) */
  t20: number;
  /** RT60 using T30 method (extrapolated from -5 to -35 dB) */
  t30: number;
  /** Early Decay Time (extrapolated from 0 to -10 dB) */
  edt: number;
  /** Clarity C50 in dB (early to late energy ratio at 50ms) */
  c50: number;
  /** Clarity C80 in dB (early to late energy ratio at 80ms) */
  c80: number;
  /** Definition D50 as percentage (early energy fraction at 50ms) */
  d50: number;
  /** Definition D80 as percentage (early energy fraction at 80ms) */
  d80: number;
  /** Centre Time Ts in ms (first moment of squared impulse response) */
  ts: number;
}

/**
 * Window function types for impulse response computation.
 */
export type WindowType = "none" | "hanning" | "tukey";

// =============================================================================
// Inverse FFT
// =============================================================================

/**
 * Bit-reverse an integer for FFT indexing.
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
 * Compute inverse FFT of complex data.
 * Input: interleaved [re0, im0, re1, im1, ...] complex array
 * Output: interleaved complex array of same format
 *
 * Uses the Cooley-Tukey radix-2 algorithm with conjugate trick:
 * IFFT(X) = conj(FFT(conj(X))) / N
 */
export function inverseFFT(complexData: Float32Array): Float32Array {
  const n = complexData.length / 2;

  if (n === 0 || (n & (n - 1)) !== 0) {
    throw new Error("Input length must be a power of 2");
  }

  const bits = Math.log2(n);

  // Separate into real and imaginary, with conjugate (negate imag)
  const re = new Float32Array(n);
  const im = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    const j = bitReverse(i, bits);
    re[j] = complexData[2 * i];
    im[j] = -complexData[2 * i + 1]; // Conjugate
  }

  // Forward FFT (Cooley-Tukey)
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

  // Conjugate again and normalize by N
  const output = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    output[2 * i] = re[i] / n;
    output[2 * i + 1] = -im[i] / n;
  }

  return output;
}

// =============================================================================
// Impulse Response Computation
// =============================================================================

/**
 * Apply a window function to a complex spectrum to reduce artifacts.
 * @param spectrum - Interleaved complex spectrum [re0, im0, re1, im1, ...]
 * @param windowType - Type of window to apply
 * @returns Windowed spectrum
 */
export function applySpectralWindow(
  spectrum: Float32Array,
  windowType: WindowType = "tukey"
): Float32Array {
  const n = spectrum.length / 2;
  const windowed = new Float32Array(spectrum.length);

  for (let i = 0; i < n; i++) {
    let w = 1.0;

    switch (windowType) {
      case "hanning":
        w = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
        break;
      case "tukey": {
        // Tukey window with alpha=0.5 (cosine-tapered)
        const alpha = 0.5;
        const x = i / (n - 1);
        if (x < alpha / 2) {
          w = 0.5 * (1 + Math.cos((2 * Math.PI / alpha) * (x - alpha / 2)));
        } else if (x > 1 - alpha / 2) {
          w = 0.5 * (1 + Math.cos((2 * Math.PI / alpha) * (x - 1 + alpha / 2)));
        }
        // else w = 1 (flat middle section)
        break;
      }
      case "none":
      default:
        w = 1.0;
    }

    windowed[2 * i] = spectrum[2 * i] * w;
    windowed[2 * i + 1] = spectrum[2 * i + 1] * w;
  }

  return windowed;
}

/**
 * Compute room impulse response from a complex transfer function.
 *
 * The impulse response h(t) is the inverse FFT of the transfer function H(f).
 * This reveals how the room "rings" after an impulse.
 *
 * @param transferReal - Real part of transfer function (positive frequencies)
 * @param transferImag - Imaginary part of transfer function (positive frequencies)
 * @param sampleRate - Sample rate in Hz
 * @param windowType - Window to apply before IFFT to reduce artifacts
 * @returns Impulse response result with time axis
 */
export function computeImpulseResponse(
  transferReal: Float32Array,
  transferImag: Float32Array,
  sampleRate: number,
  windowType: WindowType = "tukey"
): ImpulseResponseResult {
  const nPositive = transferReal.length;
  // Full FFT size is 2x the positive frequencies (Nyquist mirroring)
  const nfft = nPositive * 2;
  const n = nextPowerOf2(nfft);

  // Build full complex spectrum with conjugate symmetry for real IFFT output
  // [DC, f1, f2, ..., f(N/2-1), Nyquist, f(N/2-1)*, ..., f1*]
  const fullSpectrum = new Float32Array(n * 2);

  // DC component (index 0)
  fullSpectrum[0] = transferReal[0];
  fullSpectrum[1] = transferImag[0];

  // Positive frequencies (indices 1 to N/2-1)
  for (let i = 1; i < nPositive && i < n / 2; i++) {
    fullSpectrum[2 * i] = transferReal[i];
    fullSpectrum[2 * i + 1] = transferImag[i];

    // Mirror to negative frequencies with conjugate
    const mirrorIdx = n - i;
    fullSpectrum[2 * mirrorIdx] = transferReal[i];
    fullSpectrum[2 * mirrorIdx + 1] = -transferImag[i];
  }

  // Nyquist component (index N/2) - real only
  if (nPositive >= n / 2) {
    const nyquistIdx = n / 2;
    fullSpectrum[2 * nyquistIdx] = transferReal[nyquistIdx];
    fullSpectrum[2 * nyquistIdx + 1] = 0; // Nyquist must be real
  }

  // Apply window to reduce edge artifacts
  const windowed = applySpectralWindow(fullSpectrum, windowType);

  // Compute inverse FFT
  const ifftResult = inverseFFT(windowed);

  // Extract real part as impulse response
  const impulseResponse = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    impulseResponse[i] = ifftResult[2 * i];
  }

  // Create time axis
  const timeAxis = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    timeAxis[i] = i / sampleRate;
  }

  return {
    impulseResponse,
    timeAxis,
    sampleRate,
  };
}

// =============================================================================
// Energy Decay Curve (Schroeder Integration)
// =============================================================================

/**
 * Compute the energy decay curve using Schroeder backward integration.
 *
 * The Schroeder curve is the cumulative sum of squared impulse response
 * from the end to the beginning, converted to dB:
 *   EDC(t) = 10 * log10( ∫_t^∞ h²(τ) dτ / ∫_0^∞ h²(τ) dτ )
 *
 * This provides a smooth decay curve for accurate RT60 estimation.
 *
 * @param impulseResponse - Time-domain impulse response
 * @param sampleRate - Sample rate in Hz
 * @returns Energy decay curve in dB with time axis
 */
export function schroederIntegration(
  impulseResponse: Float32Array,
  sampleRate: number
): EnergyDecayResult {
  const n = impulseResponse.length;

  // Compute squared impulse response (energy)
  const energy = new Float32Array(n);
  let totalEnergy = 0;
  for (let i = 0; i < n; i++) {
    energy[i] = impulseResponse[i] * impulseResponse[i];
    totalEnergy += energy[i];
  }

  // Backward integration (cumulative sum from end)
  const decayCurve = new Float32Array(n);
  let cumulative = totalEnergy;
  const epsilon = 1e-10;

  // Handle zero-energy case
  if (totalEnergy < epsilon) {
    decayCurve.fill(-100); // Very low dB value for silence
    const timeAxis = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      timeAxis[i] = i / sampleRate;
    }
    return { decayCurve, timeAxis };
  }

  for (let i = 0; i < n; i++) {
    decayCurve[i] = 10 * Math.log10(Math.max(cumulative / totalEnergy, epsilon));
    cumulative -= energy[i];
  }

  // Create time axis
  const timeAxis = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    timeAxis[i] = i / sampleRate;
  }

  return {
    decayCurve,
    timeAxis,
  };
}

// =============================================================================
// Reverberation Time Estimation
// =============================================================================

/**
 * Find the index where decay curve first crosses a threshold.
 * Uses linear interpolation for sub-sample accuracy.
 */
function findDecayPoint(
  decayCurve: Float32Array,
  threshold: number
): number {
  for (let i = 0; i < decayCurve.length - 1; i++) {
    if (decayCurve[i] >= threshold && decayCurve[i + 1] < threshold) {
      // Linear interpolation
      const t = (threshold - decayCurve[i]) / (decayCurve[i + 1] - decayCurve[i]);
      return i + t;
    }
  }
  return NaN;
}

/**
 * Estimate reverberation time using linear regression on decay curve.
 *
 * @param decayCurve - Energy decay curve in dB
 * @param sampleRate - Sample rate in Hz
 * @param startDb - Start of regression range in dB (e.g., -5)
 * @param endDb - End of regression range in dB (e.g., -25 for T20, -35 for T30)
 * @returns Estimated RT60 in seconds, or NaN if estimation failed
 */
export function estimateRT60(
  decayCurve: Float32Array,
  sampleRate: number,
  startDb: number = -5,
  endDb: number = -35
): number {
  // Find start and end points
  const startIdx = findDecayPoint(decayCurve, startDb);
  const endIdx = findDecayPoint(decayCurve, endDb);

  if (isNaN(startIdx) || isNaN(endIdx) || endIdx <= startIdx) {
    return NaN;
  }

  // Linear regression on the decay range
  // We fit: decay(t) = a * t + b
  // Then RT60 = -60 / a

  const startSample = Math.floor(startIdx);
  const endSample = Math.ceil(endIdx);
  const n = endSample - startSample + 1;

  if (n < 2) return NaN;

  let sumX = 0;
  let sumY = 0;
  let sumXY = 0;
  let sumXX = 0;

  for (let i = startSample; i <= endSample; i++) {
    const x = i / sampleRate; // time in seconds
    const y = decayCurve[i];
    sumX += x;
    sumY += y;
    sumXY += x * y;
    sumXX += x * x;
  }

  const denom = n * sumXX - sumX * sumX;
  if (Math.abs(denom) < 1e-10) return NaN;

  // Slope of the regression line (dB/s)
  const slope = (n * sumXY - sumX * sumY) / denom;

  // RT60 = -60 / slope
  if (slope >= 0) return NaN; // Decay should have negative slope

  return -60 / slope;
}

/**
 * Estimate Early Decay Time (EDT).
 * EDT is the time for the first 10 dB of decay, extrapolated to 60 dB.
 *
 * @param decayCurve - Energy decay curve in dB
 * @param sampleRate - Sample rate in Hz
 * @returns EDT in seconds, or NaN if estimation failed
 */
export function estimateEDT(
  decayCurve: Float32Array,
  sampleRate: number
): number {
  return estimateRT60(decayCurve, sampleRate, 0, -10);
}

// =============================================================================
// Clarity and Definition
// =============================================================================

/**
 * Compute clarity ratio C_t (early to late energy ratio).
 *
 * C_t = 10 * log10( ∫_0^t h²(τ) dτ / ∫_t^∞ h²(τ) dτ )
 *
 * @param impulseResponse - Time-domain impulse response
 * @param sampleRate - Sample rate in Hz
 * @param boundaryMs - Time boundary in milliseconds (50 for C50, 80 for C80)
 * @returns Clarity in dB
 */
export function computeClarity(
  impulseResponse: Float32Array,
  sampleRate: number,
  boundaryMs: number
): number {
  const boundarySample = Math.round((boundaryMs / 1000) * sampleRate);
  const n = impulseResponse.length;

  let earlyEnergy = 0;
  let lateEnergy = 0;

  for (let i = 0; i < n; i++) {
    const energy = impulseResponse[i] * impulseResponse[i];
    if (i < boundarySample) {
      earlyEnergy += energy;
    } else {
      lateEnergy += energy;
    }
  }

  const epsilon = 1e-10;
  return 10 * Math.log10((earlyEnergy + epsilon) / (lateEnergy + epsilon));
}

/**
 * Compute definition D_t (early energy fraction).
 *
 * D_t = ∫_0^t h²(τ) dτ / ∫_0^∞ h²(τ) dτ
 *
 * @param impulseResponse - Time-domain impulse response
 * @param sampleRate - Sample rate in Hz
 * @param boundaryMs - Time boundary in milliseconds (50 for D50, 80 for D80)
 * @returns Definition as a value between 0 and 1
 */
export function computeDefinition(
  impulseResponse: Float32Array,
  sampleRate: number,
  boundaryMs: number
): number {
  const boundarySample = Math.round((boundaryMs / 1000) * sampleRate);
  const n = impulseResponse.length;

  let earlyEnergy = 0;
  let totalEnergy = 0;

  for (let i = 0; i < n; i++) {
    const energy = impulseResponse[i] * impulseResponse[i];
    totalEnergy += energy;
    if (i < boundarySample) {
      earlyEnergy += energy;
    }
  }

  if (totalEnergy < 1e-10) return 0;
  return earlyEnergy / totalEnergy;
}

/**
 * Compute centre time Ts (temporal centroid).
 *
 * Ts = ∫_0^∞ t * h²(t) dt / ∫_0^∞ h²(t) dt
 *
 * @param impulseResponse - Time-domain impulse response
 * @param sampleRate - Sample rate in Hz
 * @returns Centre time in seconds
 */
export function computeCentreTime(
  impulseResponse: Float32Array,
  sampleRate: number
): number {
  const n = impulseResponse.length;

  let numerator = 0;
  let denominator = 0;

  for (let i = 0; i < n; i++) {
    const t = i / sampleRate;
    const energy = impulseResponse[i] * impulseResponse[i];
    numerator += t * energy;
    denominator += energy;
  }

  if (denominator < 1e-10) return 0;
  return numerator / denominator;
}

// =============================================================================
// Complete Acoustic Analysis
// =============================================================================

/**
 * Compute all standard acoustic metrics from an impulse response.
 *
 * @param impulseResponse - Time-domain impulse response
 * @param sampleRate - Sample rate in Hz
 * @returns Complete set of acoustic metrics
 */
export function computeAcousticMetrics(
  impulseResponse: Float32Array,
  sampleRate: number
): AcousticMetrics {
  // Compute energy decay curve
  const { decayCurve } = schroederIntegration(impulseResponse, sampleRate);

  // Reverberation times
  const t20 = estimateRT60(decayCurve, sampleRate, -5, -25);
  const t30 = estimateRT60(decayCurve, sampleRate, -5, -35);
  const edt = estimateEDT(decayCurve, sampleRate);

  // Clarity metrics
  const c50 = computeClarity(impulseResponse, sampleRate, 50);
  const c80 = computeClarity(impulseResponse, sampleRate, 80);

  // Definition metrics (as percentages)
  const d50 = computeDefinition(impulseResponse, sampleRate, 50) * 100;
  const d80 = computeDefinition(impulseResponse, sampleRate, 80) * 100;

  // Centre time (in ms for display)
  const ts = computeCentreTime(impulseResponse, sampleRate) * 1000;

  return {
    t20,
    t30,
    edt,
    c50,
    c80,
    d50,
    d80,
    ts,
  };
}

/**
 * Compute impulse response and all acoustic metrics from transfer function.
 *
 * This is a convenience function that combines impulse response computation
 * with acoustic metrics analysis.
 *
 * @param transferReal - Real part of transfer function
 * @param transferImag - Imaginary part of transfer function
 * @param sampleRate - Sample rate in Hz
 * @param windowType - Window to apply before IFFT
 * @returns Impulse response, decay curve, and acoustic metrics
 */
export function analyzeTransferFunction(
  transferReal: Float32Array,
  transferImag: Float32Array,
  sampleRate: number,
  windowType: WindowType = "tukey"
): {
  impulseResponse: ImpulseResponseResult;
  energyDecay: EnergyDecayResult;
  metrics: AcousticMetrics;
} {
  // Compute impulse response
  const impulseResponse = computeImpulseResponse(
    transferReal,
    transferImag,
    sampleRate,
    windowType
  );

  // Compute energy decay curve
  const energyDecay = schroederIntegration(
    impulseResponse.impulseResponse,
    sampleRate
  );

  // Compute all metrics
  const metrics = computeAcousticMetrics(
    impulseResponse.impulseResponse,
    sampleRate
  );

  return {
    impulseResponse,
    energyDecay,
    metrics,
  };
}
