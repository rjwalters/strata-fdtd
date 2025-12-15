/**
 * Probe data loader for FDTD simulation time series.
 *
 * Loads JSON files containing pressure recordings from simulation probes.
 * Converts data to Float32Array for efficient processing/visualization.
 */

import type { ProbeData, LoadOptions } from './types';

/**
 * Raw probe data format from Python export.
 */
interface RawProbeJSON {
  sample_rate: number;
  dt: number;
  duration: number;
  n_samples: number;
  probes: {
    [name: string]: {
      position: [number, number, number];
      data: number[];
    };
  };
}

/**
 * Load probe time series data from JSON file.
 *
 * @param url - URL to probe data JSON file
 * @param options - Loading options
 * @returns Probe data with Float32Array time series
 *
 * @example
 * ```typescript
 * const probeData = await loadProbeData('/data/sim_probes.json');
 *
 * // Access a specific probe
 * const refProbe = probeData.probes['reference'];
 * console.log(`Probe at ${refProbe.position}, ${refProbe.data.length} samples`);
 *
 * // Get time array
 * const time = new Float32Array(probeData.nSamples);
 * for (let i = 0; i < probeData.nSamples; i++) {
 *   time[i] = i * probeData.dt;
 * }
 * ```
 */
export async function loadProbeData(
  url: string,
  options: LoadOptions = {}
): Promise<ProbeData> {
  const response = await fetch(url, { signal: options.signal });

  if (!response.ok) {
    throw new Error(`Failed to load probe data: ${response.status} ${response.statusText}`);
  }

  const raw: RawProbeJSON = await response.json();

  // Convert probe data to Float32Arrays
  const probes: ProbeData['probes'] = {};

  for (const [name, probe] of Object.entries(raw.probes)) {
    probes[name] = {
      position: probe.position,
      data: new Float32Array(probe.data),
    };
  }

  return {
    sampleRate: raw.sample_rate,
    dt: raw.dt,
    duration: raw.duration,
    nSamples: raw.n_samples,
    probes,
  };
}

/**
 * Get time array for probe data.
 *
 * @param probeData - Loaded probe data
 * @returns Float32Array of time values in seconds
 */
export function getTimeArray(probeData: ProbeData): Float32Array {
  const time = new Float32Array(probeData.nSamples);
  for (let i = 0; i < probeData.nSamples; i++) {
    time[i] = i * probeData.dt;
  }
  return time;
}

/**
 * Get frequency response for a probe using FFT.
 *
 * @param data - Probe pressure time series
 * @param sampleRate - Sample rate in Hz
 * @returns Object with frequencies and magnitude arrays
 */
export function getFrequencyResponse(
  data: Float32Array,
  sampleRate: number
): { frequencies: Float32Array; magnitude: Float32Array } {
  // Simple DFT for small data sets
  // For production, use a proper FFT library (e.g., fft.js)
  const n = data.length;
  const nyquist = Math.floor(n / 2);

  const frequencies = new Float32Array(nyquist);
  const magnitude = new Float32Array(nyquist);

  for (let k = 0; k < nyquist; k++) {
    let re = 0;
    let im = 0;

    for (let t = 0; t < n; t++) {
      const angle = (2 * Math.PI * k * t) / n;
      re += data[t] * Math.cos(angle);
      im -= data[t] * Math.sin(angle);
    }

    frequencies[k] = (k * sampleRate) / n;
    magnitude[k] = Math.sqrt(re * re + im * im) / n;
  }

  return { frequencies, magnitude };
}

/**
 * Find peak frequency in probe data.
 *
 * @param data - Probe pressure time series
 * @param sampleRate - Sample rate in Hz
 * @returns Peak frequency in Hz
 */
export function findPeakFrequency(
  data: Float32Array,
  sampleRate: number
): number {
  const { frequencies, magnitude } = getFrequencyResponse(data, sampleRate);

  let maxMag = 0;
  let peakFreq = 0;

  for (let i = 1; i < magnitude.length; i++) {
    if (magnitude[i] > maxMag) {
      maxMag = magnitude[i];
      peakFreq = frequencies[i];
    }
  }

  return peakFreq;
}
