/**
 * Unit tests for chart component data transformations.
 *
 * Tests the FFT computation, peak detection, and downsampling algorithms
 * used by TimeSeriesPlot and SpectrumPlot components.
 */

import { describe, it, expect } from "vitest";

/**
 * FFT computation extracted from SpectrumPlot for testability.
 * This mirrors the logic in SpectrumPlot.tsx computeSpectrum().
 */
function nextPowerOf2(n: number): number {
  return Math.pow(2, Math.ceil(Math.log2(n)));
}

function computeSpectrum(
  data: Float32Array,
  sampleRate: number,
  nfft?: number
): { frequencies: Float32Array; magnitude: Float32Array } {
  const n = nfft ?? nextPowerOf2(data.length);

  // Prepare input with Hanning window
  const input: number[] = new Array(n).fill(0);
  for (let i = 0; i < Math.min(data.length, n); i++) {
    const window = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (data.length - 1)));
    input[i] = data[i] * window;
  }

  // Compute DFT (simplified, without FFT library dependency)
  const nyquist = n / 2;
  const frequencies = new Float32Array(nyquist);
  const magnitude = new Float32Array(nyquist);

  for (let k = 0; k < nyquist; k++) {
    let re = 0;
    let im = 0;
    for (let t = 0; t < n; t++) {
      const angle = (2 * Math.PI * k * t) / n;
      re += input[t] * Math.cos(angle);
      im -= input[t] * Math.sin(angle);
    }
    frequencies[k] = (k * sampleRate) / n;
    magnitude[k] = Math.sqrt(re * re + im * im) / n;
  }

  return { frequencies, magnitude };
}

/**
 * Peak detection extracted from SpectrumPlot for testability.
 */
function findPeaks(
  frequencies: Float32Array,
  magnitude: Float32Array,
  threshold: number = 0.1,
  maxPeaks: number = 5
): Array<{ frequency: number; magnitude: number }> {
  const maxMag = Math.max(...magnitude);
  const peaks: Array<{ frequency: number; magnitude: number; index: number }> = [];

  for (let i = 1; i < magnitude.length - 1; i++) {
    if (magnitude[i] > magnitude[i - 1] && magnitude[i] > magnitude[i + 1]) {
      if (magnitude[i] > maxMag * threshold) {
        peaks.push({
          frequency: frequencies[i],
          magnitude: magnitude[i],
          index: i,
        });
      }
    }
  }

  peaks.sort((a, b) => b.magnitude - a.magnitude);
  return peaks.slice(0, maxPeaks).map(({ frequency, magnitude }) => ({
    frequency,
    magnitude,
  }));
}

/**
 * Decibel conversion extracted from SpectrumPlot.
 */
function toDecibels(value: number, ref: number = 1): number {
  return 20 * Math.log10(Math.max(value / ref, 1e-10));
}

/**
 * Downsampling logic extracted from TimeSeriesPlot.
 */
function downsampleData(data: Float32Array, maxPoints: number): number[] {
  if (data.length <= maxPoints) {
    return Array.from(data);
  }

  const step = Math.ceil(data.length / maxPoints);
  const downsampled: number[] = [];
  for (let i = 0; i < data.length; i += step) {
    downsampled.push(data[i]);
  }
  return downsampled;
}

describe("FFT Computation", () => {
  it("returns correct number of frequency bins", () => {
    const data = new Float32Array(1024);
    const sampleRate = 44100;
    const { frequencies, magnitude } = computeSpectrum(data, sampleRate);

    expect(frequencies.length).toBe(512); // nyquist = n/2
    expect(magnitude.length).toBe(512);
  });

  it("frequencies are monotonically increasing", () => {
    const data = new Float32Array(256).fill(0);
    const sampleRate = 1000;
    const { frequencies } = computeSpectrum(data, sampleRate);

    for (let i = 1; i < frequencies.length; i++) {
      expect(frequencies[i]).toBeGreaterThan(frequencies[i - 1]);
    }
  });

  it("frequencies range from 0 to nyquist", () => {
    const data = new Float32Array(256).fill(0);
    const sampleRate = 1000;
    const { frequencies } = computeSpectrum(data, sampleRate);

    expect(frequencies[0]).toBe(0);
    // Last frequency should be close to nyquist
    expect(frequencies[frequencies.length - 1]).toBeLessThan(sampleRate / 2);
  });

  it("detects pure sine wave at correct frequency", () => {
    const sampleRate = 1000;
    const freq = 100; // 100 Hz test tone
    const nSamples = 1024;

    const data = new Float32Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      data[i] = Math.sin((2 * Math.PI * freq * i) / sampleRate);
    }

    const { frequencies, magnitude } = computeSpectrum(data, sampleRate);

    // Find peak
    let maxIdx = 0;
    let maxMag = 0;
    for (let i = 0; i < magnitude.length; i++) {
      if (magnitude[i] > maxMag) {
        maxMag = magnitude[i];
        maxIdx = i;
      }
    }

    // Peak should be near 100 Hz (within resolution)
    const resolution = sampleRate / nSamples;
    expect(Math.abs(frequencies[maxIdx] - freq)).toBeLessThan(resolution * 2);
  });

  it("magnitude values are non-negative", () => {
    const data = new Float32Array(128);
    for (let i = 0; i < data.length; i++) {
      data[i] = Math.random() * 2 - 1;
    }

    const { magnitude } = computeSpectrum(data, 1000);

    for (const m of magnitude) {
      expect(m).toBeGreaterThanOrEqual(0);
    }
  });
});

describe("Peak Detection", () => {
  it("finds peaks in simple spectrum", () => {
    const n = 100;
    const frequencies = new Float32Array(n);
    const magnitude = new Float32Array(n);

    for (let i = 0; i < n; i++) {
      frequencies[i] = i * 10; // 0, 10, 20, ... 990 Hz
      magnitude[i] = 0;
    }

    // Add two peaks
    magnitude[10] = 1.0; // 100 Hz
    magnitude[30] = 0.5; // 300 Hz

    const peaks = findPeaks(frequencies, magnitude, 0.1, 5);

    expect(peaks.length).toBe(2);
    expect(peaks[0].frequency).toBe(100);
    expect(peaks[0].magnitude).toBe(1.0);
    expect(peaks[1].frequency).toBe(300);
  });

  it("respects threshold", () => {
    const n = 100;
    const frequencies = new Float32Array(n);
    const magnitude = new Float32Array(n);

    for (let i = 0; i < n; i++) {
      frequencies[i] = i * 10;
      magnitude[i] = 0;
    }

    magnitude[10] = 1.0;
    magnitude[30] = 0.05; // Below 10% threshold

    const peaks = findPeaks(frequencies, magnitude, 0.1, 5);

    expect(peaks.length).toBe(1);
    expect(peaks[0].frequency).toBe(100);
  });

  it("limits to maxPeaks", () => {
    const n = 100;
    const frequencies = new Float32Array(n);
    const magnitude = new Float32Array(n);

    for (let i = 0; i < n; i++) {
      frequencies[i] = i * 10;
      magnitude[i] = 0;
    }

    // Add 10 peaks
    for (let i = 0; i < 10; i++) {
      magnitude[10 + i * 5] = 1.0 - i * 0.05;
    }

    const peaks = findPeaks(frequencies, magnitude, 0.1, 3);

    expect(peaks.length).toBe(3);
  });

  it("returns empty array for flat spectrum", () => {
    const n = 100;
    const frequencies = new Float32Array(n);
    const magnitude = new Float32Array(n).fill(0.5);

    for (let i = 0; i < n; i++) {
      frequencies[i] = i * 10;
    }

    const peaks = findPeaks(frequencies, magnitude, 0.1, 5);

    expect(peaks.length).toBe(0);
  });
});

describe("Decibel Conversion", () => {
  it("converts 1.0 relative to 1.0 to 0 dB", () => {
    expect(toDecibels(1.0, 1.0)).toBe(0);
  });

  it("converts 0.1 relative to 1.0 to -20 dB", () => {
    expect(toDecibels(0.1, 1.0)).toBeCloseTo(-20, 5);
  });

  it("converts 0.01 relative to 1.0 to -40 dB", () => {
    expect(toDecibels(0.01, 1.0)).toBeCloseTo(-40, 5);
  });

  it("handles very small values without returning -Infinity", () => {
    const result = toDecibels(0, 1.0);
    expect(Number.isFinite(result)).toBe(true);
    expect(result).toBeLessThan(-100);
  });

  it("handles negative values by taking absolute value implicitly", () => {
    // The function takes max(value/ref, 1e-10), so negative becomes clamped
    const result = toDecibels(-0.5, 1.0);
    expect(Number.isFinite(result)).toBe(true);
  });
});

describe("Downsampling", () => {
  it("returns original array when below maxPoints", () => {
    const data = new Float32Array([1, 2, 3, 4, 5]);
    const result = downsampleData(data, 10);

    expect(result).toEqual([1, 2, 3, 4, 5]);
  });

  it("returns original array when equal to maxPoints", () => {
    const data = new Float32Array([1, 2, 3, 4, 5]);
    const result = downsampleData(data, 5);

    expect(result).toEqual([1, 2, 3, 4, 5]);
  });

  it("downsamples when above maxPoints", () => {
    const data = new Float32Array(1000);
    for (let i = 0; i < 1000; i++) {
      data[i] = i;
    }

    const result = downsampleData(data, 100);

    expect(result.length).toBeLessThanOrEqual(100);
    expect(result[0]).toBe(0);
  });

  it("preserves first value", () => {
    const data = new Float32Array(1000);
    data[0] = 42;
    for (let i = 1; i < 1000; i++) {
      data[i] = i;
    }

    const result = downsampleData(data, 50);

    expect(result[0]).toBe(42);
  });

  it("samples at regular intervals", () => {
    const data = new Float32Array(100);
    for (let i = 0; i < 100; i++) {
      data[i] = i;
    }

    const result = downsampleData(data, 10);
    const step = Math.ceil(100 / 10);

    // Check that samples are at expected positions
    for (let i = 0; i < result.length; i++) {
      expect(result[i]).toBe(i * step);
    }
  });
});

describe("Next Power of 2", () => {
  it("returns same value for powers of 2", () => {
    expect(nextPowerOf2(1)).toBe(1);
    expect(nextPowerOf2(2)).toBe(2);
    expect(nextPowerOf2(4)).toBe(4);
    expect(nextPowerOf2(1024)).toBe(1024);
  });

  it("rounds up non-powers of 2", () => {
    expect(nextPowerOf2(3)).toBe(4);
    expect(nextPowerOf2(5)).toBe(8);
    expect(nextPowerOf2(100)).toBe(128);
    expect(nextPowerOf2(1000)).toBe(1024);
  });
});
