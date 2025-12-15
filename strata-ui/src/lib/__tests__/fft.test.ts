import { describe, it, expect } from "vitest";
import { realFFT, computeSpectrum, nextPowerOf2 } from "../fft";

describe("nextPowerOf2", () => {
  it("returns power of 2 for exact powers", () => {
    expect(nextPowerOf2(1)).toBe(1);
    expect(nextPowerOf2(2)).toBe(2);
    expect(nextPowerOf2(4)).toBe(4);
    expect(nextPowerOf2(1024)).toBe(1024);
  });

  it("rounds up to next power of 2", () => {
    expect(nextPowerOf2(3)).toBe(4);
    expect(nextPowerOf2(5)).toBe(8);
    expect(nextPowerOf2(100)).toBe(128);
    expect(nextPowerOf2(1000)).toBe(1024);
  });
});

describe("realFFT", () => {
  it("transforms DC signal correctly", () => {
    const input = [1, 1, 1, 1, 1, 1, 1, 1];
    const result = realFFT(input, 8);

    // DC component should be 8 (sum of all values)
    expect(result[0]).toBeCloseTo(8, 5);
    expect(result[1]).toBeCloseTo(0, 5);

    // All other components should be near zero
    for (let i = 2; i < result.length; i++) {
      expect(Math.abs(result[i])).toBeLessThan(1e-10);
    }
  });

  it("transforms sine wave to single frequency bin", () => {
    const n = 64;
    const freq = 4; // 4 cycles in the signal
    const input: number[] = [];

    for (let i = 0; i < n; i++) {
      input.push(Math.sin((2 * Math.PI * freq * i) / n));
    }

    const result = realFFT(input, n);

    // Should have energy at bin 4 (and its mirror at bin n-4)
    const mag4 = Math.sqrt(result[8] ** 2 + result[9] ** 2);
    expect(mag4).toBeGreaterThan(20); // Strong peak at frequency 4

    // DC should be near zero
    expect(Math.abs(result[0])).toBeLessThan(1e-10);
  });

  it("handles zero input", () => {
    const input = [0, 0, 0, 0];
    const result = realFFT(input, 4);

    for (let i = 0; i < result.length; i++) {
      expect(result[i]).toBe(0);
    }
  });

  it("pads short input with zeros", () => {
    const input = [1, 2];
    const result = realFFT(input, 8);

    // Should not throw and should produce valid output
    expect(result.length).toBe(16); // n * 2 for interleaved complex
  });
});

describe("computeSpectrum", () => {
  it("computes frequencies correctly", () => {
    const sampleRate = 1000;
    const data = new Float32Array(256);

    // Create 100 Hz sine wave
    for (let i = 0; i < data.length; i++) {
      data[i] = Math.sin((2 * Math.PI * 100 * i) / sampleRate);
    }

    const { frequencies, magnitude } = computeSpectrum(data, sampleRate);

    // Check frequency array
    expect(frequencies[0]).toBe(0);
    expect(frequencies[1]).toBeCloseTo(sampleRate / 256, 2);

    // Find peak frequency
    let maxIdx = 0;
    let maxMag = 0;
    for (let i = 1; i < magnitude.length; i++) {
      if (magnitude[i] > maxMag) {
        maxMag = magnitude[i];
        maxIdx = i;
      }
    }

    // Peak should be near 100 Hz (within one bin)
    const peakFreq = frequencies[maxIdx];
    expect(peakFreq).toBeGreaterThan(90);
    expect(peakFreq).toBeLessThan(110);
  });

  it("returns correct array lengths", () => {
    const data = new Float32Array(1000);
    const { frequencies, magnitude } = computeSpectrum(data, 44100);

    // Should pad to 1024, so Nyquist bins = 512
    expect(frequencies.length).toBe(512);
    expect(magnitude.length).toBe(512);
  });

  it("handles empty input", () => {
    const data = new Float32Array(0);
    // Should handle gracefully (though the plot component checks for this)
    expect(() => computeSpectrum(data, 44100)).not.toThrow();
  });

  it("applies windowing to reduce spectral leakage", () => {
    const sampleRate = 1024;
    const data = new Float32Array(1024);

    // Create a signal that's NOT an integer number of cycles
    // Without windowing, this would cause significant spectral leakage
    const freq = 100.5; // Non-integer cycles
    for (let i = 0; i < data.length; i++) {
      data[i] = Math.sin((2 * Math.PI * freq * i) / sampleRate);
    }

    const { magnitude } = computeSpectrum(data, sampleRate);

    // With Hanning window, energy should be concentrated near the peak
    // Find peak and check that nearby bins have most of the energy
    let maxIdx = 0;
    let maxMag = 0;
    let totalEnergy = 0;

    for (let i = 1; i < magnitude.length; i++) {
      totalEnergy += magnitude[i] ** 2;
      if (magnitude[i] > maxMag) {
        maxMag = magnitude[i];
        maxIdx = i;
      }
    }

    // Energy in peak +/- 5 bins should be > 90% of total
    let peakEnergy = 0;
    for (let i = Math.max(1, maxIdx - 5); i <= Math.min(magnitude.length - 1, maxIdx + 5); i++) {
      peakEnergy += magnitude[i] ** 2;
    }

    expect(peakEnergy / totalEnergy).toBeGreaterThan(0.9);
  });

  it("respects custom nfft parameter", () => {
    const data = new Float32Array(100);
    const { frequencies } = computeSpectrum(data, 1000, 512);

    // Should use 512-point FFT, so 256 frequency bins
    expect(frequencies.length).toBe(256);
  });
});
