import { describe, it, expect } from "vitest";
import { realFFT, computeSpectrum, nextPowerOf2, welchPSD, getWindow, type WindowType } from "../fft";

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

describe("welchPSD", () => {
  it("returns correct frequency bins and PSD values", () => {
    const sampleRate = 1000;
    const data = new Float32Array(8192);

    // Create 100 Hz sine wave
    for (let i = 0; i < data.length; i++) {
      data[i] = Math.sin((2 * Math.PI * 100 * i) / sampleRate);
    }

    const result = welchPSD(data, sampleRate);

    // Check frequency array starts at 0
    expect(result.frequencies[0]).toBe(0);

    // Check we have positive PSD values
    expect(result.psd.length).toBeGreaterThan(0);

    // Find peak in PSD
    let maxIdx = 0;
    let maxPsd = 0;
    for (let i = 1; i < result.psd.length; i++) {
      if (result.psd[i] > maxPsd) {
        maxPsd = result.psd[i];
        maxIdx = i;
      }
    }

    // Peak should be near 100 Hz
    const peakFreq = result.frequencies[maxIdx];
    expect(peakFreq).toBeGreaterThan(90);
    expect(peakFreq).toBeLessThan(110);
  });

  it("returns segment count", () => {
    const sampleRate = 1000;
    const data = new Float32Array(16384);

    const result = welchPSD(data, sampleRate, { segmentSize: 4096, overlap: 0.5 });

    // With 16384 samples, 4096 segment size, 50% overlap (2048 hop):
    // numSegments = floor((16384 - 4096) / 2048) + 1 = floor(12288/2048) + 1 = 6 + 1 = 7
    expect(result.numSegments).toBe(7);
  });

  it("respects segment size option", () => {
    const sampleRate = 1000;
    const data = new Float32Array(8192);

    const result2048 = welchPSD(data, sampleRate, { segmentSize: 2048 });
    const result4096 = welchPSD(data, sampleRate, { segmentSize: 4096 });

    // Smaller segment size = more segments
    expect(result2048.numSegments).toBeGreaterThan(result4096.numSegments);

    // Smaller segment size = fewer frequency bins
    expect(result2048.frequencies.length).toBeLessThan(result4096.frequencies.length);
  });

  it("respects overlap option", () => {
    const sampleRate = 1000;
    const data = new Float32Array(8192);

    const result25 = welchPSD(data, sampleRate, { segmentSize: 4096, overlap: 0.25 });
    const result75 = welchPSD(data, sampleRate, { segmentSize: 4096, overlap: 0.75 });

    // Higher overlap = more segments
    expect(result75.numSegments).toBeGreaterThan(result25.numSegments);
  });

  it("reduces variance compared to single FFT", () => {
    const sampleRate = 1000;
    const data = new Float32Array(16384);

    // Create noisy sine wave
    for (let i = 0; i < data.length; i++) {
      data[i] = Math.sin((2 * Math.PI * 100 * i) / sampleRate) + (Math.random() - 0.5) * 0.5;
    }

    const welchResult = welchPSD(data, sampleRate, { segmentSize: 4096, overlap: 0.5 });
    const singleResult = computeSpectrum(data, sampleRate);

    // Welch averaging should have smoother spectrum
    // Calculate variance of non-peak bins
    const welchPsdSlice = Array.from(welchResult.psd.slice(10, 50));
    const singleMagSlice = Array.from(singleResult.magnitude.slice(10, 50));

    const welchMean = welchPsdSlice.reduce((a, b) => a + b, 0) / welchPsdSlice.length;
    const singleMean = singleMagSlice.reduce((a, b) => a + b, 0) / singleMagSlice.length;

    const welchVar = welchPsdSlice.reduce((a, b) => a + (b - welchMean) ** 2, 0) / welchPsdSlice.length;
    const singleVar = singleMagSlice.reduce((a, b) => a + (b - singleMean) ** 2, 0) / singleMagSlice.length;

    // Normalized variance should be lower for Welch (divided by mean^2 to normalize)
    const welchNormVar = welchVar / (welchMean ** 2 + 1e-10);
    const singleNormVar = singleVar / (singleMean ** 2 + 1e-10);

    // Welch should reduce variance
    expect(welchNormVar).toBeLessThan(singleNormVar);
  });

  it("handles short signals with single segment", () => {
    const sampleRate = 1000;
    const data = new Float32Array(1024);

    for (let i = 0; i < data.length; i++) {
      data[i] = Math.sin((2 * Math.PI * 100 * i) / sampleRate);
    }

    const result = welchPSD(data, sampleRate, { segmentSize: 4096 });

    // With 1024 samples and 4096 segment size, should have 1 segment
    expect(result.numSegments).toBe(1);
    expect(result.psd.length).toBeGreaterThan(0);
  });

  it("respects window option", () => {
    const sampleRate = 1000;
    const data = new Float32Array(8192);

    // Create a signal with known spectral content
    for (let i = 0; i < data.length; i++) {
      data[i] = Math.sin((2 * Math.PI * 100 * i) / sampleRate);
    }

    // Compute with different windows
    const resultHanning = welchPSD(data, sampleRate, { segmentSize: 4096, window: 'hanning' });
    const resultHamming = welchPSD(data, sampleRate, { segmentSize: 4096, window: 'hamming' });
    const resultBlackman = welchPSD(data, sampleRate, { segmentSize: 4096, window: 'blackman' });
    const resultBH = welchPSD(data, sampleRate, { segmentSize: 4096, window: 'blackman-harris' });

    // All should produce valid results with same frequency bins
    expect(resultHanning.frequencies.length).toBe(resultHamming.frequencies.length);
    expect(resultHanning.frequencies.length).toBe(resultBlackman.frequencies.length);
    expect(resultHanning.frequencies.length).toBe(resultBH.frequencies.length);

    // All should find the peak at around 100 Hz
    const findPeakFreq = (result: { frequencies: Float32Array; psd: Float32Array }) => {
      let maxIdx = 0;
      let maxPsd = 0;
      for (let i = 1; i < result.psd.length; i++) {
        if (result.psd[i] > maxPsd) {
          maxPsd = result.psd[i];
          maxIdx = i;
        }
      }
      return result.frequencies[maxIdx];
    };

    expect(findPeakFreq(resultHanning)).toBeCloseTo(100, -1);
    expect(findPeakFreq(resultHamming)).toBeCloseTo(100, -1);
    expect(findPeakFreq(resultBlackman)).toBeCloseTo(100, -1);
    expect(findPeakFreq(resultBH)).toBeCloseTo(100, -1);
  });
});

describe("getWindow", () => {
  const windowTypes: WindowType[] = ['hanning', 'hamming', 'blackman', 'blackman-harris'];

  it("returns array of correct length", () => {
    for (const type of windowTypes) {
      const w = getWindow(type, 1024);
      expect(w.length).toBe(1024);
    }
  });

  it("produces symmetric windows", () => {
    for (const type of windowTypes) {
      const w = getWindow(type, 256);
      // Check first and last values are approximately equal
      expect(w[0]).toBeCloseTo(w[255], 5);
      // Check second and second-to-last are approximately equal
      expect(w[1]).toBeCloseTo(w[254], 5);
      // Check values near center
      expect(w[127]).toBeCloseTo(w[128], 3);
    }
  });

  it("has maximum value of approximately 1 for all windows", () => {
    for (const type of windowTypes) {
      const w = getWindow(type, 256);
      const maxVal = Math.max(...w);
      expect(maxVal).toBeCloseTo(1, 1);
    }
  });

  it("has correct endpoint values for hanning window", () => {
    const w = getWindow('hanning', 256);
    // Hanning window should be near zero at endpoints
    expect(w[0]).toBeCloseTo(0, 5);
    expect(w[255]).toBeCloseTo(0, 5);
  });

  it("has correct endpoint values for hamming window", () => {
    const w = getWindow('hamming', 256);
    // Hamming window has non-zero endpoints (approximately 0.08)
    expect(w[0]).toBeCloseTo(0.08, 2);
    expect(w[255]).toBeCloseTo(0.08, 2);
  });

  it("has correct endpoint values for blackman window", () => {
    const w = getWindow('blackman', 256);
    // Blackman window should be near zero at endpoints
    expect(w[0]).toBeCloseTo(0, 5);
    expect(w[255]).toBeCloseTo(0, 5);
  });

  it("has correct endpoint values for blackman-harris window", () => {
    const w = getWindow('blackman-harris', 256);
    // Blackman-Harris window has very small but non-zero endpoints
    expect(w[0]).toBeCloseTo(0.00006, 3);
    expect(w[255]).toBeCloseTo(0.00006, 3);
  });

  it("produces different window shapes", () => {
    const hanning = getWindow('hanning', 256);
    const hamming = getWindow('hamming', 256);
    const blackman = getWindow('blackman', 256);
    const bh = getWindow('blackman-harris', 256);

    // Windows should be different (compare at endpoints where differences are most pronounced)
    expect(hanning[0]).not.toBe(hamming[0]);
    expect(hanning[0]).not.toBe(blackman[0]);
    expect(hamming[0]).not.toBe(bh[0]);
  });
});
