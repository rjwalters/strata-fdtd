import { describe, it, expect, beforeAll } from "vitest";
import {
  lttbDownsample,
  minMaxDownsample,
  decimateDownsample,
  logBinDownsample,
} from "./downsample";

describe("Downsampling algorithms", () => {
  // Generate test data once
  let smallData: Float32Array;
  let largeData: Float32Array;
  let veryLargeData: Float32Array;
  const sampleRate = 44100;

  beforeAll(() => {
    // Small dataset (1000 samples)
    smallData = new Float32Array(1000);
    for (let i = 0; i < smallData.length; i++) {
      smallData[i] = Math.sin((2 * Math.PI * 440 * i) / sampleRate);
    }

    // Large dataset (100k samples)
    largeData = new Float32Array(100000);
    for (let i = 0; i < largeData.length; i++) {
      largeData[i] = Math.sin((2 * Math.PI * 440 * i) / sampleRate);
    }

    // Very large dataset (1M samples)
    veryLargeData = new Float32Array(1000000);
    for (let i = 0; i < veryLargeData.length; i++) {
      veryLargeData[i] = Math.sin((2 * Math.PI * 440 * i) / sampleRate);
    }
  });

  describe("lttbDownsample", () => {
    it("should return all points when target >= data length", () => {
      const result = lttbDownsample(smallData, 2000, sampleRate);
      expect(result.length).toBe(smallData.length);
    });

    it("should return target number of points when downsampling", () => {
      const targetPoints = 100;
      const result = lttbDownsample(smallData, targetPoints, sampleRate);
      expect(result.length).toBe(targetPoints);
    });

    it("should preserve first and last points", () => {
      const result = lttbDownsample(smallData, 100, sampleRate);
      expect(result[0][0]).toBe(0); // First time
      expect(result[0][1]).toBe(smallData[0]); // First value
      const lastTime = ((smallData.length - 1) / sampleRate) * 1000;
      expect(result[result.length - 1][0]).toBeCloseTo(lastTime, 2);
      expect(result[result.length - 1][1]).toBe(smallData[smallData.length - 1]);
    });

    it("should maintain approximate waveform shape", () => {
      // For a sine wave, max and min should be approximately ±1
      const result = lttbDownsample(smallData, 100, sampleRate);
      const values = result.map(([, v]) => v);
      const max = Math.max(...values);
      const min = Math.min(...values);
      expect(max).toBeGreaterThan(0.9);
      expect(min).toBeLessThan(-0.9);
    });

    it("should handle large datasets efficiently", () => {
      const start = performance.now();
      const result = lttbDownsample(largeData, 1000, sampleRate);
      const duration = performance.now() - start;

      expect(result.length).toBe(1000);
      // Should complete in under 100ms for 100k points
      expect(duration).toBeLessThan(100);
    });

    it("should handle very large datasets", () => {
      const start = performance.now();
      const result = lttbDownsample(veryLargeData, 2000, sampleRate);
      const duration = performance.now() - start;

      expect(result.length).toBe(2000);
      // Should complete in under 500ms for 1M points
      expect(duration).toBeLessThan(500);
    });
  });

  describe("minMaxDownsample", () => {
    it("should return all points when target >= data length", () => {
      const result = minMaxDownsample(smallData, 2000, sampleRate);
      expect(result.length).toBe(smallData.length);
    });

    it("should preserve peaks and troughs", () => {
      const result = minMaxDownsample(smallData, 50, sampleRate);
      const values = result.map(([, v]) => v);
      const max = Math.max(...values);
      const min = Math.min(...values);
      // Min-max should capture exact extremes
      expect(max).toBeCloseTo(1, 2);
      expect(min).toBeCloseTo(-1, 2);
    });

    it("should return approximately 2x target points (min/max pairs)", () => {
      const targetPoints = 50;
      const result = minMaxDownsample(smallData, targetPoints, sampleRate);
      // Should have roughly 2x points (min and max per bucket)
      // but some buckets may have min === max
      expect(result.length).toBeGreaterThan(targetPoints);
      expect(result.length).toBeLessThanOrEqual(targetPoints * 2);
    });

    it("should handle large datasets efficiently", () => {
      const start = performance.now();
      const result = minMaxDownsample(largeData, 500, sampleRate);
      const duration = performance.now() - start;

      expect(result.length).toBeGreaterThan(500);
      // Should complete in under 50ms for 100k points
      expect(duration).toBeLessThan(50);
    });
  });

  describe("decimateDownsample", () => {
    it("should return all points when target >= data length", () => {
      const result = decimateDownsample(smallData, 2000, sampleRate);
      expect(result.length).toBe(smallData.length);
    });

    it("should return approximately target number of points", () => {
      const result = decimateDownsample(smallData, 100, sampleRate);
      // Decimation may not hit exact target due to stepping
      expect(result.length).toBeGreaterThan(90);
      expect(result.length).toBeLessThanOrEqual(110);
    });

    it("should be the fastest algorithm", () => {
      const iterations = 10;
      let decimateTime = 0;
      let lttbTime = 0;

      for (let i = 0; i < iterations; i++) {
        const start1 = performance.now();
        decimateDownsample(largeData, 1000, sampleRate);
        decimateTime += performance.now() - start1;

        const start2 = performance.now();
        lttbDownsample(largeData, 1000, sampleRate);
        lttbTime += performance.now() - start2;
      }

      // Decimation should be faster than LTTB
      expect(decimateTime).toBeLessThan(lttbTime);
    });
  });

  describe("logBinDownsample", () => {
    let frequencies: Float32Array;
    let magnitude: Float32Array;

    beforeAll(() => {
      // Create test spectrum data (10k frequency bins)
      const numBins = 10000;
      frequencies = new Float32Array(numBins);
      magnitude = new Float32Array(numBins);

      for (let i = 0; i < numBins; i++) {
        frequencies[i] = (i * sampleRate) / (2 * numBins); // Linear frequency spacing
        // Create a spectrum with some peaks
        magnitude[i] =
          Math.exp(-((frequencies[i] - 1000) ** 2) / 100000) + // Peak at 1kHz
          Math.exp(-((frequencies[i] - 5000) ** 2) / 500000); // Peak at 5kHz
      }
    });

    it("should downsample to target number of bins", () => {
      const result = logBinDownsample(frequencies, magnitude, 100, 20, 20000);
      expect(result.frequencies.length).toBe(100);
      expect(result.magnitude.length).toBe(100);
    });

    it("should have logarithmically spaced frequencies", () => {
      const result = logBinDownsample(frequencies, magnitude, 100, 20, 20000);

      // Check that frequency ratio between consecutive bins is roughly constant
      const ratios: number[] = [];
      for (let i = 1; i < result.frequencies.length; i++) {
        ratios.push(result.frequencies[i] / result.frequencies[i - 1]);
      }

      const avgRatio = ratios.reduce((a, b) => a + b, 0) / ratios.length;
      // All ratios should be close to the average (within 10%)
      for (const ratio of ratios) {
        expect(ratio / avgRatio).toBeGreaterThan(0.9);
        expect(ratio / avgRatio).toBeLessThan(1.1);
      }
    });

    it("should preserve peaks using max aggregation", () => {
      const result = logBinDownsample(frequencies, magnitude, 200, 20, 20000);

      // Find the bin closest to 1kHz and 5kHz
      let maxMag1k = 0;
      let maxMag5k = 0;

      for (let i = 0; i < result.frequencies.length; i++) {
        if (
          result.frequencies[i] > 800 &&
          result.frequencies[i] < 1200 &&
          result.magnitude[i] > maxMag1k
        ) {
          maxMag1k = result.magnitude[i];
        }
        if (
          result.frequencies[i] > 4000 &&
          result.frequencies[i] < 6000 &&
          result.magnitude[i] > maxMag5k
        ) {
          maxMag5k = result.magnitude[i];
        }
      }

      // Should preserve peaks (they should be significantly above zero)
      expect(maxMag1k).toBeGreaterThan(0.5);
      expect(maxMag5k).toBeGreaterThan(0.3);
    });

    it("should handle large spectrum data efficiently", () => {
      // Create very large spectrum (100k bins)
      const numBins = 100000;
      const largeFreqs = new Float32Array(numBins);
      const largeMag = new Float32Array(numBins);

      for (let i = 0; i < numBins; i++) {
        largeFreqs[i] = (i * sampleRate) / (2 * numBins);
        largeMag[i] = Math.random();
      }

      const start = performance.now();
      const result = logBinDownsample(largeFreqs, largeMag, 1000, 20, 20000);
      const duration = performance.now() - start;

      expect(result.frequencies.length).toBe(1000);
      // Should complete in under 100ms for 100k bins
      expect(duration).toBeLessThan(100);
    });
  });
});

describe("Performance benchmarks", () => {
  const sampleRate = 44100;

  it("benchmark: 10k samples with LTTB", () => {
    const data = new Float32Array(10000);
    for (let i = 0; i < data.length; i++) {
      data[i] = Math.sin((2 * Math.PI * 440 * i) / sampleRate);
    }

    const iterations = 100;
    const start = performance.now();
    for (let i = 0; i < iterations; i++) {
      lttbDownsample(data, 500, sampleRate);
    }
    const avgTime = (performance.now() - start) / iterations;

    console.log(`LTTB 10k->500: ${avgTime.toFixed(3)}ms avg`);
    expect(avgTime).toBeLessThan(5);
  });

  it("benchmark: 100k samples with LTTB", () => {
    const data = new Float32Array(100000);
    for (let i = 0; i < data.length; i++) {
      data[i] = Math.sin((2 * Math.PI * 440 * i) / sampleRate);
    }

    const iterations = 20;
    const start = performance.now();
    for (let i = 0; i < iterations; i++) {
      lttbDownsample(data, 1000, sampleRate);
    }
    const avgTime = (performance.now() - start) / iterations;

    console.log(`LTTB 100k->1000: ${avgTime.toFixed(3)}ms avg`);
    expect(avgTime).toBeLessThan(20);
  });

  it("benchmark: 1M samples with LTTB", () => {
    const data = new Float32Array(1000000);
    for (let i = 0; i < data.length; i++) {
      data[i] = Math.sin((2 * Math.PI * 440 * i) / sampleRate);
    }

    const iterations = 5;
    const start = performance.now();
    for (let i = 0; i < iterations; i++) {
      lttbDownsample(data, 2000, sampleRate);
    }
    const avgTime = (performance.now() - start) / iterations;

    console.log(`LTTB 1M->2000: ${avgTime.toFixed(3)}ms avg`);
    expect(avgTime).toBeLessThan(100);
  });

  it("benchmark: 100k samples with minMax", () => {
    const data = new Float32Array(100000);
    for (let i = 0; i < data.length; i++) {
      data[i] = Math.sin((2 * Math.PI * 440 * i) / sampleRate);
    }

    const iterations = 50;
    const start = performance.now();
    for (let i = 0; i < iterations; i++) {
      minMaxDownsample(data, 500, sampleRate);
    }
    const avgTime = (performance.now() - start) / iterations;

    console.log(`MinMax 100k->500: ${avgTime.toFixed(3)}ms avg`);
    expect(avgTime).toBeLessThan(10);
  });
});
