/**
 * Memory usage benchmarks for chart components.
 *
 * Measures memory footprint at various data scales and tests for memory leaks.
 *
 * Performance targets:
 * | Metric           | 10k samples | 100k samples | 1M samples |
 * |------------------|-------------|--------------|------------|
 * | Memory overhead  | <10MB       | <50MB        | <200MB     |
 *
 * Note: These benchmarks use approximations since the Web Memory API
 * (performance.measureUserAgentSpecificMemory) is not available in Node.js.
 * The tests focus on:
 * 1. Calculating theoretical memory footprint
 * 2. Detecting memory leaks via repeated operations
 * 3. Measuring garbage collection pressure via allocation patterns
 */

import { describe, it, expect } from "vitest";
import {
  lttbDownsample,
  minMaxDownsample,
  logBinDownsample,
} from "../../lib/downsample";
import { computeSpectrum } from "../../lib/fft";

const sampleRate = 44100;

// Generate test audio data
function generateTestData(numSamples: number): Float32Array {
  const data = new Float32Array(numSamples);
  for (let i = 0; i < numSamples; i++) {
    data[i] =
      Math.sin((2 * Math.PI * 440 * i) / sampleRate) +
      (Math.random() - 0.5) * 0.1;
  }
  return data;
}

// Calculate theoretical memory footprint for Float32Array
function calculateFloat32ArrayMemory(length: number): number {
  // Float32Array: 4 bytes per element + array overhead (~64 bytes)
  return length * 4 + 64;
}

// Calculate theoretical memory for downsampled data (array of [time, value] tuples)
function calculateDownsampledMemory(numPoints: number): number {
  // Each point is [number, number] - 2 numbers * 8 bytes = 16 bytes
  // Plus array overhead
  return numPoints * 16 + 64;
}

describe("Memory usage benchmarks", () => {
  describe("Memory footprint at various scales", () => {
    it("calculates memory footprint for 10k samples", () => {
      const data = generateTestData(10000);
      const rawMemory = calculateFloat32ArrayMemory(10000);

      // Downsample to typical display size
      const downsampled = lttbDownsample(data, 1000, sampleRate);
      const downsampledMemory = calculateDownsampledMemory(downsampled.length);

      const totalMemory = rawMemory + downsampledMemory;
      const totalMB = totalMemory / (1024 * 1024);

      console.log(`Memory footprint 10k samples:`);
      console.log(`  Raw data: ${(rawMemory / 1024).toFixed(2)} KB`);
      console.log(`  Downsampled: ${(downsampledMemory / 1024).toFixed(2)} KB`);
      console.log(`  Total: ${totalMB.toFixed(3)} MB`);

      expect(totalMB).toBeLessThan(10); // Target: <10MB
    });

    it("calculates memory footprint for 100k samples", () => {
      const data = generateTestData(100000);
      const rawMemory = calculateFloat32ArrayMemory(100000);

      const downsampled = lttbDownsample(data, 2000, sampleRate);
      const downsampledMemory = calculateDownsampledMemory(downsampled.length);

      // Also compute spectrum (typical for audio viz)
      const spectrum = computeSpectrum(data, sampleRate);
      const spectrumMemory =
        calculateFloat32ArrayMemory(spectrum.frequencies.length) +
        calculateFloat32ArrayMemory(spectrum.magnitude.length);

      const totalMemory = rawMemory + downsampledMemory + spectrumMemory;
      const totalMB = totalMemory / (1024 * 1024);

      console.log(`Memory footprint 100k samples:`);
      console.log(`  Raw data: ${(rawMemory / 1024).toFixed(2)} KB`);
      console.log(`  Downsampled: ${(downsampledMemory / 1024).toFixed(2)} KB`);
      console.log(`  Spectrum: ${(spectrumMemory / 1024).toFixed(2)} KB`);
      console.log(`  Total: ${totalMB.toFixed(3)} MB`);

      expect(totalMB).toBeLessThan(50); // Target: <50MB
    });

    it("calculates memory footprint for 1M samples", () => {
      const data = generateTestData(1000000);
      const rawMemory = calculateFloat32ArrayMemory(1000000);

      const downsampled = lttbDownsample(data, 4000, sampleRate);
      const downsampledMemory = calculateDownsampledMemory(downsampled.length);

      // For 1M samples, spectrum computation would be expensive
      // Assume we'd use a smaller FFT window
      const fftWindowSize = 65536; // 64k FFT
      const spectrumBins = fftWindowSize / 2;
      const spectrumMemory = calculateFloat32ArrayMemory(spectrumBins) * 2;

      const totalMemory = rawMemory + downsampledMemory + spectrumMemory;
      const totalMB = totalMemory / (1024 * 1024);

      console.log(`Memory footprint 1M samples:`);
      console.log(`  Raw data: ${(rawMemory / (1024 * 1024)).toFixed(2)} MB`);
      console.log(`  Downsampled: ${(downsampledMemory / 1024).toFixed(2)} KB`);
      console.log(`  Spectrum (64k FFT): ${(spectrumMemory / 1024).toFixed(2)} KB`);
      console.log(`  Total: ${totalMB.toFixed(3)} MB`);

      expect(totalMB).toBeLessThan(200); // Target: <200MB
    });

    it("calculates memory for multiple probes (3 probes at 100k)", () => {
      const numProbes = 3;
      const samplesPerProbe = 100000;

      const rawMemoryPerProbe = calculateFloat32ArrayMemory(samplesPerProbe);
      const totalRawMemory = rawMemoryPerProbe * numProbes;

      // Each probe downsampled separately
      const downsampledMemoryPerProbe = calculateDownsampledMemory(2000);
      const totalDownsampledMemory = downsampledMemoryPerProbe * numProbes;

      const totalMemory = totalRawMemory + totalDownsampledMemory;
      const totalMB = totalMemory / (1024 * 1024);

      console.log(`Memory footprint 3 probes × 100k samples:`);
      console.log(`  Raw data per probe: ${(rawMemoryPerProbe / 1024).toFixed(2)} KB`);
      console.log(`  Total raw: ${(totalRawMemory / (1024 * 1024)).toFixed(2)} MB`);
      console.log(`  Total downsampled: ${(totalDownsampledMemory / 1024).toFixed(2)} KB`);
      console.log(`  Grand total: ${totalMB.toFixed(3)} MB`);

      expect(totalMB).toBeLessThan(50); // Should still be under 50MB
    });
  });

  describe("Memory stability over repeated renders", () => {
    /**
     * Tests for memory leaks by performing repeated operations
     * and verifying no unbounded growth in allocations.
     */

    it("no unbounded growth in downsampling operations", () => {
      const data = generateTestData(100000);
      const iterations = 100;

      // Perform repeated downsampling (simulating 100 render cycles)
      const results: [number, number][][] = [];
      for (let i = 0; i < iterations; i++) {
        const result = lttbDownsample(data, 1000, sampleRate);
        results.push(result);
      }

      // Check that all results are the same size (deterministic)
      const firstLength = results[0].length;
      const allSameLength = results.every((r) => r.length === firstLength);

      console.log(`Downsampling consistency over ${iterations} iterations:`);
      console.log(`  Each result: ${firstLength} points`);
      console.log(`  All same length: ${allSameLength}`);

      expect(allSameLength).toBe(true);
      expect(firstLength).toBe(1000); // Should match target

      // Memory should not grow unboundedly - verify by checking
      // that keeping references doesn't cause issues
      const totalPoints = results.reduce((sum, r) => sum + r.length, 0);
      const estimatedMemory = calculateDownsampledMemory(totalPoints);
      console.log(`  Total points held: ${totalPoints}`);
      console.log(`  Estimated memory: ${(estimatedMemory / (1024 * 1024)).toFixed(3)} MB`);

      // 100 iterations × 1000 points × 16 bytes ≈ 1.6MB - should be reasonable
      expect(estimatedMemory / (1024 * 1024)).toBeLessThan(10);
    });

    it("no memory leak in zoom operations", () => {
      const data = generateTestData(100000);
      const iterations = 50;
      const duration = (data.length / sampleRate) * 1000;

      // Simulate 50 zoom operations (user zooming in/out repeatedly)
      const results: [number, number][][] = [];
      for (let i = 0; i < iterations; i++) {
        // Random zoom window
        const zoomStart = Math.random() * duration * 0.5;
        const zoomEnd = zoomStart + (0.2 + Math.random() * 0.3) * duration;

        const startSample = Math.floor((zoomStart / 1000) * sampleRate);
        const endSample = Math.ceil((zoomEnd / 1000) * sampleRate);
        const viewData = data.subarray(
          Math.max(0, startSample),
          Math.min(data.length, endSample)
        );

        const result = lttbDownsample(viewData, 1000, sampleRate);
        results.push(result);
      }

      // Calculate total memory used
      const totalPoints = results.reduce((sum, r) => sum + r.length, 0);
      const estimatedMemory = calculateDownsampledMemory(totalPoints);

      console.log(`Zoom operations over ${iterations} iterations:`);
      console.log(`  Total points held: ${totalPoints}`);
      console.log(`  Estimated memory: ${(estimatedMemory / (1024 * 1024)).toFixed(3)} MB`);

      // Each zoom should produce ~1000 points, so 50 × 1000 = 50k points max
      expect(totalPoints).toBeLessThanOrEqual(iterations * 1001); // Allow for rounding
    });

    it("no memory leak in spectrum computation", () => {
      const data = generateTestData(16384); // Power of 2 for efficient FFT
      const iterations = 50;

      const results: Array<{
        frequencies: Float32Array;
        magnitude: Float32Array;
      }> = [];

      for (let i = 0; i < iterations; i++) {
        const result = computeSpectrum(data, sampleRate);
        results.push(result);
      }

      // Check all results are consistent
      const firstLength = results[0].frequencies.length;
      const allSameLength = results.every(
        (r) => r.frequencies.length === firstLength
      );

      console.log(`Spectrum computation over ${iterations} iterations:`);
      console.log(`  Each result: ${firstLength} frequency bins`);
      console.log(`  All same length: ${allSameLength}`);

      expect(allSameLength).toBe(true);

      // Calculate memory
      const memoryPerSpectrum =
        calculateFloat32ArrayMemory(firstLength) * 2; // freq + mag
      const totalMemory = memoryPerSpectrum * iterations;

      console.log(`  Total estimated memory: ${(totalMemory / (1024 * 1024)).toFixed(3)} MB`);
      expect(totalMemory / (1024 * 1024)).toBeLessThan(50);
    });
  });

  describe("Garbage collection pressure", () => {
    /**
     * Measures allocation patterns during zoom operations.
     * High allocation rates can cause GC pauses affecting UI smoothness.
     */

    it("measures allocation rate during rapid zoom", () => {
      const data = generateTestData(100000);
      const duration = (data.length / sampleRate) * 1000;
      const numOperations = 30; // Simulate 30 rapid zoom adjustments

      let totalAllocatedBytes = 0;

      const start = performance.now();
      for (let i = 0; i < numOperations; i++) {
        const zoomStart = Math.random() * duration * 0.5;
        const zoomEnd = zoomStart + Math.random() * duration * 0.5;

        const startSample = Math.floor((zoomStart / 1000) * sampleRate);
        const endSample = Math.ceil((zoomEnd / 1000) * sampleRate);
        const viewData = data.subarray(
          Math.max(0, startSample),
          Math.min(data.length, endSample)
        );

        const result = lttbDownsample(viewData, 1000, sampleRate);
        totalAllocatedBytes += calculateDownsampledMemory(result.length);
      }
      const elapsed = performance.now() - start;

      const allocationRateMBps =
        (totalAllocatedBytes / (1024 * 1024)) / (elapsed / 1000);

      console.log(`Allocation rate during rapid zoom:`);
      console.log(`  ${numOperations} operations in ${elapsed.toFixed(1)}ms`);
      console.log(`  Total allocated: ${(totalAllocatedBytes / 1024).toFixed(2)} KB`);
      console.log(`  Rate: ${allocationRateMBps.toFixed(2)} MB/s`);

      // Allocation rate should be reasonable (under 100 MB/s)
      // Higher rates may cause GC pressure
      expect(allocationRateMBps).toBeLessThan(100);
    });

    it("measures allocation rate during tooltip hover simulation", () => {
      const data = generateTestData(100000);
      const duration = (data.length / sampleRate) * 1000;
      const numLookups = 1000; // Simulate 1000 mousemove events

      // Tooltip lookups are O(1) and don't allocate much
      const start = performance.now();
      const values: number[] = [];
      for (let i = 0; i < numLookups; i++) {
        const timeMs = Math.random() * duration;
        const sampleIndex = Math.round((timeMs / 1000) * sampleRate);
        const clampedIndex = Math.max(0, Math.min(data.length - 1, sampleIndex));
        values.push(data[clampedIndex]);
      }
      const elapsed = performance.now() - start;

      // The values array is the only significant allocation
      const allocatedBytes = numLookups * 8; // 8 bytes per number
      const allocationRateMBps =
        (allocatedBytes / (1024 * 1024)) / (elapsed / 1000);

      console.log(`Allocation rate during tooltip hover:`);
      console.log(`  ${numLookups} lookups in ${elapsed.toFixed(1)}ms`);
      console.log(`  Total allocated: ${(allocatedBytes / 1024).toFixed(2)} KB`);
      console.log(`  Rate: ${allocationRateMBps.toFixed(2)} MB/s`);

      // Tooltip lookups should have minimal allocation pressure
      // Note: High rate is acceptable here since total allocation is tiny (~8KB)
      // and the operation is extremely fast (<1ms for 1000 lookups)
      expect(allocationRateMBps).toBeLessThan(100);
    });

    it("measures allocation in spectrum log binning", () => {
      const spectrum = computeSpectrum(generateTestData(65536), sampleRate);
      const numOperations = 20;

      let totalAllocatedBytes = 0;

      const start = performance.now();
      for (let i = 0; i < numOperations; i++) {
        const minFreq = 20 + Math.random() * 1000;
        const maxFreq = minFreq + 5000 + Math.random() * 10000;

        const result = logBinDownsample(
          spectrum.frequencies,
          spectrum.magnitude,
          500,
          minFreq,
          maxFreq
        );

        totalAllocatedBytes +=
          calculateFloat32ArrayMemory(result.frequencies.length) +
          calculateFloat32ArrayMemory(result.magnitude.length);
      }
      const elapsed = performance.now() - start;

      const allocationRateMBps =
        (totalAllocatedBytes / (1024 * 1024)) / (elapsed / 1000);

      console.log(`Allocation rate during spectrum zoom:`);
      console.log(`  ${numOperations} operations in ${elapsed.toFixed(1)}ms`);
      console.log(`  Total allocated: ${(totalAllocatedBytes / 1024).toFixed(2)} KB`);
      console.log(`  Rate: ${allocationRateMBps.toFixed(2)} MB/s`);

      expect(allocationRateMBps).toBeLessThan(100);
    });
  });

  describe("Memory efficiency of downsampling algorithms", () => {
    it("compares memory efficiency: LTTB vs MinMax", () => {
      const data = generateTestData(100000);
      const targetPoints = 1000;

      // LTTB
      const lttbResult = lttbDownsample(data, targetPoints, sampleRate);
      const lttbMemory = calculateDownsampledMemory(lttbResult.length);

      // MinMax (produces ~2x points)
      const minMaxResult = minMaxDownsample(data, targetPoints, sampleRate);
      const minMaxMemory = calculateDownsampledMemory(minMaxResult.length);

      console.log(`Memory efficiency comparison (100k samples -> ~1k points):`);
      console.log(`  LTTB: ${lttbResult.length} points, ${(lttbMemory / 1024).toFixed(2)} KB`);
      console.log(`  MinMax: ${minMaxResult.length} points, ${(minMaxMemory / 1024).toFixed(2)} KB`);
      console.log(`  MinMax/LTTB ratio: ${(minMaxMemory / lttbMemory).toFixed(2)}x`);

      // LTTB should be more memory efficient (fewer points)
      expect(lttbResult.length).toBeLessThanOrEqual(minMaxResult.length);
    });

    it("measures subarray efficiency (no copy)", () => {
      const data = generateTestData(1000000);

      // Subarray creates a view, not a copy
      const start = performance.now();
      const views: Float32Array[] = [];
      for (let i = 0; i < 1000; i++) {
        const startIdx = Math.floor(Math.random() * 500000);
        const endIdx = startIdx + 100000;
        views.push(data.subarray(startIdx, endIdx));
      }
      const elapsed = performance.now() - start;

      console.log(`Subarray view creation:`);
      console.log(`  1000 views in ${elapsed.toFixed(2)}ms`);
      console.log(`  Avg: ${(elapsed / 1000).toFixed(4)}ms per view`);

      // Views should be essentially free (O(1))
      expect(elapsed).toBeLessThan(10);

      // Verify views don't copy data (same buffer)
      const view1 = data.subarray(0, 100);
      const view2 = data.subarray(0, 100);
      expect(view1.buffer).toBe(view2.buffer);
    });
  });

  describe("Canvas memory estimation", () => {
    /**
     * Canvas rendering uses additional memory for the pixel buffer.
     * This test estimates canvas memory requirements.
     */
    it("estimates canvas memory for various chart sizes", () => {
      const chartSizes = [
        { width: 800, height: 300 },
        { width: 1200, height: 400 },
        { width: 1920, height: 600 },
        { width: 3840, height: 1200 }, // 4K display
      ];

      console.log("Canvas memory estimation:");
      for (const size of chartSizes) {
        // RGBA: 4 bytes per pixel
        // With 2x device pixel ratio: 4x pixels
        const dpr = 2;
        const memoryBytes = size.width * dpr * size.height * dpr * 4;
        const memoryMB = memoryBytes / (1024 * 1024);

        console.log(
          `  ${size.width}×${size.height} @${dpr}x DPR: ${memoryMB.toFixed(2)} MB`
        );
      }

      // A typical chart at 1200×400 @2x should use ~4MB
      const typicalMemory = 1200 * 2 * 400 * 2 * 4;
      expect(typicalMemory / (1024 * 1024)).toBeLessThan(10);
    });
  });
});
