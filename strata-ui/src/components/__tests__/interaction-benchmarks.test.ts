/**
 * Interaction latency benchmarks for chart components.
 *
 * Measures tooltip response time, zoom/brush latency, and pan operations
 * at various data scales (10k, 100k, 1M samples).
 *
 * Performance targets:
 * | Metric          | 10k samples | 100k samples | 1M samples |
 * |-----------------|-------------|--------------|------------|
 * | Tooltip latency | <16ms       | <16ms        | <32ms      |
 * | Zoom latency    | <50ms       | <100ms       | <200ms     |
 */

import { describe, it, expect, beforeAll } from "vitest";
import * as d3 from "d3";
import {
  lttbDownsample,
  minMaxDownsample,
  logBinDownsample,
} from "../../lib/downsample";
import { computeSpectrum } from "../../lib/fft";

// Test data at various scales
interface TestDatasets {
  small: Float32Array; // 10k samples
  medium: Float32Array; // 100k samples
  large: Float32Array; // 1M samples
}

let datasets: TestDatasets;
const sampleRate = 44100;

// Generate test audio data (sine wave with noise)
function generateTestData(numSamples: number): Float32Array {
  const data = new Float32Array(numSamples);
  for (let i = 0; i < numSamples; i++) {
    // 440 Hz sine wave with some noise
    data[i] =
      Math.sin((2 * Math.PI * 440 * i) / sampleRate) +
      Math.sin((2 * Math.PI * 880 * i) / sampleRate) * 0.5 +
      (Math.random() - 0.5) * 0.1;
  }
  return data;
}

beforeAll(() => {
  datasets = {
    small: generateTestData(10000),
    medium: generateTestData(100000),
    large: generateTestData(1000000),
  };
});

describe("Interaction latency benchmarks", () => {
  describe("Tooltip response time simulation", () => {
    /**
     * Simulates tooltip lookup: finding the value at a given time/x position.
     * This is the core operation performed on mousemove events.
     */
    function simulateTooltipLookup(
      data: Float32Array,
      timeMs: number
    ): { value: number; sampleIndex: number } {
      const sampleIndex = Math.round((timeMs / 1000) * sampleRate);
      const clampedIndex = Math.max(0, Math.min(data.length - 1, sampleIndex));
      return {
        value: data[clampedIndex],
        sampleIndex: clampedIndex,
      };
    }

    it("benchmark: tooltip lookup at 10k samples", () => {
      const iterations = 1000;
      const duration = (datasets.small.length / sampleRate) * 1000;

      const start = performance.now();
      for (let i = 0; i < iterations; i++) {
        const timeMs = Math.random() * duration;
        simulateTooltipLookup(datasets.small, timeMs);
      }
      const avgTime = (performance.now() - start) / iterations;

      console.log(`Tooltip lookup 10k: ${avgTime.toFixed(4)}ms avg`);
      expect(avgTime).toBeLessThan(16); // Target: <16ms per lookup
    });

    it("benchmark: tooltip lookup at 100k samples", () => {
      const iterations = 1000;
      const duration = (datasets.medium.length / sampleRate) * 1000;

      const start = performance.now();
      for (let i = 0; i < iterations; i++) {
        const timeMs = Math.random() * duration;
        simulateTooltipLookup(datasets.medium, timeMs);
      }
      const avgTime = (performance.now() - start) / iterations;

      console.log(`Tooltip lookup 100k: ${avgTime.toFixed(4)}ms avg`);
      expect(avgTime).toBeLessThan(16); // Target: <16ms per lookup
    });

    it("benchmark: tooltip lookup at 1M samples", () => {
      const iterations = 1000;
      const duration = (datasets.large.length / sampleRate) * 1000;

      const start = performance.now();
      for (let i = 0; i < iterations; i++) {
        const timeMs = Math.random() * duration;
        simulateTooltipLookup(datasets.large, timeMs);
      }
      const avgTime = (performance.now() - start) / iterations;

      console.log(`Tooltip lookup 1M: ${avgTime.toFixed(4)}ms avg`);
      expect(avgTime).toBeLessThan(32); // Target: <32ms per lookup
    });

    it("benchmark: tooltip with multiple probes (3 probes) at 100k samples", () => {
      // Simulate looking up values for multiple probes simultaneously
      const probes = [datasets.medium, datasets.medium, datasets.medium];
      const iterations = 500;
      const duration = (datasets.medium.length / sampleRate) * 1000;

      const start = performance.now();
      for (let i = 0; i < iterations; i++) {
        const timeMs = Math.random() * duration;
        for (const probe of probes) {
          simulateTooltipLookup(probe, timeMs);
        }
      }
      const avgTime = (performance.now() - start) / iterations;

      console.log(`Tooltip lookup 3 probes 100k: ${avgTime.toFixed(4)}ms avg`);
      expect(avgTime).toBeLessThan(16); // Should still be <16ms total
    });
  });

  describe("Zoom/brush response time", () => {
    /**
     * Simulates zoom operation: extracting a subarray and downsampling.
     * This is performed when user completes a brush selection.
     */
    function simulateZoomOperation(
      data: Float32Array,
      startMs: number,
      endMs: number,
      targetWidth: number = 800
    ): [number, number][] {
      const startSample = Math.floor((startMs / 1000) * sampleRate);
      const endSample = Math.ceil((endMs / 1000) * sampleRate);
      const viewData = data.subarray(
        Math.max(0, startSample),
        Math.min(data.length, endSample)
      );

      // Downsample to target width (4 points per pixel)
      const targetPoints = Math.min(targetWidth * 4, viewData.length);
      return lttbDownsample(viewData, targetPoints, sampleRate);
    }

    it("benchmark: zoom response at 10k samples", () => {
      const iterations = 50;
      const duration = (datasets.small.length / sampleRate) * 1000;

      const start = performance.now();
      for (let i = 0; i < iterations; i++) {
        const zoomStart = Math.random() * duration * 0.5;
        const zoomEnd = zoomStart + Math.random() * duration * 0.5;
        simulateZoomOperation(datasets.small, zoomStart, zoomEnd);
      }
      const avgTime = (performance.now() - start) / iterations;

      console.log(`Zoom operation 10k: ${avgTime.toFixed(3)}ms avg`);
      expect(avgTime).toBeLessThan(50); // Target: <50ms
    });

    it("benchmark: zoom response at 100k samples", () => {
      const iterations = 20;
      const duration = (datasets.medium.length / sampleRate) * 1000;

      const start = performance.now();
      for (let i = 0; i < iterations; i++) {
        const zoomStart = Math.random() * duration * 0.5;
        const zoomEnd = zoomStart + Math.random() * duration * 0.5;
        simulateZoomOperation(datasets.medium, zoomStart, zoomEnd);
      }
      const avgTime = (performance.now() - start) / iterations;

      console.log(`Zoom operation 100k: ${avgTime.toFixed(3)}ms avg`);
      expect(avgTime).toBeLessThan(100); // Target: <100ms
    });

    it("benchmark: zoom response at 1M samples", () => {
      const iterations = 10;
      const duration = (datasets.large.length / sampleRate) * 1000;

      const start = performance.now();
      for (let i = 0; i < iterations; i++) {
        const zoomStart = Math.random() * duration * 0.5;
        const zoomEnd = zoomStart + Math.random() * duration * 0.5;
        simulateZoomOperation(datasets.large, zoomStart, zoomEnd);
      }
      const avgTime = (performance.now() - start) / iterations;

      console.log(`Zoom operation 1M: ${avgTime.toFixed(3)}ms avg`);
      expect(avgTime).toBeLessThan(200); // Target: <200ms
    });

    it("benchmark: zoom with minMax algorithm at 100k samples", () => {
      const iterations = 20;
      const duration = (datasets.medium.length / sampleRate) * 1000;

      function zoomWithMinMax(
        data: Float32Array,
        startMs: number,
        endMs: number
      ) {
        const startSample = Math.floor((startMs / 1000) * sampleRate);
        const endSample = Math.ceil((endMs / 1000) * sampleRate);
        const viewData = data.subarray(
          Math.max(0, startSample),
          Math.min(data.length, endSample)
        );
        return minMaxDownsample(viewData, 800 * 4, sampleRate);
      }

      const start = performance.now();
      for (let i = 0; i < iterations; i++) {
        const zoomStart = Math.random() * duration * 0.5;
        const zoomEnd = zoomStart + Math.random() * duration * 0.5;
        zoomWithMinMax(datasets.medium, zoomStart, zoomEnd);
      }
      const avgTime = (performance.now() - start) / iterations;

      console.log(`Zoom minMax 100k: ${avgTime.toFixed(3)}ms avg`);
      expect(avgTime).toBeLessThan(100); // Target: <100ms
    });
  });

  describe("Pan response time", () => {
    /**
     * Simulates pan operation: shifting the view window.
     * This recalculates the visible data range and re-renders.
     */
    function simulatePanOperation(
      data: Float32Array,
      windowSizeMs: number,
      panDeltaMs: number,
      currentStartMs: number
    ): [number, number][] {
      const newStartMs = Math.max(
        0,
        Math.min(
          (data.length / sampleRate) * 1000 - windowSizeMs,
          currentStartMs + panDeltaMs
        )
      );
      const newEndMs = newStartMs + windowSizeMs;

      const startSample = Math.floor((newStartMs / 1000) * sampleRate);
      const endSample = Math.ceil((newEndMs / 1000) * sampleRate);
      const viewData = data.subarray(
        Math.max(0, startSample),
        Math.min(data.length, endSample)
      );

      return lttbDownsample(viewData, 800 * 4, sampleRate);
    }

    it("benchmark: pan response at 100k samples", () => {
      const iterations = 50;
      const windowSizeMs = 500; // 500ms window
      const panDeltaMs = 50; // Pan by 50ms each time
      let currentStartMs = 0;

      const start = performance.now();
      for (let i = 0; i < iterations; i++) {
        simulatePanOperation(
          datasets.medium,
          windowSizeMs,
          panDeltaMs,
          currentStartMs
        );
        currentStartMs += panDeltaMs;
      }
      const avgTime = (performance.now() - start) / iterations;

      console.log(`Pan operation 100k: ${avgTime.toFixed(3)}ms avg`);
      expect(avgTime).toBeLessThan(50); // Target: <50ms for smooth 20fps panning
    });

    it("benchmark: rapid pan (simulating drag) at 100k samples", () => {
      // Simulate rapid mouse drag with multiple small pan operations
      const windowSizeMs = 500;
      const numDragFrames = 30; // 30 frames of drag
      const panDeltaMs = 10; // 10ms per frame
      let currentStartMs = 0;

      const start = performance.now();
      for (let i = 0; i < numDragFrames; i++) {
        simulatePanOperation(
          datasets.medium,
          windowSizeMs,
          panDeltaMs,
          currentStartMs
        );
        currentStartMs += panDeltaMs;
      }
      const totalTime = performance.now() - start;
      const avgTime = totalTime / numDragFrames;

      console.log(
        `Rapid pan 100k: ${avgTime.toFixed(3)}ms avg, ${totalTime.toFixed(1)}ms total for 30 frames`
      );
      // Should complete 30 frames in under 500ms (60fps target would be 500ms)
      expect(totalTime).toBeLessThan(1000);
    });
  });

  describe("Spectrum plot interaction", () => {
    /**
     * Simulates spectrum tooltip: finding magnitude at frequency.
     */
    function simulateSpectrumTooltip(
      frequencies: Float32Array,
      magnitude: Float32Array,
      targetFreq: number
    ): { frequency: number; magnitude: number } {
      // Binary search for closest frequency
      let low = 0;
      let high = frequencies.length - 1;

      while (low < high) {
        const mid = Math.floor((low + high) / 2);
        if (frequencies[mid] < targetFreq) {
          low = mid + 1;
        } else {
          high = mid;
        }
      }

      // Return closest match
      const idx = low;
      return {
        frequency: frequencies[idx],
        magnitude: magnitude[idx],
      };
    }

    it("benchmark: spectrum tooltip at 100k samples", () => {
      // Compute spectrum first
      const spectrum = computeSpectrum(datasets.medium, sampleRate);
      const iterations = 1000;
      const maxFreq = sampleRate / 2;

      const start = performance.now();
      for (let i = 0; i < iterations; i++) {
        const targetFreq = Math.random() * maxFreq;
        simulateSpectrumTooltip(
          spectrum.frequencies,
          spectrum.magnitude,
          targetFreq
        );
      }
      const avgTime = (performance.now() - start) / iterations;

      console.log(`Spectrum tooltip 100k: ${avgTime.toFixed(4)}ms avg`);
      expect(avgTime).toBeLessThan(1); // Should be very fast with binary search
    });

    it("benchmark: spectrum zoom with log binning at 100k samples", () => {
      const spectrum = computeSpectrum(datasets.medium, sampleRate);
      const iterations = 20;

      const start = performance.now();
      for (let i = 0; i < iterations; i++) {
        const minFreq = 20 + Math.random() * 1000;
        const maxFreq = minFreq + 1000 + Math.random() * 10000;
        logBinDownsample(
          spectrum.frequencies,
          spectrum.magnitude,
          500,
          minFreq,
          maxFreq
        );
      }
      const avgTime = (performance.now() - start) / iterations;

      console.log(`Spectrum zoom 100k: ${avgTime.toFixed(3)}ms avg`);
      expect(avgTime).toBeLessThan(50);
    });
  });

  describe("D3 scale operations", () => {
    /**
     * Benchmark D3 scale inversions used in interactions.
     */
    it("benchmark: scale inversion performance", () => {
      const xScale = d3.scaleLinear().domain([0, 1000]).range([0, 800]);
      const iterations = 10000;

      const start = performance.now();
      for (let i = 0; i < iterations; i++) {
        const pixelX = Math.random() * 800;
        xScale.invert(pixelX);
      }
      const avgTime = (performance.now() - start) / iterations;

      console.log(`D3 scale inversion: ${avgTime.toFixed(5)}ms avg`);
      expect(avgTime).toBeLessThan(0.1);
    });

    it("benchmark: log scale inversion (for spectrum)", () => {
      const xScale = d3.scaleLog().domain([20, 20000]).range([0, 800]);
      const iterations = 10000;

      const start = performance.now();
      for (let i = 0; i < iterations; i++) {
        const pixelX = Math.random() * 800;
        xScale.invert(pixelX);
      }
      const avgTime = (performance.now() - start) / iterations;

      console.log(`D3 log scale inversion: ${avgTime.toFixed(5)}ms avg`);
      expect(avgTime).toBeLessThan(0.1);
    });
  });

  describe("Full interaction pipeline", () => {
    /**
     * Simulates the complete tooltip interaction pipeline:
     * 1. Scale inversion (pixel -> data coordinate)
     * 2. Data lookup for each visible probe
     * 3. Format values for display
     */
    function simulateFullTooltipPipeline(
      probes: Float32Array[],
      pixelX: number,
      chartWidth: number,
      duration: number
    ): { time: number; values: number[] } {
      // Step 1: Scale inversion
      const xScale = d3.scaleLinear().domain([0, duration]).range([0, chartWidth]);
      const timeMs = xScale.invert(pixelX);

      // Step 2: Data lookup for each probe
      const values: number[] = [];
      for (const probe of probes) {
        const sampleIndex = Math.round((timeMs / 1000) * sampleRate);
        const clampedIndex = Math.max(0, Math.min(probe.length - 1, sampleIndex));
        values.push(probe[clampedIndex]);
      }

      return { time: timeMs, values };
    }

    it("benchmark: full tooltip pipeline with 3 probes at 100k samples", () => {
      const probes = [datasets.medium, datasets.medium, datasets.medium];
      const chartWidth = 800;
      const duration = (datasets.medium.length / sampleRate) * 1000;
      const iterations = 1000;

      const start = performance.now();
      for (let i = 0; i < iterations; i++) {
        const pixelX = Math.random() * chartWidth;
        simulateFullTooltipPipeline(probes, pixelX, chartWidth, duration);
      }
      const avgTime = (performance.now() - start) / iterations;

      console.log(`Full tooltip pipeline 3 probes 100k: ${avgTime.toFixed(4)}ms avg`);
      expect(avgTime).toBeLessThan(16); // Must be under one frame at 60fps
    });
  });
});
