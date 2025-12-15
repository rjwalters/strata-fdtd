/**
 * Tests for performance optimization utilities.
 */

import { describe, it, expect, beforeEach } from "vitest";
import {
  calculateDownsampleFactor,
  downsamplePressure,
  filterByThreshold,
  PerformanceTracker,
  suggestQualitySettings,
  DEFAULT_DOWNSAMPLE_OPTIONS,
  PERFORMANCE_THRESHOLDS,
} from "../performance";

describe("calculateDownsampleFactor", () => {
  it("returns 1 when grid is smaller than target", () => {
    expect(calculateDownsampleFactor([32, 32, 32], 262144)).toBe(1);
    expect(calculateDownsampleFactor([50, 50, 50], 262144)).toBe(1);
  });

  it("returns correct factor for large grids", () => {
    // 100³ = 1,000,000 voxels, target 262,144 (64³)
    // factor = cbrt(1000000/262144) ≈ 1.56, ceil = 2
    expect(calculateDownsampleFactor([100, 100, 100], 262144)).toBe(2);

    // 150³ = 3,375,000 voxels
    // factor = cbrt(3375000/262144) ≈ 2.35, ceil = 3
    expect(calculateDownsampleFactor([150, 150, 150], 262144)).toBe(3);

    // 200³ = 8,000,000 voxels
    // factor = cbrt(8000000/262144) ≈ 3.1, ceil = 4
    expect(calculateDownsampleFactor([200, 200, 200], 262144)).toBe(4);
  });

  it("handles non-cubic grids", () => {
    // 100x50x50 = 250,000 voxels, just under target
    expect(calculateDownsampleFactor([100, 50, 50], 262144)).toBe(1);

    // 200x100x50 = 1,000,000 voxels
    expect(calculateDownsampleFactor([200, 100, 50], 262144)).toBe(2);
  });
});

describe("downsamplePressure", () => {
  it("returns original data when below target", () => {
    const data = new Float32Array(8000); // 20³
    const shape: [number, number, number] = [20, 20, 20];

    const result = downsamplePressure(data, shape, { targetVoxels: 262144 });

    expect(result.factor).toBe(1);
    expect(result.shape).toEqual(shape);
    expect(result.data).toBe(data);
  });

  it("downsamples using average method", () => {
    // Create a simple 4x4x4 grid
    const shape: [number, number, number] = [4, 4, 4];
    const data = new Float32Array(64);
    // Fill with known values
    for (let i = 0; i < 64; i++) {
      data[i] = i;
    }

    const result = downsamplePressure(data, shape, {
      targetVoxels: 8, // Force 2x downsampling to 2x2x2
      method: "average",
    });

    expect(result.factor).toBe(2);
    expect(result.shape).toEqual([2, 2, 2]);
    expect(result.data.length).toBe(8);
  });

  it("downsamples using max method", () => {
    const shape: [number, number, number] = [4, 4, 4];
    const data = new Float32Array(64);
    // Set one high value in first block
    data[0] = 100;
    data[1] = 10;

    const result = downsamplePressure(data, shape, {
      targetVoxels: 8,
      method: "max",
    });

    expect(result.factor).toBe(2);
    // First block should have max value 100
    expect(Math.abs(result.data[0])).toBe(100);
  });

  it("downsamples using nearest method", () => {
    const shape: [number, number, number] = [4, 4, 4];
    const data = new Float32Array(64);
    // Set center values
    for (let i = 0; i < 64; i++) {
      data[i] = i;
    }

    const result = downsamplePressure(data, shape, {
      targetVoxels: 8,
      method: "nearest",
    });

    expect(result.factor).toBe(2);
    expect(result.shape).toEqual([2, 2, 2]);
  });

  it("preserves original shape in result", () => {
    const shape: [number, number, number] = [100, 100, 100];
    const data = new Float32Array(1000000);

    const result = downsamplePressure(data, shape, { targetVoxels: 262144 });

    expect(result.originalShape).toEqual(shape);
  });
});

describe("filterByThreshold", () => {
  it("filters values below threshold", () => {
    const data = new Float32Array([0.1, 0.5, -0.8, 0.2, -0.1, 0.9]);
    const maxPressure = 1.0;
    const threshold = 0.3; // 30%

    const result = filterByThreshold(data, threshold, maxPressure);

    // Values >= 0.3: 0.5, -0.8, 0.9 (3 values)
    expect(result.count).toBe(3);
    expect(result.indices.length).toBe(3);
    expect(result.values.length).toBe(3);

    // Check indices are correct
    expect(Array.from(result.indices)).toEqual([1, 2, 5]);
    // Float32Array has limited precision, so use approximate comparison
    expect(result.values[0]).toBeCloseTo(0.5, 5);
    expect(result.values[1]).toBeCloseTo(-0.8, 5);
    expect(result.values[2]).toBeCloseTo(0.9, 5);
  });

  it("returns all values when threshold is 0", () => {
    const data = new Float32Array([0.1, 0.5, -0.8]);
    const result = filterByThreshold(data, 0, 1.0);

    expect(result.count).toBe(3);
  });

  it("returns empty arrays when threshold is high", () => {
    const data = new Float32Array([0.1, 0.2, 0.3]);
    const result = filterByThreshold(data, 0.5, 1.0);

    expect(result.count).toBe(0);
    expect(result.indices.length).toBe(0);
  });
});

describe("PerformanceTracker", () => {
  let tracker: PerformanceTracker;

  beforeEach(() => {
    tracker = new PerformanceTracker();
  });

  it("tracks frame times", () => {
    // Simulate a few frames
    for (let i = 0; i < 10; i++) {
      tracker.startFrame();
      // Simulate some work time
      tracker.endFrame();
    }

    const metrics = tracker.getMetrics();
    expect(metrics.fps).toBeGreaterThan(0);
    expect(metrics.frameTime).toBeGreaterThanOrEqual(0);
  });

  it("tracks voxel counts", () => {
    tracker.updateVoxelCounts(10000, 100000, 2);

    const metrics = tracker.getMetrics();
    expect(metrics.renderedVoxels).toBe(10000);
    expect(metrics.totalVoxels).toBe(100000);
    expect(metrics.downsampleFactor).toBe(2);
  });

  it("calculates memory usage", () => {
    tracker.updateVoxelCounts(100000, 1000000, 1);

    const metrics = tracker.getMetrics();
    // ~12 bytes per voxel, 100000 voxels ≈ 1.14 MB
    expect(metrics.memoryMB).toBeGreaterThan(1);
    expect(metrics.memoryMB).toBeLessThan(2);
  });

  it("resets correctly", () => {
    tracker.updateVoxelCounts(10000, 100000, 2);
    tracker.reset();

    const metrics = tracker.getMetrics();
    expect(metrics.renderedVoxels).toBe(0);
    expect(metrics.totalVoxels).toBe(0);
    expect(metrics.downsampleFactor).toBe(1);
  });
});

describe("suggestQualitySettings", () => {
  it("suggests conservative settings for very large grids", () => {
    const settings = suggestQualitySettings([200, 200, 200], 0);

    expect(settings.targetVoxels).toBeLessThan(200000);
    expect(settings.minThreshold).toBeGreaterThan(0);
    expect(settings.geometry).toBe("point");
  });

  it("suggests moderate settings for medium grids", () => {
    const settings = suggestQualitySettings([100, 100, 100], 0);

    expect(settings.targetVoxels).toBeLessThanOrEqual(262144);
    // 100³ = 1M voxels, which is in the >1M range, so should suggest point geometry
    expect(settings.minThreshold).toBeGreaterThan(0);
  });

  it("suggests high quality for small grids", () => {
    const settings = suggestQualitySettings([50, 50, 50], 0);

    expect(settings.geometry).toBe("cube");
    expect(settings.minThreshold).toBe(0);
  });

  it("adjusts for low FPS", () => {
    const settingsLowFPS = suggestQualitySettings([100, 100, 100], 15);
    const settingsHighFPS = suggestQualitySettings([100, 100, 100], 60);

    expect(settingsLowFPS.targetVoxels).toBeLessThan(settingsHighFPS.targetVoxels);
  });
});

describe("Constants", () => {
  it("has reasonable default options", () => {
    expect(DEFAULT_DOWNSAMPLE_OPTIONS.targetVoxels).toBe(262144); // 64³
    expect(DEFAULT_DOWNSAMPLE_OPTIONS.method).toBe("average");
  });

  it("has reasonable performance thresholds", () => {
    expect(PERFORMANCE_THRESHOLDS.targetFPS).toBe(30);
    expect(PERFORMANCE_THRESHOLDS.maxVoxelsForRealtime).toBe(500000);
  });
});
