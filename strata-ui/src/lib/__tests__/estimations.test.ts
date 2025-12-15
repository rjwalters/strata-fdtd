import { describe, it, expect } from "vitest";
import { calculateEstimates } from "../estimations";
import type { GridInfo } from "../scriptParser";

// Constants from estimations.ts
const SPEED_OF_SOUND = 343.0;
const CFL_FACTOR = 0.5;

describe("estimations", () => {
  describe("calculateEstimates", () => {
    it("should return null for null grid", () => {
      const result = calculateEstimates(null);
      expect(result).toBeNull();
    });

    it("should calculate all estimates for valid grid", () => {
      const grid: GridInfo = {
        shape: [100, 100, 100],
        resolution: 1e-3,
        extent: [0.1, 0.1, 0.1],
        type: "uniform",
      };

      const result = calculateEstimates(grid);

      expect(result).not.toBeNull();
      expect(result?.memory).toBeDefined();
      expect(result?.timesteps).toBeDefined();
      expect(result?.runtime).toBeDefined();
      expect(result?.warnings).toBeDefined();
    });
  });

  describe("estimateMemory", () => {
    it("should calculate correct bytes for typical grid", () => {
      const grid: GridInfo = {
        shape: [100, 100, 100],
        resolution: 1e-3,
        extent: [0.1, 0.1, 0.1],
        type: "uniform",
      };

      const result = calculateEstimates(grid);

      // Total cells = 100 * 100 * 100 = 1,000,000
      // Pressure: 1M * 4 bytes = 4 MB
      // Velocity: 1M * 3 * 4 bytes = 12 MB
      // Aux: 1M * 2 * 4 bytes = 8 MB
      // Total: 24 MB
      const expectedBytes = 1_000_000 * (4 + 3 * 4 + 2 * 4);

      expect(result?.memory.bytes).toBe(expectedBytes);
      expect(result?.memory.bytes).toBe(24_000_000);
    });

    it("should format bytes as MB/GB", () => {
      // Small grid (< 1 GB)
      const smallGrid: GridInfo = {
        shape: [50, 50, 50],
        resolution: 1e-3,
        extent: [0.05, 0.05, 0.05],
        type: "uniform",
      };

      const smallResult = calculateEstimates(smallGrid);
      expect(smallResult?.memory.formatted).toContain("MB");

      // Large grid (> 1 GB)
      const largeGrid: GridInfo = {
        shape: [500, 500, 500],
        resolution: 1e-3,
        extent: [0.5, 0.5, 0.5],
        type: "uniform",
      };

      const largeResult = calculateEstimates(largeGrid);
      expect(largeResult?.memory.formatted).toContain("GB");
    });

    it("should handle different grid sizes", () => {
      const grid1: GridInfo = {
        shape: [64, 64, 64],
        resolution: 1e-3,
        extent: [0.064, 0.064, 0.064],
        type: "uniform",
      };

      const grid2: GridInfo = {
        shape: [128, 128, 128],
        resolution: 1e-3,
        extent: [0.128, 0.128, 0.128],
        type: "uniform",
      };

      const result1 = calculateEstimates(grid1);
      const result2 = calculateEstimates(grid2);

      // Grid2 is 2x larger in each dimension, so 8x more cells, 8x more memory
      expect(result2?.memory.bytes).toBe(result1!.memory.bytes * 8);
    });
  });

  describe("estimateTimestep", () => {
    it("should calculate CFL-limited timestep", () => {
      const grid: GridInfo = {
        shape: [100, 100, 100],
        resolution: 1e-3,
        extent: [0.1, 0.1, 0.1],
        type: "uniform",
      };

      const result = calculateEstimates(grid);

      // dt = (CFL_FACTOR * dx) / (c * sqrt(3))
      // dt = (0.5 * 1e-3) / (343 * sqrt(3))
      const expectedDt = (CFL_FACTOR * 1e-3) / (SPEED_OF_SOUND * Math.sqrt(3));

      // Calculate timesteps and work backwards to verify dt
      const maxDimension = 0.1;
      const frequency = 40e3;
      const wavelength = SPEED_OF_SOUND / frequency;
      const propagationTime = (maxDimension + 10 * wavelength) / SPEED_OF_SOUND;
      const expectedTimesteps = Math.ceil(propagationTime / expectedDt);

      expect(result?.timesteps).toBe(expectedTimesteps);
    });

    it("should use safety factor of 0.5", () => {
      const grid: GridInfo = {
        shape: [100, 100, 100],
        resolution: 2e-3,
        extent: [0.2, 0.2, 0.2],
        type: "uniform",
      };

      const result = calculateEstimates(grid);

      // Verify that doubling resolution approximately doubles timesteps
      // (for same physical extent)
      const grid2: GridInfo = {
        shape: [200, 200, 200],
        resolution: 1e-3,
        extent: [0.2, 0.2, 0.2],
        type: "uniform",
      };

      const result2 = calculateEstimates(grid2);

      // With half the resolution, we need approximately twice as many timesteps
      expect(result2!.timesteps).toBeGreaterThan(result!.timesteps * 1.8);
      expect(result2!.timesteps).toBeLessThan(result!.timesteps * 2.2);
    });
  });

  describe("estimateTimesteps", () => {
    it("should estimate steps for 10 wavelengths of propagation", () => {
      const grid: GridInfo = {
        shape: [100, 100, 100],
        resolution: 1e-3,
        extent: [0.1, 0.1, 0.1],
        type: "uniform",
      };

      const result = calculateEstimates(grid);

      // Should have reasonable number of timesteps
      expect(result?.timesteps).toBeGreaterThan(0);
      expect(result?.timesteps).toBeLessThan(1_000_000);
    });

    it("should scale with grid extent", () => {
      const smallGrid: GridInfo = {
        shape: [50, 50, 50],
        resolution: 1e-3,
        extent: [0.05, 0.05, 0.05],
        type: "uniform",
      };

      const largeGrid: GridInfo = {
        shape: [100, 100, 100],
        resolution: 1e-3,
        extent: [0.1, 0.1, 0.1],
        type: "uniform",
      };

      const smallResult = calculateEstimates(smallGrid);
      const largeResult = calculateEstimates(largeGrid);

      // Larger grid should need more timesteps (longer propagation distance)
      expect(largeResult!.timesteps).toBeGreaterThan(smallResult!.timesteps);
    });
  });

  describe("estimateRuntime", () => {
    it("should project runtime for Python backend", () => {
      const grid: GridInfo = {
        shape: [50, 50, 50],
        resolution: 1e-3,
        extent: [0.05, 0.05, 0.05],
        type: "uniform",
      };

      const result = calculateEstimates(grid);

      // Should have reasonable runtime estimate
      expect(result?.runtime.seconds).toBeGreaterThan(0);
      expect(result?.runtime.formatted).toBeDefined();
    });

    it("should format time as seconds/minutes/hours", () => {
      // Fast simulation (< 60 seconds)
      const fastGrid: GridInfo = {
        shape: [20, 20, 20],
        resolution: 1e-3,
        extent: [0.02, 0.02, 0.02],
        type: "uniform",
      };

      const fastResult = calculateEstimates(fastGrid);
      expect(fastResult?.runtime.formatted).toMatch(/\d+\.\d+s/);

      // Medium simulation (minutes)
      const mediumGrid: GridInfo = {
        shape: [100, 100, 100],
        resolution: 1e-3,
        extent: [0.1, 0.1, 0.1],
        type: "uniform",
      };

      const mediumResult = calculateEstimates(mediumGrid);
      // Could be in seconds, minutes or hours depending on the exact calculation
      expect(mediumResult?.runtime.formatted).toMatch(/(\d+\.\d+s|\d+m|\d+h)/);

      // Slow simulation (should be minutes or hours)
      const slowGrid: GridInfo = {
        shape: [150, 150, 150],
        resolution: 1e-3,
        extent: [0.15, 0.15, 0.15],
        type: "uniform",
      };

      const slowResult = calculateEstimates(slowGrid);
      // Should be in minutes or hours format
      expect(slowResult?.runtime.formatted).toMatch(/(\d+m|\d+h)/);
    });

    it("should scale with grid size and timesteps", () => {
      const grid1: GridInfo = {
        shape: [50, 50, 50],
        resolution: 1e-3,
        extent: [0.05, 0.05, 0.05],
        type: "uniform",
      };

      const grid2: GridInfo = {
        shape: [100, 100, 100],
        resolution: 1e-3,
        extent: [0.1, 0.1, 0.1],
        type: "uniform",
      };

      const result1 = calculateEstimates(grid1);
      const result2 = calculateEstimates(grid2);

      // Grid2 has 8x more cells, but also needs more timesteps
      // Runtime should be significantly higher
      expect(result2!.runtime.seconds).toBeGreaterThan(result1!.runtime.seconds * 8);
    });
  });

  describe("generateWarnings", () => {
    it("should warn when memory exceeds 8GB", () => {
      // Create grid that needs > 8GB
      // Need total cells such that: cells * 24 bytes > 8 * 1024^3
      // cells > 8 * 1024^3 / 24 ≈ 357,913,941
      // So shape ≈ 712^3 = 361,043,328 (safely over the threshold)
      const largeGrid: GridInfo = {
        shape: [712, 712, 712],
        resolution: 1e-3,
        extent: [0.712, 0.712, 0.712],
        type: "uniform",
      };

      const result = calculateEstimates(largeGrid);

      expect(result?.warnings.length).toBeGreaterThan(0);
      expect(result?.warnings.some((w) => w.includes("memory"))).toBe(true);
    });

    it("should warn when runtime exceeds 10 minutes", () => {
      // Create grid with long runtime (> 600 seconds)
      const slowGrid: GridInfo = {
        shape: [200, 200, 200],
        resolution: 1e-3,
        extent: [0.2, 0.2, 0.2],
        type: "uniform",
      };

      const result = calculateEstimates(slowGrid);

      if (result!.runtime.seconds > 600) {
        expect(result?.warnings.length).toBeGreaterThan(0);
        expect(result?.warnings.some((w) => w.includes("runtime"))).toBe(true);
      }
    });

    it("should return empty array for reasonable estimates", () => {
      const reasonableGrid: GridInfo = {
        shape: [50, 50, 50],
        resolution: 1e-3,
        extent: [0.05, 0.05, 0.05],
        type: "uniform",
      };

      const result = calculateEstimates(reasonableGrid);

      // Small grid should have no warnings
      expect(result?.warnings).toEqual([]);
    });

    it("should include helpful suggestions in warnings", () => {
      const largeGrid: GridInfo = {
        shape: [710, 710, 710],
        resolution: 1e-3,
        extent: [0.71, 0.71, 0.71],
        type: "uniform",
      };

      const result = calculateEstimates(largeGrid);

      // Warnings should suggest solutions
      const hasUsefulSuggestion = result?.warnings.some(
        (w) =>
          w.includes("reducing grid size") ||
          w.includes("coarser resolution") ||
          w.includes("native backend") ||
          w.includes("smaller grid")
      );

      expect(hasUsefulSuggestion).toBe(true);
    });
  });

  describe("Complete Workflow", () => {
    it("should provide complete estimates for typical use case", () => {
      const grid: GridInfo = {
        shape: [100, 100, 100],
        resolution: 1e-3,
        extent: [0.1, 0.1, 0.1],
        type: "uniform",
      };

      const result = calculateEstimates(grid);

      // Verify all fields are present and reasonable
      expect(result).not.toBeNull();
      expect(result?.memory.bytes).toBe(24_000_000);
      expect(result?.memory.formatted).toBe("22.9 MB");
      expect(result?.timesteps).toBeGreaterThan(0);
      expect(result?.runtime.seconds).toBeGreaterThan(0);
      expect(result?.warnings).toBeDefined();
    });

    it("should handle edge cases gracefully", () => {
      // Very small grid
      const tinyGrid: GridInfo = {
        shape: [10, 10, 10],
        resolution: 1e-4,
        extent: [1e-3, 1e-3, 1e-3],
        type: "uniform",
      };

      const tinyResult = calculateEstimates(tinyGrid);
      expect(tinyResult).not.toBeNull();
      expect(tinyResult?.memory.bytes).toBeGreaterThan(0);
      expect(tinyResult?.timesteps).toBeGreaterThan(0);

      // Non-cubic grid
      const rectangularGrid: GridInfo = {
        shape: [50, 100, 200],
        resolution: 1e-3,
        extent: [0.05, 0.1, 0.2],
        type: "uniform",
      };

      const rectangularResult = calculateEstimates(rectangularGrid);
      expect(rectangularResult).not.toBeNull();
      expect(rectangularResult?.memory.bytes).toBe(50 * 100 * 200 * 24);
    });
  });
});
