import { describe, it, expect } from "vitest";
import {
  rectangleTemplate,
  sphereTemplate,
  gaussianPulseTemplate,
  continuousWaveTemplate,
  probeTemplate,
} from "../templates";
import type { GridInfo } from "../scriptParser";

describe("templates", () => {
  describe("rectangleTemplate", () => {
    it("should generate rectangle template with grid context", () => {
      const grid: GridInfo = {
        shape: [100, 100, 100],
        resolution: 1e-3,
        extent: [0.1, 0.1, 0.1],
        type: "uniform",
      };

      const result = rectangleTemplate.generate(grid);

      // Check that it contains the expected function call
      expect(result).toContain("scene.add_rectangle");

      // Check that it uses grid-based positioning (center should be at extent/2)
      expect(result).toContain("5.000e-2"); // 0.1 / 2 = 0.05 = 5.000e-2

      // Check that size is based on resolution * 10
      expect(result).toContain("1.000e-2"); // 1e-3 * 10 = 0.01 = 1.000e-2

      // Check that it includes material parameter
      expect(result).toContain('material="pzt5"');
    });

    it("should generate rectangle template without grid", () => {
      const result = rectangleTemplate.generate(null);

      // Should use hardcoded defaults
      expect(result).toContain("scene.add_rectangle");
      expect(result).toContain("center=(0.05, 0.05, 0.05)");
      expect(result).toContain("size=(0.01, 0.01, 0.01)");
      expect(result).toContain('material="pzt5"');
    });
  });

  describe("sphereTemplate", () => {
    it("should generate sphere template with correct radius", () => {
      const grid: GridInfo = {
        shape: [100, 100, 100],
        resolution: 2e-3,
        extent: [0.2, 0.2, 0.2],
        type: "uniform",
      };

      const result = sphereTemplate.generate(grid);

      // Check that it contains the expected function call
      expect(result).toContain("scene.add_sphere");

      // Check that it uses grid-based positioning (center should be at extent/2)
      expect(result).toContain("1.000e-1"); // 0.2 / 2 = 0.1 = 1.000e-1

      // Check that radius is based on resolution * 5
      expect(result).toContain("1.000e-2"); // 2e-3 * 5 = 0.01 = 1.000e-2

      // Check that it includes material parameter
      expect(result).toContain('material="water"');
    });

    it("should generate sphere template without grid", () => {
      const result = sphereTemplate.generate(null);

      // Should use hardcoded defaults
      expect(result).toContain("scene.add_sphere");
      expect(result).toContain("center=(0.05, 0.05, 0.05)");
      expect(result).toContain("radius=0.005");
      expect(result).toContain('material="water"');
    });
  });

  describe("gaussianPulseTemplate", () => {
    it("should generate source template with appropriate positions", () => {
      const grid: GridInfo = {
        shape: [100, 100, 100],
        resolution: 1e-3,
        extent: [0.1, 0.1, 0.1],
        type: "uniform",
      };

      const result = gaussianPulseTemplate.generate(grid);

      // Check that it contains the expected function call
      expect(result).toContain("scene.add_source");
      expect(result).toContain("GaussianPulse");

      // Check that x position is at 25% of extent
      expect(result).toContain("2.500e-2"); // 0.1 * 0.25 = 0.025 = 2.500e-2

      // Check that y and z are at 50% of extent
      expect(result).toContain("5.000e-2"); // 0.1 * 0.5 = 0.05 = 5.000e-2

      // Check frequency
      expect(result).toContain("frequency=40e3");
    });

    it("should generate source template without grid", () => {
      const result = gaussianPulseTemplate.generate(null);

      expect(result).toContain("GaussianPulse");
      expect(result).toContain("frequency=40e3");
      expect(result).toContain("position=(0.025, 0.05, 0.05)");
    });
  });

  describe("continuousWaveTemplate", () => {
    it("should generate continuous wave template with grid context", () => {
      const grid: GridInfo = {
        shape: [200, 200, 200],
        resolution: 5e-4,
        extent: [0.1, 0.1, 0.1],
        type: "uniform",
      };

      const result = continuousWaveTemplate.generate(grid);

      // Check that it contains the expected function call
      expect(result).toContain("scene.add_source");
      expect(result).toContain("ContinuousWave");

      // Check that x position is at 25% of extent
      expect(result).toContain("2.500e-2"); // 0.1 * 0.25 = 0.025

      // Check amplitude parameter
      expect(result).toContain("amplitude=1.0");
    });

    it("should generate continuous wave template without grid", () => {
      const result = continuousWaveTemplate.generate(null);

      expect(result).toContain("ContinuousWave");
      expect(result).toContain("frequency=40e3");
      expect(result).toContain("position=(0.025, 0.05, 0.05)");
      expect(result).toContain("amplitude=1.0");
    });
  });

  describe("probeTemplate", () => {
    it("should generate probe template at 75% of grid extent", () => {
      const grid: GridInfo = {
        shape: [100, 100, 100],
        resolution: 1e-3,
        extent: [0.1, 0.1, 0.1],
        type: "uniform",
      };

      const result = probeTemplate.generate(grid);

      // Check that it contains the expected function call
      expect(result).toContain("scene.add_probe");

      // Check that x position is at 75% of extent
      expect(result).toContain("7.500e-2"); // 0.1 * 0.75 = 0.075 = 7.500e-2

      // Check that y and z are at 50% of extent
      expect(result).toContain("5.000e-2"); // 0.1 * 0.5 = 0.05 = 5.000e-2

      // Check name parameter
      expect(result).toContain('name="probe1"');
    });

    it("should generate probe template without grid", () => {
      const result = probeTemplate.generate(null);

      expect(result).toContain("scene.add_probe");
      expect(result).toContain("position=(0.075, 0.05, 0.05)");
      expect(result).toContain('name="probe1"');
    });
  });

  describe("Exponential Notation", () => {
    it("should use exponential notation for small values", () => {
      const grid: GridInfo = {
        shape: [1000, 1000, 1000],
        resolution: 1e-5, // Very small resolution
        extent: [0.01, 0.01, 0.01],
        type: "uniform",
      };

      const rectResult = rectangleTemplate.generate(grid);
      const sphereResult = sphereTemplate.generate(grid);

      // Should use exponential notation (e.g., 5.000e-3, 1.000e-4)
      expect(rectResult).toMatch(/\d+\.\d+e[+-]\d+/);
      expect(sphereResult).toMatch(/\d+\.\d+e[+-]\d+/);
    });

    it("should use 3 decimal places in exponential notation", () => {
      const grid: GridInfo = {
        shape: [100, 100, 100],
        resolution: 1.234e-3,
        extent: [0.1234, 0.1234, 0.1234],
        type: "uniform",
      };

      const result = rectangleTemplate.generate(grid);

      // Should use 3 decimal places (e.g., 6.170e-2)
      expect(result).toMatch(/\d+\.\d{3}e[+-]\d+/);
    });
  });

  describe("Template Properties", () => {
    it("should have proper name and description", () => {
      expect(rectangleTemplate.name).toBe("Rectangle");
      expect(rectangleTemplate.description).toContain("rectangular");

      expect(sphereTemplate.name).toBe("Sphere");
      expect(sphereTemplate.description).toContain("spherical");

      expect(gaussianPulseTemplate.name).toBe("Gaussian Pulse");
      expect(gaussianPulseTemplate.description).toContain("Gaussian pulse");

      expect(continuousWaveTemplate.name).toBe("Continuous Wave");
      expect(continuousWaveTemplate.description).toContain("continuous wave");

      expect(probeTemplate.name).toBe("Probe");
      expect(probeTemplate.description).toContain("pressure probe");
    });
  });

  describe("Edge Cases", () => {
    it("should handle grids with different aspect ratios", () => {
      const grid: GridInfo = {
        shape: [50, 100, 200],
        resolution: 1e-3,
        extent: [0.05, 0.1, 0.2],
        type: "uniform",
      };

      const sourceResult = gaussianPulseTemplate.generate(grid);
      const probeResult = probeTemplate.generate(grid);

      // Source at 25% x, 50% y, 50% z
      expect(sourceResult).toContain("1.250e-2"); // 0.05 * 0.25
      expect(sourceResult).toContain("5.000e-2"); // 0.1 * 0.5
      expect(sourceResult).toContain("1.000e-1"); // 0.2 * 0.5

      // Probe at 75% x, 50% y, 50% z
      expect(probeResult).toContain("3.750e-2"); // 0.05 * 0.75
    });

    it("should handle very small grids", () => {
      const grid: GridInfo = {
        shape: [10, 10, 10],
        resolution: 1e-5,
        extent: [1e-4, 1e-4, 1e-4],
        type: "uniform",
      };

      const result = rectangleTemplate.generate(grid);

      // Should still generate valid code
      expect(result).toContain("scene.add_rectangle");
      expect(result).toMatch(/\d+\.\d+e[+-]\d+/);
    });

    it("should handle large grids", () => {
      const grid: GridInfo = {
        shape: [1000, 1000, 1000],
        resolution: 1e-2,
        extent: [10, 10, 10],
        type: "uniform",
      };

      const result = sphereTemplate.generate(grid);

      // Should still generate valid code
      expect(result).toContain("scene.add_sphere");
      expect(result).toContain("5.000e+0"); // 10 / 2 = 5
    });
  });
});
