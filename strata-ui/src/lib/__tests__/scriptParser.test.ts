import { describe, it, expect } from "vitest";
import { parseScript } from "../scriptParser";

describe("scriptParser", () => {
  describe("parseGrid", () => {
    it("should parse UniformGrid with standard format", () => {
      const script = `
from metamaterial import UniformGrid

grid = UniformGrid(
    shape=(100, 50, 200),
    resolution=1e-3
)
      `;

      const result = parseScript(script);

      expect(result.hasValidGrid).toBe(true);
      expect(result.grid).not.toBeNull();
      expect(result.grid?.type).toBe("uniform");
      expect(result.grid?.shape).toEqual([100, 50, 200]);
      expect(result.grid?.resolution).toBe(1e-3);
      expect(result.grid?.extent).toEqual([0.1, 0.05, 0.2]);
    });

    it("should parse UniformGrid with variable assignment", () => {
      const script = `
grid = UniformGrid(shape=(64, 64, 128), resolution=5e-4)
      `;

      const result = parseScript(script);

      expect(result.hasValidGrid).toBe(true);
      expect(result.grid).not.toBeNull();
      expect(result.grid?.shape).toEqual([64, 64, 128]);
      expect(result.grid?.resolution).toBe(5e-4);
    });

    it("should detect NonuniformGrid and return error", () => {
      const script = `
from metamaterial import NonuniformGrid

grid = NonuniformGrid.from_stretch(
    shape=(100, 100, 200),
    base_resolution=1e-3,
    stretch_z=1.05
)
      `;

      const result = parseScript(script);

      expect(result.hasValidGrid).toBe(false);
      expect(result.grid).toBeNull();
      expect(result.errors).toContain(
        "NonuniformGrid detected but not fully supported in preview yet"
      );
    });

    it("should handle missing grid configuration", () => {
      const script = `
from metamaterial import Scene

scene = Scene()
      `;

      const result = parseScript(script);

      expect(result.hasValidGrid).toBe(false);
      expect(result.grid).toBeNull();
      expect(result.errors).toContain("No valid grid configuration found");
    });

    it("should extract grid extent correctly", () => {
      const script = `
grid = UniformGrid(shape=(200, 100, 50), resolution=2e-3)
      `;

      const result = parseScript(script);

      expect(result.grid?.extent).toEqual([
        200 * 2e-3, // 0.4
        100 * 2e-3, // 0.2
        50 * 2e-3, // 0.1
      ]);
    });
  });

  describe("parseMaterials", () => {
    it("should parse rectangle definitions", () => {
      const script = `
grid = UniformGrid(shape=(100, 100, 100), resolution=1e-3)
scene = Scene(grid)

scene.add_rectangle(
    center=(0.05, 0.05, 0.05),
    size=(0.02, 0.02, 0.02),
    material="pzt5"
)
      `;

      const result = parseScript(script);

      expect(result.materials).toHaveLength(1);
      expect(result.materials[0]).toEqual({
        id: "rect-0",
        type: "rectangle",
        center: [0.05, 0.05, 0.05],
        size: [0.02, 0.02, 0.02],
        material: "pzt5",
      });
    });

    it("should parse sphere definitions", () => {
      const script = `
scene.add_sphere(
    center=(0.03, 0.04, 0.05),
    radius=0.01,
    material="water"
)
      `;

      const result = parseScript(script);

      expect(result.materials).toHaveLength(1);
      expect(result.materials[0]).toEqual({
        id: "sphere-0",
        type: "sphere",
        center: [0.03, 0.04, 0.05],
        radius: 0.01,
        material: "water",
      });
    });

    it("should handle multiple materials", () => {
      const script = `
scene.add_rectangle(
    center=(0.05, 0.05, 0.05),
    size=(0.01, 0.01, 0.01),
    material="pzt5"
)

scene.add_sphere(
    center=(0.07, 0.07, 0.07),
    radius=0.005,
    material="water"
)

scene.add_rectangle(
    center=(0.03, 0.03, 0.03),
    size=(0.02, 0.02, 0.02),
    material="aluminum"
)
      `;

      const result = parseScript(script);

      expect(result.materials).toHaveLength(3);
      // Parser processes all rectangles first, then all spheres
      expect(result.materials[0].type).toBe("rectangle");
      expect(result.materials[1].type).toBe("rectangle");
      expect(result.materials[2].type).toBe("sphere");
    });

    it("should generate unique IDs", () => {
      const script = `
scene.add_rectangle(center=(0.05, 0.05, 0.05), size=(0.01, 0.01, 0.01), material="pzt5")
scene.add_rectangle(center=(0.06, 0.06, 0.06), size=(0.01, 0.01, 0.01), material="pzt5")
scene.add_sphere(center=(0.07, 0.07, 0.07), radius=0.005, material="water")
scene.add_sphere(center=(0.08, 0.08, 0.08), radius=0.005, material="water")
      `;

      const result = parseScript(script);

      expect(result.materials).toHaveLength(4);
      expect(result.materials[0].id).toBe("rect-0");
      expect(result.materials[1].id).toBe("rect-1");
      expect(result.materials[2].id).toBe("sphere-0");
      expect(result.materials[3].id).toBe("sphere-1");
    });
  });

  describe("parseSources", () => {
    it("should parse GaussianPulse sources", () => {
      const script = `
scene.add_source(
    GaussianPulse(
        frequency=40e3,
        position=(0.025, 0.05, 0.05)
    )
)
      `;

      const result = parseScript(script);

      expect(result.sources).toHaveLength(1);
      expect(result.sources[0]).toEqual({
        id: "source-0",
        type: "GaussianPulse",
        position: [0.025, 0.05, 0.05],
        frequency: 40e3,
      });
    });

    it("should parse ContinuousWave sources", () => {
      const script = `
scene.add_source(
    ContinuousWave(
        frequency=50e3,
        position=(0.01, 0.02, 0.03),
        amplitude=1.5
    )
)
      `;

      const result = parseScript(script);

      expect(result.sources).toHaveLength(1);
      expect(result.sources[0]).toEqual({
        id: "source-0",
        type: "ContinuousWave",
        position: [0.01, 0.02, 0.03],
        frequency: 50e3,
        amplitude: 1.5,
      });
    });

    it("should parse ContinuousWave without amplitude", () => {
      const script = `
scene.add_source(
    ContinuousWave(
        frequency=60e3,
        position=(0.01, 0.02, 0.03)
    )
)
      `;

      const result = parseScript(script);

      expect(result.sources).toHaveLength(1);
      expect(result.sources[0].amplitude).toBeUndefined();
    });

    it("should extract frequency and position correctly", () => {
      const script = `
scene.add_source(GaussianPulse(frequency=1.5e6, position=(0.001, 0.002, 0.003)))
      `;

      const result = parseScript(script);

      expect(result.sources[0].frequency).toBe(1.5e6);
      expect(result.sources[0].position).toEqual([0.001, 0.002, 0.003]);
    });

    it("should handle multiple sources", () => {
      const script = `
scene.add_source(GaussianPulse(frequency=40e3, position=(0.01, 0.02, 0.03)))
scene.add_source(ContinuousWave(frequency=50e3, position=(0.04, 0.05, 0.06), amplitude=2.0))
      `;

      const result = parseScript(script);

      expect(result.sources).toHaveLength(2);
      expect(result.sources[0].id).toBe("source-0");
      expect(result.sources[1].id).toBe("source-1");
    });
  });

  describe("parseProbes", () => {
    it("should parse probes with names", () => {
      const script = `
scene.add_probe(
    position=(0.075, 0.05, 0.05),
    name="receiver"
)
      `;

      const result = parseScript(script);

      expect(result.probes).toHaveLength(1);
      expect(result.probes[0]).toEqual({
        id: "probe-0",
        position: [0.075, 0.05, 0.05],
        name: "receiver",
      });
    });

    it("should parse probes without names", () => {
      const script = `
scene.add_probe(position=(0.08, 0.06, 0.04))
      `;

      const result = parseScript(script);

      expect(result.probes).toHaveLength(1);
      expect(result.probes[0].name).toBeUndefined();
    });

    it("should extract positions correctly", () => {
      const script = `
scene.add_probe(position=(1.23e-2, 4.56e-3, 7.89e-4), name="probe1")
      `;

      const result = parseScript(script);

      expect(result.probes[0].position).toEqual([1.23e-2, 4.56e-3, 7.89e-4]);
    });

    it("should handle multiple probes", () => {
      const script = `
scene.add_probe(position=(0.01, 0.02, 0.03), name="probe1")
scene.add_probe(position=(0.04, 0.05, 0.06), name="probe2")
scene.add_probe(position=(0.07, 0.08, 0.09))
      `;

      const result = parseScript(script);

      expect(result.probes).toHaveLength(3);
      expect(result.probes[0].id).toBe("probe-0");
      expect(result.probes[1].id).toBe("probe-1");
      expect(result.probes[2].id).toBe("probe-2");
      expect(result.probes[2].name).toBeUndefined();
    });
  });

  describe("Error Handling", () => {
    it("should handle malformed input gracefully", () => {
      const script = "this is not valid python code @#$%";

      const result = parseScript(script);

      // Should not throw, but should have errors
      expect(result.errors.length).toBeGreaterThan(0);
    });

    it("should accumulate multiple errors", () => {
      const script = `
from metamaterial import NonuniformGrid

grid = NonuniformGrid.from_stretch(
    shape=(100, 100, 200),
    base_resolution=1e-3
)
      `;

      const result = parseScript(script);

      // Should have error about NonuniformGrid
      expect(result.errors).toContain(
        "NonuniformGrid detected but not fully supported in preview yet"
      );
    });
  });

  describe("Complete Script Parsing", () => {
    it("should parse a complete simulation script", () => {
      const script = `
from metamaterial import UniformGrid, Scene, GaussianPulse

# Create a uniform grid
grid = UniformGrid(
    shape=(100, 100, 100),
    resolution=1e-3
)

# Create a scene
scene = Scene(grid)

# Add materials
scene.add_rectangle(
    center=(0.05, 0.05, 0.05),
    size=(0.02, 0.02, 0.02),
    material="pzt5"
)

scene.add_sphere(
    center=(0.07, 0.07, 0.07),
    radius=0.01,
    material="water"
)

# Add sources
scene.add_source(
    GaussianPulse(
        frequency=40e3,
        position=(0.025, 0.05, 0.05)
    )
)

# Add probes
scene.add_probe(
    position=(0.075, 0.05, 0.05),
    name="receiver"
)
      `;

      const result = parseScript(script);

      expect(result.hasValidGrid).toBe(true);
      expect(result.grid).not.toBeNull();
      expect(result.materials).toHaveLength(2);
      expect(result.sources).toHaveLength(1);
      expect(result.probes).toHaveLength(1);
      expect(result.errors).toHaveLength(0);
    });
  });
});
