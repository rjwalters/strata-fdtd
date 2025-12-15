/**
 * Tests for GeometryOverlay mesh generation.
 */

import { describe, it, expect } from "vitest";
import * as THREE from "three";
import {
  generateBoundaryMesh,
  createGeometryMesh,
  computeGeometryStats,
} from "../GeometryOverlay";

describe("generateBoundaryMesh", () => {
  it("generates no geometry for all-air grid", () => {
    const shape: [number, number, number] = [5, 5, 5];
    const geometry = new Uint8Array(125).fill(1); // all air

    const mesh = generateBoundaryMesh(geometry, shape, 0.001);

    expect(mesh.getAttribute("position")).toBeDefined();
    expect(mesh.getAttribute("position").count).toBe(0);
  });

  it("generates outer boundary faces for all-solid grid", () => {
    const shape: [number, number, number] = [5, 5, 5];
    const geometry = new Uint8Array(125).fill(0); // all solid

    const mesh = generateBoundaryMesh(geometry, shape, 0.001);

    // For a 5x5x5 solid block, we get faces on all outer surfaces.
    // Each outer face of the block consists of 5x5 = 25 voxel faces.
    // 6 block faces * 25 voxel faces = 150 faces total.
    // Each face has 2 triangles * 3 vertices = 6 vertices.
    // 150 * 6 = 900 vertices
    const positions = mesh.getAttribute("position");
    expect(positions.count).toBe(150 * 6); // 150 faces, 6 vertices each
  });

  it("generates correct faces for single solid voxel", () => {
    const shape: [number, number, number] = [3, 3, 3];
    const geometry = new Uint8Array(27).fill(1); // all air
    // Place single solid voxel in center
    const idx = (x: number, y: number, z: number) => x + y * 3 + z * 9;
    geometry[idx(1, 1, 1)] = 0; // solid

    const mesh = generateBoundaryMesh(geometry, shape, 0.001);

    // Single voxel exposed on all 6 sides = 6 faces
    // 6 faces * 2 triangles * 3 vertices = 36 vertices
    const positions = mesh.getAttribute("position");
    expect(positions.count).toBe(36);
  });

  it("generates correct faces for L-shaped solid", () => {
    const shape: [number, number, number] = [3, 3, 3];
    const geometry = new Uint8Array(27).fill(1); // all air
    const idx = (x: number, y: number, z: number) => x + y * 3 + z * 9;

    // L-shape: two adjacent solid voxels
    geometry[idx(1, 1, 1)] = 0; // solid
    geometry[idx(2, 1, 1)] = 0; // solid (adjacent in +x)

    const mesh = generateBoundaryMesh(geometry, shape, 0.001);

    // Two adjacent voxels share one face, so:
    // First voxel: 5 exposed faces (not +x)
    // Second voxel: 5 exposed faces (not -x)
    // Total: 10 faces * 6 vertices = 60 vertices
    const positions = mesh.getAttribute("position");
    expect(positions.count).toBe(60);
  });

  it("scales geometry by resolution", () => {
    const shape: [number, number, number] = [3, 3, 3];
    const geometry = new Uint8Array(27).fill(0); // all solid
    const resolution = 0.01; // 1cm

    const mesh = generateBoundaryMesh(geometry, shape, resolution);

    const positions = mesh.getAttribute("position");
    // Check that vertex positions are scaled by resolution
    // Grid is 3x3x3, centered, so range should be [-1.5, 1.5] * resolution
    const posArray = positions.array as Float32Array;
    const maxCoord = Math.max(...posArray);
    const minCoord = Math.min(...posArray);

    expect(maxCoord).toBeCloseTo(1.5 * resolution, 5);
    expect(minCoord).toBeCloseTo(-1.5 * resolution, 5);
  });

  it("generates normals for all vertices", () => {
    const shape: [number, number, number] = [5, 5, 5];
    const geometry = new Uint8Array(125).fill(0); // all solid

    const mesh = generateBoundaryMesh(geometry, shape, 0.001);

    const positions = mesh.getAttribute("position");
    const normals = mesh.getAttribute("normal");

    expect(normals).toBeDefined();
    expect(normals.count).toBe(positions.count);
  });
});

describe("createGeometryMesh", () => {
  it("returns empty group for hidden mode", () => {
    const shape: [number, number, number] = [5, 5, 5];
    const geometry = new Uint8Array(125).fill(0);

    const group = createGeometryMesh(geometry, shape, 0.001, "hidden");

    expect(group.children.length).toBe(0);
  });

  it("creates wireframe lines for wireframe mode", () => {
    const shape: [number, number, number] = [5, 5, 5];
    const geometry = new Uint8Array(125).fill(0);

    const group = createGeometryMesh(geometry, shape, 0.001, "wireframe");

    expect(group.children.length).toBe(1);
    expect(group.children[0]).toBeInstanceOf(THREE.LineSegments);
    expect(group.children[0].name).toBe("geometry-wireframe");
  });

  it("creates mesh for solid mode", () => {
    const shape: [number, number, number] = [5, 5, 5];
    const geometry = new Uint8Array(125).fill(0);

    const group = createGeometryMesh(geometry, shape, 0.001, "solid");

    expect(group.children.length).toBe(1);
    expect(group.children[0]).toBeInstanceOf(THREE.Mesh);
    expect(group.children[0].name).toBe("geometry-solid");
  });

  it("creates transparent mesh with edges for transparent mode", () => {
    const shape: [number, number, number] = [5, 5, 5];
    const geometry = new Uint8Array(125).fill(0);

    const group = createGeometryMesh(geometry, shape, 0.001, "transparent");

    // Should have both mesh and wireframe
    expect(group.children.length).toBe(2);
    const names = group.children.map((c) => c.name);
    expect(names).toContain("geometry-transparent");
    expect(names).toContain("geometry-transparent-edges");
  });

  it("respects custom color", () => {
    const shape: [number, number, number] = [5, 5, 5];
    const geometry = new Uint8Array(125).fill(0);
    const customColor = new THREE.Color(0xff0000);

    const group = createGeometryMesh(
      geometry,
      shape,
      0.001,
      "solid",
      customColor
    );

    const mesh = group.children[0] as THREE.Mesh;
    const material = mesh.material as THREE.MeshStandardMaterial;
    expect(material.color.getHex()).toBe(0xff0000);
  });

  it("respects custom opacity for transparent mode", () => {
    const shape: [number, number, number] = [5, 5, 5];
    const geometry = new Uint8Array(125).fill(0);

    const group = createGeometryMesh(
      geometry,
      shape,
      0.001,
      "transparent",
      undefined,
      0.5
    );

    const mesh = group.children.find(
      (c) => c.name === "geometry-transparent"
    ) as THREE.Mesh;
    const material = mesh.material as THREE.MeshBasicMaterial;
    expect(material.opacity).toBe(0.5);
    expect(material.transparent).toBe(true);
    expect(material.depthWrite).toBe(false);
  });
});

describe("computeGeometryStats", () => {
  it("computes correct stats for all-air grid", () => {
    const shape: [number, number, number] = [5, 5, 5];
    const geometry = new Uint8Array(125).fill(1);

    const stats = computeGeometryStats(geometry, shape);

    expect(stats.totalCells).toBe(125);
    expect(stats.airCells).toBe(125);
    expect(stats.solidCells).toBe(0);
    expect(stats.solidFraction).toBe(0);
  });

  it("computes correct stats for all-solid grid", () => {
    const shape: [number, number, number] = [5, 5, 5];
    const geometry = new Uint8Array(125).fill(0);

    const stats = computeGeometryStats(geometry, shape);

    expect(stats.totalCells).toBe(125);
    expect(stats.airCells).toBe(0);
    expect(stats.solidCells).toBe(125);
    expect(stats.solidFraction).toBe(1);
  });

  it("computes correct stats for mixed grid", () => {
    const shape: [number, number, number] = [10, 10, 10];
    const geometry = new Uint8Array(1000);
    // Fill half with air, half with solid
    for (let i = 0; i < 1000; i++) {
      geometry[i] = i < 500 ? 1 : 0;
    }

    const stats = computeGeometryStats(geometry, shape);

    expect(stats.totalCells).toBe(1000);
    expect(stats.airCells).toBe(500);
    expect(stats.solidCells).toBe(500);
    expect(stats.solidFraction).toBe(0.5);
  });
});
