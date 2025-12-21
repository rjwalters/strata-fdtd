/**
 * GeometryOverlay - Renders FDTD geometry mask (solid/air boundaries) as a 3D mesh.
 *
 * Supports wireframe mode toggle and opacity-based rendering:
 * - wireframe: Edge lines of solid regions
 * - opacity 0: Hidden (don't render)
 * - opacity 1-99: Semi-transparent solids
 * - opacity 100: Solid opaque surfaces
 */

import { useMemo } from "react";
import * as THREE from "three";

/** @deprecated Use showWireframe and boundaryOpacity instead */
export type GeometryMode = "wireframe" | "solid" | "transparent" | "hidden";

export interface GeometryOverlayProps {
  /** Boolean mask (1=air, 0=solid) */
  geometry: Uint8Array;
  /** Grid dimensions [nx, ny, nz] */
  shape: [number, number, number];
  /** Grid resolution in meters */
  resolution: number;
  /** Visualization mode */
  mode: GeometryMode;
  /** Solid color (default: gray #4a4a4a) */
  color?: THREE.Color;
  /** Transparency opacity for transparent mode (default: 0.3) */
  opacity?: number;
}

/** Face direction for mesh generation */
type FaceDirection = "+x" | "-x" | "+y" | "-y" | "+z" | "-z";

/** Normal vectors for each face direction */
const FACE_NORMALS: Record<FaceDirection, [number, number, number]> = {
  "+x": [1, 0, 0],
  "-x": [-1, 0, 0],
  "+y": [0, 1, 0],
  "-y": [0, -1, 0],
  "+z": [0, 0, 1],
  "-z": [0, 0, -1],
};

/**
 * Get the 4 vertices for a face of a unit cube at position (x, y, z).
 * Vertices are ordered for counter-clockwise winding when viewed from outside.
 */
function getFaceVertices(
  dir: FaceDirection,
  x: number,
  y: number,
  z: number
): [number, number, number][] {
  switch (dir) {
    case "+x":
      return [
        [x + 1, y, z],
        [x + 1, y + 1, z],
        [x + 1, y + 1, z + 1],
        [x + 1, y, z + 1],
      ];
    case "-x":
      return [
        [x, y, z + 1],
        [x, y + 1, z + 1],
        [x, y + 1, z],
        [x, y, z],
      ];
    case "+y":
      return [
        [x, y + 1, z],
        [x, y + 1, z + 1],
        [x + 1, y + 1, z + 1],
        [x + 1, y + 1, z],
      ];
    case "-y":
      return [
        [x, y, z + 1],
        [x, y, z],
        [x + 1, y, z],
        [x + 1, y, z + 1],
      ];
    case "+z":
      return [
        [x, y, z + 1],
        [x + 1, y, z + 1],
        [x + 1, y + 1, z + 1],
        [x, y + 1, z + 1],
      ];
    case "-z":
      return [
        [x, y + 1, z],
        [x + 1, y + 1, z],
        [x + 1, y, z],
        [x, y, z],
      ];
  }
}

/**
 * Generate boundary mesh from geometry mask.
 *
 * Only renders faces of solid voxels that are adjacent to air cells,
 * creating an efficient surface mesh rather than rendering every voxel.
 */
export function generateBoundaryMesh(
  geometry: Uint8Array,
  shape: [number, number, number],
  resolution: number
): THREE.BufferGeometry {
  const [nx, ny, nz] = shape;
  const positions: number[] = [];
  const normals: number[] = [];

  // Helper to get linear index
  const idx = (x: number, y: number, z: number): number =>
    x + y * nx + z * nx * ny;

  // Helper to check if cell is air (or out of bounds = air)
  const isAir = (x: number, y: number, z: number): boolean => {
    if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) {
      return true; // Out of bounds treated as air
    }
    return geometry[idx(x, y, z)] === 1;
  };

  // Helper to check if cell is solid
  const isSolid = (x: number, y: number, z: number): boolean => {
    if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) {
      return false;
    }
    return geometry[idx(x, y, z)] === 0;
  };

  // Add a quad face (2 triangles)
  const addFace = (dir: FaceDirection, x: number, y: number, z: number) => {
    const verts = getFaceVertices(dir, x, y, z);
    const normal = FACE_NORMALS[dir];

    // Triangle 1: v0, v1, v2
    // Triangle 2: v0, v2, v3
    const indices = [0, 1, 2, 0, 2, 3];

    for (const i of indices) {
      const [vx, vy, vz] = verts[i];
      // Scale by resolution and center the geometry
      positions.push(
        (vx - nx / 2) * resolution,
        (vy - ny / 2) * resolution,
        (vz - nz / 2) * resolution
      );
      normals.push(...normal);
    }
  };

  // Iterate through all cells
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        if (isSolid(x, y, z)) {
          // Check each neighbor - add face if neighbor is air
          if (isAir(x + 1, y, z)) addFace("+x", x, y, z);
          if (isAir(x - 1, y, z)) addFace("-x", x, y, z);
          if (isAir(x, y + 1, z)) addFace("+y", x, y, z);
          if (isAir(x, y - 1, z)) addFace("-y", x, y, z);
          if (isAir(x, y, z + 1)) addFace("+z", x, y, z);
          if (isAir(x, y, z - 1)) addFace("-z", x, y, z);
        }
      }
    }
  }

  const bufferGeometry = new THREE.BufferGeometry();
  bufferGeometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3)
  );
  bufferGeometry.setAttribute(
    "normal",
    new THREE.Float32BufferAttribute(normals, 3)
  );

  return bufferGeometry;
}

/**
 * Create mesh group for geometry boundary visualization.
 * @param geometry - Boolean mask (1=air, 0=solid)
 * @param shape - Grid dimensions [nx, ny, nz]
 * @param resolution - Grid resolution in meters
 * @param showWireframe - If true, render as wireframe edges only
 * @param opacity - Boundary opacity 0-100 (0=hidden, 100=solid)
 * @param color - Mesh color (default: gray)
 */
export function createGeometryMesh(
  geometry: Uint8Array,
  shape: [number, number, number],
  resolution: number,
  showWireframe: boolean,
  opacity: number, // 0-100
  color: THREE.Color = new THREE.Color(0x4a4a4a)
): THREE.Group {
  const group = new THREE.Group();
  group.name = "geometry-overlay";

  // Hidden: opacity 0 and not wireframe
  if (opacity === 0 && !showWireframe) {
    return group;
  }

  const bufferGeometry = generateBoundaryMesh(geometry, shape, resolution);

  if (showWireframe) {
    // Wireframe: use EdgesGeometry for clean edge lines
    const normalizedOpacity = opacity / 100;
    const edges = new THREE.EdgesGeometry(bufferGeometry, 1);
    const lineMaterial = new THREE.LineBasicMaterial({
      color: new THREE.Color(0xcccccc), // Light gray for wireframe
      linewidth: 1,
      transparent: opacity < 100,
      opacity: normalizedOpacity,
    });
    const wireframe = new THREE.LineSegments(edges, lineMaterial);
    wireframe.name = "geometry-wireframe";
    group.add(wireframe);
  } else if (opacity === 100) {
    // Solid: opaque mesh with lighting
    const material = new THREE.MeshStandardMaterial({
      color,
      metalness: 0.1,
      roughness: 0.8,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(bufferGeometry, material);
    mesh.name = "geometry-solid";
    group.add(mesh);
  } else if (opacity > 0) {
    // Transparent: semi-transparent with proper depth handling
    const normalizedOpacity = opacity / 100;
    const material = new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: normalizedOpacity,
      side: THREE.DoubleSide,
      depthWrite: false, // Prevent z-fighting with pressure voxels
    });
    const mesh = new THREE.Mesh(bufferGeometry, material);
    mesh.name = "geometry-transparent";
    mesh.renderOrder = -1; // Render before pressure voxels
    group.add(mesh);

    // Add subtle wireframe on top for depth cues
    const edges = new THREE.EdgesGeometry(bufferGeometry, 30);
    const lineMaterial = new THREE.LineBasicMaterial({
      color: new THREE.Color(0x888888),
      transparent: true,
      opacity: Math.min(1, normalizedOpacity * 1.5),
    });
    const wireframe = new THREE.LineSegments(edges, lineMaterial);
    wireframe.name = "geometry-transparent-edges";
    wireframe.renderOrder = 0;
    group.add(wireframe);
  }

  return group;
}

/**
 * React hook to create and manage geometry mesh.
 * Memoizes the mesh creation to avoid regenerating on every render.
 */
export function useGeometryMesh(
  geometry: Uint8Array | null,
  shape: [number, number, number] | null,
  resolution: number,
  showWireframe: boolean,
  opacity: number, // 0-100
  color?: THREE.Color
): THREE.Group | null {
  return useMemo(() => {
    if (!geometry || !shape || (opacity === 0 && !showWireframe)) {
      return null;
    }

    return createGeometryMesh(
      geometry,
      shape,
      resolution,
      showWireframe,
      opacity,
      color
    );
  }, [geometry, shape, resolution, showWireframe, opacity, color]);
}

/**
 * Compute statistics about the geometry mesh for debugging/display.
 */
export function computeGeometryStats(
  geometry: Uint8Array,
  shape: [number, number, number]
): {
  totalCells: number;
  solidCells: number;
  airCells: number;
  solidFraction: number;
} {
  const totalCells = shape[0] * shape[1] * shape[2];
  let airCells = 0;

  for (let i = 0; i < geometry.length; i++) {
    if (geometry[i] === 1) airCells++;
  }

  const solidCells = totalCells - airCells;

  return {
    totalCells,
    solidCells,
    airCells,
    solidFraction: solidCells / totalCells,
  };
}
