/**
 * Utilities for extracting 2D slices from 3D pressure field data.
 */

import type { SliceAxis } from "../stores/simulationStore";

/**
 * Result of slice extraction
 */
export interface SliceData {
  /** 2D pressure data for the slice */
  data: Float32Array;
  /** Width of the slice in cells */
  width: number;
  /** Height of the slice in cells */
  height: number;
  /** Physical width in meters */
  physicalWidth: number;
  /** Physical height in meters */
  physicalHeight: number;
  /** Axis labels for the slice dimensions */
  axisLabels: [string, string];
}

/**
 * Extract a 2D slice from a 3D pressure field.
 *
 * The 3D data is assumed to be in row-major order with the indexing:
 * index = x + y * nx + z * nx * ny
 *
 * @param data - 3D pressure field as flat Float32Array
 * @param shape - Grid dimensions [nx, ny, nz]
 * @param axis - Axis perpendicular to the slice plane ('x', 'y', or 'z')
 * @param normalizedPosition - Position along the axis (0-1)
 * @param resolution - Grid resolution in meters
 * @returns SliceData containing the 2D slice
 */
export function extractSlice(
  data: Float32Array,
  shape: [number, number, number],
  axis: SliceAxis,
  normalizedPosition: number,
  resolution: number
): SliceData {
  const [nx, ny, nz] = shape;

  // Clamp position to valid range
  const clampedPosition = Math.max(0, Math.min(1, normalizedPosition));

  if (axis === "z") {
    // XY plane at constant Z (top-down view)
    const zIndex = Math.min(
      nz - 1,
      Math.max(0, Math.round(clampedPosition * (nz - 1)))
    );
    const sliceData = new Float32Array(nx * ny);

    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const idx3d = x + y * nx + zIndex * nx * ny;
        sliceData[x + y * nx] = data[idx3d];
      }
    }

    return {
      data: sliceData,
      width: nx,
      height: ny,
      physicalWidth: nx * resolution,
      physicalHeight: ny * resolution,
      axisLabels: ["X", "Y"],
    };
  } else if (axis === "y") {
    // XZ plane at constant Y (front/back view)
    const yIndex = Math.min(
      ny - 1,
      Math.max(0, Math.round(clampedPosition * (ny - 1)))
    );
    const sliceData = new Float32Array(nx * nz);

    for (let z = 0; z < nz; z++) {
      for (let x = 0; x < nx; x++) {
        const idx3d = x + yIndex * nx + z * nx * ny;
        // Store with Z as height (row), X as width (column)
        sliceData[x + z * nx] = data[idx3d];
      }
    }

    return {
      data: sliceData,
      width: nx,
      height: nz,
      physicalWidth: nx * resolution,
      physicalHeight: nz * resolution,
      axisLabels: ["X", "Z"],
    };
  } else {
    // YZ plane at constant X (left/right view)
    const xIndex = Math.min(
      nx - 1,
      Math.max(0, Math.round(clampedPosition * (nx - 1)))
    );
    const sliceData = new Float32Array(ny * nz);

    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        const idx3d = xIndex + y * nx + z * nx * ny;
        // Store with Z as height (row), Y as width (column)
        sliceData[y + z * ny] = data[idx3d];
      }
    }

    return {
      data: sliceData,
      width: ny,
      height: nz,
      physicalWidth: ny * resolution,
      physicalHeight: nz * resolution,
      axisLabels: ["Y", "Z"],
    };
  }
}

/**
 * Calculate the slice index from a normalized position.
 *
 * @param normalizedPosition - Position along axis (0-1)
 * @param axisSize - Number of cells along the axis
 * @returns The integer slice index
 */
export function getSliceIndex(
  normalizedPosition: number,
  axisSize: number
): number {
  return Math.min(
    axisSize - 1,
    Math.max(0, Math.round(normalizedPosition * (axisSize - 1)))
  );
}

/**
 * Get the size of a dimension based on the slice axis.
 *
 * @param shape - Grid dimensions [nx, ny, nz]
 * @param axis - The axis perpendicular to the slice
 * @returns The number of cells along that axis
 */
export function getAxisSize(
  shape: [number, number, number],
  axis: SliceAxis
): number {
  const [nx, ny, nz] = shape;
  switch (axis) {
    case "x":
      return nx;
    case "y":
      return ny;
    case "z":
      return nz;
  }
}

/**
 * Get a human-readable label for the slice plane.
 *
 * @param axis - The axis perpendicular to the slice
 * @returns Description of the slice plane (e.g., "XY plane (Z slice)")
 */
export function getSlicePlaneLabel(axis: SliceAxis): string {
  switch (axis) {
    case "x":
      return "YZ plane (X slice)";
    case "y":
      return "XZ plane (Y slice)";
    case "z":
      return "XY plane (Z slice)";
  }
}

/**
 * Result of geometry slice extraction
 */
export interface GeometrySliceData {
  /** 2D geometry mask for the slice (1=air, 0=solid) */
  data: Uint8Array;
  /** Width of the slice in cells */
  width: number;
  /** Height of the slice in cells */
  height: number;
}

/**
 * Extract a 2D slice from a 3D geometry mask.
 *
 * The 3D data is assumed to be in row-major order with the indexing:
 * index = x + y * nx + z * nx * ny
 *
 * @param mask - 3D geometry mask as flat Uint8Array (1=air, 0=solid)
 * @param shape - Grid dimensions [nx, ny, nz]
 * @param axis - Axis perpendicular to the slice plane ('x', 'y', or 'z')
 * @param normalizedPosition - Position along the axis (0-1)
 * @returns GeometrySliceData containing the 2D slice
 */
export function extractGeometrySlice(
  mask: Uint8Array,
  shape: [number, number, number],
  axis: SliceAxis,
  normalizedPosition: number
): GeometrySliceData {
  const [nx, ny, nz] = shape;

  // Clamp position to valid range
  const clampedPosition = Math.max(0, Math.min(1, normalizedPosition));

  if (axis === "z") {
    // XY plane at constant Z (top-down view)
    const zIndex = Math.min(
      nz - 1,
      Math.max(0, Math.round(clampedPosition * (nz - 1)))
    );
    const sliceData = new Uint8Array(nx * ny);

    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const idx3d = x + y * nx + zIndex * nx * ny;
        sliceData[x + y * nx] = mask[idx3d];
      }
    }

    return {
      data: sliceData,
      width: nx,
      height: ny,
    };
  } else if (axis === "y") {
    // XZ plane at constant Y (front/back view)
    const yIndex = Math.min(
      ny - 1,
      Math.max(0, Math.round(clampedPosition * (ny - 1)))
    );
    const sliceData = new Uint8Array(nx * nz);

    for (let z = 0; z < nz; z++) {
      for (let x = 0; x < nx; x++) {
        const idx3d = x + yIndex * nx + z * nx * ny;
        // Store with Z as height (row), X as width (column)
        sliceData[x + z * nx] = mask[idx3d];
      }
    }

    return {
      data: sliceData,
      width: nx,
      height: nz,
    };
  } else {
    // YZ plane at constant X (left/right view)
    const xIndex = Math.min(
      nx - 1,
      Math.max(0, Math.round(clampedPosition * (nx - 1)))
    );
    const sliceData = new Uint8Array(ny * nz);

    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        const idx3d = xIndex + y * nx + z * nx * ny;
        // Store with Z as height (row), Y as width (column)
        sliceData[y + z * ny] = mask[idx3d];
      }
    }

    return {
      data: sliceData,
      width: ny,
      height: nz,
    };
  }
}
