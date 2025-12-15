/**
 * Demo geometry generator for testing and examples
 */

import type { BufferGeometry } from 'three'
import { BoxGeometry, SphereGeometry, CylinderGeometry } from 'three'

export type DemoGeometryType = 'box' | 'sphere' | 'cylinder'

export interface VoxelData {
  positions: Float32Array
  values: Float32Array
  dimensions: [number, number, number]
  spacing: [number, number, number]
}

/**
 * Generate a demo voxel dataset
 */
export function generateDemoVoxels(
  dimensions: [number, number, number] = [32, 32, 32],
  spacing: [number, number, number] = [1, 1, 1]
): VoxelData {
  const [nx, ny, nz] = dimensions
  const totalVoxels = nx * ny * nz

  const positions = new Float32Array(totalVoxels * 3)
  const values = new Float32Array(totalVoxels)

  // Generate positions and values
  let index = 0
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        positions[index * 3] = x * spacing[0]
        positions[index * 3 + 1] = y * spacing[1]
        positions[index * 3 + 2] = z * spacing[2]

        // Generate a sphere pattern
        const cx = (nx - 1) / 2
        const cy = (ny - 1) / 2
        const cz = (nz - 1) / 2
        const r = Math.sqrt(
          Math.pow(x - cx, 2) + Math.pow(y - cy, 2) + Math.pow(z - cz, 2)
        )

        // Gaussian falloff
        const maxR = Math.min(cx, cy, cz) * 0.8
        values[index] = Math.exp(-((r / maxR) ** 2))

        index++
      }
    }
  }

  return { positions, values, dimensions, spacing }
}

/**
 * Generate wave propagation demo
 */
export function generateWavePropagation(
  dimensions: [number, number, number] = [64, 64, 32],
  time: number = 0
): VoxelData {
  const [nx, ny, nz] = dimensions
  const spacing: [number, number, number] = [1, 1, 1]
  const totalVoxels = nx * ny * nz

  const positions = new Float32Array(totalVoxels * 3)
  const values = new Float32Array(totalVoxels)

  const frequency = 0.1
  const wavelength = 10
  const decay = 0.02

  let index = 0
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        positions[index * 3] = x * spacing[0]
        positions[index * 3 + 1] = y * spacing[1]
        positions[index * 3 + 2] = z * spacing[2]

        // Distance from source (center of xy plane at z=0)
        const cx = nx / 2
        const cy = ny / 2
        const r = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + z ** 2)

        // Spherical wave with decay
        const phase = (r / wavelength) * 2 * Math.PI - time * frequency * 2 * Math.PI
        values[index] = Math.sin(phase) * Math.exp(-r * decay)

        index++
      }
    }
  }

  return { positions, values, dimensions, spacing }
}

/**
 * Generate simple geometry for testing
 */
export function createDemoGeometry(type: 'box' | 'sphere' | 'cylinder'): BufferGeometry {
  switch (type) {
    case 'box':
      return new BoxGeometry(1, 1, 1)
    case 'sphere':
      return new SphereGeometry(0.5, 32, 32)
    case 'cylinder':
      return new CylinderGeometry(0.5, 0.5, 1, 32)
    default:
      return new BoxGeometry(1, 1, 1)
  }
}

/**
 * Get extent of voxel data
 */
export function getVoxelExtent(data: VoxelData): {
  min: [number, number, number]
  max: [number, number, number]
  center: [number, number, number]
} {
  const [nx, ny, nz] = data.dimensions
  const [sx, sy, sz] = data.spacing

  const max: [number, number, number] = [(nx - 1) * sx, (ny - 1) * sy, (nz - 1) * sz]
  const min: [number, number, number] = [0, 0, 0]
  const center: [number, number, number] = [max[0] / 2, max[1] / 2, max[2] / 2]

  return { min, max, center }
}
