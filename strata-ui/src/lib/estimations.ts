/**
 * Estimation utilities for FDTD simulations
 */

import type { GridInfo } from './scriptParser'

export interface SimulationEstimates {
  memory: {
    bytes: number
    formatted: string
  }
  timesteps: number
  runtime: {
    seconds: number
    formatted: string
  }
  warnings: string[]
}

/**
 * Speed of sound in air (m/s)
 */
const SPEED_OF_SOUND = 343.0

/**
 * CFL condition safety factor
 */
const CFL_FACTOR = 0.5

/**
 * Estimate memory usage for grid
 */
function estimateMemory(grid: GridInfo): { bytes: number; formatted: string } {
  const [nx, ny, nz] = grid.shape
  const totalCells = nx * ny * nz

  // Pressure field: Float32 per cell
  const pressureBytes = totalCells * 4

  // Velocity field: 3x Float32 per cell
  const velocityBytes = totalCells * 3 * 4

  // Material properties and auxiliary fields: ~2x Float32 per cell
  const auxBytes = totalCells * 2 * 4

  const totalBytes = pressureBytes + velocityBytes + auxBytes

  return {
    bytes: totalBytes,
    formatted: formatBytes(totalBytes),
  }
}

/**
 * Estimate timestep from CFL condition
 */
function estimateTimestep(grid: GridInfo): number {
  const dx = grid.resolution
  const dt = (CFL_FACTOR * dx) / (SPEED_OF_SOUND * Math.sqrt(3))
  return dt
}

/**
 * Estimate number of timesteps for typical simulation
 * Assumes ~10 wavelengths of propagation at 40 kHz
 */
function estimateTimesteps(grid: GridInfo): number {
  const dt = estimateTimestep(grid)
  const maxDimension = Math.max(...grid.extent)
  const frequency = 40e3  // Typical frequency
  const wavelength = SPEED_OF_SOUND / frequency
  const propagationTime = (maxDimension + 10 * wavelength) / SPEED_OF_SOUND
  return Math.ceil(propagationTime / dt)
}

/**
 * Estimate runtime based on grid size and backend
 */
function estimateRuntime(grid: GridInfo, timesteps: number): { seconds: number; formatted: string } {
  const [nx, ny, nz] = grid.shape
  const totalCells = nx * ny * nz

  // Rough benchmarks (cells/sec on typical hardware):
  // - Python backend: ~5M cells/sec (single thread)
  // - Native backend: ~100M cells/sec (8 threads)
  // Use conservative estimate for Python backend
  const cellsPerSecond = 5e6

  const totalOps = totalCells * timesteps
  const seconds = totalOps / cellsPerSecond

  return {
    seconds,
    formatted: formatTime(seconds),
  }
}

/**
 * Format bytes to human-readable string
 */
function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`
}

/**
 * Format time to human-readable string
 */
function formatTime(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  if (seconds < 3600) {
    const minutes = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${minutes}m ${secs}s`
  }
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  return `${hours}h ${minutes}m`
}

/**
 * Generate warnings based on estimates
 */
function generateWarnings(estimates: Omit<SimulationEstimates, 'warnings'>): string[] {
  const warnings: string[] = []

  // Memory warnings
  if (estimates.memory.bytes > 8 * 1024 ** 3) {
    warnings.push(`High memory usage (${estimates.memory.formatted}). Consider reducing grid size or using coarser resolution.`)
  }

  // Runtime warnings
  if (estimates.runtime.seconds > 600) {
    warnings.push(`Long runtime estimated (${estimates.runtime.formatted}). Consider using native backend or smaller grid.`)
  }

  return warnings
}

/**
 * Calculate all simulation estimates
 */
export function calculateEstimates(grid: GridInfo | null): SimulationEstimates | null {
  if (!grid) {
    return null
  }

  const memory = estimateMemory(grid)
  const timesteps = estimateTimesteps(grid)
  const runtime = estimateRuntime(grid, timesteps)

  const estimates: Omit<SimulationEstimates, 'warnings'> = {
    memory,
    timesteps,
    runtime,
  }

  const warnings = generateWarnings(estimates)

  return {
    ...estimates,
    warnings,
  }
}
