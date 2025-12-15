/**
 * Performance utilities for large grid optimization.
 *
 * Provides adaptive downsampling and performance metrics tracking.
 */

// =============================================================================
// Types
// =============================================================================

export interface DownsampleOptions {
  /** Target maximum voxels for real-time rendering (default: 262144 = 64³) */
  targetVoxels: number;
  /** Downsampling method */
  method: "nearest" | "average" | "max";
}

export interface DownsampleResult {
  /** Downsampled pressure data */
  data: Float32Array;
  /** New grid shape after downsampling */
  shape: [number, number, number];
  /** Downsample factor applied (1 = no downsampling) */
  factor: number;
  /** Original shape */
  originalShape: [number, number, number];
}

export interface PerformanceMetrics {
  /** Frames per second */
  fps: number;
  /** Frame time in milliseconds */
  frameTime: number;
  /** Number of voxels being rendered */
  renderedVoxels: number;
  /** Total voxels in data */
  totalVoxels: number;
  /** Downsampling factor (1 = none) */
  downsampleFactor: number;
  /** Estimated memory usage in MB */
  memoryMB: number;
}

// =============================================================================
// Default Configuration
// =============================================================================

export const DEFAULT_DOWNSAMPLE_OPTIONS: DownsampleOptions = {
  targetVoxels: 262144, // 64³ - good balance of quality and performance
  method: "average",
};

// Performance thresholds
export const PERFORMANCE_THRESHOLDS = {
  /** Target FPS for smooth playback */
  targetFPS: 30,
  /** Maximum voxels before forcing downsampling */
  maxVoxelsForRealtime: 500000,
  /** Warning threshold for rendered voxels */
  warnVoxelCount: 200000,
} as const;

// =============================================================================
// Downsampling
// =============================================================================

/**
 * Calculate optimal downsample factor based on grid size and target voxels.
 */
export function calculateDownsampleFactor(
  shape: [number, number, number],
  targetVoxels: number
): number {
  const totalVoxels = shape[0] * shape[1] * shape[2];
  if (totalVoxels <= targetVoxels) {
    return 1;
  }
  // Calculate factor to reduce to target voxels
  // factor³ * target ≈ total, so factor = cbrt(total/target)
  return Math.ceil(Math.pow(totalVoxels / targetVoxels, 1 / 3));
}

/**
 * Downsample 3D pressure data to reduce voxel count.
 *
 * Uses block-based sampling with configurable method (nearest, average, max).
 */
export function downsamplePressure(
  data: Float32Array,
  shape: [number, number, number],
  options: Partial<DownsampleOptions> = {}
): DownsampleResult {
  const opts = { ...DEFAULT_DOWNSAMPLE_OPTIONS, ...options };
  const [nx, ny, nz] = shape;
  const totalVoxels = nx * ny * nz;

  // Check if downsampling is needed
  if (totalVoxels <= opts.targetVoxels) {
    return {
      data,
      shape,
      factor: 1,
      originalShape: shape,
    };
  }

  const factor = calculateDownsampleFactor(shape, opts.targetVoxels);

  // Calculate new dimensions
  const newNx = Math.ceil(nx / factor);
  const newNy = Math.ceil(ny / factor);
  const newNz = Math.ceil(nz / factor);
  const newShape: [number, number, number] = [newNx, newNy, newNz];
  const newTotal = newNx * newNy * newNz;

  const result = new Float32Array(newTotal);

  // Index helper for original data (row-major, z varies fastest)
  const getIndex = (x: number, y: number, z: number): number =>
    x * ny * nz + y * nz + z;

  // Index helper for result data
  const setIndex = (x: number, y: number, z: number): number =>
    x * newNy * newNz + y * newNz + z;

  // Downsample each block
  for (let newX = 0; newX < newNx; newX++) {
    for (let newY = 0; newY < newNy; newY++) {
      for (let newZ = 0; newZ < newNz; newZ++) {
        // Calculate block bounds in original data
        const startX = newX * factor;
        const startY = newY * factor;
        const startZ = newZ * factor;
        const endX = Math.min(startX + factor, nx);
        const endY = Math.min(startY + factor, ny);
        const endZ = Math.min(startZ + factor, nz);

        let value: number;

        switch (opts.method) {
          case "nearest": {
            // Use center of block
            const centerX = Math.min(startX + Math.floor(factor / 2), nx - 1);
            const centerY = Math.min(startY + Math.floor(factor / 2), ny - 1);
            const centerZ = Math.min(startZ + Math.floor(factor / 2), nz - 1);
            value = data[getIndex(centerX, centerY, centerZ)];
            break;
          }

          case "max": {
            // Use maximum absolute value (preserves peaks)
            let maxAbs = 0;
            let maxVal = 0;
            for (let x = startX; x < endX; x++) {
              for (let y = startY; y < endY; y++) {
                for (let z = startZ; z < endZ; z++) {
                  const v = data[getIndex(x, y, z)];
                  const absV = Math.abs(v);
                  if (absV > maxAbs) {
                    maxAbs = absV;
                    maxVal = v;
                  }
                }
              }
            }
            value = maxVal;
            break;
          }

          case "average":
          default: {
            // Average all values in block
            let sum = 0;
            let count = 0;
            for (let x = startX; x < endX; x++) {
              for (let y = startY; y < endY; y++) {
                for (let z = startZ; z < endZ; z++) {
                  sum += data[getIndex(x, y, z)];
                  count++;
                }
              }
            }
            value = count > 0 ? sum / count : 0;
            break;
          }
        }

        result[setIndex(newX, newY, newZ)] = value;
      }
    }
  }

  return {
    data: result,
    shape: newShape,
    factor,
    originalShape: shape,
  };
}

// =============================================================================
// Threshold Filtering
// =============================================================================

export interface FilterResult {
  /** Indices of voxels above threshold */
  indices: Uint32Array;
  /** Pressure values at those indices */
  values: Float32Array;
  /** Count of voxels above threshold */
  count: number;
}

/**
 * Filter pressure data to only include voxels above threshold.
 *
 * Returns indices and values for efficient sparse rendering.
 */
export function filterByThreshold(
  data: Float32Array,
  threshold: number,
  maxPressure: number
): FilterResult {
  const thresholdValue = threshold * maxPressure;

  // First pass: count matching voxels
  let count = 0;
  for (let i = 0; i < data.length; i++) {
    if (Math.abs(data[i]) >= thresholdValue) {
      count++;
    }
  }

  // Allocate result arrays
  const indices = new Uint32Array(count);
  const values = new Float32Array(count);

  // Second pass: collect indices and values
  let idx = 0;
  for (let i = 0; i < data.length; i++) {
    if (Math.abs(data[i]) >= thresholdValue) {
      indices[idx] = i;
      values[idx] = data[i];
      idx++;
    }
  }

  return { indices, values, count };
}

// =============================================================================
// Performance Tracking
// =============================================================================

/**
 * Performance tracker for measuring render performance.
 */
export class PerformanceTracker {
  private frameTimes: number[] = [];
  private maxSamples = 60;
  private lastFrameTime = 0;
  private _renderedVoxels = 0;
  private _totalVoxels = 0;
  private _downsampleFactor = 1;

  /**
   * Call at the start of each frame.
   */
  startFrame(): void {
    this.lastFrameTime = performance.now();
  }

  /**
   * Call at the end of each frame.
   */
  endFrame(): void {
    const frameTime = performance.now() - this.lastFrameTime;
    this.frameTimes.push(frameTime);
    if (this.frameTimes.length > this.maxSamples) {
      this.frameTimes.shift();
    }
  }

  /**
   * Update voxel counts for metrics.
   */
  updateVoxelCounts(
    rendered: number,
    total: number,
    factor: number = 1
  ): void {
    this._renderedVoxels = rendered;
    this._totalVoxels = total;
    this._downsampleFactor = factor;
  }

  /**
   * Get current performance metrics.
   */
  getMetrics(): PerformanceMetrics {
    const avgFrameTime =
      this.frameTimes.length > 0
        ? this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length
        : 0;

    const fps = avgFrameTime > 0 ? 1000 / avgFrameTime : 0;

    // Estimate memory: ~12 bytes per voxel (position + color + matrix ref)
    const memoryMB = (this._renderedVoxels * 12) / (1024 * 1024);

    return {
      fps: Math.round(fps * 10) / 10,
      frameTime: Math.round(avgFrameTime * 100) / 100,
      renderedVoxels: this._renderedVoxels,
      totalVoxels: this._totalVoxels,
      downsampleFactor: this._downsampleFactor,
      memoryMB: Math.round(memoryMB * 100) / 100,
    };
  }

  /**
   * Reset all metrics.
   */
  reset(): void {
    this.frameTimes = [];
    this._renderedVoxels = 0;
    this._totalVoxels = 0;
    this._downsampleFactor = 1;
  }
}

// =============================================================================
// Auto Quality Adjustment
// =============================================================================

export interface QualitySettings {
  /** Target voxels for downsampling */
  targetVoxels: number;
  /** Minimum threshold to apply */
  minThreshold: number;
  /** Voxel geometry type */
  geometry: "cube" | "sphere" | "point";
}

/**
 * Suggest quality settings based on grid size and performance.
 */
export function suggestQualitySettings(
  shape: [number, number, number],
  currentFPS: number
): QualitySettings {
  const totalVoxels = shape[0] * shape[1] * shape[2];

  // Default settings
  const settings: QualitySettings = {
    targetVoxels: 262144, // 64³
    minThreshold: 0,
    geometry: "cube",
  };

  // Adjust based on grid size
  if (totalVoxels > 3000000) {
    // > 150³: aggressive downsampling
    settings.targetVoxels = 125000; // ~50³
    settings.minThreshold = 0.05;
    settings.geometry = "point";
  } else if (totalVoxels > 1000000) {
    // > 100³: moderate downsampling
    settings.targetVoxels = 216000; // ~60³
    settings.minThreshold = 0.02;
    settings.geometry = "point";
  } else if (totalVoxels > 500000) {
    // > 80³: light downsampling
    settings.targetVoxels = 262144; // 64³
    settings.minThreshold = 0.01;
    settings.geometry = "cube";
  }

  // Further adjust if FPS is too low
  if (currentFPS > 0 && currentFPS < 20) {
    settings.targetVoxels = Math.floor(settings.targetVoxels * 0.7);
    settings.minThreshold = Math.max(settings.minThreshold, 0.03);
  }

  return settings;
}
