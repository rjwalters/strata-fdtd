/**
 * Performance monitoring utilities
 */

export interface PerformanceMetrics {
  fps: number
  frameTime: number
  memory?: number
  gpuMemory?: number
}

export interface PerformanceEntry {
  timestamp: number
  metrics: PerformanceMetrics
}

/**
 * Performance monitor for tracking FPS and memory
 */
export class PerformanceMonitor {
  private entries: PerformanceEntry[] = []
  private maxEntries: number
  private lastFrameTime: number = 0
  private frameCount: number = 0
  private lastFpsUpdate: number = 0
  private currentFps: number = 0

  constructor(maxEntries: number = 100) {
    this.maxEntries = maxEntries
  }

  /**
   * Record a frame
   */
  recordFrame(): void {
    const now = performance.now()

    if (this.lastFrameTime > 0) {
      this.frameCount++

      if (now - this.lastFpsUpdate >= 1000) {
        this.currentFps = (this.frameCount * 1000) / (now - this.lastFpsUpdate)
        this.frameCount = 0
        this.lastFpsUpdate = now

        const entry: PerformanceEntry = {
          timestamp: now,
          metrics: {
            fps: this.currentFps,
            frameTime: now - this.lastFrameTime,
            memory: this.getMemoryUsage(),
          },
        }

        this.entries.push(entry)

        if (this.entries.length > this.maxEntries) {
          this.entries.shift()
        }
      }
    }

    this.lastFrameTime = now
  }

  /**
   * Get memory usage if available
   */
  private getMemoryUsage(): number | undefined {
    const perf = performance as Performance & {
      memory?: {
        usedJSHeapSize: number
        totalJSHeapSize: number
        jsHeapSizeLimit: number
      }
    }

    if (perf.memory) {
      return perf.memory.usedJSHeapSize / (1024 * 1024) // MB
    }
    return undefined
  }

  /**
   * Get current metrics
   */
  getCurrentMetrics(): PerformanceMetrics {
    return {
      fps: this.currentFps,
      frameTime: this.lastFrameTime > 0 ? performance.now() - this.lastFrameTime : 0,
      memory: this.getMemoryUsage(),
    }
  }

  /**
   * Get all recorded entries
   */
  getEntries(): PerformanceEntry[] {
    return [...this.entries]
  }

  /**
   * Clear all entries
   */
  clear(): void {
    this.entries = []
    this.frameCount = 0
    this.lastFpsUpdate = 0
    this.currentFps = 0
  }
}

/**
 * Global performance monitor instance
 */
export const globalMonitor = new PerformanceMonitor()

/**
 * Record a timing measurement
 */
export function measureTime<T>(name: string, fn: () => T): T {
  const start = performance.now()
  const result = fn()
  const end = performance.now()
  console.debug(`[Performance] ${name}: ${(end - start).toFixed(2)}ms`)
  return result
}

/**
 * Create a throttled function
 */
export function throttle<T extends (...args: unknown[]) => unknown>(
  fn: T,
  delay: number
): T {
  let lastCall = 0

  return ((...args: Parameters<T>) => {
    const now = Date.now()
    if (now - lastCall >= delay) {
      lastCall = now
      return fn(...args)
    }
  }) as T
}

/**
 * Create a debounced function
 */
export function debounce<T extends (...args: unknown[]) => unknown>(
  fn: T,
  delay: number
): T {
  let timeoutId: ReturnType<typeof setTimeout> | null = null

  return ((...args: Parameters<T>) => {
    if (timeoutId) {
      clearTimeout(timeoutId)
    }
    timeoutId = setTimeout(() => {
      fn(...args)
      timeoutId = null
    }, delay)
  }) as T
}

// Data processing types and functions for workers

export interface DownsampleOptions {
  shape: [number, number, number]
  targetVoxels: number
  method: 'nearest' | 'average' | 'max'
}

export interface DownsampleResult {
  data: Float32Array
  shape: [number, number, number]
  scale: [number, number, number]
}

export interface FilterResult {
  indices: Uint32Array
  values: Float32Array
  count: number
}

/**
 * Downsample pressure data for faster rendering
 */
export function downsamplePressure(
  data: Float32Array,
  options: DownsampleOptions
): DownsampleResult {
  const [nx, ny, nz] = options.shape
  const targetVoxels = options.targetVoxels

  // Calculate scale factor to reach target
  const totalVoxels = nx * ny * nz
  if (totalVoxels <= targetVoxels) {
    return {
      data,
      shape: options.shape,
      scale: [1, 1, 1],
    }
  }

  const scaleFactor = Math.cbrt(totalVoxels / targetVoxels)
  const scale: [number, number, number] = [
    Math.ceil(scaleFactor),
    Math.ceil(scaleFactor),
    Math.ceil(scaleFactor),
  ]

  const newShape: [number, number, number] = [
    Math.ceil(nx / scale[0]),
    Math.ceil(ny / scale[1]),
    Math.ceil(nz / scale[2]),
  ]

  const newData = new Float32Array(newShape[0] * newShape[1] * newShape[2])

  // Downsample based on method
  for (let z = 0; z < newShape[2]; z++) {
    for (let y = 0; y < newShape[1]; y++) {
      for (let x = 0; x < newShape[0]; x++) {
        const newIdx = x + y * newShape[0] + z * newShape[0] * newShape[1]
        const srcX = Math.min(x * scale[0], nx - 1)
        const srcY = Math.min(y * scale[1], ny - 1)
        const srcZ = Math.min(z * scale[2], nz - 1)
        const srcIdx = srcX + srcY * nx + srcZ * nx * ny

        newData[newIdx] = data[srcIdx]
      }
    }
  }

  return { data: newData, shape: newShape, scale }
}

/**
 * Filter data by threshold
 */
export function filterByThreshold(
  data: Float32Array,
  threshold: number
): FilterResult {
  const indices: number[] = []
  const values: number[] = []

  for (let i = 0; i < data.length; i++) {
    if (Math.abs(data[i]) >= threshold) {
      indices.push(i)
      values.push(data[i])
    }
  }

  return {
    indices: new Uint32Array(indices),
    values: new Float32Array(values),
    count: indices.length,
  }
}
