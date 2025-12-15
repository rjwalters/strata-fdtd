import * as THREE from "three"

/**
 * Colormap function type - maps a value to a THREE.Color
 * @param value - The value to map
 * @param min - Minimum value in the range
 * @param max - Maximum value in the range
 * @returns THREE.Color for the value
 */
export type Colormap = (value: number, min: number, max: number) => THREE.Color

// Pre-allocated colors to avoid GC pressure in hot loops
const BLUE = new THREE.Color(0x2563eb) // Blue-600
const WHITE = new THREE.Color(0xffffff)
const RED = new THREE.Color(0xdc2626) // Red-600

/**
 * Diverging blue-white-red colormap for pressure visualization
 * - Negative values: blue → white
 * - Zero: white
 * - Positive values: white → red
 */
export const pressureColormap: Colormap = (
  value: number,
  min: number,
  max: number
): THREE.Color => {
  // Handle edge case where min === max
  if (max === min) {
    return WHITE.clone()
  }

  // Normalize to 0-1 range
  const normalized = (value - min) / (max - min)
  // Center around zero: -1 to 1
  const centered = normalized * 2 - 1

  const result = new THREE.Color()
  if (centered < 0) {
    // Blue to white (negative values)
    result.lerpColors(BLUE, WHITE, centered + 1)
  } else {
    // White to red (positive values)
    result.lerpColors(WHITE, RED, centered)
  }
  return result
}

/**
 * Optimized colormap that writes to existing color to avoid allocation
 * Use this in animation loops for better performance
 */
export function applyPressureColormap(
  value: number,
  min: number,
  max: number,
  target: THREE.Color
): void {
  if (max === min) {
    target.copy(WHITE)
    return
  }

  const normalized = (value - min) / (max - min)
  const centered = normalized * 2 - 1

  if (centered < 0) {
    target.lerpColors(BLUE, WHITE, centered + 1)
  } else {
    target.lerpColors(WHITE, RED, centered)
  }
}

/**
 * Sequential viridis-like colormap for magnitude visualization
 * Maps 0-1 to dark purple → blue → green → yellow
 */
export const magnitudeColormap: Colormap = (
  value: number,
  min: number,
  max: number
): THREE.Color => {
  if (max === min) {
    return new THREE.Color(0x440154) // Viridis dark purple
  }

  const t = Math.max(0, Math.min(1, (value - min) / (max - min)))

  // Simplified viridis approximation
  const r = Math.max(0, Math.min(1, t < 0.5 ? t * 0.5 : 0.25 + (t - 0.5) * 1.5))
  const g = Math.max(0, Math.min(1, t < 0.3 ? 0 : (t - 0.3) * 1.43))
  const b = Math.max(0, Math.min(1, t < 0.7 ? 0.5 + t * 0.5 : 0.85 - (t - 0.7) * 2.8))

  return new THREE.Color(r, g, b)
}

/**
 * Get the min and max values from a Float32Array
 * Skips NaN and Infinity values
 */
export function getRange(data: Float32Array): [number, number] {
  let min = Infinity
  let max = -Infinity

  for (let i = 0; i < data.length; i++) {
    const v = data[i]
    if (Number.isFinite(v)) {
      if (v < min) min = v
      if (v > max) max = v
    }
  }

  // Handle all NaN/Infinity case
  if (!Number.isFinite(min)) min = 0
  if (!Number.isFinite(max)) max = 0

  return [min, max]
}

/**
 * Get symmetric range centered on zero (for diverging colormaps)
 * The range will be [-absMax, +absMax] where absMax is the largest absolute value
 */
export function getSymmetricRange(data: Float32Array): [number, number] {
  let absMax = 0

  for (let i = 0; i < data.length; i++) {
    const v = data[i]
    if (Number.isFinite(v)) {
      const abs = Math.abs(v)
      if (abs > absMax) absMax = abs
    }
  }

  return [-absMax, absMax]
}
