/**
 * Colormap utilities for visualization
 */

import type { Color } from 'three'
import { Color as ThreeColor } from 'three'

export type ColormapName = 'viridis' | 'plasma' | 'inferno' | 'magma' | 'turbo' | 'coolwarm'

/**
 * Viridis colormap data points
 */
const VIRIDIS: [number, number, number][] = [
  [0.267004, 0.004874, 0.329415],
  [0.282327, 0.140926, 0.457517],
  [0.253935, 0.265254, 0.529983],
  [0.206756, 0.371758, 0.553117],
  [0.163625, 0.471133, 0.558148],
  [0.127568, 0.566949, 0.550556],
  [0.134692, 0.658636, 0.517649],
  [0.266941, 0.748751, 0.440573],
  [0.477504, 0.821444, 0.318195],
  [0.741388, 0.873449, 0.149561],
  [0.993248, 0.906157, 0.143936],
]

/**
 * Turbo colormap data points
 */
const TURBO: [number, number, number][] = [
  [0.18995, 0.07176, 0.23217],
  [0.19483, 0.28394, 0.67254],
  [0.08771, 0.48096, 0.86745],
  [0.11817, 0.65989, 0.88698],
  [0.38295, 0.80525, 0.76992],
  [0.65830, 0.90615, 0.51413],
  [0.88773, 0.93190, 0.26776],
  [0.99490, 0.83091, 0.15458],
  [0.96200, 0.60998, 0.12082],
  [0.84228, 0.36990, 0.12941],
  [0.64720, 0.16314, 0.16314],
]

/**
 * Coolwarm colormap data points
 */
const COOLWARM: [number, number, number][] = [
  [0.2298, 0.2987, 0.7537],
  [0.4069, 0.5038, 0.8485],
  [0.5827, 0.6883, 0.9111],
  [0.7351, 0.8248, 0.9399],
  [0.8627, 0.9177, 0.9440],
  [0.9578, 0.9578, 0.9578],
  [0.9628, 0.8734, 0.8167],
  [0.9386, 0.7373, 0.6403],
  [0.8769, 0.5683, 0.4519],
  [0.7693, 0.3731, 0.2713],
  [0.6196, 0.1683, 0.1265],
]

const COLORMAPS: Record<ColormapName, [number, number, number][]> = {
  viridis: VIRIDIS,
  plasma: VIRIDIS, // Placeholder
  inferno: VIRIDIS, // Placeholder
  magma: VIRIDIS, // Placeholder
  turbo: TURBO,
  coolwarm: COOLWARM,
}

/**
 * Linear interpolation between two values
 */
function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

/**
 * Get color from colormap for normalized value [0, 1]
 */
export function getColor(value: number, colormap: ColormapName = 'viridis'): Color {
  const colors = COLORMAPS[colormap]
  const clampedValue = Math.max(0, Math.min(1, value))

  // Find segment
  const segmentIndex = clampedValue * (colors.length - 1)
  const i = Math.floor(segmentIndex)
  const t = segmentIndex - i

  // Handle edge case
  if (i >= colors.length - 1) {
    const [r, g, b] = colors[colors.length - 1]
    return new ThreeColor(r, g, b)
  }

  // Interpolate
  const [r1, g1, b1] = colors[i]
  const [r2, g2, b2] = colors[i + 1]

  return new ThreeColor(lerp(r1, r2, t), lerp(g1, g2, t), lerp(b1, b2, t))
}

/**
 * Get hex color from colormap
 */
export function getColorHex(value: number, colormap: ColormapName = 'viridis'): number {
  return getColor(value, colormap).getHex()
}

/**
 * Generate colormap texture data
 */
export function generateColormapData(
  colormap: ColormapName,
  width: number = 256
): Uint8Array {
  const data = new Uint8Array(width * 4)

  for (let i = 0; i < width; i++) {
    const t = i / (width - 1)
    const color = getColor(t, colormap)

    data[i * 4] = Math.round(color.r * 255)
    data[i * 4 + 1] = Math.round(color.g * 255)
    data[i * 4 + 2] = Math.round(color.b * 255)
    data[i * 4 + 3] = 255
  }

  return data
}
