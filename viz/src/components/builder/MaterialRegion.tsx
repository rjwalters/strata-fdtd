/**
 * Material region visualization (rectangles and spheres)
 */

import { useMemo } from 'react'
import type { MaterialRegion as MaterialRegionType } from '@/lib/scriptParser'

interface MaterialRegionProps {
  region: MaterialRegionType
  sliceAxis: 'none' | 'xy' | 'xz' | 'yz'
  slicePosition: number
  extent: [number, number, number]
}

/**
 * Material color mapping
 */
const MATERIAL_COLORS: Record<string, number> = {
  pzt5: 0x4a90e2,      // Blue
  water: 0x64b5f6,     // Light blue
  air: 0xeeeeee,       // Light gray
  steel: 0x757575,     // Gray
  aluminum: 0xbdbdbd,  // Light gray
  concrete: 0x9e9e9e,  // Medium gray
  wood: 0xa1887f,      // Brown
}

const DEFAULT_COLOR = 0x9c27b0  // Purple

function getMaterialColor(materialName: string): number {
  return MATERIAL_COLORS[materialName.toLowerCase()] ?? DEFAULT_COLOR
}

export function MaterialRegion({ region, sliceAxis, slicePosition, extent }: MaterialRegionProps) {
  const color = getMaterialColor(region.material)

  // Check if region intersects slice plane
  const isVisible = useMemo(() => {
    if (sliceAxis === 'none') return true

    const axisIndex = sliceAxis === 'yz' ? 0 : sliceAxis === 'xz' ? 1 : 2
    const threshold = extent[axisIndex] * slicePosition

    // Get region bounds
    const center = region.center
    let halfSize: number

    if (region.type === 'rectangle' && region.size) {
      halfSize = region.size[axisIndex] / 2
    } else if (region.type === 'sphere' && region.radius) {
      halfSize = region.radius
    } else {
      return false
    }

    // Check if material intersects the slice plane (within tolerance)
    const minBound = center[axisIndex] - halfSize
    const maxBound = center[axisIndex] + halfSize
    return threshold >= minBound && threshold <= maxBound
  }, [region, sliceAxis, slicePosition, extent])

  if (!isVisible) return null

  if (region.type === 'rectangle' && region.size) {
    return (
      <mesh position={region.center as [number, number, number]}>
        <boxGeometry args={region.size as [number, number, number]} />
        <meshStandardMaterial
          color={color}
          transparent
          opacity={0.6}
          depthWrite={false}
        />
      </mesh>
    )
  }

  if (region.type === 'sphere' && region.radius) {
    return (
      <mesh position={region.center as [number, number, number]}>
        <sphereGeometry args={[region.radius, 32, 32]} />
        <meshStandardMaterial
          color={color}
          transparent
          opacity={0.6}
          depthWrite={false}
        />
      </mesh>
    )
  }

  return null
}
