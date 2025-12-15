/**
 * Source marker visualization
 */

import { useMemo } from 'react'
import type { SourceInfo } from '@/lib/scriptParser'

interface SourceMarkerProps {
  source: SourceInfo
  sliceAxis: 'none' | 'xy' | 'xz' | 'yz'
  slicePosition: number
  extent: [number, number, number]
  dualSliceMode: boolean
  slice1Position: number
  slice2Position: number
}

/**
 * Source type color mapping
 */
const SOURCE_COLORS: Record<string, number> = {
  GaussianPulse: 0xff9800,     // Orange
  ContinuousWave: 0xf44336,    // Red
  ToneBurst: 0xff5722,         // Deep orange
}

const DEFAULT_COLOR = 0xffc107  // Amber

function getSourceColor(sourceType: string): number {
  return SOURCE_COLORS[sourceType] ?? DEFAULT_COLOR
}

export function SourceMarker({ source, sliceAxis, slicePosition, extent, dualSliceMode, slice1Position, slice2Position }: SourceMarkerProps) {
  const color = getSourceColor(source.type)

  // Check if source is near slice plane(s)
  const isVisible = useMemo(() => {
    if (sliceAxis === 'none') return true

    const axisIndex = sliceAxis === 'yz' ? 0 : sliceAxis === 'xz' ? 1 : 2
    const sourcePos = source.position[axisIndex]
    const tolerance = 0.01  // 10mm tolerance for point visibility

    if (dualSliceMode) {
      // In dual slice mode, show source if within the slab between the two slices
      const threshold1 = extent[axisIndex] * slice1Position
      const threshold2 = extent[axisIndex] * slice2Position
      const minThreshold = Math.min(threshold1, threshold2)
      const maxThreshold = Math.max(threshold1, threshold2)
      // Show if within the slab (with tolerance at boundaries)
      return sourcePos >= minThreshold - tolerance && sourcePos <= maxThreshold + tolerance
    } else {
      // Single slice mode: check if source is near the slice plane
      const threshold = extent[axisIndex] * slicePosition
      return Math.abs(sourcePos - threshold) < tolerance
    }
  }, [source, sliceAxis, slicePosition, extent, dualSliceMode, slice1Position, slice2Position])

  if (!isVisible) return null

  return (
    <mesh position={source.position as [number, number, number]}>
      <sphereGeometry args={[0.003, 16, 16]} />  {/* 3mm radius */}
      <meshBasicMaterial color={color} />
    </mesh>
  )
}
