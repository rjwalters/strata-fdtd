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

export function SourceMarker({ source, sliceAxis, slicePosition, extent }: SourceMarkerProps) {
  const color = getSourceColor(source.type)

  // Check if source is near slice plane
  const isVisible = useMemo(() => {
    if (sliceAxis === 'none') return true

    const axisIndex = sliceAxis === 'yz' ? 0 : sliceAxis === 'xz' ? 1 : 2
    const threshold = extent[axisIndex] * slicePosition
    const tolerance = 0.01  // 10mm tolerance for point visibility

    // Check if source is near the slice plane (within tolerance)
    return Math.abs(source.position[axisIndex] - threshold) < tolerance
  }, [source, sliceAxis, slicePosition, extent])

  if (!isVisible) return null

  return (
    <mesh position={source.position as [number, number, number]}>
      <sphereGeometry args={[0.003, 16, 16]} />  {/* 3mm radius */}
      <meshBasicMaterial color={color} />
    </mesh>
  )
}
