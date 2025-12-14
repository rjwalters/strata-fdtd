/**
 * Probe marker visualization
 */

import { useMemo } from 'react'
import type { ProbeInfo } from '@/lib/scriptParser'

interface ProbeMarkerProps {
  probe: ProbeInfo
  sliceAxis: 'none' | 'xy' | 'xz' | 'yz'
  slicePosition: number
  extent: [number, number, number]
}

export function ProbeMarker({ probe, sliceAxis, slicePosition, extent }: ProbeMarkerProps) {
  // Check if probe is near slice plane
  const isVisible = useMemo(() => {
    if (sliceAxis === 'none') return true

    const axisIndex = sliceAxis === 'yz' ? 0 : sliceAxis === 'xz' ? 1 : 2
    const threshold = extent[axisIndex] * slicePosition
    const tolerance = 0.01  // 10mm tolerance for point visibility

    // Check if probe is near the slice plane (within tolerance)
    return Math.abs(probe.position[axisIndex] - threshold) < tolerance
  }, [probe, sliceAxis, slicePosition, extent])

  if (!isVisible) return null

  return (
    <mesh position={probe.position as [number, number, number]}>
      <boxGeometry args={[0.002, 0.002, 0.002]} />  {/* 2mm cube */}
      <meshBasicMaterial color={0x00e676} />  {/* Green */}
    </mesh>
  )
}
