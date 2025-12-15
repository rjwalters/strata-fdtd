/**
 * Probe marker visualization
 */

import { useMemo } from 'react'
import type { ProbeInfo } from '../../lib/scriptParser'

interface ProbeMarkerProps {
  probe: ProbeInfo
  sliceAxis: 'none' | 'xy' | 'xz' | 'yz'
  slicePosition: number
  extent: [number, number, number]
  dualSliceMode: boolean
  slice1Position: number
  slice2Position: number
}

export function ProbeMarker({ probe, sliceAxis, slicePosition, extent, dualSliceMode, slice1Position, slice2Position }: ProbeMarkerProps) {
  // Check if probe is near slice plane(s)
  const isVisible = useMemo(() => {
    if (sliceAxis === 'none') return true

    const axisIndex = sliceAxis === 'yz' ? 0 : sliceAxis === 'xz' ? 1 : 2
    const probePos = probe.position[axisIndex]
    const tolerance = 0.01  // 10mm tolerance for point visibility

    if (dualSliceMode) {
      // In dual slice mode, show probe if within the slab between the two slices
      const threshold1 = extent[axisIndex] * slice1Position
      const threshold2 = extent[axisIndex] * slice2Position
      const minThreshold = Math.min(threshold1, threshold2)
      const maxThreshold = Math.max(threshold1, threshold2)
      // Show if within the slab (with tolerance at boundaries)
      return probePos >= minThreshold - tolerance && probePos <= maxThreshold + tolerance
    } else {
      // Single slice mode: check if probe is near the slice plane
      const threshold = extent[axisIndex] * slicePosition
      return Math.abs(probePos - threshold) < tolerance
    }
  }, [probe, sliceAxis, slicePosition, extent, dualSliceMode, slice1Position, slice2Position])

  if (!isVisible) return null

  return (
    <mesh position={probe.position as [number, number, number]}>
      <boxGeometry args={[0.002, 0.002, 0.002]} />  {/* 2mm cube */}
      <meshBasicMaterial color={0x00e676} />  {/* Green */}
    </mesh>
  )
}
