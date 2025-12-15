/**
 * Slice plane visualization for 3D preview
 */

import { useMemo } from 'react'
import * as THREE from 'three'
import type { ThreeEvent } from '@react-three/fiber'
import type { MeasurementPoint } from '../../stores/builderStore'

interface SlicePlaneProps {
  axis: 'xy' | 'xz' | 'yz'
  position: number  // 0-1 range
  extent: [number, number, number]
  onClick?: (point: MeasurementPoint) => void
  measurementMode?: boolean
  color?: string
}

export function SlicePlane({ axis, position, extent, onClick, measurementMode = false, color = '#4a90e2' }: SlicePlaneProps) {
  const [planePosition, planeRotation, planeSize] = useMemo(() => {
    switch (axis) {
      case 'xy':
        return [
          [extent[0] / 2, extent[1] / 2, extent[2] * position] as [number, number, number],
          [0, 0, 0] as [number, number, number],
          [extent[0], extent[1]] as [number, number],
        ]
      case 'xz':
        return [
          [extent[0] / 2, extent[1] * position, extent[2] / 2] as [number, number, number],
          [Math.PI / 2, 0, 0] as [number, number, number],
          [extent[0], extent[2]] as [number, number],
        ]
      case 'yz':
        return [
          [extent[0] * position, extent[1] / 2, extent[2] / 2] as [number, number, number],
          [0, 0, Math.PI / 2] as [number, number, number],
          [extent[1], extent[2]] as [number, number],
        ]
    }
  }, [axis, position, extent])

  const handleClick = (event: ThreeEvent<MouseEvent>) => {
    if (!onClick || !measurementMode) return

    event.stopPropagation()
    const point = event.point
    onClick({ x: point.x, y: point.y, z: point.z })
  }

  return (
    <mesh
      position={planePosition}
      rotation={planeRotation}
      onClick={handleClick}
    >
      <planeGeometry args={planeSize} />
      <meshBasicMaterial
        color={color}
        transparent
        opacity={0.2}
        side={THREE.DoubleSide}
        depthWrite={false}
      />
    </mesh>
  )
}
