/**
 * Grid overlay for slice planes - provides visual reference for spatial reasoning
 */

import { useMemo } from 'react'
import * as THREE from 'three'

interface SlicePlaneGridProps {
  axis: 'xy' | 'xz' | 'yz'
  position: number  // 0-1 range
  extent: [number, number, number]
  gridSpacing?: number  // Grid cell size in meters
  color?: string
  opacity?: number
}

export function SlicePlaneGrid({
  axis,
  position,
  extent,
  gridSpacing,
  color = '#666666',
  opacity = 0.5,
}: SlicePlaneGridProps) {
  const geometry = useMemo(() => {
    // Determine plane dimensions and position based on axis
    let size1: number, size2: number
    let planePosition: [number, number, number]

    switch (axis) {
      case 'xy':
        size1 = extent[0]
        size2 = extent[1]
        planePosition = [extent[0] / 2, extent[1] / 2, extent[2] * position]
        break
      case 'xz':
        size1 = extent[0]
        size2 = extent[2]
        planePosition = [extent[0] / 2, extent[1] * position, extent[2] / 2]
        break
      case 'yz':
        size1 = extent[1]
        size2 = extent[2]
        planePosition = [extent[0] * position, extent[1] / 2, extent[2] / 2]
        break
    }

    // Calculate adaptive grid spacing based on plane size (target ~20-40 divisions)
    const maxDim = Math.max(size1, size2)
    const defaultSpacing = gridSpacing ?? maxDim / 20

    const divisions1 = Math.ceil(size1 / defaultSpacing)
    const divisions2 = Math.ceil(size2 / defaultSpacing)
    const actualSpacing1 = size1 / divisions1
    const actualSpacing2 = size2 / divisions2

    // Generate grid line vertices
    const points: number[] = []

    // Lines along first axis
    for (let i = 0; i <= divisions1; i++) {
      const t = i * actualSpacing1 - size1 / 2
      points.push(t, -size2 / 2, 0)
      points.push(t, size2 / 2, 0)
    }

    // Lines along second axis
    for (let j = 0; j <= divisions2; j++) {
      const t = j * actualSpacing2 - size2 / 2
      points.push(-size1 / 2, t, 0)
      points.push(size1 / 2, t, 0)
    }

    const geom = new THREE.BufferGeometry()
    geom.setAttribute('position', new THREE.Float32BufferAttribute(points, 3))

    return { geometry: geom, planePosition }
  }, [axis, position, extent, gridSpacing])

  // Calculate rotation to align grid with slice plane
  const rotation = useMemo((): [number, number, number] => {
    switch (axis) {
      case 'xy':
        return [0, 0, 0]
      case 'xz':
        return [Math.PI / 2, 0, 0]
      case 'yz':
        return [0, 0, Math.PI / 2]
    }
  }, [axis])

  return (
    <lineSegments
      position={geometry.planePosition}
      rotation={rotation}
      geometry={geometry.geometry}
      renderOrder={1}
    >
      <lineBasicMaterial
        color={color}
        opacity={opacity}
        transparent
        depthTest={true}
        depthWrite={false}
      />
    </lineSegments>
  )
}
