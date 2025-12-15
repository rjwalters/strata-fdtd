/**
 * Grid bounding box visualization for 3D preview
 */

import { useMemo } from 'react'
import * as THREE from 'three'
import type { GridInfo } from '@/lib/scriptParser'

interface GridBoxProps {
  grid: GridInfo
}

export function GridBox({ grid }: GridBoxProps) {
  const { geometry, material } = useMemo(() => {
    const [ex, ey, ez] = grid.extent

    // Create edges geometry for wireframe box
    const edges = new THREE.EdgesGeometry(
      new THREE.BoxGeometry(ex, ey, ez)
    )

    const lineMaterial = new THREE.LineBasicMaterial({
      color: 0x888888,
      linewidth: 1,
    })

    return { geometry: edges, material: lineMaterial }
  }, [grid])

  return (
    <lineSegments
      geometry={geometry}
      material={material}
      position={[grid.extent[0] / 2, grid.extent[1] / 2, grid.extent[2] / 2]}
    />
  )
}
