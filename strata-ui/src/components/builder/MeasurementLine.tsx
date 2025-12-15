/**
 * Measurement line visualization for distance measurement on slice plane
 */

import { useMemo } from 'react'
import { Line, Html } from '@react-three/drei'
import * as THREE from 'three'
import type { MeasurementPoint } from '../../stores/builderStore'

interface MeasurementLineProps {
  points: MeasurementPoint[]
}

export function MeasurementLine({ points }: MeasurementLineProps) {
  // Memoize all Vector3 calculations to avoid recreating on every render
  const { start, end, midpoint, formattedDistance } = useMemo(() => {
    if (points.length < 2) {
      return { start: null, end: null, midpoint: null, formattedDistance: '' }
    }

    const start = new THREE.Vector3(points[0].x, points[0].y, points[0].z)
    const end = new THREE.Vector3(points[1].x, points[1].y, points[1].z)
    const distance = start.distanceTo(end)
    const midpoint = new THREE.Vector3().lerpVectors(start, end, 0.5)

    // Format distance based on magnitude
    let formattedDistance: string
    if (distance < 0.01) {
      formattedDistance = `${(distance * 1000).toFixed(2)} mm`
    } else if (distance < 1) {
      formattedDistance = `${(distance * 100).toFixed(1)} cm`
    } else {
      formattedDistance = `${distance.toFixed(3)} m`
    }

    return { start, end, midpoint, formattedDistance }
  }, [points])

  if (!start || !end || !midpoint) {
    return null
  }

  return (
    <group>
      {/* Measurement line */}
      <Line
        points={[start, end]}
        color="yellow"
        lineWidth={2}
      />

      {/* Start point marker */}
      <mesh position={start}>
        <sphereGeometry args={[0.002]} />
        <meshBasicMaterial color="yellow" />
      </mesh>

      {/* End point marker */}
      <mesh position={end}>
        <sphereGeometry args={[0.002]} />
        <meshBasicMaterial color="yellow" />
      </mesh>

      {/* Distance label */}
      <Html position={midpoint} center>
        <div className="bg-black bg-opacity-75 text-yellow-400 px-2 py-1 rounded text-xs whitespace-nowrap pointer-events-none">
          {formattedDistance}
        </div>
      </Html>
    </group>
  )
}
