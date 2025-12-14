/**
 * 3D preview component for simulation builder
 */

import { Suspense } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Grid as DreiGrid } from '@react-three/drei'
import { GridBox } from './GridBox'
import { MaterialRegion } from './MaterialRegion'
import { SourceMarker } from './SourceMarker'
import { ProbeMarker } from './ProbeMarker'
import { SlicePlane } from './SlicePlane'
import { MeasurementLine } from './MeasurementLine'
import { useBuilderStore, type MeasurementPoint } from '@/stores/builderStore'
import type { SimulationAST } from '@/lib/scriptParser'

interface Preview3DProps {
  ast: SimulationAST | null
  showGrid: boolean
  showMaterials: boolean
  showSources: boolean
  showProbes: boolean
  sliceAxis: 'none' | 'xy' | 'xz' | 'yz'
  slicePosition: number
  measurementMode: boolean
  measurementPoints: MeasurementPoint[]
}

function Scene({
  ast,
  showGrid,
  showMaterials,
  showSources,
  showProbes,
  sliceAxis,
  slicePosition,
  measurementMode,
  measurementPoints,
}: Preview3DProps) {
  const addMeasurementPoint = useBuilderStore((s) => s.addMeasurementPoint)

  if (!ast?.grid) {
    return (
      <group>
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} />
      </group>
    )
  }

  const grid = ast.grid

  // Handle click on slice plane for measurements
  const handleSliceClick = (point: MeasurementPoint) => {
    if (measurementMode && sliceAxis !== 'none') {
      addMeasurementPoint(point)
    }
  }

  // Calculate camera distance based on grid size
  const maxExtent = Math.max(...grid.extent)
  const cameraDistance = maxExtent * 1.5

  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.6} />
      <pointLight position={[cameraDistance, cameraDistance, cameraDistance]} intensity={0.8} />
      <pointLight position={[-cameraDistance, -cameraDistance, cameraDistance]} intensity={0.4} />

      {/* Grid floor */}
      {showGrid && (
        <DreiGrid
          args={[grid.extent[0], grid.extent[1]]}
          cellSize={grid.resolution * 10}
          cellThickness={0.5}
          cellColor="#6e6e6e"
          sectionSize={grid.resolution * 50}
          sectionThickness={1}
          sectionColor="#9d4b4b"
          fadeDistance={maxExtent * 3}
          fadeStrength={1}
          position={[grid.extent[0] / 2, 0, grid.extent[2] / 2]}
          rotation={[-Math.PI / 2, 0, 0]}
        />
      )}

      {/* Grid bounding box */}
      {showGrid && <GridBox grid={grid} />}

      {/* Slice plane */}
      {sliceAxis !== 'none' && (
        <SlicePlane
          axis={sliceAxis}
          position={slicePosition}
          extent={grid.extent}
          onClick={handleSliceClick}
          measurementMode={measurementMode}
        />
      )}

      {/* Measurement line */}
      {measurementMode && measurementPoints.length > 0 && (
        <MeasurementLine points={measurementPoints} />
      )}

      {/* Material regions */}
      {showMaterials &&
        ast.materials.map((material) => (
          <MaterialRegion
            key={material.id}
            region={material}
            sliceAxis={sliceAxis}
            slicePosition={slicePosition}
            extent={grid.extent}
          />
        ))}

      {/* Sources */}
      {showSources &&
        ast.sources.map((source) => (
          <SourceMarker
            key={source.id}
            source={source}
            sliceAxis={sliceAxis}
            slicePosition={slicePosition}
            extent={grid.extent}
          />
        ))}

      {/* Probes */}
      {showProbes &&
        ast.probes.map((probe) => (
          <ProbeMarker
            key={probe.id}
            probe={probe}
            sliceAxis={sliceAxis}
            slicePosition={slicePosition}
            extent={grid.extent}
          />
        ))}
    </>
  )
}

function Fallback() {
  return (
    <mesh>
      <boxGeometry args={[1, 1, 1]} />
      <meshBasicMaterial color="#666" wireframe />
    </mesh>
  )
}

export function Preview3D(props: Preview3DProps) {
  const { ast } = props

  // Calculate camera position based on grid
  const cameraPosition: [number, number, number] = ast?.grid
    ? [
        ast.grid.extent[0] * 1.2,
        ast.grid.extent[1] * 1.2,
        ast.grid.extent[2] * 1.2,
      ]
    : [0.15, 0.15, 0.15]

  const cameraTarget: [number, number, number] = ast?.grid
    ? [
        ast.grid.extent[0] / 2,
        ast.grid.extent[1] / 2,
        ast.grid.extent[2] / 2,
      ]
    : [0.05, 0.05, 0.05]

  return (
    <div className="w-full h-full bg-gray-900">
      {!ast?.hasValidGrid ? (
        <div className="w-full h-full flex items-center justify-center text-gray-400">
          <div className="text-center">
            <p className="text-lg font-medium">Waiting for valid grid configuration...</p>
            <p className="text-sm mt-2">
              Define a UniformGrid in your script to see the preview
            </p>
            {ast?.errors && ast.errors.length > 0 && (
              <div className="mt-4 text-sm text-yellow-400">
                {ast.errors.map((error, i) => (
                  <p key={i}>• {error}</p>
                ))}
              </div>
            )}
          </div>
        </div>
      ) : (
        <Canvas
          camera={{
            position: cameraPosition,
            fov: 50,
          }}
          gl={{ antialias: true, alpha: true }}
        >
          <Suspense fallback={<Fallback />}>
            <Scene {...props} />
            <OrbitControls
              target={cameraTarget}
              enableDamping
              dampingFactor={0.05}
              minDistance={0.01}
              maxDistance={10}
            />
          </Suspense>
        </Canvas>
      )}
    </div>
  )
}
