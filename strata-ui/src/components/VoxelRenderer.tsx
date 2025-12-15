import { useEffect, useRef, useMemo, useImperativeHandle, forwardRef } from "react"
import * as THREE from "three"
import { OrbitControls } from "three/addons/controls/OrbitControls.js"
import {
  type Colormap,
  pressureColormap,
  applyPressureColormap,
  getSymmetricRange,
} from "../lib/colormap"
import {
  createGeometryMesh,
  type GeometryMode,
} from "./GeometryOverlay"
import { createDemoGeometry, type DemoGeometryType } from "../lib/demoGeometry"

export type VoxelGeometry = "point" | "mesh" | "hidden"

export interface VoxelRendererHandle {
  /** Get the WebGL canvas element for capture */
  getCanvas: () => HTMLCanvasElement | null
  /** Force a render (useful for capturing specific frames) */
  render: () => void
  /** Get the Three.js scene (for overlays like FlowParticleRenderer) */
  getScene: () => THREE.Scene | null
}

export interface VoxelRendererProps {
  /** Pressure data as flat Float32Array in row-major order (z varies fastest) */
  pressure: Float32Array | null
  /** Grid dimensions [nx, ny, nz] */
  shape: [number, number, number]
  /** Meters per grid cell */
  resolution: number
  /** Colormap function to use */
  colormap?: Colormap
  /** Hide voxels with |pressure| below this threshold (0-1 normalized) */
  threshold?: number
  /** Type of geometry to render */
  geometry?: VoxelGeometry
  /** Voxel scale relative to cell size (0-1) */
  voxelScale?: number
  /** Show grid helper */
  showGrid?: boolean
  /** Show axis helper */
  showAxes?: boolean
  /** Called when renderer is ready */
  onReady?: () => void
  /** Boundary geometry visualization mode */
  geometryMode?: GeometryMode
  /** Demo geometry type for boundary overlay */
  demoType?: DemoGeometryType
  /** Opacity for transparent geometry mode */
  geometryOpacity?: number
}

// Pre-allocated objects for color updates (avoid GC pressure)
const tempColor = new THREE.Color()

export const VoxelRenderer = forwardRef<VoxelRendererHandle, VoxelRendererProps>(function VoxelRenderer({
  pressure,
  shape,
  resolution,
  colormap = pressureColormap,
  threshold = 0,
  geometry = "point",
  voxelScale = 0.8,
  showGrid = true,
  showAxes = true,
  onReady,
  geometryMode = "wireframe",
  demoType = "helmholtz",
  geometryOpacity = 0.3,
}, ref) {
  const containerRef = useRef<HTMLDivElement>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const sceneRef = useRef<THREE.Scene | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const controlsRef = useRef<OrbitControls | null>(null)
  const meshRef = useRef<THREE.InstancedMesh | null>(null)
  const pointsRef = useRef<THREE.Points | null>(null)
  const gridRef = useRef<THREE.GridHelper | null>(null)
  const axesRef = useRef<THREE.AxesHelper | null>(null)
  const boundaryMeshRef = useRef<THREE.Group | null>(null)
  const animationIdRef = useRef<number>(0)
  const instanceCountRef = useRef<number>(0)

  // Expose canvas, render method, and scene via ref
  useImperativeHandle(ref, () => ({
    getCanvas: () => rendererRef.current?.domElement ?? null,
    render: () => {
      if (rendererRef.current && sceneRef.current && cameraRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current)
      }
    },
    getScene: () => sceneRef.current,
  }), [])

  const [nx, ny, nz] = shape
  const totalCells = nx * ny * nz

  // No longer needed - point and mesh geometries are created inline

  // Initialize scene
  useEffect(() => {
    if (!containerRef.current) return

    const container = containerRef.current
    const width = container.clientWidth
    const height = container.clientHeight

    // Scene setup
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x1a1a1a)
    sceneRef.current = scene

    // Camera setup - position to view the entire grid
    const maxDim = Math.max(nx, ny, nz) * resolution
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.01, maxDim * 10)
    // Position camera to see the entire grid
    const cameraDistance = maxDim * 1.5
    camera.position.set(cameraDistance, cameraDistance * 0.8, cameraDistance)
    camera.lookAt(0, 0, 0)
    cameraRef.current = camera

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(width, height)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    container.appendChild(renderer.domElement)
    rendererRef.current = renderer

    // OrbitControls
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.05
    controls.target.set(0, 0, 0)
    controlsRef.current = controls

    // Lighting (for non-point geometries)
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6)
    scene.add(ambientLight)

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8)
    directionalLight.position.set(maxDim, maxDim, maxDim)
    scene.add(directionalLight)

    // Animation loop
    const animate = () => {
      animationIdRef.current = requestAnimationFrame(animate)
      controls.update()
      renderer.render(scene, camera)
    }
    animate()

    // Handle resize
    const handleResize = () => {
      if (!containerRef.current || !rendererRef.current || !cameraRef.current) return
      const newWidth = containerRef.current.clientWidth
      const newHeight = containerRef.current.clientHeight
      cameraRef.current.aspect = newWidth / newHeight
      cameraRef.current.updateProjectionMatrix()
      rendererRef.current.setSize(newWidth, newHeight)
    }

    const resizeObserver = new ResizeObserver(handleResize)
    resizeObserver.observe(container)

    onReady?.()

    // Cleanup
    return () => {
      cancelAnimationFrame(animationIdRef.current)
      resizeObserver.disconnect()
      controls.dispose()
      renderer.dispose()
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement)
      }
    }
  }, [nx, ny, nz, resolution, onReady])

  // Create/update instanced mesh when shape or geometry changes
  useEffect(() => {
    if (!sceneRef.current) return

    const scene = sceneRef.current

    // Remove old mesh/points
    if (meshRef.current) {
      scene.remove(meshRef.current)
      meshRef.current.geometry.dispose()
      if (meshRef.current.material instanceof THREE.Material) {
        meshRef.current.material.dispose()
      }
      meshRef.current = null
    }
    if (pointsRef.current) {
      scene.remove(pointsRef.current)
      pointsRef.current.geometry.dispose()
      if (pointsRef.current.material instanceof THREE.Material) {
        pointsRef.current.material.dispose()
      }
      pointsRef.current = null
    }

    instanceCountRef.current = totalCells

    // Calculate grid offset to center at origin
    const offsetX = ((nx - 1) * resolution) / 2
    const offsetY = ((ny - 1) * resolution) / 2
    const offsetZ = ((nz - 1) * resolution) / 2

    if (geometry === "hidden") {
      // Don't render any voxel geometry
    } else if (geometry === "point") {
      // Use Points for maximum performance
      const positions = new Float32Array(totalCells * 3)
      const colors = new Float32Array(totalCells * 3)

      let idx = 0
      for (let x = 0; x < nx; x++) {
        for (let y = 0; y < ny; y++) {
          for (let z = 0; z < nz; z++) {
            positions[idx * 3] = x * resolution - offsetX
            positions[idx * 3 + 1] = y * resolution - offsetY
            positions[idx * 3 + 2] = z * resolution - offsetZ
            // Initialize colors to white
            colors[idx * 3] = 1
            colors[idx * 3 + 1] = 1
            colors[idx * 3 + 2] = 1
            idx++
          }
        }
      }

      const pointsGeometry = new THREE.BufferGeometry()
      pointsGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3))
      pointsGeometry.setAttribute("color", new THREE.BufferAttribute(colors, 3))

      const pointsMaterial = new THREE.PointsMaterial({
        size: voxelScale * resolution * 0.1, // Small points like tiny spheres
        vertexColors: true,
        sizeAttenuation: true,
      })

      const points = new THREE.Points(pointsGeometry, pointsMaterial)
      scene.add(points)
      pointsRef.current = points
    } else if (geometry === "mesh") {
      // Wireframe mesh - create lines connecting voxel centers
      const linePositions: number[] = []
      const lineColors: number[] = []
      const orangeColor = new THREE.Color(0xff6600)

      for (let x = 0; x < nx; x++) {
        for (let y = 0; y < ny; y++) {
          for (let z = 0; z < nz; z++) {
            const px = x * resolution - offsetX
            const py = y * resolution - offsetY
            const pz = z * resolution - offsetZ

            // Connect to +X neighbor
            if (x < nx - 1) {
              linePositions.push(px, py, pz)
              linePositions.push(px + resolution, py, pz)
              lineColors.push(orangeColor.r, orangeColor.g, orangeColor.b)
              lineColors.push(orangeColor.r, orangeColor.g, orangeColor.b)
            }
            // Connect to +Y neighbor
            if (y < ny - 1) {
              linePositions.push(px, py, pz)
              linePositions.push(px, py + resolution, pz)
              lineColors.push(orangeColor.r, orangeColor.g, orangeColor.b)
              lineColors.push(orangeColor.r, orangeColor.g, orangeColor.b)
            }
            // Connect to +Z neighbor
            if (z < nz - 1) {
              linePositions.push(px, py, pz)
              linePositions.push(px, py, pz + resolution)
              lineColors.push(orangeColor.r, orangeColor.g, orangeColor.b)
              lineColors.push(orangeColor.r, orangeColor.g, orangeColor.b)
            }
          }
        }
      }

      const lineGeometry = new THREE.BufferGeometry()
      lineGeometry.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(linePositions, 3)
      )
      lineGeometry.setAttribute(
        "color",
        new THREE.Float32BufferAttribute(lineColors, 3)
      )

      const lineMaterial = new THREE.LineBasicMaterial({
        vertexColors: true,
        linewidth: 1,
        transparent: true,
        opacity: 0.6,
      })

      const lineSegments = new THREE.LineSegments(lineGeometry, lineMaterial)
      scene.add(lineSegments)
      meshRef.current = lineSegments as unknown as THREE.InstancedMesh
    }
  }, [nx, ny, nz, totalCells, resolution, geometry, voxelScale])

  // Update colors when pressure data changes
  useEffect(() => {
    if (!pressure || pressure.length !== totalCells) return

    // Hidden and mesh modes don't need color updates
    if (geometry === "hidden" || geometry === "mesh") {
      return
    }

    const [min, max] = getSymmetricRange(pressure)
    const thresholdValue = threshold * max

    if (geometry === "point" && pointsRef.current) {
      const colors = pointsRef.current.geometry.attributes.color as THREE.BufferAttribute
      const colorArray = colors.array as Float32Array

      for (let i = 0; i < totalCells; i++) {
        const value = pressure[i]

        // Apply threshold - set to black/transparent if below threshold
        if (Math.abs(value) < thresholdValue) {
          colorArray[i * 3] = 0.1
          colorArray[i * 3 + 1] = 0.1
          colorArray[i * 3 + 2] = 0.1
        } else {
          applyPressureColormap(value, min, max, tempColor)
          colorArray[i * 3] = tempColor.r
          colorArray[i * 3 + 1] = tempColor.g
          colorArray[i * 3 + 2] = tempColor.b
        }
      }

      colors.needsUpdate = true
    }
  }, [pressure, totalCells, threshold, colormap, geometry])

  // Update grid helper visibility
  useEffect(() => {
    if (!sceneRef.current) return
    const scene = sceneRef.current

    // Remove old grid
    if (gridRef.current) {
      scene.remove(gridRef.current)
      gridRef.current.dispose()
      gridRef.current = null
    }

    if (showGrid) {
      const gridSize = Math.max(nx, nz) * resolution * 1.2
      const divisions = Math.max(nx, nz)
      const grid = new THREE.GridHelper(gridSize, divisions, 0x444444, 0x333333)
      // Position grid at bottom of volume
      grid.position.y = -((ny - 1) * resolution) / 2 - resolution / 2
      scene.add(grid)
      gridRef.current = grid
    }
  }, [showGrid, nx, ny, nz, resolution])

  // Update axes helper visibility
  useEffect(() => {
    if (!sceneRef.current) return
    const scene = sceneRef.current

    // Remove old axes
    if (axesRef.current) {
      scene.remove(axesRef.current)
      axesRef.current.dispose()
      axesRef.current = null
    }

    if (showAxes) {
      const axesSize = Math.max(nx, ny, nz) * resolution * 0.6
      const axes = new THREE.AxesHelper(axesSize)
      scene.add(axes)
      axesRef.current = axes
    }
  }, [showAxes, nx, ny, nz, resolution])

  // Update boundary geometry overlay
  useEffect(() => {
    if (!sceneRef.current) return
    const scene = sceneRef.current

    // Remove old boundary mesh
    if (boundaryMeshRef.current) {
      scene.remove(boundaryMeshRef.current)
      boundaryMeshRef.current.traverse((child) => {
        if (child instanceof THREE.Mesh) {
          child.geometry.dispose()
          if (Array.isArray(child.material)) {
            child.material.forEach((m) => m.dispose())
          } else {
            child.material.dispose()
          }
        } else if (child instanceof THREE.LineSegments) {
          child.geometry.dispose()
          if (Array.isArray(child.material)) {
            child.material.forEach((m) => m.dispose())
          } else {
            child.material.dispose()
          }
        }
      })
      boundaryMeshRef.current = null
    }

    if (geometryMode !== "hidden") {
      // Create demo boundary geometry
      const boundaryData = createDemoGeometry(demoType, shape)
      const boundaryGroup = createGeometryMesh(
        boundaryData,
        shape,
        resolution,
        geometryMode,
        undefined,
        geometryOpacity
      )
      scene.add(boundaryGroup)
      boundaryMeshRef.current = boundaryGroup
    }
  }, [geometryMode, demoType, shape, resolution, geometryOpacity])

  // Memoize stats for display
  const stats = useMemo(() => {
    if (!pressure) return null
    const [min, max] = getSymmetricRange(pressure)
    return { min, max, count: totalCells }
  }, [pressure, totalCells])

  return (
    <div className="relative w-full h-full">
      <div ref={containerRef} className="w-full h-full" style={{ minHeight: "200px" }} />
      {stats && (
        <div className="absolute bottom-2 left-2 bg-black/60 text-white text-xs px-2 py-1 rounded font-mono">
          Range: [{stats.min.toFixed(3)}, {stats.max.toFixed(3)}] | Voxels: {stats.count.toLocaleString()}
        </div>
      )}
    </div>
  )
})
