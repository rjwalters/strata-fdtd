/**
 * Optimized VoxelRenderer with adaptive downsampling and sparse rendering.
 *
 * Key optimizations:
 * 1. Adaptive downsampling for large grids (>64³) via Web Worker
 * 2. Threshold-based sparse rendering (only render significant voxels)
 * 3. Dynamic instance count based on visible voxels
 * 4. Performance tracking and metrics
 * 5. Off-main-thread data processing with automatic fallback
 *
 * The component uses useDataWorker to process downsampling in a Web Worker,
 * keeping the main thread responsive during playback. Double-buffering ensures
 * smooth visual updates by only swapping to new data when processing completes.
 */

import {
  useEffect,
  useRef,
  useMemo,
  useImperativeHandle,
  forwardRef,
  useState,
} from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  type Colormap,
  pressureColormap,
  applyPressureColormap,
  getSymmetricRange,
} from "../lib/colormap";
import { createGeometryMesh, type GeometryMode } from "./GeometryOverlay";
import { createDemoGeometry, createDemoPressure, type DemoGeometryType } from "../lib/demoGeometry";
import {
  PerformanceTracker,
  type PerformanceMetrics,
  type DownsampleResult,
} from "../lib/performance";
import { PerformanceMetrics as PerformanceMetricsDisplay } from "./PerformanceMetrics";
import { useDataWorker } from "../hooks/useDataWorker";

export type VoxelGeometry = "point" | "mesh" | "hidden";

export interface VoxelRendererHandle {
  getCanvas: () => HTMLCanvasElement | null;
  render: () => void;
  getScene: () => THREE.Scene | null;
}

export interface OptimizedVoxelRendererProps {
  pressure: Float32Array | null;
  shape: [number, number, number];
  resolution: number;
  colormap?: Colormap;
  threshold?: number;
  displayFill?: number; // 0-1, percentage of voxels to display
  geometry?: VoxelGeometry;
  voxelScale?: number;
  showGrid?: boolean;
  showAxes?: boolean;
  onReady?: () => void;
  geometryMode?: GeometryMode;
  demoType?: DemoGeometryType;
  geometryOpacity?: number;
  // Performance options
  enableDownsampling?: boolean;
  targetVoxels?: number;
  downsampleMethod?: "nearest" | "average" | "max";
  showPerformanceMetrics?: boolean;
  onPerformanceUpdate?: (metrics: PerformanceMetrics) => void;
}

// Pre-allocated objects for color updates
const tempColor = new THREE.Color();

export const OptimizedVoxelRenderer = forwardRef<
  VoxelRendererHandle,
  OptimizedVoxelRendererProps
>(function OptimizedVoxelRenderer(
  {
    pressure,
    shape,
    resolution,
    colormap = pressureColormap,
    threshold = 0,
    displayFill = 1,
    geometry = "cube",
    voxelScale = 0.8,
    showGrid = true,
    showAxes = true,
    onReady,
    geometryMode = "wireframe",
    demoType = "helmholtz",
    geometryOpacity = 0.3,
    enableDownsampling = true,
    targetVoxels = 262144, // 64³
    downsampleMethod = "average",
    showPerformanceMetrics = false,
    onPerformanceUpdate,
  },
  ref
) {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const meshRef = useRef<THREE.InstancedMesh | null>(null);
  const pointsRef = useRef<THREE.Points | null>(null);
  const gridRef = useRef<THREE.GridHelper | null>(null);
  const axesRef = useRef<THREE.AxesHelper | null>(null);
  const boundaryMeshRef = useRef<THREE.Group | null>(null);
  const animationIdRef = useRef<number>(0);
  const performanceTrackerRef = useRef(new PerformanceTracker());
  const lastRenderCountRef = useRef(0);

  // Performance metrics state
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);

  // Web Worker for off-main-thread data processing
  // The hook handles fallback to main thread if worker is unavailable
  const { downsample } = useDataWorker();

  // Double-buffering state for async data processing
  // effectiveData holds the currently displayed data
  // We use a request ID to handle race conditions from rapid updates
  const [effectiveData, setEffectiveData] = useState<DownsampleResult | null>(null);
  const requestIdRef = useRef(0);
  const processingRef = useRef(false);

  // Expose canvas, scene, and render method via ref
  useImperativeHandle(
    ref,
    () => ({
      getCanvas: () => rendererRef.current?.domElement ?? null,
      render: () => {
        if (rendererRef.current && sceneRef.current && cameraRef.current) {
          rendererRef.current.render(sceneRef.current, cameraRef.current);
        }
      },
      getScene: () => sceneRef.current,
    }),
    []
  );

  const [nx, ny, nz] = shape;
  const totalCells = nx * ny * nz;

  // Track demo animation time
  const [demoTime, setDemoTime] = useState(0);

  // Animate demo pressure when no simulation is loaded
  useEffect(() => {
    if (pressure) return; // Don't animate if real data is loaded

    const interval = setInterval(() => {
      setDemoTime((t) => (t + 0.02) % 1);
    }, 50); // 20 FPS animation

    return () => clearInterval(interval);
  }, [pressure]);

  // Async data processing with Web Worker
  // Uses double-buffering: only updates effectiveData when new result is ready
  useEffect(() => {
    // Use demo pressure if no simulation data is loaded
    const sourceData = pressure ?? createDemoPressure(demoType, shape, demoTime);

    // No downsampling needed - update synchronously for responsiveness
    if (!enableDownsampling || totalCells <= targetVoxels) {
      setEffectiveData({
        data: sourceData,
        shape: shape,
        factor: 1,
        originalShape: shape,
      } as DownsampleResult);
      return;
    }

    // Increment request ID to handle race conditions
    const currentRequestId = ++requestIdRef.current;

    // Skip if already processing and this isn't a higher priority request
    if (processingRef.current) {
      // New request will supersede the in-flight one via requestId check
    }

    processingRef.current = true;

    // Process in Web Worker (or fallback to main thread if unavailable)
    downsample(sourceData, shape, {
      targetVoxels,
      method: downsampleMethod,
    })
      .then((result) => {
        // Only update if this is still the latest request (handles race conditions)
        if (currentRequestId === requestIdRef.current) {
          setEffectiveData(result);
        }
      })
      .catch((error) => {
        console.error("Downsampling error:", error);
        // On error, fall back to showing unprocessed data
        if (currentRequestId === requestIdRef.current) {
          setEffectiveData({
            data: sourceData,
            shape: shape,
            factor: 1,
            originalShape: shape,
          } as DownsampleResult);
        }
      })
      .finally(() => {
        if (currentRequestId === requestIdRef.current) {
          processingRef.current = false;
        }
      });
  }, [pressure, demoType, demoTime, shape, totalCells, enableDownsampling, targetVoxels, downsampleMethod, downsample]);

  // Create a stable random display mask based on displayFill
  // Uses a seeded approach so the same voxels are hidden consistently
  const displayMask = useMemo(() => {
    if (!effectiveData) return null;
    const total = effectiveData.shape[0] * effectiveData.shape[1] * effectiveData.shape[2];
    const mask = new Uint8Array(total);

    if (displayFill >= 1) {
      // Show all voxels
      mask.fill(1);
    } else {
      // Use deterministic pseudo-random selection based on index
      // This ensures consistent voxel selection across frames
      for (let i = 0; i < total; i++) {
        // Simple hash function for deterministic randomness
        const hash = ((i * 2654435761) >>> 0) / 4294967296; // Knuth multiplicative hash
        mask[i] = hash < displayFill ? 1 : 0;
      }
    }
    return mask;
  }, [effectiveData?.shape[0], effectiveData?.shape[1], effectiveData?.shape[2], displayFill]);

  // No longer needed - point and mesh geometries are created inline

  // Initialize scene
  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);
    sceneRef.current = scene;

    // Camera setup
    const maxDim = Math.max(nx, ny, nz) * resolution;
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.01, maxDim * 10);
    const cameraDistance = maxDim * 1.5;
    camera.position.set(cameraDistance, cameraDistance * 0.8, cameraDistance);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // OrbitControls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.target.set(0, 0, 0);
    controlsRef.current = controls;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(maxDim, maxDim, maxDim);
    scene.add(directionalLight);

    // Performance tracking animation loop
    const tracker = performanceTrackerRef.current;
    let frameCount = 0;

    const animate = () => {
      animationIdRef.current = requestAnimationFrame(animate);
      tracker.startFrame();

      controls.update();
      renderer.render(scene, camera);

      tracker.endFrame();
      frameCount++;

      // Update metrics every 30 frames
      if (frameCount % 30 === 0) {
        const currentMetrics = tracker.getMetrics();
        setMetrics(currentMetrics);
        onPerformanceUpdate?.(currentMetrics);
      }
    };
    animate();

    // Handle resize
    const handleResize = () => {
      if (!containerRef.current || !rendererRef.current || !cameraRef.current)
        return;
      const newWidth = containerRef.current.clientWidth;
      const newHeight = containerRef.current.clientHeight;
      cameraRef.current.aspect = newWidth / newHeight;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(newWidth, newHeight);
    };

    const resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(container);

    onReady?.();

    return () => {
      cancelAnimationFrame(animationIdRef.current);
      resizeObserver.disconnect();
      controls.dispose();
      renderer.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [nx, ny, nz, resolution, onReady, onPerformanceUpdate]);

  // Create/update instanced mesh when shape, geometry, or downsample settings change
  useEffect(() => {
    if (!sceneRef.current || !effectiveData) return;

    const scene = sceneRef.current;
    const [ex, ey, ez] = effectiveData.shape;
    const effectiveTotal = ex * ey * ez;

    // Calculate resolution scaling for downsampled data
    const scaleFactor = effectiveData.factor;
    const effectiveResolution = resolution * scaleFactor;

    // Remove old mesh/points
    if (meshRef.current) {
      scene.remove(meshRef.current);
      meshRef.current.geometry.dispose();
      if (meshRef.current.material instanceof THREE.Material) {
        meshRef.current.material.dispose();
      }
      meshRef.current = null;
    }
    if (pointsRef.current) {
      scene.remove(pointsRef.current);
      pointsRef.current.geometry.dispose();
      if (pointsRef.current.material instanceof THREE.Material) {
        pointsRef.current.material.dispose();
      }
      pointsRef.current = null;
    }

    // Calculate grid offset to center at origin (use original shape for centering)
    const offsetX = ((nx - 1) * resolution) / 2;
    const offsetY = ((ny - 1) * resolution) / 2;
    const offsetZ = ((nz - 1) * resolution) / 2;

    if (geometry === "hidden") {
      // Don't render any voxel geometry
    } else if (geometry === "point") {
      // Points for maximum performance
      const positions = new Float32Array(effectiveTotal * 3);
      const colors = new Float32Array(effectiveTotal * 3);

      let idx = 0;
      for (let x = 0; x < ex; x++) {
        for (let y = 0; y < ey; y++) {
          for (let z = 0; z < ez; z++) {
            positions[idx * 3] = x * effectiveResolution - offsetX;
            positions[idx * 3 + 1] = y * effectiveResolution - offsetY;
            positions[idx * 3 + 2] = z * effectiveResolution - offsetZ;
            colors[idx * 3] = 1;
            colors[idx * 3 + 1] = 1;
            colors[idx * 3 + 2] = 1;
            idx++;
          }
        }
      }

      const pointsGeometry = new THREE.BufferGeometry();
      pointsGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      pointsGeometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

      const pointsMaterial = new THREE.PointsMaterial({
        size: voxelScale * effectiveResolution * 0.1, // Small points like tiny spheres
        vertexColors: true,
        sizeAttenuation: true,
      });

      const points = new THREE.Points(pointsGeometry, pointsMaterial);
      scene.add(points);
      pointsRef.current = points;
    } else if (geometry === "mesh") {
      // Wireframe mesh - create lines connecting voxel centers
      // Each voxel connects to its +X, +Y, +Z neighbors (to avoid duplicate lines)
      const linePositions: number[] = [];
      const lineColors: number[] = [];
      const orangeColor = new THREE.Color(0xff6600);

      for (let x = 0; x < ex; x++) {
        for (let y = 0; y < ey; y++) {
          for (let z = 0; z < ez; z++) {
            const px = x * effectiveResolution - offsetX;
            const py = y * effectiveResolution - offsetY;
            const pz = z * effectiveResolution - offsetZ;

            // Connect to +X neighbor
            if (x < ex - 1) {
              linePositions.push(px, py, pz);
              linePositions.push(px + effectiveResolution, py, pz);
              lineColors.push(orangeColor.r, orangeColor.g, orangeColor.b);
              lineColors.push(orangeColor.r, orangeColor.g, orangeColor.b);
            }
            // Connect to +Y neighbor
            if (y < ey - 1) {
              linePositions.push(px, py, pz);
              linePositions.push(px, py + effectiveResolution, pz);
              lineColors.push(orangeColor.r, orangeColor.g, orangeColor.b);
              lineColors.push(orangeColor.r, orangeColor.g, orangeColor.b);
            }
            // Connect to +Z neighbor
            if (z < ez - 1) {
              linePositions.push(px, py, pz);
              linePositions.push(px, py, pz + effectiveResolution);
              lineColors.push(orangeColor.r, orangeColor.g, orangeColor.b);
              lineColors.push(orangeColor.r, orangeColor.g, orangeColor.b);
            }
          }
        }
      }

      const lineGeometry = new THREE.BufferGeometry();
      lineGeometry.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(linePositions, 3)
      );
      lineGeometry.setAttribute(
        "color",
        new THREE.Float32BufferAttribute(lineColors, 3)
      );

      const lineMaterial = new THREE.LineBasicMaterial({
        vertexColors: true,
        linewidth: 1, // Note: linewidth > 1 only works on some platforms
        transparent: true,
        opacity: 0.6,
      });

      const lineSegments = new THREE.LineSegments(lineGeometry, lineMaterial);
      scene.add(lineSegments);
      // Store in meshRef for cleanup (reusing the ref)
      meshRef.current = lineSegments as unknown as THREE.InstancedMesh;
    }

    // Update performance tracker
    performanceTrackerRef.current.updateVoxelCounts(
      effectiveTotal,
      totalCells,
      scaleFactor
    );
  }, [
    effectiveData,
    nx,
    ny,
    nz,
    totalCells,
    resolution,
    geometry,
    voxelScale,
  ]);

  // Update colors when pressure data changes (optimized with threshold and display fill filtering)
  useEffect(() => {
    if (!effectiveData || !displayMask) return;

    const { data, shape: effShape } = effectiveData;
    const [ex, ey, ez] = effShape;
    const effectiveTotal = ex * ey * ez;

    if (data.length !== effectiveTotal) return;

    // Hidden and mesh modes don't need color updates
    if (geometry === "hidden" || geometry === "mesh") {
      lastRenderCountRef.current = geometry === "mesh" ? effectiveTotal : 0;
      performanceTrackerRef.current.updateVoxelCounts(
        lastRenderCountRef.current,
        totalCells,
        effectiveData.factor
      );
      return;
    }

    const [min, max] = getSymmetricRange(data);
    const thresholdValue = threshold * max;

    if (geometry === "point" && pointsRef.current) {
      const colors = pointsRef.current.geometry.attributes.color as THREE.BufferAttribute;
      const colorArray = colors.array as Float32Array;

      let visibleCount = 0;
      for (let i = 0; i < effectiveTotal; i++) {
        const value = data[i];
        // Check both threshold AND display mask
        if (Math.abs(value) < thresholdValue || displayMask[i] === 0) {
          colorArray[i * 3] = 0.1;
          colorArray[i * 3 + 1] = 0.1;
          colorArray[i * 3 + 2] = 0.1;
        } else {
          applyPressureColormap(value, min, max, tempColor);
          colorArray[i * 3] = tempColor.r;
          colorArray[i * 3 + 1] = tempColor.g;
          colorArray[i * 3 + 2] = tempColor.b;
          visibleCount++;
        }
      }

      colors.needsUpdate = true;
      lastRenderCountRef.current = visibleCount;
    }

    // Update performance tracker with actual rendered count
    performanceTrackerRef.current.updateVoxelCounts(
      lastRenderCountRef.current,
      totalCells,
      effectiveData.factor
    );
  }, [effectiveData, displayMask, threshold, colormap, geometry, resolution, nx, ny, nz, totalCells]);

  // Update grid helper visibility
  useEffect(() => {
    if (!sceneRef.current) return;
    const scene = sceneRef.current;

    if (gridRef.current) {
      scene.remove(gridRef.current);
      gridRef.current.dispose();
      gridRef.current = null;
    }

    if (showGrid) {
      const gridSize = Math.max(nx, nz) * resolution * 1.2;
      const divisions = Math.max(nx, nz);
      const grid = new THREE.GridHelper(gridSize, divisions, 0x444444, 0x333333);
      grid.position.y = -((ny - 1) * resolution) / 2 - resolution / 2;
      scene.add(grid);
      gridRef.current = grid;
    }
  }, [showGrid, nx, ny, nz, resolution]);

  // Update axes helper visibility
  useEffect(() => {
    if (!sceneRef.current) return;
    const scene = sceneRef.current;

    if (axesRef.current) {
      scene.remove(axesRef.current);
      axesRef.current.dispose();
      axesRef.current = null;
    }

    if (showAxes) {
      const axesSize = Math.max(nx, ny, nz) * resolution * 0.6;
      const axes = new THREE.AxesHelper(axesSize);
      scene.add(axes);
      axesRef.current = axes;
    }
  }, [showAxes, nx, ny, nz, resolution]);

  // Update boundary geometry overlay
  useEffect(() => {
    if (!sceneRef.current) return;
    const scene = sceneRef.current;

    if (boundaryMeshRef.current) {
      scene.remove(boundaryMeshRef.current);
      boundaryMeshRef.current.traverse((child) => {
        if (child instanceof THREE.Mesh) {
          child.geometry.dispose();
          if (Array.isArray(child.material)) {
            child.material.forEach((m) => m.dispose());
          } else {
            child.material.dispose();
          }
        } else if (child instanceof THREE.LineSegments) {
          child.geometry.dispose();
          if (Array.isArray(child.material)) {
            child.material.forEach((m) => m.dispose());
          } else {
            child.material.dispose();
          }
        }
      });
      boundaryMeshRef.current = null;
    }

    if (geometryMode !== "hidden") {
      const boundaryData = createDemoGeometry(demoType, shape);
      const boundaryGroup = createGeometryMesh(
        boundaryData,
        shape,
        resolution,
        geometryMode,
        undefined,
        geometryOpacity
      );
      scene.add(boundaryGroup);
      boundaryMeshRef.current = boundaryGroup;
    }
  }, [geometryMode, demoType, shape, resolution, geometryOpacity]);

  // Memoize stats for display
  const stats = useMemo(() => {
    if (!effectiveData) return null;
    const [min, max] = getSymmetricRange(effectiveData.data);
    return {
      min,
      max,
      count: lastRenderCountRef.current,
      total: totalCells,
      factor: effectiveData.factor,
    };
  }, [effectiveData, totalCells]);

  return (
    <div className="relative w-full h-full">
      <div ref={containerRef} className="w-full h-full" style={{ minHeight: "200px" }} />

      {/* Stats overlay */}
      {stats && (
        <div className="absolute bottom-2 left-2 bg-black/60 text-white text-xs px-2 py-1 rounded font-mono">
          Range: [{stats.min.toFixed(3)}, {stats.max.toFixed(3)}] | Rendered:{" "}
          {stats.count.toLocaleString()}
          {stats.factor > 1 && (
            <span className="text-blue-400"> ({stats.factor}x downsampled)</span>
          )}
        </div>
      )}

      {/* Performance metrics overlay */}
      {showPerformanceMetrics && (
        <div className="absolute top-2 right-2">
          <PerformanceMetricsDisplay metrics={metrics} />
        </div>
      )}
    </div>
  );
});
