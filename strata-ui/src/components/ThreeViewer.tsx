import { useEffect, useRef, useMemo, useCallback } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  createGeometryMesh,
  type GeometryMode,
} from "./GeometryOverlay";
import { createDemoGeometry, type DemoGeometryType } from "../lib/demoGeometry";

interface ThreeViewerProps {
  /** Grid shape for demo geometry */
  demoShape?: [number, number, number];
  /** Demo geometry type */
  demoType?: DemoGeometryType;
  /** External geometry data (overrides demo) */
  geometry?: Uint8Array | null;
  /** Grid shape for external geometry */
  shape?: [number, number, number] | null;
  /** Grid resolution in meters */
  resolution?: number;
  /** Geometry visualization mode */
  geometryMode?: GeometryMode;
  /** Geometry color */
  geometryColor?: THREE.Color;
  /** Geometry opacity for transparent mode */
  geometryOpacity?: number;
  /** Show grid helper */
  showGrid?: boolean;
  /** Show axes helper */
  showAxes?: boolean;
}

export function ThreeViewer({
  demoShape = [50, 50, 50],
  demoType = "helmholtz",
  geometry: externalGeometry,
  shape: externalShape,
  resolution = 0.002, // 2mm default
  geometryMode = "wireframe",
  geometryColor,
  geometryOpacity = 0.3,
  showGrid = true,
  showAxes = true,
}: ThreeViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const animationIdRef = useRef<number>(0);
  const geometryGroupRef = useRef<THREE.Group | null>(null);
  const gridRef = useRef<THREE.GridHelper | null>(null);
  const axesRef = useRef<THREE.AxesHelper | null>(null);

  // Compute geometry data synchronously (no setState in effect)
  const { currentGeometry, currentShape } = useMemo(() => {
    if (externalGeometry && externalShape) {
      return { currentGeometry: externalGeometry, currentShape: externalShape };
    }
    // Use demo geometry
    const demoGeom = createDemoGeometry(demoType, demoShape);
    return { currentGeometry: demoGeom, currentShape: demoShape };
  }, [externalGeometry, externalShape, demoType, demoShape]);

  // Update geometry mesh when data or mode changes
  const updateGeometryMesh = useCallback(() => {
    if (!sceneRef.current) return;

    // Remove old geometry group
    if (geometryGroupRef.current) {
      sceneRef.current.remove(geometryGroupRef.current);
      geometryGroupRef.current.traverse((child) => {
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
    }

    // Create new geometry mesh
    const newGroup = createGeometryMesh(
      currentGeometry,
      currentShape,
      resolution,
      geometryMode,
      geometryColor,
      geometryOpacity
    );

    geometryGroupRef.current = newGroup;
    sceneRef.current.add(newGroup);
  }, [
    currentGeometry,
    currentShape,
    resolution,
    geometryMode,
    geometryColor,
    geometryOpacity,
  ]);

  // Scene initialization
  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);
    sceneRef.current = scene;

    // Camera setup
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.001, 100);
    // Position camera based on geometry size
    const maxDim = Math.max(...currentShape) * resolution;
    camera.position.set(maxDim * 2, maxDim * 2, maxDim * 2);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.sortObjects = true; // Enable object sorting for transparency
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // OrbitControls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controlsRef.current = controls;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 5, 5);
    scene.add(directionalLight);

    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
    directionalLight2.position.set(-3, -3, -3);
    scene.add(directionalLight2);

    // Grid helper (sized to geometry)
    const gridSize = Math.max(...currentShape) * resolution * 2;
    const grid = new THREE.GridHelper(
      gridSize,
      20,
      0x444444,
      0x333333
    );
    grid.visible = showGrid;
    scene.add(grid);
    gridRef.current = grid;

    // Axes helper
    const axes = new THREE.AxesHelper(gridSize / 2);
    axes.visible = showAxes;
    scene.add(axes);
    axesRef.current = axes;

    // Animation loop
    const animate = () => {
      animationIdRef.current = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // Handle resize
    const handleResize = () => {
      if (
        !containerRef.current ||
        !rendererRef.current ||
        !cameraRef.current
      )
        return;

      const newWidth = containerRef.current.clientWidth;
      const newHeight = containerRef.current.clientHeight;

      cameraRef.current.aspect = newWidth / newHeight;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(newWidth, newHeight);
    };

    const resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(container);

    // Cleanup
    return () => {
      cancelAnimationFrame(animationIdRef.current);
      resizeObserver.disconnect();
      controls.dispose();
      renderer.dispose();

      // Dispose geometry group
      if (geometryGroupRef.current) {
        geometryGroupRef.current.traverse((child) => {
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
      }

      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
    // Note: We intentionally only run scene setup once on mount.
    // Grid/axes/geometry updates are handled by separate effects.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Update geometry when it changes
  useEffect(() => {
    updateGeometryMesh();
  }, [updateGeometryMesh]);

  // Update grid visibility
  useEffect(() => {
    if (gridRef.current) {
      gridRef.current.visible = showGrid;
    }
  }, [showGrid]);

  // Update axes visibility
  useEffect(() => {
    if (axesRef.current) {
      axesRef.current.visible = showAxes;
    }
  }, [showAxes]);

  return (
    <div
      ref={containerRef}
      className="w-full h-full"
      style={{ minHeight: "200px" }}
    />
  );
}
