/**
 * Benchmark runner component for measuring rendering performance.
 *
 * Runs through various grid sizes and collects performance metrics.
 */

import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  PerformanceTracker,
  downsamplePressure,
} from "@/lib/performance";
import { applyPressureColormap, getSymmetricRange } from "@/lib/colormap";

// Benchmark configuration
export interface BenchmarkConfig {
  /** Grid sizes to benchmark (e.g., [50, 100, 150, 200]) */
  gridSizes: number[];
  /** Number of frames to collect per benchmark */
  framesPerBenchmark: number;
  /** Warmup frames before collecting data */
  warmupFrames: number;
  /** Whether to test with downsampling enabled */
  testDownsampling: boolean;
  /** Target voxels for downsampling tests */
  targetVoxels: number;
}

export interface BenchmarkResult {
  gridSize: number;
  voxelCount: number;
  downsampled: boolean;
  renderedVoxels: number;
  fps: {
    mean: number;
    min: number;
    max: number;
    p50: number;
    p95: number;
    p99: number;
  };
  frameTime: {
    mean: number;
    min: number;
    max: number;
    p50: number;
    p95: number;
    p99: number;
  };
  memoryMB: number;
  timeToFirstFrame: number;
}

export interface BenchmarkOutput {
  timestamp: string;
  platform: {
    userAgent: string;
    hardwareConcurrency: number;
    devicePixelRatio: number;
    renderer: string;
  };
  config: BenchmarkConfig;
  results: BenchmarkResult[];
}

const DEFAULT_CONFIG: BenchmarkConfig = {
  gridSizes: [50, 100, 150, 200],
  framesPerBenchmark: 120,
  warmupFrames: 30,
  testDownsampling: true,
  targetVoxels: 262144, // 64³
};

// Pre-allocated objects for rendering
const tempColor = new THREE.Color();
const tempMatrix = new THREE.Matrix4();
const tempScale = new THREE.Vector3();

function percentile(arr: number[], p: number): number {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = Math.ceil((p / 100) * sorted.length) - 1;
  return sorted[Math.max(0, idx)];
}

function mean(arr: number[]): number {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

// Generate synthetic pressure data for benchmarking
function generatePressureData(size: number): Float32Array {
  const total = size * size * size;
  const data = new Float32Array(total);
  const center = size / 2;

  for (let x = 0; x < size; x++) {
    for (let y = 0; y < size; y++) {
      for (let z = 0; z < size; z++) {
        const idx = x * size * size + y * size + z;
        // Create a spherical wave pattern
        const dx = x - center;
        const dy = y - center;
        const dz = z - center;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        data[idx] = Math.sin(dist * 0.5) * Math.exp(-dist * 0.02);
      }
    }
  }

  return data;
}

interface BenchmarkRunnerProps {
  config?: Partial<BenchmarkConfig>;
  onComplete?: (results: BenchmarkOutput) => void;
  onProgress?: (current: number, total: number, message: string) => void;
}

export function BenchmarkRunner({
  config: configOverrides,
  onComplete,
  onProgress,
}: BenchmarkRunnerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState<string>("Initializing...");
  const [progress, setProgress] = useState<number>(0);
  const [results, setResults] = useState<BenchmarkResult[]>([]);
  const [isComplete, setIsComplete] = useState(false);
  const [output, setOutput] = useState<BenchmarkOutput | null>(null);

  const config = useMemo(
    () => ({ ...DEFAULT_CONFIG, ...configOverrides }),
    [configOverrides]
  );

  const runBenchmark = useCallback(async () => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);

    const camera = new THREE.PerspectiveCamera(75, width / height, 0.01, 1000);
    camera.position.set(5, 5, 5);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 10);
    scene.add(directionalLight);

    // Get renderer info
    const gl = renderer.getContext();
    const debugInfo = gl.getExtension("WEBGL_debug_renderer_info");
    const rendererInfo = debugInfo
      ? gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL)
      : "Unknown";

    const allResults: BenchmarkResult[] = [];
    const testCases: { size: number; downsampled: boolean }[] = [];

    // Build test cases
    for (const size of config.gridSizes) {
      testCases.push({ size, downsampled: false });
      if (config.testDownsampling && size > 64) {
        testCases.push({ size, downsampled: true });
      }
    }

    const totalTests = testCases.length;
    let currentTest = 0;

    for (const testCase of testCases) {
      const { size, downsampled } = testCase;
      currentTest++;

      const testName = `${size}³ ${downsampled ? "(downsampled)" : ""}`;
      setStatus(`Benchmarking ${testName}...`);
      onProgress?.(currentTest, totalTests, `Benchmarking ${testName}`);

      // Generate pressure data
      const pressureData = generatePressureData(size);
      const shape: [number, number, number] = [size, size, size];

      // Apply downsampling if enabled
      let effectiveData = {
        data: pressureData,
        shape: shape,
        factor: 1,
        originalShape: shape,
      };

      if (downsampled) {
        effectiveData = downsamplePressure(pressureData, shape, {
          targetVoxels: config.targetVoxels,
          method: "average",
        });
      }

      const [ex, ey, ez] = effectiveData.shape;
      const effectiveTotal = ex * ey * ez;
      const resolution = 1 / size;
      const effectiveResolution = resolution * effectiveData.factor;

      // Clean up previous mesh
      scene.children
        .filter((c) => c instanceof THREE.InstancedMesh)
        .forEach((mesh) => {
          scene.remove(mesh);
          (mesh as THREE.InstancedMesh).geometry.dispose();
          ((mesh as THREE.InstancedMesh).material as THREE.Material).dispose();
        });

      // Create instanced mesh
      const voxelGeometry = new THREE.BoxGeometry(0.8, 0.8, 0.8);
      const material = new THREE.MeshStandardMaterial({
        vertexColors: false,
        metalness: 0.1,
        roughness: 0.8,
      });

      const mesh = new THREE.InstancedMesh(voxelGeometry, material, effectiveTotal);
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

      // Calculate offset for centering
      const offsetX = ((size - 1) * resolution) / 2;
      const offsetY = ((size - 1) * resolution) / 2;
      const offsetZ = ((size - 1) * resolution) / 2;

      // Set up instances
      const [min, max] = getSymmetricRange(effectiveData.data);
      let idx = 0;

      for (let x = 0; x < ex; x++) {
        for (let y = 0; y < ey; y++) {
          for (let z = 0; z < ez; z++) {
            const dataIdx = x * ey * ez + y * ez + z;
            const value = effectiveData.data[dataIdx];

            tempMatrix.makeTranslation(
              x * effectiveResolution - offsetX,
              y * effectiveResolution - offsetY,
              z * effectiveResolution - offsetZ
            );
            tempScale.set(effectiveResolution, effectiveResolution, effectiveResolution);
            tempMatrix.scale(tempScale);
            mesh.setMatrixAt(idx, tempMatrix);

            applyPressureColormap(value, min, max, tempColor);
            mesh.setColorAt(idx, tempColor);
            idx++;
          }
        }
      }

      mesh.count = idx;
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) {
        mesh.instanceColor.needsUpdate = true;
      }

      scene.add(mesh);

      // Adjust camera
      const maxDim = Math.max(ex, ey, ez) * effectiveResolution;
      camera.position.set(maxDim * 1.5, maxDim * 1.2, maxDim * 1.5);
      camera.lookAt(0, 0, 0);
      controls.target.set(0, 0, 0);
      controls.update();

      // Performance tracking
      const tracker = new PerformanceTracker();
      tracker.updateVoxelCounts(idx, size * size * size, effectiveData.factor);

      const frameTimes: number[] = [];
      let frameCount = 0;
      let timeToFirstFrame = 0;
      const startTime = performance.now();

      // Run benchmark loop
      await new Promise<void>((resolve) => {
        const totalFrames = config.warmupFrames + config.framesPerBenchmark;

        const animate = () => {
          tracker.startFrame();

          // Rotate camera slowly for more realistic benchmark
          const time = performance.now() * 0.0001;
          camera.position.x = Math.cos(time) * maxDim * 1.5;
          camera.position.z = Math.sin(time) * maxDim * 1.5;
          camera.lookAt(0, 0, 0);

          controls.update();
          renderer.render(scene, camera);

          tracker.endFrame();

          if (frameCount === 0) {
            timeToFirstFrame = performance.now() - startTime;
          }

          frameCount++;

          // Only collect metrics after warmup
          if (frameCount > config.warmupFrames) {
            const metrics = tracker.getMetrics();
            frameTimes.push(metrics.frameTime);
          }

          // Update progress
          const testProgress = (currentTest - 1) / totalTests;
          const frameProgress = frameCount / totalFrames / totalTests;
          setProgress(Math.round((testProgress + frameProgress) * 100));

          if (frameCount < totalFrames) {
            requestAnimationFrame(animate);
          } else {
            resolve();
          }
        };

        requestAnimationFrame(animate);
      });

      // Calculate statistics
      const fpsValues = frameTimes.map((ft) => (ft > 0 ? 1000 / ft : 0));
      const metrics = tracker.getMetrics();

      const result: BenchmarkResult = {
        gridSize: size,
        voxelCount: size * size * size,
        downsampled,
        renderedVoxels: idx,
        fps: {
          mean: Math.round(mean(fpsValues) * 10) / 10,
          min: Math.round(Math.min(...fpsValues) * 10) / 10,
          max: Math.round(Math.max(...fpsValues) * 10) / 10,
          p50: Math.round(percentile(fpsValues, 50) * 10) / 10,
          p95: Math.round(percentile(fpsValues, 95) * 10) / 10,
          p99: Math.round(percentile(fpsValues, 99) * 10) / 10,
        },
        frameTime: {
          mean: Math.round(mean(frameTimes) * 100) / 100,
          min: Math.round(Math.min(...frameTimes) * 100) / 100,
          max: Math.round(Math.max(...frameTimes) * 100) / 100,
          p50: Math.round(percentile(frameTimes, 50) * 100) / 100,
          p95: Math.round(percentile(frameTimes, 95) * 100) / 100,
          p99: Math.round(percentile(frameTimes, 99) * 100) / 100,
        },
        memoryMB: metrics.memoryMB,
        timeToFirstFrame: Math.round(timeToFirstFrame * 100) / 100,
      };

      allResults.push(result);
      setResults([...allResults]);
    }

    // Clean up
    renderer.dispose();
    container.removeChild(renderer.domElement);

    // Build final output
    const finalOutput: BenchmarkOutput = {
      timestamp: new Date().toISOString(),
      platform: {
        userAgent: navigator.userAgent,
        hardwareConcurrency: navigator.hardwareConcurrency,
        devicePixelRatio: window.devicePixelRatio,
        renderer: rendererInfo,
      },
      config,
      results: allResults,
    };

    setOutput(finalOutput);
    setIsComplete(true);
    setStatus("Complete");
    setProgress(100);
    onComplete?.(finalOutput);

    // Expose results globally for Puppeteer to read
    (window as unknown as { __benchmarkResults: BenchmarkOutput }).__benchmarkResults = finalOutput;
  }, [config, onComplete, onProgress]);

  useEffect(() => {
    // Small delay to ensure container is mounted
    const timer = setTimeout(runBenchmark, 100);
    return () => clearTimeout(timer);
  }, [runBenchmark]);

  return (
    <div className="w-full h-full flex flex-col bg-gray-900 text-white">
      <div className="p-4 border-b border-gray-700">
        <h1 className="text-xl font-bold">Performance Benchmark</h1>
        <p className="text-sm text-gray-400">
          Testing rendering performance at various grid sizes
        </p>
      </div>

      <div className="flex-1 flex">
        {/* Canvas container */}
        <div ref={containerRef} className="flex-1" style={{ minHeight: "400px" }} />

        {/* Results panel */}
        <div className="w-96 p-4 border-l border-gray-700 overflow-auto">
          <div className="space-y-4">
            {/* Progress */}
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>{status}</span>
                <span>{progress}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded h-2">
                <div
                  className="bg-blue-500 h-2 rounded transition-all"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>

            {/* Results table */}
            {results.length > 0 && (
              <div>
                <h2 className="text-lg font-semibold mb-2">Results</h2>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-gray-400">
                      <th className="pb-2">Grid</th>
                      <th className="pb-2">FPS (mean)</th>
                      <th className="pb-2">Frame (p95)</th>
                      <th className="pb-2">Memory</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((r, i) => (
                      <tr key={i} className="border-t border-gray-700">
                        <td className="py-2">
                          {r.gridSize}³
                          {r.downsampled && (
                            <span className="text-blue-400 text-xs ml-1">DS</span>
                          )}
                        </td>
                        <td
                          className={
                            r.fps.mean >= 30
                              ? "text-green-400"
                              : r.fps.mean >= 15
                              ? "text-yellow-400"
                              : "text-red-400"
                          }
                        >
                          {r.fps.mean}
                        </td>
                        <td>{r.frameTime.p95}ms</td>
                        <td>{r.memoryMB}MB</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* JSON output */}
            {isComplete && output && (
              <div>
                <h2 className="text-lg font-semibold mb-2">JSON Output</h2>
                <pre className="bg-gray-800 p-2 rounded text-xs overflow-auto max-h-64">
                  {JSON.stringify(output, null, 2)}
                </pre>
                <button
                  className="mt-2 px-4 py-2 bg-blue-600 rounded text-sm hover:bg-blue-700"
                  onClick={() => {
                    navigator.clipboard.writeText(JSON.stringify(output, null, 2));
                  }}
                >
                  Copy JSON
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
