/**
 * Zustand store for centralized simulation state management.
 *
 * Coordinates interactions between the 3D viewer, time series plots,
 * and playback controls.
 */
import { create } from "zustand";
import { useShallow } from "zustand/react/shallow";
import { subscribeWithSelector } from "zustand/middleware";
import {
  loadManifest,
  loadMetadata,
  loadSnapshot,
  loadVelocitySnapshot,
  loadProbeData,
  loadGeometry,
} from "@/lib/loaders";
import type {
  SimulationManifest,
  SimulationMetadata,
  ProbeData,
  Geometry,
  VelocitySnapshot,
} from "@/lib/loaders";
import type { PerformanceMetrics } from "@/lib/performance";

// =============================================================================
// Types
// =============================================================================

/** Available colormap types */
export type ColormapType = "diverging" | "magnitude" | "viridis";

/** @deprecated Use showWireframe and boundaryOpacity instead */
export type GeometryMode = "wireframe" | "solid" | "transparent" | "hidden";

/** Voxel geometry types */
export type VoxelGeometry = "point" | "mesh" | "hidden";

/** Downsampling method */
export type DownsampleMethod = "nearest" | "average" | "max";

/** Visualization mode */
export type VisualizationMode = "voxels" | "flow_particles";

/** View mode - 3D voxels or 2D slice */
export type ViewMode = "3d" | "slice";

/** Slice axis */
export type SliceAxis = "x" | "y" | "z";

/** Flow particle configuration */
export interface FlowParticleConfig {
  /** Number of particles to render */
  particleCount: number;
  /** Time scale factor (e.g., 0.001 = 1000x slower than real-time) */
  timeScale: number;
  /** Particle size in world units */
  particleSize: number;
  /** Whether to show particle trails */
  showTrails: boolean;
  /** Trail length in frames */
  trailLength: number;
}

/** Simulation state interface */
export interface SimulationState {
  // Data
  manifest: SimulationManifest | null;
  metadata: SimulationMetadata | null;
  probeData: ProbeData | null;
  geometry: Geometry | null;
  snapshots: Map<number, Float32Array>;
  velocitySnapshots: Map<number, VelocitySnapshot>;

  // Flow particle visualization
  visualizationMode: VisualizationMode;
  flowParticleConfig: FlowParticleConfig;
  hasVelocityData: boolean;

  // Derived data (computed from metadata)
  shape: [number, number, number];
  resolution: number;
  totalFrames: number;

  // Playback
  currentFrame: number;
  isPlaying: boolean;
  playbackSpeed: number;
  isLooping: boolean;

  // View options
  colormap: ColormapType;
  pressureRange: [number, number] | "auto";
  voxelGeometry: VoxelGeometry;
  showWireframe: boolean;
  boundaryOpacity: number; // 0-100, 0=hidden, 100=solid
  threshold: number;
  displayFill: number; // 0-1, percentage of voxels to display (sparseness control)
  showAxes: boolean;
  showGrid: boolean;

  // Selection
  selectedProbes: string[];
  hoveredVoxel: [number, number, number] | null;

  // Slice view
  viewMode: ViewMode;
  sliceAxis: SliceAxis;
  slicePosition: number; // 0-1 normalized position along axis
  showSliceGeometry: boolean; // Whether to show geometry overlay on slice

  // Loading state
  isLoading: boolean;
  loadingProgress: number;
  error: string | null;

  // Background preloading state
  isBackgroundLoading: boolean;
  backgroundLoadingProgress: number; // 0-1, fraction of frames cached
  cacheMaxFrames: number; // Max frames to keep in cache (0 = unlimited)
  cacheAccessOrder: number[]; // Frame indices ordered by access time (most recent last)

  // Performance settings
  enableDownsampling: boolean;
  targetVoxels: number;
  downsampleMethod: DownsampleMethod;
  showPerformanceMetrics: boolean;
  performanceMetrics: PerformanceMetrics | null;
}

/** Simulation actions interface */
export interface SimulationActions {
  // Data loading
  loadSimulation: (basePath: string) => Promise<void>;
  loadSnapshot: (frame: number) => Promise<Float32Array | null>;
  loadVelocityForFrame: (frame: number) => Promise<VelocitySnapshot | null>;

  // Flow particle visualization
  setVisualizationMode: (mode: VisualizationMode) => void;
  setFlowParticleConfig: (config: Partial<FlowParticleConfig>) => void;
  getCurrentVelocity: () => VelocitySnapshot | null;

  // Playback
  setCurrentFrame: (frame: number) => void;
  play: () => void;
  pause: () => void;
  togglePlayback: () => void;
  stepForward: () => void;
  stepBackward: () => void;
  setPlaybackSpeed: (speed: number) => void;
  setLooping: (loop: boolean) => void;

  // View options
  setColormap: (colormap: ColormapType) => void;
  setPressureRange: (range: [number, number] | "auto") => void;
  setVoxelGeometry: (geometry: VoxelGeometry) => void;
  setShowWireframe: (show: boolean) => void;
  setBoundaryOpacity: (opacity: number) => void;
  setThreshold: (threshold: number) => void;
  setDisplayFill: (fill: number) => void;
  toggleAxes: () => void;
  toggleGrid: () => void;

  // Selection
  selectProbe: (name: string) => void;
  deselectProbe: (name: string) => void;
  toggleProbe: (name: string) => void;
  setSelectedProbes: (names: string[]) => void;
  setHoveredVoxel: (coords: [number, number, number] | null) => void;

  // Slice view
  setViewMode: (mode: ViewMode) => void;
  setSliceAxis: (axis: SliceAxis) => void;
  setSlicePosition: (position: number) => void;
  setShowSliceGeometry: (show: boolean) => void;

  // Performance
  setEnableDownsampling: (enable: boolean) => void;
  setTargetVoxels: (target: number) => void;
  setDownsampleMethod: (method: DownsampleMethod) => void;
  setShowPerformanceMetrics: (show: boolean) => void;
  updatePerformanceMetrics: (metrics: PerformanceMetrics) => void;

  // Background preloading
  startBackgroundPreload: () => void;
  stopBackgroundPreload: () => void;
  setCacheMaxFrames: (max: number) => void;

  // Utility
  reset: () => void;
  getCurrentPressure: () => Float32Array | null;
}

/** Combined store type */
export type SimulationStore = SimulationState & SimulationActions;

// =============================================================================
// Default State
// =============================================================================

const DEFAULT_STATE: SimulationState = {
  // Data
  manifest: null,
  metadata: null,
  probeData: null,
  geometry: null,
  snapshots: new Map(),
  velocitySnapshots: new Map(),

  // Flow particle visualization
  visualizationMode: "voxels",
  flowParticleConfig: {
    particleCount: 50000,
    timeScale: 0.001, // 1000x slower than real-time
    particleSize: 0.002,
    showTrails: false,
    trailLength: 10,
  },
  hasVelocityData: false,

  // Derived
  shape: [10, 10, 10],
  resolution: 0.01,
  totalFrames: 0,

  // Playback
  currentFrame: 0,
  isPlaying: false,
  playbackSpeed: 1,
  isLooping: true,

  // View options
  colormap: "diverging",
  pressureRange: "auto",
  voxelGeometry: "point",
  showWireframe: false,
  boundaryOpacity: 30, // Default to 30% opacity (transparent)
  threshold: 0,
  displayFill: 1, // 100% - show all voxels by default
  showAxes: true,
  showGrid: false,

  // Selection
  selectedProbes: [],
  hoveredVoxel: null,

  // Slice view
  viewMode: "3d",
  sliceAxis: "y",
  slicePosition: 0.5,
  showSliceGeometry: true, // Show geometry overlay by default

  // Loading
  isLoading: false,
  loadingProgress: 0,
  error: null,

  // Background preloading
  isBackgroundLoading: false,
  backgroundLoadingProgress: 0,
  cacheMaxFrames: 0, // 0 = unlimited
  cacheAccessOrder: [],

  // Performance
  enableDownsampling: true,
  targetVoxels: 262144, // 64³
  downsampleMethod: "average",
  showPerformanceMetrics: false,
  performanceMetrics: null,
};

// =============================================================================
// Background Preloader State
// =============================================================================

// Module-level state for background preloader (not in Zustand to avoid rerenders)
let backgroundPreloadAbort: AbortController | null = null;
let backgroundPreloadTimeoutId: ReturnType<typeof setTimeout> | null = null;

/**
 * Get frames to preload sorted by priority (closest to current frame first).
 * Uses bidirectional expansion from current position.
 */
function getPreloadPriorityQueue(
  currentFrame: number,
  totalFrames: number,
  cachedFrames: Set<number>
): number[] {
  const queue: number[] = [];

  // Bidirectional expansion: alternate between forward and backward
  for (let distance = 1; distance < totalFrames; distance++) {
    const forward = currentFrame + distance;
    const backward = currentFrame - distance;

    // Add forward frame if valid and not cached
    if (forward < totalFrames && !cachedFrames.has(forward)) {
      queue.push(forward);
    }

    // Add backward frame if valid and not cached
    if (backward >= 0 && !cachedFrames.has(backward)) {
      queue.push(backward);
    }
  }

  return queue;
}

// =============================================================================
// Store Implementation
// =============================================================================

export const useSimulationStore = create<SimulationStore>()(
  subscribeWithSelector((set, get) => ({
    // Initial state
    ...DEFAULT_STATE,

    // =========================================================================
    // Data Loading
    // =========================================================================

    loadSimulation: async (basePath: string) => {
      set({ isLoading: true, loadingProgress: 0, error: null });

      try {
        // Load manifest
        const manifest = await loadManifest(`${basePath}/manifest.json`);
        manifest.basePath = basePath;
        set({ manifest, loadingProgress: 10 });

        // Load metadata
        const metadataFilename =
          manifest.metadata.split("/").pop() || manifest.metadata;
        const metadata = await loadMetadata(`${basePath}/${metadataFilename}`);
        set({ metadata, loadingProgress: 30 });

        // Load probe data and geometry in parallel
        const probesFilename =
          manifest.probes.split("/").pop() || manifest.probes;
        const geometryFilename =
          manifest.geometry.split("/").pop() || manifest.geometry;

        const [probeData, geometry] = await Promise.all([
          loadProbeData(`${basePath}/${probesFilename}`),
          loadGeometry(
            `${basePath}/${geometryFilename}`,
            {
              shape: metadata.grid.shape,
              format: "packed_bits",
              file: manifest.geometry,
            }
          ),
        ]);
        set({
          probeData,
          geometry,
          selectedProbes: Object.keys(probeData.probes),
          loadingProgress: 70,
        });

        // Preload first snapshot
        const firstSnapshotInfo = manifest.snapshots[0];
        if (firstSnapshotInfo) {
          const filename =
            firstSnapshotInfo.file.split("/").pop() || firstSnapshotInfo.file;
          const snapshot = await loadSnapshot(
            `${basePath}/${filename}`,
            firstSnapshotInfo
          );

          const newSnapshots = new Map<number, Float32Array>();
          newSnapshots.set(0, snapshot.pressure);

          // Check if velocity data is available
          const hasVelocityData =
            manifest.velocitySnapshots !== undefined &&
            manifest.velocitySnapshots.length > 0;

          set({
            snapshots: newSnapshots,
            shape: metadata.grid.shape,
            resolution: metadata.grid.resolution,
            totalFrames: manifest.snapshots.length,
            currentFrame: 0,
            hasVelocityData,
            isLoading: false,
            loadingProgress: 100,
          });
        } else {
          // Check if velocity data is available
          const hasVelocityData =
            manifest.velocitySnapshots !== undefined &&
            manifest.velocitySnapshots.length > 0;

          set({
            shape: metadata.grid.shape,
            resolution: metadata.grid.resolution,
            totalFrames: 0,
            hasVelocityData,
            isLoading: false,
            loadingProgress: 100,
          });
        }
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Failed to load simulation";
        set({ error: message, isLoading: false });
      }
    },

    loadSnapshot: async (frame: number) => {
      const { manifest, snapshots } = get();

      // Return cached snapshot if available (and update access order)
      if (snapshots.has(frame)) {
        // Move frame to end of access order (most recently accessed)
        set((state) => {
          const newOrder = state.cacheAccessOrder.filter((f) => f !== frame);
          newOrder.push(frame);
          return { cacheAccessOrder: newOrder };
        });
        return snapshots.get(frame)!;
      }

      // Check if frame is valid
      if (!manifest || frame < 0 || frame >= manifest.snapshots.length) {
        return null;
      }

      const basePath = manifest.basePath ?? "";
      const snapshotInfo = manifest.snapshots[frame];
      const filename = snapshotInfo.file.split("/").pop() || snapshotInfo.file;

      try {
        const snapshot = await loadSnapshot(
          `${basePath}/${filename}`,
          snapshotInfo
        );

        // Update cache with LRU eviction
        set((state) => {
          const newSnapshots = new Map(state.snapshots);
          newSnapshots.set(frame, snapshot.pressure);

          // Update access order
          const newOrder = state.cacheAccessOrder.filter((f) => f !== frame);
          newOrder.push(frame);

          // Evict oldest frames if cache limit exceeded
          const maxFrames = state.cacheMaxFrames;
          if (maxFrames > 0 && newSnapshots.size > maxFrames) {
            const framesToEvict = newSnapshots.size - maxFrames;
            const evicted = newOrder.splice(0, framesToEvict);
            for (const evictFrame of evicted) {
              newSnapshots.delete(evictFrame);
            }
          }

          // Update background loading progress
          const totalFrames = state.totalFrames;
          const backgroundLoadingProgress =
            totalFrames > 0 ? newSnapshots.size / totalFrames : 0;

          return {
            snapshots: newSnapshots,
            cacheAccessOrder: newOrder,
            backgroundLoadingProgress,
          };
        });

        return snapshot.pressure;
      } catch (err) {
        console.error(`Failed to load snapshot ${frame}:`, err);
        return null;
      }
    },

    loadVelocityForFrame: async (frame: number) => {
      const { manifest, velocitySnapshots } = get();

      // Return cached velocity if available
      if (velocitySnapshots.has(frame)) {
        return velocitySnapshots.get(frame)!;
      }

      // Check if velocity data exists for this frame
      if (
        !manifest ||
        !manifest.velocitySnapshots ||
        frame < 0 ||
        frame >= manifest.velocitySnapshots.length
      ) {
        return null;
      }

      const basePath = manifest.basePath ?? "";
      const velocityInfo = manifest.velocitySnapshots[frame];

      // Only support interleaved format for now
      if (velocityInfo.format !== "interleaved" || !velocityInfo.file) {
        console.warn(`Unsupported velocity format: ${velocityInfo.format}`);
        return null;
      }

      const filename = velocityInfo.file.split("/").pop() || velocityInfo.file;

      try {
        const velocity = await loadVelocitySnapshot(
          `${basePath}/${filename}`,
          velocityInfo
        );

        // Cache the velocity snapshot
        set((state) => {
          const newVelocitySnapshots = new Map(state.velocitySnapshots);
          newVelocitySnapshots.set(frame, velocity);
          return { velocitySnapshots: newVelocitySnapshots };
        });

        return velocity;
      } catch (err) {
        console.error(`Failed to load velocity for frame ${frame}:`, err);
        return null;
      }
    },

    // =========================================================================
    // Flow Particle Visualization
    // =========================================================================

    setVisualizationMode: (mode: VisualizationMode) => {
      set({ visualizationMode: mode });
    },

    setFlowParticleConfig: (config: Partial<FlowParticleConfig>) => {
      set((state) => ({
        flowParticleConfig: { ...state.flowParticleConfig, ...config },
      }));
    },

    getCurrentVelocity: () => {
      const { velocitySnapshots, currentFrame } = get();
      return velocitySnapshots.get(currentFrame) || null;
    },

    // =========================================================================
    // Playback Controls
    // =========================================================================

    setCurrentFrame: (frame: number) => {
      const { totalFrames } = get();
      const clampedFrame = Math.max(0, Math.min(totalFrames - 1, frame));
      set({ currentFrame: clampedFrame });
    },

    play: () => set({ isPlaying: true }),

    pause: () => set({ isPlaying: false }),

    togglePlayback: () => set((state) => ({ isPlaying: !state.isPlaying })),

    stepForward: () => {
      const { currentFrame, totalFrames } = get();
      if (currentFrame < totalFrames - 1) {
        set({ currentFrame: currentFrame + 1 });
      }
    },

    stepBackward: () => {
      const { currentFrame } = get();
      if (currentFrame > 0) {
        set({ currentFrame: currentFrame - 1 });
      }
    },

    setPlaybackSpeed: (speed: number) => {
      set({ playbackSpeed: Math.max(0.25, Math.min(4, speed)) });
    },

    setLooping: (loop: boolean) => set({ isLooping: loop }),

    // =========================================================================
    // View Options
    // =========================================================================

    setColormap: (colormap: ColormapType) => set({ colormap }),

    setPressureRange: (range: [number, number] | "auto") =>
      set({ pressureRange: range }),

    setVoxelGeometry: (voxelGeometry: VoxelGeometry) => set({ voxelGeometry }),

    setShowWireframe: (showWireframe: boolean) => set({ showWireframe }),
    setBoundaryOpacity: (boundaryOpacity: number) =>
      set({ boundaryOpacity: Math.max(0, Math.min(100, boundaryOpacity)) }),

    setThreshold: (threshold: number) =>
      set({ threshold: Math.max(0, Math.min(1, threshold)) }),

    setDisplayFill: (fill: number) =>
      set({ displayFill: Math.max(0.01, Math.min(1, fill)) }), // Min 1% to avoid empty display

    toggleAxes: () => set((state) => ({ showAxes: !state.showAxes })),

    toggleGrid: () => set((state) => ({ showGrid: !state.showGrid })),

    // =========================================================================
    // Selection
    // =========================================================================

    selectProbe: (name: string) => {
      set((state) => {
        if (state.selectedProbes.includes(name)) {
          return state; // Already selected
        }
        return { selectedProbes: [...state.selectedProbes, name] };
      });
    },

    deselectProbe: (name: string) => {
      set((state) => ({
        selectedProbes: state.selectedProbes.filter((p) => p !== name),
      }));
    },

    toggleProbe: (name: string) => {
      set((state) => {
        if (state.selectedProbes.includes(name)) {
          return { selectedProbes: state.selectedProbes.filter((p) => p !== name) };
        }
        return { selectedProbes: [...state.selectedProbes, name] };
      });
    },

    setSelectedProbes: (names: string[]) => set({ selectedProbes: names }),

    setHoveredVoxel: (coords: [number, number, number] | null) =>
      set({ hoveredVoxel: coords }),

    // =========================================================================
    // Slice View
    // =========================================================================

    setViewMode: (mode: ViewMode) => set({ viewMode: mode }),

    setSliceAxis: (axis: SliceAxis) => set({ sliceAxis: axis }),

    setSlicePosition: (position: number) =>
      set({ slicePosition: Math.max(0, Math.min(1, position)) }),

    setShowSliceGeometry: (show: boolean) => set({ showSliceGeometry: show }),

    // =========================================================================
    // Performance
    // =========================================================================

    setEnableDownsampling: (enable: boolean) => set({ enableDownsampling: enable }),

    setTargetVoxels: (target: number) =>
      set({ targetVoxels: Math.max(1000, Math.min(1000000, target)) }),

    setDownsampleMethod: (method: DownsampleMethod) =>
      set({ downsampleMethod: method }),

    setShowPerformanceMetrics: (show: boolean) =>
      set({ showPerformanceMetrics: show }),

    updatePerformanceMetrics: (metrics: PerformanceMetrics) =>
      set({ performanceMetrics: metrics }),

    // =========================================================================
    // Background Preloading
    // =========================================================================

    startBackgroundPreload: () => {
      const state = get();
      if (state.isBackgroundLoading || !state.manifest || state.isPlaying) {
        return;
      }

      // Create abort controller for this preload session
      backgroundPreloadAbort = new AbortController();
      set({ isBackgroundLoading: true });

      const runPreload = async () => {
        const { currentFrame, totalFrames, snapshots, loadSnapshot } = get();
        const signal = backgroundPreloadAbort?.signal;

        if (signal?.aborted) {
          set({ isBackgroundLoading: false });
          return;
        }

        // Get priority queue of frames to load
        const cachedFrames = new Set(snapshots.keys());
        const queue = getPreloadPriorityQueue(currentFrame, totalFrames, cachedFrames);

        if (queue.length === 0) {
          // All frames loaded
          set({ isBackgroundLoading: false, backgroundLoadingProgress: 1 });
          return;
        }

        // Load next frame in queue
        const nextFrame = queue[0];
        try {
          await loadSnapshot(nextFrame);
        } catch {
          // Ignore load errors during background preload
        }

        // Check if we should continue
        if (signal?.aborted) {
          set({ isBackgroundLoading: false });
          return;
        }

        // Schedule next frame load using requestIdleCallback if available, else setTimeout
        if (typeof requestIdleCallback !== "undefined") {
          requestIdleCallback(
            () => {
              if (!backgroundPreloadAbort?.signal.aborted) {
                runPreload();
              }
            },
            { timeout: 1000 }
          );
        } else {
          backgroundPreloadTimeoutId = setTimeout(runPreload, 50);
        }
      };

      // Start the preload loop
      runPreload();
    },

    stopBackgroundPreload: () => {
      if (backgroundPreloadAbort) {
        backgroundPreloadAbort.abort();
        backgroundPreloadAbort = null;
      }
      if (backgroundPreloadTimeoutId) {
        clearTimeout(backgroundPreloadTimeoutId);
        backgroundPreloadTimeoutId = null;
      }
      set({ isBackgroundLoading: false });
    },

    setCacheMaxFrames: (max: number) => {
      set({ cacheMaxFrames: Math.max(0, max) });

      // Immediately evict if over limit
      const state = get();
      if (max > 0 && state.snapshots.size > max) {
        set((s) => {
          const newSnapshots = new Map(s.snapshots);
          const newOrder = [...s.cacheAccessOrder];
          const framesToEvict = newSnapshots.size - max;
          const evicted = newOrder.splice(0, framesToEvict);
          for (const frame of evicted) {
            newSnapshots.delete(frame);
          }
          return {
            snapshots: newSnapshots,
            cacheAccessOrder: newOrder,
            backgroundLoadingProgress:
              s.totalFrames > 0 ? newSnapshots.size / s.totalFrames : 0,
          };
        });
      }
    },

    // =========================================================================
    // Utility
    // =========================================================================

    reset: () => {
      // Stop any background preloading
      if (backgroundPreloadAbort) {
        backgroundPreloadAbort.abort();
        backgroundPreloadAbort = null;
      }
      if (backgroundPreloadTimeoutId) {
        clearTimeout(backgroundPreloadTimeoutId);
        backgroundPreloadTimeoutId = null;
      }
      set({
        ...DEFAULT_STATE,
        snapshots: new Map(), // Create new Map instance
        velocitySnapshots: new Map(), // Create new Map instance
      });
    },

    getCurrentPressure: () => {
      const { snapshots, currentFrame } = get();
      return snapshots.get(currentFrame) ?? null;
    },
  }))
);

// =============================================================================
// Subscriptions for Side Effects
// =============================================================================

/**
 * Auto-preload adjacent frames during playback for smoother animation.
 * Uses bidirectional preloading with forward bias during playback.
 */
useSimulationStore.subscribe(
  (state) => state.currentFrame,
  async (currentFrame) => {
    const store = useSimulationStore.getState();
    const { manifest, isPlaying, snapshots, loadSnapshot, totalFrames } = store;

    if (!manifest || !isPlaying) return;

    // During playback: preload 5 forward, 2 backward (bias toward play direction)
    const forwardCount = 5;
    const backwardCount = 2;

    // Preload forward frames (higher priority)
    for (let i = 1; i <= forwardCount; i++) {
      const frame = currentFrame + i;
      if (frame < totalFrames && !snapshots.has(frame)) {
        loadSnapshot(frame);
      }
    }

    // Preload backward frames (lower priority, for scrubbing back)
    for (let i = 1; i <= backwardCount; i++) {
      const frame = currentFrame - i;
      if (frame >= 0 && !snapshots.has(frame)) {
        loadSnapshot(frame);
      }
    }
  }
);

/**
 * Auto-start/stop background preloading based on playback state.
 * When paused with a loaded simulation, start progressive background loading.
 */
useSimulationStore.subscribe(
  (state) => ({ isPlaying: state.isPlaying, manifest: state.manifest }),
  ({ isPlaying, manifest }) => {
    const { startBackgroundPreload, stopBackgroundPreload } =
      useSimulationStore.getState();

    if (!manifest) return;

    if (isPlaying) {
      // Stop background preload when playing
      stopBackgroundPreload();
    } else {
      // Start background preload when paused (after a short delay to allow UI to settle)
      setTimeout(() => {
        const currentState = useSimulationStore.getState();
        if (!currentState.isPlaying && currentState.manifest) {
          startBackgroundPreload();
        }
      }, 500);
    }
  },
  { equalityFn: (a, b) => a.isPlaying === b.isPlaying && a.manifest === b.manifest }
);

/**
 * Handle loop/pause at end of playback.
 */
useSimulationStore.subscribe(
  (state) => ({ currentFrame: state.currentFrame, isPlaying: state.isPlaying }),
  ({ currentFrame, isPlaying }) => {
    if (!isPlaying) return;

    const { totalFrames, isLooping, setCurrentFrame, pause } =
      useSimulationStore.getState();

    if (currentFrame >= totalFrames - 1) {
      if (isLooping) {
        setCurrentFrame(0);
      } else {
        pause();
      }
    }
  },
  { equalityFn: (a, b) => a.currentFrame === b.currentFrame && a.isPlaying === b.isPlaying }
);

// =============================================================================
// Selector Hooks (for optimized re-renders)
// =============================================================================

/** Select current pressure data */
export const useCurrentPressure = () =>
  useSimulationStore((state) => state.snapshots.get(state.currentFrame) ?? null);

// Selectors for individual primitives (no re-render optimization needed)
export const selectCurrentFrame = (state: SimulationStore) => state.currentFrame;
export const selectIsPlaying = (state: SimulationStore) => state.isPlaying;
export const selectTotalFrames = (state: SimulationStore) => state.totalFrames;
export const selectPlaybackSpeed = (state: SimulationStore) => state.playbackSpeed;
export const selectIsLooping = (state: SimulationStore) => state.isLooping;
export const selectShape = (state: SimulationStore) => state.shape;
export const selectResolution = (state: SimulationStore) => state.resolution;
export const selectIsLoading = (state: SimulationStore) => state.isLoading;
export const selectLoadingProgress = (state: SimulationStore) => state.loadingProgress;
export const selectError = (state: SimulationStore) => state.error;
export const selectProbeData = (state: SimulationStore) => state.probeData;
export const selectSelectedProbes = (state: SimulationStore) => state.selectedProbes;
export const selectVoxelGeometry = (state: SimulationStore) => state.voxelGeometry;
export const selectThreshold = (state: SimulationStore) => state.threshold;
export const selectDisplayFill = (state: SimulationStore) => state.displayFill;
export const selectShowAxes = (state: SimulationStore) => state.showAxes;
export const selectShowGrid = (state: SimulationStore) => state.showGrid;
export const selectShowWireframe = (state: SimulationStore) => state.showWireframe;
export const selectBoundaryOpacity = (state: SimulationStore) => state.boundaryOpacity;
export const selectColormap = (state: SimulationStore) => state.colormap;
export const selectPressureRange = (state: SimulationStore) => state.pressureRange;
export const selectGeometry = (state: SimulationStore) => state.geometry;
export const selectEnableDownsampling = (state: SimulationStore) => state.enableDownsampling;
export const selectTargetVoxels = (state: SimulationStore) => state.targetVoxels;
export const selectDownsampleMethod = (state: SimulationStore) => state.downsampleMethod;
export const selectShowPerformanceMetrics = (state: SimulationStore) => state.showPerformanceMetrics;
export const selectPerformanceMetrics = (state: SimulationStore) => state.performanceMetrics;
export const selectViewMode = (state: SimulationStore) => state.viewMode;
export const selectSliceAxis = (state: SimulationStore) => state.sliceAxis;
export const selectSlicePosition = (state: SimulationStore) => state.slicePosition;
export const selectShowSliceGeometry = (state: SimulationStore) => state.showSliceGeometry;

/** @deprecated Use individual selectors with useSimulationStore instead */
export const usePlaybackState = () =>
  useSimulationStore(
    useShallow((state) => ({
      currentFrame: state.currentFrame,
      isPlaying: state.isPlaying,
      totalFrames: state.totalFrames,
      playbackSpeed: state.playbackSpeed,
      isLooping: state.isLooping,
    }))
  );

/** @deprecated Use individual selectors with useSimulationStore instead */
export const useViewOptions = () =>
  useSimulationStore(
    useShallow((state) => ({
      colormap: state.colormap,
      pressureRange: state.pressureRange,
      voxelGeometry: state.voxelGeometry,
      showWireframe: state.showWireframe,
      boundaryOpacity: state.boundaryOpacity,
      threshold: state.threshold,
      displayFill: state.displayFill,
      showAxes: state.showAxes,
      showGrid: state.showGrid,
    }))
  );

/** @deprecated Use individual selectors with useSimulationStore instead */
export const useGridInfo = () =>
  useSimulationStore(
    useShallow((state) => ({
      shape: state.shape,
      resolution: state.resolution,
    }))
  );

/** @deprecated Use individual selectors with useSimulationStore instead */
export const useLoadingState = () =>
  useSimulationStore(
    useShallow((state) => ({
      isLoading: state.isLoading,
      loadingProgress: state.loadingProgress,
      error: state.error,
    }))
  );

/** @deprecated Use individual selectors with useSimulationStore instead */
export const useProbeData = () =>
  useSimulationStore(
    useShallow((state) => ({
      probeData: state.probeData,
      selectedProbes: state.selectedProbes,
    }))
  );

/** @deprecated Use individual selectors with useSimulationStore instead */
export const usePerformanceSettings = () =>
  useSimulationStore(
    useShallow((state) => ({
      enableDownsampling: state.enableDownsampling,
      targetVoxels: state.targetVoxels,
      downsampleMethod: state.downsampleMethod,
      showPerformanceMetrics: state.showPerformanceMetrics,
      performanceMetrics: state.performanceMetrics,
    }))
  );

/** Select background loading state */
export const useBackgroundLoadingState = () =>
  useSimulationStore(
    useShallow((state) => ({
      isBackgroundLoading: state.isBackgroundLoading,
      backgroundLoadingProgress: state.backgroundLoadingProgress,
      cachedFrameCount: state.snapshots.size,
      totalFrames: state.totalFrames,
      cacheMaxFrames: state.cacheMaxFrames,
    }))
  );

// =============================================================================
// Global Exposure for Playwright/Testing
// =============================================================================

/**
 * Expose store methods to window for programmatic control from Playwright.
 * This enables video capture scripts to directly trigger snapshot loading
 * without relying on DOM manipulation or keyboard events.
 */
declare global {
  interface Window {
    __SIMULATION_STORE__?: {
      getState: () => SimulationStore;
      setFrame: (frame: number) => Promise<void>;
      getPressureRange: () => [number, number];
    };
  }
}

if (typeof window !== "undefined") {
  window.__SIMULATION_STORE__ = {
    /** Get current store state */
    getState: () => useSimulationStore.getState(),

    /**
     * Set frame and wait for snapshot to load.
     * Returns a promise that resolves when the snapshot is ready.
     */
    setFrame: async (frame: number): Promise<void> => {
      const store = useSimulationStore.getState();
      store.setCurrentFrame(frame);
      await store.loadSnapshot(frame);
    },

    /** Get the min/max pressure values from current snapshot */
    getPressureRange: (): [number, number] => {
      const pressure = useSimulationStore.getState().getCurrentPressure();
      if (!pressure || pressure.length === 0) {
        return [0, 0];
      }
      let min = Infinity;
      let max = -Infinity;
      for (let i = 0; i < pressure.length; i++) {
        const v = pressure[i];
        if (v < min) min = v;
        if (v > max) max = v;
      }
      return [min, max];
    },
  };
}
