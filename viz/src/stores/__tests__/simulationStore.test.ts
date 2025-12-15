import { describe, it, expect, beforeEach } from "vitest";
import { useSimulationStore } from "../simulationStore";

// Reset store before each test
beforeEach(() => {
  useSimulationStore.getState().reset();
});

describe("SimulationStore", () => {
  describe("Initial State", () => {
    it("initializes with correct defaults", () => {
      const state = useSimulationStore.getState();

      expect(state.manifest).toBeNull();
      expect(state.metadata).toBeNull();
      expect(state.probeData).toBeNull();
      expect(state.geometry).toBeNull();
      expect(state.snapshots.size).toBe(0);

      expect(state.shape).toEqual([10, 10, 10]);
      expect(state.resolution).toBe(0.01);
      expect(state.totalFrames).toBe(0);

      expect(state.currentFrame).toBe(0);
      expect(state.isPlaying).toBe(false);
      expect(state.playbackSpeed).toBe(1);
      expect(state.isLooping).toBe(true);

      expect(state.colormap).toBe("diverging");
      expect(state.pressureRange).toBe("auto");
      expect(state.voxelGeometry).toBe("point");
      expect(state.geometryMode).toBe("wireframe");
      expect(state.threshold).toBe(0);
      expect(state.showAxes).toBe(true);
      expect(state.showGrid).toBe(false);

      expect(state.selectedProbes).toEqual([]);
      expect(state.hoveredVoxel).toBeNull();

      expect(state.isLoading).toBe(false);
      expect(state.loadingProgress).toBe(0);
      expect(state.error).toBeNull();
    });
  });

  describe("Playback Controls", () => {
    it("setCurrentFrame clamps to valid range", () => {
      const store = useSimulationStore.getState();

      // Set some totalFrames first
      useSimulationStore.setState({ totalFrames: 100 });

      store.setCurrentFrame(50);
      expect(useSimulationStore.getState().currentFrame).toBe(50);

      store.setCurrentFrame(-10);
      expect(useSimulationStore.getState().currentFrame).toBe(0);

      store.setCurrentFrame(150);
      expect(useSimulationStore.getState().currentFrame).toBe(99);
    });

    it("play/pause/toggle work correctly", () => {
      const store = useSimulationStore.getState();

      expect(store.isPlaying).toBe(false);

      store.play();
      expect(useSimulationStore.getState().isPlaying).toBe(true);

      store.pause();
      expect(useSimulationStore.getState().isPlaying).toBe(false);

      store.togglePlayback();
      expect(useSimulationStore.getState().isPlaying).toBe(true);

      store.togglePlayback();
      expect(useSimulationStore.getState().isPlaying).toBe(false);
    });

    it("stepForward/stepBackward work correctly", () => {
      useSimulationStore.setState({ totalFrames: 10, currentFrame: 5 });
      const store = useSimulationStore.getState();

      store.stepForward();
      expect(useSimulationStore.getState().currentFrame).toBe(6);

      store.stepBackward();
      expect(useSimulationStore.getState().currentFrame).toBe(5);

      // Test boundaries
      useSimulationStore.setState({ currentFrame: 9 });
      store.stepForward();
      expect(useSimulationStore.getState().currentFrame).toBe(9); // Should stay at max

      useSimulationStore.setState({ currentFrame: 0 });
      store.stepBackward();
      expect(useSimulationStore.getState().currentFrame).toBe(0); // Should stay at min
    });

    it("setPlaybackSpeed clamps to valid range", () => {
      const store = useSimulationStore.getState();

      store.setPlaybackSpeed(2);
      expect(useSimulationStore.getState().playbackSpeed).toBe(2);

      store.setPlaybackSpeed(0.1);
      expect(useSimulationStore.getState().playbackSpeed).toBe(0.25);

      store.setPlaybackSpeed(10);
      expect(useSimulationStore.getState().playbackSpeed).toBe(4);
    });

    it("setLooping works correctly", () => {
      const store = useSimulationStore.getState();

      expect(store.isLooping).toBe(true);

      store.setLooping(false);
      expect(useSimulationStore.getState().isLooping).toBe(false);

      store.setLooping(true);
      expect(useSimulationStore.getState().isLooping).toBe(true);
    });
  });

  describe("View Options", () => {
    it("setColormap updates colormap", () => {
      const store = useSimulationStore.getState();

      store.setColormap("magnitude");
      expect(useSimulationStore.getState().colormap).toBe("magnitude");

      store.setColormap("viridis");
      expect(useSimulationStore.getState().colormap).toBe("viridis");
    });

    it("setPressureRange updates pressure range", () => {
      const store = useSimulationStore.getState();

      store.setPressureRange([-1, 1]);
      expect(useSimulationStore.getState().pressureRange).toEqual([-1, 1]);

      store.setPressureRange("auto");
      expect(useSimulationStore.getState().pressureRange).toBe("auto");
    });

    it("setVoxelGeometry updates geometry", () => {
      const store = useSimulationStore.getState();

      store.setVoxelGeometry("sphere");
      expect(useSimulationStore.getState().voxelGeometry).toBe("sphere");

      store.setVoxelGeometry("point");
      expect(useSimulationStore.getState().voxelGeometry).toBe("point");
    });

    it("setGeometryMode updates geometry mode", () => {
      const store = useSimulationStore.getState();

      store.setGeometryMode("solid");
      expect(useSimulationStore.getState().geometryMode).toBe("solid");

      store.setGeometryMode("transparent");
      expect(useSimulationStore.getState().geometryMode).toBe("transparent");
    });

    it("setThreshold clamps to 0-1", () => {
      const store = useSimulationStore.getState();

      store.setThreshold(0.5);
      expect(useSimulationStore.getState().threshold).toBe(0.5);

      store.setThreshold(-0.5);
      expect(useSimulationStore.getState().threshold).toBe(0);

      store.setThreshold(1.5);
      expect(useSimulationStore.getState().threshold).toBe(1);
    });

    it("toggleAxes/toggleGrid work correctly", () => {
      const store = useSimulationStore.getState();

      expect(store.showAxes).toBe(true);
      store.toggleAxes();
      expect(useSimulationStore.getState().showAxes).toBe(false);
      store.toggleAxes();
      expect(useSimulationStore.getState().showAxes).toBe(true);

      expect(store.showGrid).toBe(false);
      store.toggleGrid();
      expect(useSimulationStore.getState().showGrid).toBe(true);
      store.toggleGrid();
      expect(useSimulationStore.getState().showGrid).toBe(false);
    });
  });

  describe("Selection", () => {
    it("selectProbe adds probe to selection", () => {
      const store = useSimulationStore.getState();

      store.selectProbe("upstream");
      expect(useSimulationStore.getState().selectedProbes).toEqual(["upstream"]);

      store.selectProbe("downstream");
      expect(useSimulationStore.getState().selectedProbes).toEqual([
        "upstream",
        "downstream",
      ]);
    });

    it("selectProbe does not duplicate existing selection", () => {
      const store = useSimulationStore.getState();

      store.selectProbe("upstream");
      store.selectProbe("upstream");
      expect(useSimulationStore.getState().selectedProbes).toEqual(["upstream"]);
    });

    it("deselectProbe removes probe from selection", () => {
      useSimulationStore.setState({
        selectedProbes: ["upstream", "downstream", "cavity"],
      });
      const store = useSimulationStore.getState();

      store.deselectProbe("downstream");
      expect(useSimulationStore.getState().selectedProbes).toEqual([
        "upstream",
        "cavity",
      ]);
    });

    it("toggleProbe toggles probe selection", () => {
      const store = useSimulationStore.getState();

      store.toggleProbe("upstream");
      expect(useSimulationStore.getState().selectedProbes).toContain("upstream");

      store.toggleProbe("upstream");
      expect(useSimulationStore.getState().selectedProbes).not.toContain(
        "upstream"
      );
    });

    it("setSelectedProbes replaces all selections", () => {
      useSimulationStore.setState({ selectedProbes: ["old1", "old2"] });
      const store = useSimulationStore.getState();

      store.setSelectedProbes(["new1", "new2", "new3"]);
      expect(useSimulationStore.getState().selectedProbes).toEqual([
        "new1",
        "new2",
        "new3",
      ]);
    });

    it("setHoveredVoxel updates hovered voxel", () => {
      const store = useSimulationStore.getState();

      store.setHoveredVoxel([10, 20, 30]);
      expect(useSimulationStore.getState().hoveredVoxel).toEqual([10, 20, 30]);

      store.setHoveredVoxel(null);
      expect(useSimulationStore.getState().hoveredVoxel).toBeNull();
    });
  });

  describe("Utility", () => {
    it("reset restores initial state", () => {
      useSimulationStore.setState({
        currentFrame: 50,
        isPlaying: true,
        selectedProbes: ["probe1"],
        threshold: 0.5,
      });

      useSimulationStore.getState().reset();
      const state = useSimulationStore.getState();

      expect(state.currentFrame).toBe(0);
      expect(state.isPlaying).toBe(false);
      expect(state.selectedProbes).toEqual([]);
      expect(state.threshold).toBe(0);
    });

    it("getCurrentPressure returns current frame pressure", () => {
      const pressure = new Float32Array([1, 2, 3]);
      const snapshots = new Map<number, Float32Array>();
      snapshots.set(5, pressure);

      useSimulationStore.setState({ snapshots, currentFrame: 5 });

      const result = useSimulationStore.getState().getCurrentPressure();
      expect(result).toBe(pressure);
    });

    it("getCurrentPressure returns null for missing frame", () => {
      useSimulationStore.setState({ currentFrame: 10 });

      const result = useSimulationStore.getState().getCurrentPressure();
      expect(result).toBeNull();
    });
  });

  describe("Snapshot Cache", () => {
    it("loadSnapshot caches snapshots", async () => {
      // Mock the manifest and snapshot loading
      const mockPressure = new Float32Array([1, 2, 3]);
      const mockManifest = {
        basePath: "/test",
        metadata: "metadata.json",
        probes: "probes.json",
        geometry: "geometry.bin",
        snapshots: [
          {
            time: 0,
            shape: [2, 2, 2] as [number, number, number],
            dtype: "float32" as const,
            format: "raw",
            downsample: 1,
            file: "snapshot_0.bin",
            bytes: 32,
          },
        ],
      };

      useSimulationStore.setState({
        manifest: mockManifest,
        totalFrames: 1,
      });

      // Test that loadSnapshot returns cached value if present
      const snapshots = new Map<number, Float32Array>();
      snapshots.set(0, mockPressure);
      useSimulationStore.setState({ snapshots });

      const store = useSimulationStore.getState();
      const result = await store.loadSnapshot(0);
      expect(result).toBe(mockPressure);
    });

    it("loadSnapshot returns null for invalid frame", async () => {
      const store = useSimulationStore.getState();
      const result = await store.loadSnapshot(999);
      expect(result).toBeNull();
    });
  });
});

describe("Selector Hooks", () => {
  beforeEach(() => {
    useSimulationStore.getState().reset();
  });

  it("selector hooks return correct values", () => {
    useSimulationStore.setState({
      currentFrame: 10,
      isPlaying: true,
      totalFrames: 100,
      playbackSpeed: 2,
      isLooping: false,
      colormap: "magnitude",
      voxelGeometry: "sphere",
      threshold: 0.5,
    });

    // We can't use hooks directly in tests, but we can verify the state
    const state = useSimulationStore.getState();
    expect(state.currentFrame).toBe(10);
    expect(state.isPlaying).toBe(true);
    expect(state.totalFrames).toBe(100);
    expect(state.playbackSpeed).toBe(2);
    expect(state.isLooping).toBe(false);
    expect(state.colormap).toBe("magnitude");
    expect(state.voxelGeometry).toBe("sphere");
    expect(state.threshold).toBe(0.5);
  });
});

describe("Background Preloading", () => {
  beforeEach(() => {
    useSimulationStore.getState().reset();
  });

  it("initializes with correct background loading defaults", () => {
    const state = useSimulationStore.getState();

    expect(state.isBackgroundLoading).toBe(false);
    expect(state.backgroundLoadingProgress).toBe(0);
    expect(state.cacheMaxFrames).toBe(0);
    expect(state.cacheAccessOrder).toEqual([]);
  });

  it("setCacheMaxFrames updates cache limit", () => {
    const store = useSimulationStore.getState();

    store.setCacheMaxFrames(50);
    expect(useSimulationStore.getState().cacheMaxFrames).toBe(50);

    store.setCacheMaxFrames(0);
    expect(useSimulationStore.getState().cacheMaxFrames).toBe(0);

    // Should not allow negative values
    store.setCacheMaxFrames(-10);
    expect(useSimulationStore.getState().cacheMaxFrames).toBe(0);
  });

  it("tracks cache access order", async () => {
    // Set up state with cached snapshots
    const snapshots = new Map<number, Float32Array>();
    snapshots.set(0, new Float32Array([1, 2, 3]));
    snapshots.set(1, new Float32Array([4, 5, 6]));
    snapshots.set(2, new Float32Array([7, 8, 9]));

    useSimulationStore.setState({
      snapshots,
      cacheAccessOrder: [0, 1, 2],
      totalFrames: 10,
    });

    const store = useSimulationStore.getState();

    // Access frame 0 (should move to end of access order)
    await store.loadSnapshot(0);
    expect(useSimulationStore.getState().cacheAccessOrder).toEqual([1, 2, 0]);

    // Access frame 1 (should move to end)
    await store.loadSnapshot(1);
    expect(useSimulationStore.getState().cacheAccessOrder).toEqual([2, 0, 1]);
  });

  it("evicts LRU frames when cache limit exceeded", () => {
    // Set up state with multiple cached snapshots
    const snapshots = new Map<number, Float32Array>();
    snapshots.set(0, new Float32Array([1]));
    snapshots.set(1, new Float32Array([2]));
    snapshots.set(2, new Float32Array([3]));
    snapshots.set(3, new Float32Array([4]));
    snapshots.set(4, new Float32Array([5]));

    useSimulationStore.setState({
      snapshots,
      cacheAccessOrder: [0, 1, 2, 3, 4],
      totalFrames: 10,
    });

    const store = useSimulationStore.getState();

    // Set cache limit to 3 - should evict frames 0 and 1
    store.setCacheMaxFrames(3);

    const state = useSimulationStore.getState();
    expect(state.snapshots.size).toBe(3);
    expect(state.snapshots.has(0)).toBe(false);
    expect(state.snapshots.has(1)).toBe(false);
    expect(state.snapshots.has(2)).toBe(true);
    expect(state.snapshots.has(3)).toBe(true);
    expect(state.snapshots.has(4)).toBe(true);
    expect(state.cacheAccessOrder).toEqual([2, 3, 4]);
  });

  it("calculates background loading progress from snapshot count", () => {
    // Progress is calculated from snapshots.size / totalFrames
    // This is updated when new snapshots are loaded (in the loadSnapshot setter)

    // Set up state with multiple cached snapshots
    const snapshots = new Map<number, Float32Array>();
    snapshots.set(0, new Float32Array([1]));
    snapshots.set(1, new Float32Array([2]));
    snapshots.set(2, new Float32Array([3]));
    snapshots.set(3, new Float32Array([4]));
    snapshots.set(4, new Float32Array([5]));

    useSimulationStore.setState({
      snapshots,
      cacheAccessOrder: [0, 1, 2, 3, 4],
      totalFrames: 10,
      backgroundLoadingProgress: 0.5, // 5/10
    });

    // Verify the progress was set correctly
    expect(useSimulationStore.getState().backgroundLoadingProgress).toBe(0.5);

    // When cache limit is set and eviction happens, progress is recalculated
    useSimulationStore.getState().setCacheMaxFrames(3);

    // After evicting 2 frames, we have 3/10 = 0.3
    expect(useSimulationStore.getState().backgroundLoadingProgress).toBe(0.3);
  });

  it("stopBackgroundPreload sets isBackgroundLoading to false", () => {
    useSimulationStore.setState({ isBackgroundLoading: true });
    expect(useSimulationStore.getState().isBackgroundLoading).toBe(true);

    useSimulationStore.getState().stopBackgroundPreload();
    expect(useSimulationStore.getState().isBackgroundLoading).toBe(false);
  });
});
