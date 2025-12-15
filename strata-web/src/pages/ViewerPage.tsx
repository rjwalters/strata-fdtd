/**
 * ViewerPage - Interactive 3D visualization for HDF5 simulation results.
 *
 * This page allows users to:
 * - Upload HDF5 files via drag-drop or file picker
 * - Load HDF5 files from URLs
 * - Visualize 3D pressure field evolution
 * - View probe time series and spectra
 * - Export data and visualizations
 */

import { useState, useCallback, useEffect, useRef, useMemo } from "react";
import { useParams } from "react-router-dom";
import {
  Layout,
  Panel,
  FileUpload,
  OptimizedVoxelRenderer,
  PlaybackControls,
  BackgroundLoadingIndicator,
  TimeSeriesPlot,
  SpectrumPlot,
  ExportPanel,
  Button,
  Badge,
  Slider,
  useSimulationStore,
  useCurrentPressure,
  usePlaybackState,
  useViewOptions,
  useGridInfo,
  useProbeData,
  usePerformanceSettings,
  useExport,
} from "@strata/ui";
import type {
  VoxelRendererHandle,
  VoxelGeometry,
  GeometryMode,
  DownsampleMethod,
  ViewState,
  DemoGeometryType,
} from "@strata/ui";
import {
  ArrowLeft,
  Settings,
  Info,
  Sparkles,
  Repeat,
  Gauge,
  ChevronDown,
  Grid3x3,
  EyeOff,
  FileText,
  Waves,
} from "lucide-react";
import { useHDF5Simulation } from "../hooks/useHDF5Simulation";
import { getDemoBasePath, getDemoConfig } from "../config/demos";

// =============================================================================
// Types
// =============================================================================

interface ViewerPageProps {
  /** Callback to navigate back to home */
  onBack?: () => void;
}

// =============================================================================
// Constants
// =============================================================================

const GEOMETRY_MODES: { value: GeometryMode; label: string }[] = [
  { value: "wireframe", label: "Wireframe" },
  { value: "solid", label: "Solid" },
  { value: "transparent", label: "Transparent" },
  { value: "hidden", label: "Hidden" },
];

const DOWNSAMPLE_METHODS: { value: DownsampleMethod; label: string }[] = [
  { value: "average", label: "Average" },
  { value: "max", label: "Max" },
  { value: "nearest", label: "Nearest" },
];

const TARGET_VOXEL_PRESETS = [
  { value: 32768, label: "32³ (Low)" },
  { value: 125000, label: "50³ (Med)" },
  { value: 262144, label: "64³ (High)" },
  { value: 512000, label: "80³ (Ultra)" },
];

// =============================================================================
// Main Component
// =============================================================================

export default function ViewerPage({ onBack }: ViewerPageProps) {
  // Get demoId from route params
  const { demoId } = useParams<{ demoId?: string }>();

  // HDF5 loading state (for file uploads)
  const {
    isLoaded: isHDF5Loaded,
    isLoading: isHDF5Loading,
    progress: hdf5Progress,
    error: hdf5Error,
    hdf5Data,
    loadFile,
    loadURL,
    loadTimestep,
    reset: hdf5Reset,
  } = useHDF5Simulation();

  // Store-based loading (for demos)
  const storeLoadSimulation = useSimulationStore((s) => s.loadSimulation);
  const storeIsLoading = useSimulationStore((s) => s.isLoading);
  const storeLoadingProgress = useSimulationStore((s) => s.loadingProgress);
  const storeError = useSimulationStore((s) => s.error);
  const storeReset = useSimulationStore((s) => s.reset);
  const storeManifest = useSimulationStore((s) => s.manifest);

  // Demo loading state
  const [demoLoadAttempted, setDemoLoadAttempted] = useState(false);
  const [demoLoadError, setDemoLoadError] = useState<string | null>(null);

  // Combined loading state
  const isLoaded = isHDF5Loaded || storeManifest !== null;
  const isLoading = isHDF5Loading || storeIsLoading;
  const progress = isHDF5Loading ? hdf5Progress : storeLoadingProgress / 100;
  const error = hdf5Error || storeError || demoLoadError;

  // Reset function that handles both loading methods
  const reset = useCallback(() => {
    hdf5Reset();
    storeReset();
    setDemoLoadAttempted(false);
    setDemoLoadError(null);
  }, [hdf5Reset, storeReset]);

  // Auto-load demo when demoId is present
  useEffect(() => {
    if (!demoId || demoLoadAttempted || isLoaded) {
      return;
    }

    const loadDemo = async () => {
      setDemoLoadAttempted(true);
      setDemoLoadError(null);

      const basePath = getDemoBasePath(demoId);
      if (!basePath) {
        const config = getDemoConfig(demoId);
        if (!config) {
          setDemoLoadError(`Unknown demo: ${demoId}`);
        } else {
          setDemoLoadError('Demo storage not configured. Please check VITE_R2_BUCKET_URL.');
        }
        return;
      }

      try {
        await storeLoadSimulation(basePath);
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to load demo';
        setDemoLoadError(message);
      }
    };

    loadDemo();
  }, [demoId, demoLoadAttempted, isLoaded, storeLoadSimulation]);

  // Get demo title for header
  const demoConfig = demoId ? getDemoConfig(demoId) : null;

  // Store selectors
  const pressure = useCurrentPressure();
  const { currentFrame, isPlaying, totalFrames, playbackSpeed, isLooping } = usePlaybackState();
  const { voxelGeometry, threshold, displayFill, showAxes, showGrid, geometryMode } = useViewOptions();
  const { shape, resolution } = useGridInfo();
  const { probeData } = useProbeData();
  const {
    enableDownsampling,
    targetVoxels,
    downsampleMethod,
    showPerformanceMetrics,
  } = usePerformanceSettings();

  // Local state
  const [geometryOpacity, setGeometryOpacity] = useState(30);
  const [viewMode, setViewMode] = useState<"time" | "spectrum">("time");
  const [showMetadata, setShowMetadata] = useState(false);

  // Derive selected probe from probe data
  const probeNames = probeData ? Object.keys(probeData.probes) : [];
  const [selectedProbeOverride, setSelectedProbeOverride] = useState<string | null>(null);
  const selectedProbe = selectedProbeOverride ?? probeNames[0] ?? null;
  const setSelectedProbe = (name: string | null) => setSelectedProbeOverride(name);

  // Renderer ref for export
  const rendererRef = useRef<VoxelRendererHandle>(null);

  // Export hook
  const [exportState, exportActions] = useExport();

  // Actions
  const setCurrentFrame = useSimulationStore((s) => s.setCurrentFrame);
  const play = useSimulationStore((s) => s.play);
  const pause = useSimulationStore((s) => s.pause);
  const setVoxelGeometry = useSimulationStore((s) => s.setVoxelGeometry);
  const setThreshold = useSimulationStore((s) => s.setThreshold);
  const setDisplayFill = useSimulationStore((s) => s.setDisplayFill);
  const toggleAxes = useSimulationStore((s) => s.toggleAxes);
  const toggleGrid = useSimulationStore((s) => s.toggleGrid);
  const setLooping = useSimulationStore((s) => s.setLooping);
  const setPlaybackSpeed = useSimulationStore((s) => s.setPlaybackSpeed);
  const setGeometryMode = useSimulationStore((s) => s.setGeometryMode);
  const setEnableDownsampling = useSimulationStore((s) => s.setEnableDownsampling);
  const setTargetVoxels = useSimulationStore((s) => s.setTargetVoxels);
  const setDownsampleMethod = useSimulationStore((s) => s.setDownsampleMethod);
  const setShowPerformanceMetrics = useSimulationStore((s) => s.setShowPerformanceMetrics);
  const updatePerformanceMetrics = useSimulationStore((s) => s.updatePerformanceMetrics);
  const storeLoadSnapshot = useSimulationStore((s) => s.loadSnapshot);

  // Load snapshot when frame changes
  useEffect(() => {
    if (isLoaded && totalFrames > 0) {
      // Use HDF5 timestep loading for file uploads, store loading for demos
      if (isHDF5Loaded) {
        loadTimestep(currentFrame);
      } else if (storeManifest) {
        storeLoadSnapshot(currentFrame);
      }
    }
  }, [currentFrame, isLoaded, totalFrames, loadTimestep, isHDF5Loaded, storeManifest, storeLoadSnapshot]);


  // Playback animation loop
  const animationRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(0);

  useEffect(() => {
    if (!isPlaying || totalFrames === 0) {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
      return;
    }

    const fps = 15 * playbackSpeed;
    const interval = 1000 / fps;

    const animate = (time: number) => {
      if (time - lastTimeRef.current >= interval) {
        lastTimeRef.current = time;
        const nextFrame = currentFrame + 1;
        if (nextFrame >= totalFrames) {
          if (isLooping) {
            setCurrentFrame(0);
          } else {
            pause();
            return;
          }
        } else {
          setCurrentFrame(nextFrame);
        }
      }
      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isPlaying, totalFrames, playbackSpeed, isLooping, currentFrame, setCurrentFrame, pause]);

  const handlePlayingChange = useCallback(
    (playing: boolean) => {
      if (playing) {
        play();
      } else {
        pause();
      }
    },
    [play, pause]
  );

  // Calculate current time for time series marker
  const currentTime = useMemo(() => {
    if (!probeData) return undefined;
    return (currentFrame / totalFrames) * probeData.duration;
  }, [currentFrame, totalFrames, probeData]);

  // Handle time selection from chart
  const handleTimeSelect = useCallback(
    (time: number) => {
      if (!probeData) return;
      const frame = Math.round((time / probeData.duration) * totalFrames);
      setCurrentFrame(Math.max(0, Math.min(totalFrames - 1, frame)));
    },
    [setCurrentFrame, totalFrames, probeData]
  );

  // Export callbacks
  const getCanvas = useCallback(() => {
    return rendererRef.current?.getCanvas() ?? null;
  }, []);

  const renderFrameForExport = useCallback(
    async (frameIndex: number) => {
      await loadTimestep(frameIndex);
      await new Promise((resolve) => setTimeout(resolve, 50));
      rendererRef.current?.render();
    },
    [loadTimestep]
  );

  const getViewState = useCallback((): ViewState => {
    return {
      cameraPosition: [0, 0, 0],
      cameraTarget: [0, 0, 0],
      cameraFov: 75,
      viewOptions: {
        threshold,
        displayFill,
        voxelGeometry,
        geometryMode,
        showGrid,
        showAxes,
      },
      simulation: {
        currentFrame,
        totalFrames,
        shape,
        resolution,
      },
      timestamp: new Date().toISOString(),
    };
  }, [threshold, displayFill, voxelGeometry, geometryMode, showGrid, showAxes, currentFrame, totalFrames, shape, resolution]);

  // Handle file/URL loading with error handling
  const handleLoadFile = useCallback(
    async (file: File) => {
      try {
        await loadFile(file);
      } catch {
        // Error is already set in the hook
      }
    },
    [loadFile]
  );

  const handleLoadURL = useCallback(
    async (url: string) => {
      try {
        await loadURL(url);
      } catch {
        // Error is already set in the hook
      }
    },
    [loadURL]
  );

  // If not loaded, show file upload UI or demo loading state
  if (!isLoaded) {
    return (
      <div className="h-screen w-screen flex flex-col bg-background">
        {/* Header */}
        <header className="flex items-center justify-between px-6 py-4 border-b">
          <div className="flex items-center gap-4">
            {onBack && (
              <Button variant="ghost" size="icon" onClick={onBack}>
                <ArrowLeft className="h-4 w-4" />
              </Button>
            )}
            <div>
              <h1 className="text-lg font-bold">
                {demoConfig ? demoConfig.title : 'Simulation Viewer'}
              </h1>
              <p className="text-sm text-muted-foreground">
                {demoConfig
                  ? demoConfig.description
                  : 'Load HDF5 simulation results'}
              </p>
            </div>
          </div>
        </header>

        {/* File upload area or demo loading state */}
        <main className="flex-1 flex items-center justify-center p-8">
          {demoId && isLoading ? (
            // Demo loading state
            <div className="text-center space-y-4">
              <div className="animate-pulse">
                <div className="h-16 w-16 mx-auto rounded-full bg-primary/20 flex items-center justify-center">
                  <Waves className="h-8 w-8 text-primary animate-pulse" />
                </div>
              </div>
              <div>
                <h2 className="text-lg font-semibold">Loading Demo</h2>
                <p className="text-sm text-muted-foreground">
                  {demoConfig?.title || demoId}
                </p>
              </div>
              <div className="w-64 mx-auto">
                <div className="h-2 bg-secondary rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary transition-all duration-300"
                    style={{ width: `${progress * 100}%` }}
                  />
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  {Math.round(progress * 100)}%
                </p>
              </div>
            </div>
          ) : demoId && error ? (
            // Demo error state
            <div className="text-center space-y-4 max-w-md">
              <div className="h-16 w-16 mx-auto rounded-full bg-destructive/20 flex items-center justify-center">
                <Info className="h-8 w-8 text-destructive" />
              </div>
              <div>
                <h2 className="text-lg font-semibold">Failed to Load Demo</h2>
                <p className="text-sm text-muted-foreground mt-1">{error}</p>
              </div>
              <div className="flex gap-2 justify-center">
                {onBack && (
                  <Button variant="outline" onClick={onBack}>
                    <ArrowLeft className="h-4 w-4 mr-2" />
                    Back to Gallery
                  </Button>
                )}
                <Button onClick={() => {
                  setDemoLoadAttempted(false);
                  setDemoLoadError(null);
                }}>
                  Retry
                </Button>
              </div>
            </div>
          ) : (
            // File upload UI (when no demoId or as fallback)
            <FileUpload
              onFile={handleLoadFile}
              onURL={handleLoadURL}
              isLoading={isLoading}
              progress={progress}
              error={error}
              className="w-full"
            />
          )}
        </main>
      </div>
    );
  }

  // Main visualization view
  return (
    <Layout
      sidebar={
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-lg font-bold text-foreground">
                {demoConfig?.title || 'Simulation Viewer'}
              </h1>
              <p className="text-sm text-muted-foreground">
                {demoConfig ? demoConfig.category : 'HDF5 Visualization'}
              </p>
            </div>
            {onBack && (
              <Button variant="ghost" size="icon" onClick={onBack}>
                <ArrowLeft className="h-4 w-4" />
              </Button>
            )}
          </div>

          {/* File Info */}
          <Panel title="Simulation">
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Grid</span>
                <span>{shape.join(" × ")}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Frames</span>
                <span>{totalFrames}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Resolution</span>
                <span>{(resolution * 1000).toFixed(2)} mm</span>
              </div>
              {hdf5Data?.simulation.totalTime && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Duration</span>
                  <span>{(hdf5Data.simulation.totalTime * 1e6).toFixed(1)} μs</span>
                </div>
              )}
            </div>

            <Button
              variant="outline"
              size="sm"
              className="w-full mt-3 gap-2"
              onClick={() => setShowMetadata(!showMetadata)}
            >
              <FileText className="h-3 w-3" />
              {showMetadata ? "Hide" : "Show"} Metadata
            </Button>

            {showMetadata && hdf5Data && (
              <div className="mt-3 p-3 bg-secondary/30 rounded-md text-xs space-y-1">
                <div><strong>Created:</strong> {hdf5Data.metadata.createdAt}</div>
                <div><strong>Solver:</strong> {hdf5Data.metadata.solverVersion}</div>
                {hdf5Data.metadata.backend && (
                  <div><strong>Backend:</strong> {hdf5Data.metadata.backend}</div>
                )}
                {hdf5Data.metadata.totalRuntimeSeconds && (
                  <div><strong>Runtime:</strong> {hdf5Data.metadata.totalRuntimeSeconds.toFixed(1)}s</div>
                )}
              </div>
            )}

            <Button
              variant="destructive"
              size="sm"
              className="w-full mt-3"
              onClick={reset}
            >
              Unload Simulation
            </Button>
          </Panel>

          {/* Boundary Display */}
          <Panel title="Boundary Display">
            <div className="space-y-3">
              <div className="flex flex-wrap gap-1">
                {GEOMETRY_MODES.map((mode) => (
                  <Badge
                    key={mode.value}
                    variant={geometryMode === mode.value ? "default" : "secondary"}
                    className="cursor-pointer"
                    onClick={() => setGeometryMode(mode.value)}
                  >
                    {mode.label}
                  </Badge>
                ))}
              </div>
              {geometryMode === "transparent" && (
                <div className="space-y-1">
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Opacity</span>
                    <span>{geometryOpacity}%</span>
                  </div>
                  <Slider
                    value={[geometryOpacity]}
                    onValueChange={(v) => setGeometryOpacity(v[0])}
                    min={10}
                    max={80}
                    step={5}
                  />
                </div>
              )}
            </div>
          </Panel>

          {/* Voxel Display */}
          <Panel title="Voxel Display">
            <div className="space-y-3">
              <div>
                <span className="text-sm mb-2 block">Geometry</span>
                <div className="flex gap-1">
                  {[
                    { value: "point" as VoxelGeometry, label: "Points", icon: <Sparkles className="h-3 w-3" /> },
                    { value: "mesh" as VoxelGeometry, label: "Mesh", icon: <Grid3x3 className="h-3 w-3" /> },
                    { value: "hidden" as VoxelGeometry, label: "Hidden", icon: <EyeOff className="h-3 w-3" /> },
                  ].map((opt) => (
                    <Button
                      key={opt.value}
                      variant={voxelGeometry === opt.value ? "default" : "outline"}
                      size="sm"
                      className="flex-1 gap-1"
                      onClick={() => setVoxelGeometry(opt.value)}
                    >
                      {opt.icon}
                      {opt.label}
                    </Button>
                  ))}
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Colormap</span>
                <Badge variant="secondary">Diverging</Badge>
              </div>
            </div>
          </Panel>

          {/* Threshold */}
          <Panel title="Threshold">
            <Slider
              value={[threshold * 100]}
              max={100}
              step={1}
              onValueChange={([v]) => setThreshold(v / 100)}
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>0%</span>
              <span>{Math.round(threshold * 100)}%</span>
              <span>100%</span>
            </div>
          </Panel>

          {/* Display Fill */}
          <Panel title="Display Fill">
            <Slider
              value={[displayFill * 100]}
              min={1}
              max={100}
              step={1}
              onValueChange={([v]) => setDisplayFill(v / 100)}
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>1%</span>
              <span>{Math.round(displayFill * 100)}%</span>
              <span>100%</span>
            </div>
          </Panel>

          {/* Playback */}
          <Panel title="Playback">
            <div className="space-y-3">
              <label className="flex items-center gap-2 text-sm cursor-pointer">
                <input
                  type="checkbox"
                  checked={isLooping}
                  onChange={(e) => setLooping(e.target.checked)}
                  className="rounded"
                />
                <Repeat className="h-3 w-3" />
                Loop
              </label>
              <div>
                <span className="text-sm mb-2 block">Speed</span>
                <div className="flex gap-1">
                  {[0.25, 0.5, 1, 2, 4].map((speed) => (
                    <Button
                      key={speed}
                      variant={playbackSpeed === speed ? "default" : "outline"}
                      size="sm"
                      className="flex-1 text-xs"
                      onClick={() => setPlaybackSpeed(speed)}
                    >
                      {speed}x
                    </Button>
                  ))}
                </div>
              </div>
            </div>
          </Panel>

          {/* View Options */}
          <Panel title="View Options">
            <div className="space-y-2">
              <label className="flex items-center gap-2 text-sm cursor-pointer">
                <input
                  type="checkbox"
                  checked={showGrid}
                  onChange={(e) => e.target.checked !== showGrid && toggleGrid()}
                  className="rounded"
                />
                Show Grid
              </label>
              <label className="flex items-center gap-2 text-sm cursor-pointer">
                <input
                  type="checkbox"
                  checked={showAxes}
                  onChange={(e) => e.target.checked !== showAxes && toggleAxes()}
                  className="rounded"
                />
                Show Axes
              </label>
            </div>
          </Panel>

          {/* Performance */}
          <PerformancePanel
            enableDownsampling={enableDownsampling}
            targetVoxels={targetVoxels}
            downsampleMethod={downsampleMethod}
            showPerformanceMetrics={showPerformanceMetrics}
            onEnableDownsamplingChange={setEnableDownsampling}
            onTargetVoxelsChange={setTargetVoxels}
            onDownsampleMethodChange={setDownsampleMethod}
            onShowPerformanceMetricsChange={setShowPerformanceMetrics}
          />

          {/* Export */}
          <Panel title="Export">
            <ExportPanel
              exportState={exportState}
              exportActions={exportActions}
              getCanvas={getCanvas}
              totalFrames={totalFrames}
              currentFrame={currentFrame}
              renderFrame={renderFrameForExport}
              pressure={pressure}
              shape={shape}
              resolution={resolution}
              probeData={probeData}
              getViewState={getViewState}
            />
          </Panel>

          {/* Footer */}
          <div className="pt-4 border-t border-border">
            <div className="flex gap-2">
              <Button variant="ghost" size="icon" title="Settings">
                <Settings className="h-4 w-4" />
              </Button>
              <Button variant="ghost" size="icon" title="Info">
                <Info className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      }
      main={
        <OptimizedVoxelRenderer
          ref={rendererRef}
          pressure={pressure}
          shape={shape}
          resolution={resolution}
          geometry={voxelGeometry}
          threshold={threshold}
          displayFill={displayFill}
          showGrid={showGrid}
          showAxes={showAxes}
          geometryMode={geometryMode}
          demoType={"helmholtz" as DemoGeometryType}
          geometryOpacity={geometryOpacity / 100}
          enableDownsampling={enableDownsampling}
          targetVoxels={targetVoxels}
          downsampleMethod={downsampleMethod}
          showPerformanceMetrics={showPerformanceMetrics}
          onPerformanceUpdate={updatePerformanceMetrics}
        />
      }
      bottom={
        <BottomPanel
          totalFrames={totalFrames}
          currentFrame={currentFrame}
          isPlaying={isPlaying}
          playbackSpeed={playbackSpeed}
          probeData={probeData}
          selectedProbe={selectedProbe}
          viewMode={viewMode}
          currentTime={currentTime}
          onFrameChange={setCurrentFrame}
          onPlayingChange={handlePlayingChange}
          onTimeSelect={handleTimeSelect}
          onViewModeChange={setViewMode}
          onSelectProbe={setSelectedProbe}
        />
      }
    />
  );
}

// =============================================================================
// Sub-components
// =============================================================================

interface PerformancePanelProps {
  enableDownsampling: boolean;
  targetVoxels: number;
  downsampleMethod: DownsampleMethod;
  showPerformanceMetrics: boolean;
  onEnableDownsamplingChange: (v: boolean) => void;
  onTargetVoxelsChange: (v: number) => void;
  onDownsampleMethodChange: (v: DownsampleMethod) => void;
  onShowPerformanceMetricsChange: (v: boolean) => void;
}

function PerformancePanel({
  enableDownsampling,
  targetVoxels,
  downsampleMethod,
  showPerformanceMetrics,
  onEnableDownsamplingChange,
  onTargetVoxelsChange,
  onDownsampleMethodChange,
  onShowPerformanceMetricsChange,
}: PerformancePanelProps) {
  const [showPerfSettings, setShowPerfSettings] = useState(false);

  return (
    <Panel title="Performance">
      <div className="space-y-3">
        <label className="flex items-center gap-2 text-sm cursor-pointer">
          <input
            type="checkbox"
            checked={showPerformanceMetrics}
            onChange={(e) => onShowPerformanceMetricsChange(e.target.checked)}
            className="rounded"
          />
          <Gauge className="h-3 w-3" />
          Show Metrics
        </label>
        <label className="flex items-center gap-2 text-sm cursor-pointer">
          <input
            type="checkbox"
            checked={enableDownsampling}
            onChange={(e) => onEnableDownsamplingChange(e.target.checked)}
            className="rounded"
          />
          Auto Downsampling
        </label>

        {enableDownsampling && (
          <>
            <Button
              variant="ghost"
              size="sm"
              className="w-full justify-between text-xs"
              onClick={() => setShowPerfSettings(!showPerfSettings)}
            >
              Advanced Settings
              <ChevronDown className={`h-3 w-3 transition-transform ${showPerfSettings ? "rotate-180" : ""}`} />
            </Button>

            {showPerfSettings && (
              <div className="space-y-3 pl-2 border-l-2 border-border">
                <div>
                  <span className="text-xs text-muted-foreground mb-1 block">Target Voxels</span>
                  <div className="flex flex-wrap gap-1">
                    {TARGET_VOXEL_PRESETS.map((preset) => (
                      <Badge
                        key={preset.value}
                        variant={targetVoxels === preset.value ? "default" : "secondary"}
                        className="cursor-pointer text-xs"
                        onClick={() => onTargetVoxelsChange(preset.value)}
                      >
                        {preset.label}
                      </Badge>
                    ))}
                  </div>
                </div>
                <div>
                  <span className="text-xs text-muted-foreground mb-1 block">Method</span>
                  <div className="flex gap-1">
                    {DOWNSAMPLE_METHODS.map((method) => (
                      <Badge
                        key={method.value}
                        variant={downsampleMethod === method.value ? "default" : "secondary"}
                        className="cursor-pointer text-xs"
                        onClick={() => onDownsampleMethodChange(method.value)}
                      >
                        {method.label}
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </Panel>
  );
}

interface BottomPanelProps {
  totalFrames: number;
  currentFrame: number;
  isPlaying: boolean;
  playbackSpeed: number;
  probeData: ReturnType<typeof useSimulationStore.getState>["probeData"];
  selectedProbe: string | null;
  viewMode: "time" | "spectrum";
  currentTime?: number;
  onFrameChange: (frame: number) => void;
  onPlayingChange: (playing: boolean) => void;
  onTimeSelect: (time: number) => void;
  onViewModeChange: (mode: "time" | "spectrum") => void;
  onSelectProbe: (name: string | null) => void;
}

function BottomPanel({
  totalFrames,
  currentFrame,
  isPlaying,
  playbackSpeed,
  probeData,
  selectedProbe,
  viewMode,
  currentTime,
  onFrameChange,
  onPlayingChange,
  onTimeSelect,
  onViewModeChange,
  onSelectProbe,
}: BottomPanelProps) {
  const probeNames = probeData ? Object.keys(probeData.probes) : [];

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between mb-2">
        <div className="flex gap-2">
          <Button
            variant={viewMode === "time" ? "default" : "outline"}
            size="sm"
            className="text-xs h-6"
            onClick={() => onViewModeChange("time")}
          >
            Time Series
          </Button>
          <Button
            variant={viewMode === "spectrum" ? "default" : "outline"}
            size="sm"
            className="text-xs h-6"
            onClick={() => onViewModeChange("spectrum")}
          >
            Spectrum
          </Button>
        </div>
        {viewMode === "spectrum" && probeNames.length > 0 && (
          <div className="flex gap-1">
            {probeNames.map((name) => (
              <Badge
                key={name}
                variant={selectedProbe === name ? "default" : "outline"}
                className="cursor-pointer text-xs"
                onClick={() => onSelectProbe(name)}
              >
                {name}
              </Badge>
            ))}
          </div>
        )}
      </div>

      <div className="flex-1 min-h-0">
        {!probeData ? (
          <div className="h-full bg-secondary/30 rounded-md flex items-center justify-center text-muted-foreground text-sm">
            No probe data available
          </div>
        ) : viewMode === "time" ? (
          <TimeSeriesPlot
            probes={probeData.probes}
            sampleRate={probeData.sampleRate}
            currentTime={currentTime}
            onTimeSelect={onTimeSelect}
          />
        ) : (
          <SpectrumPlot
            data={
              selectedProbe && probeData.probes[selectedProbe]
                ? probeData.probes[selectedProbe].data
                : new Float32Array(0)
            }
            sampleRate={probeData.sampleRate}
          />
        )}
      </div>

      <div className="mt-2 flex items-center gap-4">
        <PlaybackControls
          totalFrames={totalFrames}
          currentFrame={currentFrame}
          isPlaying={isPlaying}
          onFrameChange={onFrameChange}
          onPlayingChange={onPlayingChange}
          disabled={totalFrames === 0}
          fps={15 * playbackSpeed}
        />
        <BackgroundLoadingIndicator />
      </div>
    </div>
  );
}
