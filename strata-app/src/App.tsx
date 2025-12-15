import { useState, useCallback, useEffect, useRef, useMemo, lazy, Suspense } from "react"
import * as THREE from "three"
import {
  Layout,
  Panel,
  OptimizedVoxelRenderer,
  FlowParticleRenderer,
  VisualizationModePanel,
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
  useLoadingState,
  useProbeData,
  usePerformanceSettings,
  useUrlState,
  useExport,
} from "@strata/ui"
import type {
  VoxelRendererHandle,
  VoxelGeometry,
  GeometryMode,
  DownsampleMethod,
  VisualizationMode,
  FlowParticleConfig,
  ExportState,
  ExportActions,
  ViewState,
  DemoGeometryType,
} from "@strata/ui"
import { FolderOpen, Settings, Info, Sparkles, Repeat, Music, Gauge, ChevronDown, Link2, Loader2, Grid3x3, EyeOff, Hammer, FileUp } from "lucide-react"

// Lazy load demo pages for code splitting
const OrganPipeDemo = lazy(() => import("./pages/OrganPipeDemo"))
const SimulationBuilder = lazy(() => import("./pages/SimulationBuilder"))
const ViewerPage = lazy(() => import("./pages/ViewerPage"))

// Loading fallback component
function PageLoadingFallback() {
  return (
    <div className="h-screen w-screen flex items-center justify-center bg-background">
      <div className="flex flex-col items-center gap-4">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <p className="text-muted-foreground">Loading demo...</p>
      </div>
    </div>
  )
}

type Page = "home" | "organ-pipes" | "builder" | "viewer"

const GEOMETRY_MODES: { value: GeometryMode; label: string }[] = [
  { value: "wireframe", label: "Wireframe" },
  { value: "solid", label: "Solid" },
  { value: "transparent", label: "Transparent" },
  { value: "hidden", label: "Hidden" },
]

const DEMO_GEOMETRIES: { value: DemoGeometryType; label: string }[] = [
  { value: "helmholtz", label: "Helmholtz" },
  { value: "duct", label: "Duct" },
  { value: "sphere", label: "Sphere" },
]

const DOWNSAMPLE_METHODS: { value: DownsampleMethod; label: string }[] = [
  { value: "average", label: "Average" },
  { value: "max", label: "Max" },
  { value: "nearest", label: "Nearest" },
]

const TARGET_VOXEL_PRESETS = [
  { value: 32768, label: "32³ (Low)" },
  { value: 125000, label: "50³ (Med)" },
  { value: 262144, label: "64³ (High)" },
  { value: 512000, label: "80³ (Ultra)" },
]

function App() {
  const [page, setPage] = useState<Page>("home")

  // Optimized selectors for minimal re-renders
  const pressure = useCurrentPressure()
  const { currentFrame, isPlaying, totalFrames, playbackSpeed, isLooping } = usePlaybackState()
  const { voxelGeometry, threshold, displayFill, showAxes, showGrid, geometryMode } = useViewOptions()
  const { shape, resolution } = useGridInfo()
  const { isLoading, error } = useLoadingState()
  const { probeData } = useProbeData()
  const {
    enableDownsampling,
    targetVoxels,
    downsampleMethod,
    showPerformanceMetrics,
  } = usePerformanceSettings()
  const manifest = useSimulationStore((s) => s.manifest)

  // Flow particle visualization state
  const visualizationMode = useSimulationStore((s) => s.visualizationMode)
  const flowParticleConfig = useSimulationStore((s) => s.flowParticleConfig)
  const hasVelocityData = useSimulationStore((s) => s.hasVelocityData)
  const geometry = useSimulationStore((s) => s.geometry)
  const velocitySnapshots = useSimulationStore((s) => s.velocitySnapshots)

  // URL state synchronization
  const { getShareableUrl } = useUrlState()
  const [linkCopied, setLinkCopied] = useState(false)

  // Local state for features not yet in the store
  const [demoType, setDemoType] = useState<DemoGeometryType>("helmholtz")
  const [geometryOpacity, setGeometryOpacity] = useState(30)
  const [viewMode, setViewMode] = useState<"time" | "spectrum">("time")
  const [selectedProbe, setSelectedProbe] = useState<string | null>(null)

  // Renderer ref for export
  const rendererRef = useRef<VoxelRendererHandle>(null)
  // Scene state for FlowParticleRenderer (avoid accessing ref during render)
  const [rendererScene, setRendererScene] = useState<THREE.Scene | null>(null)

  // Memoized onReady callback to prevent recreation on every render
  const handleRendererReady = useCallback(() => {
    setRendererScene(rendererRef.current?.getScene() ?? null);
  }, []);

  // Export hook
  const [exportState, exportActions] = useExport()

  // Actions
  const loadSimulation = useSimulationStore((s) => s.loadSimulation)
  const loadSnapshot = useSimulationStore((s) => s.loadSnapshot)
  const setCurrentFrame = useSimulationStore((s) => s.setCurrentFrame)
  const play = useSimulationStore((s) => s.play)
  const pause = useSimulationStore((s) => s.pause)
  const setVoxelGeometry = useSimulationStore((s) => s.setVoxelGeometry)
  const setThreshold = useSimulationStore((s) => s.setThreshold)
  const setDisplayFill = useSimulationStore((s) => s.setDisplayFill)
  const toggleAxes = useSimulationStore((s) => s.toggleAxes)
  const toggleGrid = useSimulationStore((s) => s.toggleGrid)
  const setLooping = useSimulationStore((s) => s.setLooping)
  const setPlaybackSpeed = useSimulationStore((s) => s.setPlaybackSpeed)
  const setGeometryMode = useSimulationStore((s) => s.setGeometryMode)
  const setEnableDownsampling = useSimulationStore((s) => s.setEnableDownsampling)
  const setTargetVoxels = useSimulationStore((s) => s.setTargetVoxels)
  const setDownsampleMethod = useSimulationStore((s) => s.setDownsampleMethod)
  const setShowPerformanceMetrics = useSimulationStore((s) => s.setShowPerformanceMetrics)
  const updatePerformanceMetrics = useSimulationStore((s) => s.updatePerformanceMetrics)

  // Flow particle visualization actions
  const setVisualizationMode = useSimulationStore((s) => s.setVisualizationMode)
  const setFlowParticleConfig = useSimulationStore((s) => s.setFlowParticleConfig)
  const loadVelocityForFrame = useSimulationStore((s) => s.loadVelocityForFrame)

  // Load snapshot when frame changes
  useEffect(() => {
    if (manifest && totalFrames > 0) {
      loadSnapshot(currentFrame)
    }
  }, [currentFrame, manifest, totalFrames, loadSnapshot])

  // Load velocity when in flow mode and frame changes
  useEffect(() => {
    if (manifest && hasVelocityData && visualizationMode === "flow_particles") {
      loadVelocityForFrame(currentFrame)
    }
  }, [currentFrame, manifest, hasVelocityData, visualizationMode, loadVelocityForFrame])

  // Auto-select first probe when probe data loads
  useEffect(() => {
    if (probeData && !selectedProbe) {
      const probeNames = Object.keys(probeData.probes)
      if (probeNames.length > 0) {
        setSelectedProbe(probeNames[0])
      }
    }
  }, [probeData, selectedProbe])

  // Playback animation loop
  const animationRef = useRef<number | null>(null)
  const lastTimeRef = useRef<number>(0)
  // Use ref to track current frame in animation loop to avoid re-running effect
  const currentFrameRef = useRef(currentFrame)
  currentFrameRef.current = currentFrame

  useEffect(() => {
    if (!isPlaying || totalFrames === 0) {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
        animationRef.current = null
      }
      return
    }

    const fps = 15 * playbackSpeed
    const interval = 1000 / fps

    const animate = (time: number) => {
      if (time - lastTimeRef.current >= interval) {
        lastTimeRef.current = time
        const nextFrame = currentFrameRef.current + 1
        if (nextFrame >= totalFrames) {
          if (isLooping) {
            setCurrentFrame(0)
          } else {
            pause()
            return
          }
        } else {
          setCurrentFrame(nextFrame)
        }
      }
      animationRef.current = requestAnimationFrame(animate)
    }

    animationRef.current = requestAnimationFrame(animate)

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isPlaying, totalFrames, playbackSpeed, isLooping, setCurrentFrame, pause])

  const handleLoadSample = useCallback(async () => {
    await loadSimulation("/sample-data")
  }, [loadSimulation])

  const handlePlayingChange = useCallback(
    (playing: boolean) => {
      if (playing) {
        play()
      } else {
        pause()
      }
    },
    [play, pause]
  )

  // Calculate current time for time series marker
  const currentTime = useMemo(() => {
    if (!probeData) return undefined
    return (currentFrame / totalFrames) * probeData.duration
  }, [currentFrame, totalFrames, probeData])

  // Compute pressure range for flow particle coloring
  const pressureRange = useMemo((): [number, number] => {
    if (!pressure || pressure.length === 0) return [-1, 1]
    let min = Infinity
    let max = -Infinity
    for (let i = 0; i < pressure.length; i++) {
      const v = pressure[i]
      if (v < min) min = v
      if (v > max) max = v
    }
    // Make it symmetric for diverging colormap
    const absMax = Math.max(Math.abs(min), Math.abs(max))
    return [-absMax, absMax]
  }, [pressure])

  // Get current velocity for flow particles
  const currentVelocity = velocitySnapshots.get(currentFrame) ?? null

  // Delta time for flow particles (scaled by playback speed)
  const flowDeltaTime = (1 / 60) / playbackSpeed

  // Handle time selection from chart
  const handleTimeSelect = useCallback(
    (time: number) => {
      if (!probeData) return
      const frame = Math.round((time / probeData.duration) * totalFrames)
      setCurrentFrame(Math.max(0, Math.min(totalFrames - 1, frame)))
    },
    [setCurrentFrame, totalFrames, probeData]
  )

  // Copy shareable link to clipboard
  const handleCopyLink = useCallback(() => {
    const url = getShareableUrl()
    navigator.clipboard.writeText(url).then(() => {
      setLinkCopied(true)
      setTimeout(() => setLinkCopied(false), 2000)
    })
  }, [getShareableUrl])

  // Export callbacks
  const getCanvas = useCallback(() => {
    return rendererRef.current?.getCanvas() ?? null
  }, [])

  const renderFrameForExport = useCallback(
    async (frameIndex: number) => {
      if (manifest) {
        await loadSnapshot(frameIndex)
      }
      // Small delay to ensure render
      await new Promise((resolve) => setTimeout(resolve, 50))
      rendererRef.current?.render()
    },
    [manifest, loadSnapshot]
  )

  const getViewState = useCallback((): ViewState => {
    return {
      cameraPosition: [0, 0, 0], // Would need camera ref to get actual position
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
      simulation: manifest ? {
        currentFrame,
        totalFrames,
        shape,
        resolution,
      } : undefined,
      timestamp: new Date().toISOString(),
    }
  }, [threshold, displayFill, voxelGeometry, geometryMode, showGrid, showAxes, manifest, currentFrame, totalFrames, shape, resolution])

  if (page === "organ-pipes") {
    return (
      <Suspense fallback={<PageLoadingFallback />}>
        <OrganPipeDemo onBack={() => setPage("home")} />
      </Suspense>
    )
  }

if (page === "builder") {
    return (
      <Suspense fallback={<PageLoadingFallback />}>
        <SimulationBuilder onBack={() => setPage("home")} />
      </Suspense>
    )
  }

  if (page === "viewer") {
    return (
      <Suspense fallback={<PageLoadingFallback />}>
        <ViewerPage onBack={() => setPage("home")} />
      </Suspense>
    )
  }

  return (
    <Layout
      sidebar={
        <Sidebar
          isLoading={isLoading}
          error={error}
          manifest={manifest}
          shape={shape}
          totalFrames={totalFrames}
          voxelGeometry={voxelGeometry}
          threshold={threshold}
          displayFill={displayFill}
          showGrid={showGrid}
          showAxes={showAxes}
          geometryMode={geometryMode}
          demoType={demoType}
          geometryOpacity={geometryOpacity}
          isLooping={isLooping}
          playbackSpeed={playbackSpeed}
          onLoadSample={handleLoadSample}
          onVoxelGeometryChange={setVoxelGeometry}
          onThresholdChange={setThreshold}
          onDisplayFillChange={setDisplayFill}
          onShowGridChange={(v) => v !== showGrid && toggleGrid()}
          onShowAxesChange={(v) => v !== showAxes && toggleAxes()}
          onGeometryModeChange={setGeometryMode}
          onDemoTypeChange={setDemoType}
          onGeometryOpacityChange={setGeometryOpacity}
          onLoopingChange={setLooping}
          onPlaybackSpeedChange={setPlaybackSpeed}
          onNavigate={setPage}
          enableDownsampling={enableDownsampling}
          targetVoxels={targetVoxels}
          downsampleMethod={downsampleMethod}
          showPerformanceMetrics={showPerformanceMetrics}
          onEnableDownsamplingChange={setEnableDownsampling}
          onTargetVoxelsChange={setTargetVoxels}
          onDownsampleMethodChange={setDownsampleMethod}
          onCopyLink={handleCopyLink}
          linkCopied={linkCopied}
          onShowPerformanceMetricsChange={setShowPerformanceMetrics}
          // Flow particle visualization props
          visualizationMode={visualizationMode}
          flowParticleConfig={flowParticleConfig}
          hasVelocityData={hasVelocityData}
          onVisualizationModeChange={setVisualizationMode}
          onFlowParticleConfigChange={setFlowParticleConfig}
          // Export props
          exportState={exportState}
          exportActions={exportActions}
          getCanvas={getCanvas}
          currentFrame={currentFrame}
          renderFrame={renderFrameForExport}
          pressure={pressure}
          resolution={resolution}
          probeData={probeData}
          getViewState={getViewState}
        />
      }
      main={
        <>
          <OptimizedVoxelRenderer
            ref={rendererRef}
            pressure={visualizationMode === "voxels" ? pressure : null}
            shape={shape}
            resolution={resolution}
            geometry={voxelGeometry}
            threshold={threshold}
            displayFill={displayFill}
            showGrid={showGrid}
            showAxes={showAxes}
            geometryMode={geometryMode}
            demoType={demoType}
            geometryOpacity={geometryOpacity / 100}
            enableDownsampling={enableDownsampling}
            targetVoxels={targetVoxels}
            downsampleMethod={downsampleMethod}
            showPerformanceMetrics={showPerformanceMetrics}
            onPerformanceUpdate={updatePerformanceMetrics}
            onReady={handleRendererReady}
          />
          {visualizationMode === "flow_particles" && (
            <FlowParticleRenderer
              pressure={pressure}
              velocity={currentVelocity}
              shape={shape}
              resolution={resolution}
              geometryMask={geometry?.mask ?? null}
              config={flowParticleConfig}
              scene={rendererScene}
              pressureRange={pressureRange}
              deltaTime={flowDeltaTime}
            />
          )}
        </>
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
  )
}

interface SidebarProps {
  isLoading: boolean
  error: string | null
  manifest: ReturnType<typeof useSimulationStore.getState>["manifest"]
  shape: [number, number, number]
  totalFrames: number
  voxelGeometry: VoxelGeometry
  threshold: number
  displayFill: number
  showGrid: boolean
  showAxes: boolean
  geometryMode: GeometryMode
  demoType: DemoGeometryType
  geometryOpacity: number
  isLooping: boolean
  playbackSpeed: number
  enableDownsampling: boolean
  targetVoxels: number
  downsampleMethod: DownsampleMethod
  showPerformanceMetrics: boolean
  onLoadSample: () => void
  onVoxelGeometryChange: (g: VoxelGeometry) => void
  onThresholdChange: (t: number) => void
  onDisplayFillChange: (f: number) => void
  onShowGridChange: (v: boolean) => void
  onShowAxesChange: (v: boolean) => void
  onGeometryModeChange: (mode: GeometryMode) => void
  onDemoTypeChange: (type: DemoGeometryType) => void
  onGeometryOpacityChange: (opacity: number) => void
  onLoopingChange: (v: boolean) => void
  onPlaybackSpeedChange: (v: number) => void
  onNavigate: (page: Page) => void
  onEnableDownsamplingChange: (v: boolean) => void
  onTargetVoxelsChange: (v: number) => void
  onDownsampleMethodChange: (v: DownsampleMethod) => void
  onCopyLink: () => void
  linkCopied: boolean
  onShowPerformanceMetricsChange: (v: boolean) => void
  // Flow particle visualization props
  visualizationMode: VisualizationMode
  flowParticleConfig: FlowParticleConfig
  hasVelocityData: boolean
  onVisualizationModeChange: (mode: VisualizationMode) => void
  onFlowParticleConfigChange: (config: Partial<FlowParticleConfig>) => void
  // Export props
  exportState: ExportState
  exportActions: ExportActions
  getCanvas: () => HTMLCanvasElement | null
  currentFrame: number
  renderFrame: (frameIndex: number) => Promise<void>
  pressure: Float32Array | null
  resolution: number
  probeData: {
    probes: Record<string, { data: Float32Array; position?: [number, number, number] }>
    sampleRate: number
    duration: number
  } | null
  getViewState: () => ViewState
}

function Sidebar({
  isLoading,
  error,
  manifest,
  shape,
  totalFrames,
  voxelGeometry,
  threshold,
  displayFill,
  showGrid,
  showAxes,
  geometryMode,
  demoType,
  geometryOpacity,
  isLooping,
  playbackSpeed,
  enableDownsampling,
  targetVoxels,
  downsampleMethod,
  showPerformanceMetrics,
  onLoadSample,
  onVoxelGeometryChange,
  onThresholdChange,
  onDisplayFillChange,
  onShowGridChange,
  onShowAxesChange,
  onGeometryModeChange,
  onDemoTypeChange,
  onGeometryOpacityChange,
  onLoopingChange,
  onPlaybackSpeedChange,
  onNavigate,
  onEnableDownsamplingChange,
  onTargetVoxelsChange,
  onDownsampleMethodChange,
  onCopyLink,
  linkCopied,
  onShowPerformanceMetricsChange,
  // Flow particle visualization props
  visualizationMode,
  flowParticleConfig,
  hasVelocityData,
  onVisualizationModeChange,
  onFlowParticleConfigChange,
  // Export props
  exportState,
  exportActions,
  getCanvas,
  currentFrame,
  renderFrame,
  pressure,
  resolution,
  probeData,
  getViewState,
}: SidebarProps) {
  const [showPerfSettings, setShowPerfSettings] = useState(false)
  const geometryOptions: { value: VoxelGeometry; label: string; icon: React.ReactNode }[] = [
    { value: "point", label: "Points", icon: <Sparkles className="h-3 w-3" /> },
    { value: "mesh", label: "Mesh", icon: <Grid3x3 className="h-3 w-3" /> },
    { value: "hidden", label: "Hidden", icon: <EyeOff className="h-3 w-3" /> },
  ]

  const speedOptions = [0.25, 0.5, 1, 2, 4]

  // Memoize slider values to prevent infinite re-render loops
  // (creating new arrays on each render triggers Radix Slider's onValueChange)
  const thresholdValue = useMemo(() => [Math.round(threshold * 100)], [threshold])
  const displayFillValue = useMemo(() => [Math.round(displayFill * 100)], [displayFill])
  const geometryOpacityValue = useMemo(() => [geometryOpacity], [geometryOpacity])

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-lg font-bold text-foreground">FDTD Visualizer</h1>
        <p className="text-sm text-muted-foreground">Acoustic simulation viewer</p>
      </div>

<Panel title="Tools">
        <div className="space-y-3">
          <div>
            <Button
              variant="outline"
              className="w-full justify-start gap-2"
              onClick={() => onNavigate("builder")}
            >
              <Hammer className="h-4 w-4" />
              Simulation Builder
            </Button>
            <p className="text-xs text-muted-foreground mt-1">
              Build simulations visually with live 3D preview
            </p>
          </div>
          <div>
            <Button
              variant="outline"
              className="w-full justify-start gap-2"
              onClick={() => onNavigate("viewer")}
            >
              <FileUp className="h-4 w-4" />
              Open HDF5 File
            </Button>
            <p className="text-xs text-muted-foreground mt-1">
              Load and visualize simulation results
            </p>
          </div>
          <div>
            <Button
              variant="outline"
              className="w-full justify-start gap-2"
              onClick={() => onNavigate("organ-pipes")}
            >
              <Music className="h-4 w-4" />
              Organ Pipe Modes
            </Button>
            <p className="text-xs text-muted-foreground mt-1">
              Explore standing waves in open/closed pipes
            </p>
          </div>
        </div>
      </Panel>

      <Panel title="Data">
        <Button
          variant="outline"
          className="w-full justify-start gap-2"
          onClick={onLoadSample}
          disabled={isLoading}
        >
          <FolderOpen className="h-4 w-4" />
          {isLoading ? "Loading..." : "Load Sample Data"}
        </Button>
        {error && <div className="text-xs text-destructive">{error}</div>}
        {manifest ? (
          <div className="text-xs text-muted-foreground">
            Grid: {shape.join(" x ")} | {totalFrames} frames
          </div>
        ) : (
          <div className="text-xs text-muted-foreground">
            Demo: {DEMO_GEOMETRIES.find(g => g.value === demoType)?.label} geometry
          </div>
        )}
        <Button
          variant="outline"
          className="w-full justify-start gap-2 mt-2"
          onClick={onCopyLink}
        >
          <Link2 className="h-4 w-4" />
          {linkCopied ? "Link Copied!" : "Copy Shareable Link"}
        </Button>
        <p className="text-xs text-muted-foreground mt-1">
          Share current view (frame, colormap, settings)
        </p>
      </Panel>

      <Panel title="Demo Geometry">
        <div className="flex flex-wrap gap-1">
          {DEMO_GEOMETRIES.map((geo) => (
            <Badge
              key={geo.value}
              variant={demoType === geo.value ? "default" : "secondary"}
              className="cursor-pointer"
              onClick={() => onDemoTypeChange(geo.value)}
            >
              {geo.label}
            </Badge>
          ))}
        </div>
      </Panel>

      <VisualizationModePanel
        visualizationMode={visualizationMode}
        hasVelocityData={hasVelocityData}
        flowParticleConfig={flowParticleConfig}
        onModeChange={onVisualizationModeChange}
        onConfigChange={onFlowParticleConfigChange}
      />

      <Panel title="Boundary Display">
        <div className="space-y-3">
          <div className="flex flex-wrap gap-1">
            {GEOMETRY_MODES.map((mode) => (
              <Badge
                key={mode.value}
                variant={geometryMode === mode.value ? "default" : "secondary"}
                className="cursor-pointer"
                onClick={() => onGeometryModeChange(mode.value)}
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
                value={geometryOpacityValue}
                onValueChange={(v) => onGeometryOpacityChange(v[0])}
                min={10}
                max={80}
                step={5}
              />
            </div>
          )}
        </div>
      </Panel>

      <Panel title="Voxel Display">
        <div className="space-y-3">
          <div>
            <span className="text-sm mb-2 block">Geometry</span>
            <div className="flex gap-1">
              {geometryOptions.map((opt) => (
                <Button
                  key={opt.value}
                  variant={voxelGeometry === opt.value ? "default" : "outline"}
                  size="sm"
                  className="flex-1 gap-1"
                  onClick={() => onVoxelGeometryChange(opt.value)}
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

      <Panel title="Threshold">
        <Slider
          value={thresholdValue}
          max={100}
          step={1}
          onValueChange={([v]) => onThresholdChange(v / 100)}
        />
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>0%</span>
          <span>{Math.round(threshold * 100)}%</span>
          <span>100%</span>
        </div>
      </Panel>

      <Panel title="Display Fill">
        <Slider
          value={displayFillValue}
          min={1}
          max={100}
          step={1}
          onValueChange={([v]) => onDisplayFillChange(v / 100)}
        />
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>1%</span>
          <span>{Math.round(displayFill * 100)}%</span>
          <span>100%</span>
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          Randomly hide voxels to reduce visual density
        </p>
      </Panel>

      <Panel title="Playback">
        <div className="space-y-3">
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={isLooping}
              onChange={(e) => onLoopingChange(e.target.checked)}
              className="rounded"
            />
            <Repeat className="h-3 w-3" />
            Loop
          </label>
          <div>
            <span className="text-sm mb-2 block">Speed</span>
            <div className="flex gap-1">
              {speedOptions.map((speed) => (
                <Button
                  key={speed}
                  variant={playbackSpeed === speed ? "default" : "outline"}
                  size="sm"
                  className="flex-1 text-xs"
                  onClick={() => onPlaybackSpeedChange(speed)}
                >
                  {speed}x
                </Button>
              ))}
            </div>
          </div>
        </div>
      </Panel>

      <Panel title="View Options">
        <div className="space-y-2">
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={showGrid}
              onChange={(e) => onShowGridChange(e.target.checked)}
              className="rounded"
            />
            Show Grid
          </label>
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={showAxes}
              onChange={(e) => onShowAxesChange(e.target.checked)}
              className="rounded"
            />
            Show Axes
          </label>
        </div>
      </Panel>

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

      <Panel title="Export">
        <ExportPanel
          exportState={exportState}
          exportActions={exportActions}
          getCanvas={getCanvas}
          totalFrames={totalFrames}
          currentFrame={currentFrame}
          renderFrame={renderFrame}
          pressure={pressure}
          shape={shape}
          resolution={resolution}
          probeData={probeData}
          getViewState={getViewState}
        />
      </Panel>

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
  )
}

interface BottomPanelProps {
  totalFrames: number
  currentFrame: number
  isPlaying: boolean
  playbackSpeed: number
  probeData: ReturnType<typeof useSimulationStore.getState>["probeData"]
  selectedProbe: string | null
  viewMode: "time" | "spectrum"
  currentTime?: number
  onFrameChange: (frame: number) => void
  onPlayingChange: (playing: boolean) => void
  onTimeSelect: (time: number) => void
  onViewModeChange: (mode: "time" | "spectrum") => void
  onSelectProbe: (name: string | null) => void
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
  const probeNames = probeData ? Object.keys(probeData.probes) : []

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
            Load simulation to view charts
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
  )
}

export default App
