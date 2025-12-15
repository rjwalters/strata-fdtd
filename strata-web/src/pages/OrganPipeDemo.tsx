import { useState, useCallback, useEffect, useRef, useMemo } from "react"
import * as THREE from "three"
// Shared UI components from strata-ui
import {
  Layout,
  Panel,
  VoxelRenderer,
  FlowParticleRenderer,
  VisualizationModePanel,
  PlaybackControls,
  BackgroundLoadingIndicator,
  VideoExportButton,
  Button,
  Badge,
  Slider,
  useSimulationStore,
  useCurrentPressure,
  usePlaybackState,
  useGridInfo,
  useLoadingState,
  type VoxelRendererHandle,
  type VoxelGeometry,
  type VisualizationMode,
  type FlowParticleConfig,
} from "strata-ui"
import { Sparkles, Grid3x3, EyeOff, ArrowLeft, Info } from "lucide-react"
// Local app-specific hooks
import { useVideoExport } from "@/hooks/useVideoExport"

interface PipeConfig {
  id: string
  name: string
  description: string
  path: string
  physics: string
  modeFormula: string
}

const PIPE_CONFIGS: PipeConfig[] = [
  {
    id: "closed",
    name: "Closed Pipe",
    description: "Both ends closed (rigid walls)",
    path: "/demos/organ-pipes/closed-pipe",
    physics: "Standing waves with pressure antinodes at both ends. Creates all harmonics.",
    modeFormula: "fₙ = n × c / (2L)",
  },
  {
    id: "open",
    name: "Open Pipe",
    description: "Both ends open (radiating)",
    path: "/demos/organ-pipes/open-pipe",
    physics: "Standing waves with pressure nodes at both ends. Creates all harmonics.",
    modeFormula: "fₙ = n × c / (2L)",
  },
  {
    id: "half-open",
    name: "Half-Open Pipe",
    description: "One end closed, one open",
    path: "/demos/organ-pipes/half-open-pipe",
    physics: "Pressure antinode at closed end, node at open end. Only odd harmonics.",
    modeFormula: "fₙ = (2n-1) × c / (4L)",
  },
]

interface OrganPipeDemoProps {
  onBack?: () => void
}

export function OrganPipeDemo({ onBack }: OrganPipeDemoProps) {
  // Zustand store selectors
  const pressure = useCurrentPressure()
  const { currentFrame, isPlaying, totalFrames } = usePlaybackState()
  const { shape, resolution } = useGridInfo()
  const { isLoading, error } = useLoadingState()
  const manifest = useSimulationStore((s) => s.manifest)

  // Flow visualization state
  const visualizationMode = useSimulationStore((s) => s.visualizationMode)
  const flowParticleConfig = useSimulationStore((s) => s.flowParticleConfig)
  const hasVelocityData = useSimulationStore((s) => s.hasVelocityData)
  const velocitySnapshots = useSimulationStore((s) => s.velocitySnapshots)
  const geometry = useSimulationStore((s) => s.geometry)
  const playbackSpeed = useSimulationStore((s) => s.playbackSpeed)

  // Zustand store actions
  const loadSimulation = useSimulationStore((s) => s.loadSimulation)
  const loadSnapshot = useSimulationStore((s) => s.loadSnapshot)
  const setCurrentFrame = useSimulationStore((s) => s.setCurrentFrame)
  const play = useSimulationStore((s) => s.play)
  const pause = useSimulationStore((s) => s.pause)
  const loadVelocityForFrame = useSimulationStore((s) => s.loadVelocityForFrame)
  const setVisualizationMode = useSimulationStore((s) => s.setVisualizationMode)
  const setFlowParticleConfig = useSimulationStore((s) => s.setFlowParticleConfig)

  const [exportState, exportActions] = useVideoExport()
  const rendererRef = useRef<VoxelRendererHandle>(null)
  // Scene state for FlowParticleRenderer (avoid accessing ref during render)
  const [rendererScene, setRendererScene] = useState<THREE.Scene | null>(null)
  const [voxelGeometry, setVoxelGeometry] = useState<VoxelGeometry>("point")
  const [threshold, setThreshold] = useState(0.1)
  const [showGrid, setShowGrid] = useState(true)
  const [showAxes, setShowAxes] = useState(true)
  const [selectedPipe, setSelectedPipe] = useState<PipeConfig>(PIPE_CONFIGS[0])
  const [modeAnalysis, setModeAnalysis] = useState<{
    expected_modes_hz: number[]
    measured_modes_hz: number[]
    mode_errors_percent: number[]
  } | null>(null)

  // Load snapshot when frame changes
  useEffect(() => {
    if (manifest && totalFrames > 0) {
      loadSnapshot(currentFrame)
    }
  }, [currentFrame, manifest, totalFrames, loadSnapshot])

  // Load velocity data when in flow mode
  useEffect(() => {
    if (manifest && hasVelocityData && visualizationMode === "flow_particles") {
      loadVelocityForFrame(currentFrame)
    }
  }, [currentFrame, manifest, hasVelocityData, visualizationMode, loadVelocityForFrame])

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
    const absMax = Math.max(Math.abs(min), Math.abs(max))
    return [-absMax, absMax]
  }, [pressure])

  // Get current velocity for flow particles
  const currentVelocity = velocitySnapshots.get(currentFrame) ?? null

  // Delta time for flow particles (scaled by playback speed)
  const flowDeltaTime = (1 / 60) / playbackSpeed

  // Callback to render a specific frame for video export
  const renderFrameForExport = useCallback(async (frameIndex: number) => {
    setCurrentFrame(frameIndex)
    // Wait for the frame to load and render
    await new Promise(resolve => setTimeout(resolve, 50))
    rendererRef.current?.render()
  }, [setCurrentFrame])

  const loadPipeSimulation = useCallback(async (pipe: PipeConfig) => {
    setSelectedPipe(pipe)
    setModeAnalysis(null)
    await loadSimulation(pipe.path)

    // Load mode analysis if available
    try {
      const prefix = pipe.id === "half-open" ? "half-open-pipe" : `${pipe.id}-pipe`
      const response = await fetch(`${pipe.path}/${prefix}_mode_analysis.json`)
      if (response.ok) {
        const analysis = await response.json()
        setModeAnalysis(analysis)
      }
    } catch (e) {
      console.warn("Mode analysis not available:", e)
    }
  }, [loadSimulation])

  // Load initial simulation
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    loadPipeSimulation(PIPE_CONFIGS[0])
  }, []) // Only run on mount

  // Handle play/pause callbacks
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
          showGrid={showGrid}
          showAxes={showAxes}
          selectedPipe={selectedPipe}
          modeAnalysis={modeAnalysis}
          visualizationMode={visualizationMode}
          hasVelocityData={hasVelocityData}
          flowParticleConfig={flowParticleConfig}
          onBack={onBack}
          onPipeSelect={loadPipeSimulation}
          onVoxelGeometryChange={setVoxelGeometry}
          onThresholdChange={setThreshold}
          onShowGridChange={setShowGrid}
          onShowAxesChange={setShowAxes}
          onVisualizationModeChange={setVisualizationMode}
          onFlowConfigChange={setFlowParticleConfig}
        />
      }
      main={
        <>
          <VoxelRenderer
            ref={rendererRef}
            pressure={visualizationMode === "voxels" ? pressure : null}
            shape={shape}
            resolution={resolution}
            geometry={voxelGeometry}
            threshold={threshold}
            showGrid={showGrid}
            showAxes={showAxes}
            onReady={() => setRendererScene(rendererRef.current?.getScene() ?? null)}
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
          currentFrame={currentFrame}
          isPlaying={isPlaying}
          totalFrames={totalFrames}
          selectedPipe={selectedPipe}
          onFrameChange={setCurrentFrame}
          onPlayingChange={handlePlayingChange}
          exportState={exportState}
          exportActions={exportActions}
          getCanvas={() => rendererRef.current?.getCanvas() ?? null}
          renderFrameForExport={renderFrameForExport}
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
  showGrid: boolean
  showAxes: boolean
  selectedPipe: PipeConfig
  modeAnalysis: {
    expected_modes_hz: number[]
    measured_modes_hz: number[]
    mode_errors_percent: number[]
  } | null
  // Flow visualization
  visualizationMode: VisualizationMode
  hasVelocityData: boolean
  flowParticleConfig: FlowParticleConfig
  onBack?: () => void
  onPipeSelect: (pipe: PipeConfig) => void
  onVoxelGeometryChange: (g: VoxelGeometry) => void
  onThresholdChange: (t: number) => void
  onShowGridChange: (v: boolean) => void
  onShowAxesChange: (v: boolean) => void
  onVisualizationModeChange: (mode: VisualizationMode) => void
  onFlowConfigChange: (config: Partial<FlowParticleConfig>) => void
}

function Sidebar({
  isLoading,
  error,
  manifest,
  shape,
  totalFrames,
  voxelGeometry,
  threshold,
  showGrid,
  showAxes,
  selectedPipe,
  modeAnalysis,
  visualizationMode,
  hasVelocityData,
  flowParticleConfig,
  onBack,
  onPipeSelect,
  onVoxelGeometryChange,
  onThresholdChange,
  onShowGridChange,
  onShowAxesChange,
  onVisualizationModeChange,
  onFlowConfigChange,
}: SidebarProps) {
  const geometryOptions: { value: VoxelGeometry; label: string; icon: React.ReactNode }[] = [
    { value: "point", label: "Points", icon: <Sparkles className="h-3 w-3" /> },
    { value: "mesh", label: "Mesh", icon: <Grid3x3 className="h-3 w-3" /> },
    { value: "hidden", label: "Hidden", icon: <EyeOff className="h-3 w-3" /> },
  ]

  return (
    <div className="space-y-6">
      <div>
        {onBack && (
          <Button
            variant="ghost"
            size="sm"
            className="mb-2 -ml-2"
            onClick={onBack}
          >
            <ArrowLeft className="h-4 w-4 mr-1" />
            Back
          </Button>
        )}
        <h1 className="text-lg font-bold text-foreground">Organ Pipe Demo</h1>
        <p className="text-sm text-muted-foreground">Standing wave modes in pipes</p>
      </div>

      <Panel title="Pipe Type">
        <div className="space-y-2">
          {PIPE_CONFIGS.map((pipe) => (
            <Button
              key={pipe.id}
              variant={selectedPipe.id === pipe.id ? "default" : "outline"}
              className="w-full justify-start text-left h-auto py-2"
              onClick={() => onPipeSelect(pipe)}
              disabled={isLoading}
            >
              <div>
                <div className="font-medium">{pipe.name}</div>
                <div className="text-xs opacity-70">{pipe.description}</div>
              </div>
            </Button>
          ))}
        </div>
        {error && (
          <div className="text-xs text-destructive mt-2">{error}</div>
        )}
        {manifest && (
          <div className="text-xs text-muted-foreground mt-2">
            Grid: {shape.join(" × ")} | {totalFrames} frames
          </div>
        )}
      </Panel>

      <Panel title="Physics">
        <div className="space-y-2 text-sm">
          <p className="text-muted-foreground">{selectedPipe.physics}</p>
          <div className="bg-secondary/50 rounded px-2 py-1 font-mono text-xs">
            {selectedPipe.modeFormula}
          </div>
        </div>
      </Panel>

      {modeAnalysis && (
        <Panel title="Mode Analysis">
          <div className="space-y-1 text-xs">
            {modeAnalysis.expected_modes_hz.slice(0, 3).map((_, i) => {
              const measured = modeAnalysis.measured_modes_hz[i]
              const measurementError = modeAnalysis.mode_errors_percent[i]
              const isGood = measurementError < 10
              return (
                <div key={i} className="flex justify-between items-center">
                  <span className="text-muted-foreground">Mode {i + 1}:</span>
                  <span className={isGood ? "text-green-500" : "text-yellow-500"}>
                    {measured?.toFixed(0) ?? "?"} Hz
                    <span className="text-muted-foreground ml-1">
                      ({measurementError?.toFixed(1) ?? "?"}% err)
                    </span>
                  </span>
                </div>
              )
            })}
          </div>
        </Panel>
      )}

      <VisualizationModePanel
        visualizationMode={visualizationMode}
        hasVelocityData={hasVelocityData}
        flowParticleConfig={flowParticleConfig}
        onModeChange={onVisualizationModeChange}
        onConfigChange={onFlowConfigChange}
      />

      <Panel title="Display">
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
        </div>
      </Panel>

      <Panel title="Threshold">
        <Slider
          value={[threshold * 100]}
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
    </div>
  )
}

interface BottomPanelProps {
  currentFrame: number
  isPlaying: boolean
  totalFrames: number
  selectedPipe: PipeConfig
  onFrameChange: (frame: number) => void
  onPlayingChange: (playing: boolean) => void
  exportState: ReturnType<typeof useVideoExport>[0]
  exportActions: ReturnType<typeof useVideoExport>[1]
  getCanvas: () => HTMLCanvasElement | null
  renderFrameForExport: (frameIndex: number) => Promise<void>
}

function BottomPanel({
  currentFrame,
  isPlaying,
  totalFrames,
  selectedPipe,
  onFrameChange,
  onPlayingChange,
  exportState,
  exportActions,
  getCanvas,
  renderFrameForExport,
}: BottomPanelProps) {
  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold text-muted-foreground">
            {selectedPipe.name}
          </h3>
          <Badge variant="outline" className="text-xs">
            {selectedPipe.id}
          </Badge>
        </div>
        <div className="flex items-center gap-4">
          <VideoExportButton
            exportState={exportState}
            exportActions={exportActions}
            getCanvas={getCanvas}
            totalFrames={totalFrames}
            renderFrame={renderFrameForExport}
            disabled={totalFrames === 0 || isPlaying}
            options={{ fps: 15, filename: `${selectedPipe.id}-pipe.mp4` }}
          />
          <div className="flex items-center gap-1 text-xs text-muted-foreground">
            <Info className="h-3 w-3" />
            <span>Pipe length: 200mm | Radius: 15mm | Resolution: 2mm</span>
          </div>
        </div>
      </div>

      <div className="flex-1 bg-secondary/30 rounded-md flex items-center justify-center text-muted-foreground text-sm">
        <div className="text-center">
          <p className="mb-1">{selectedPipe.physics}</p>
          <p className="font-mono text-xs">{selectedPipe.modeFormula}</p>
        </div>
      </div>

      <div className="mt-2 flex items-center gap-4">
        <PlaybackControls
          totalFrames={totalFrames}
          currentFrame={currentFrame}
          isPlaying={isPlaying}
          onFrameChange={onFrameChange}
          onPlayingChange={onPlayingChange}
          disabled={totalFrames === 0}
          fps={15}
        />
        <BackgroundLoadingIndicator />
      </div>
    </div>
  )
}

export default OrganPipeDemo
