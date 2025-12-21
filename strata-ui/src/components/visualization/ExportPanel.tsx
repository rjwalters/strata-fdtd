/**
 * Export panel component with screenshot, animation, and data export options.
 */

import { useState, useCallback, useMemo, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Slider } from "@/components/ui/slider"
import {
  Camera,
  Film,
  FileSpreadsheet,
  ChevronDown,
  ChevronRight,
  X,
  Loader2,
  ImageIcon,
  FileJson,
  FileText,
  Layers,
} from "lucide-react"
import { type ExportState, type ExportActions } from "@/hooks/useExport"
import type { ViewState } from "@/lib/export"
import type { SliceAxis } from "@/stores/simulationStore"

interface ExportPanelProps {
  /** Export state from useExport hook */
  exportState: ExportState
  /** Export actions from useExport hook */
  exportActions: ExportActions
  /** Get canvas element for screenshot/animation export */
  getCanvas: () => HTMLCanvasElement | null
  /** Total frames available */
  totalFrames: number
  /** Current frame index */
  currentFrame: number
  /** Render a specific frame */
  renderFrame: (frameIndex: number) => Promise<void>
  /** Current pressure data */
  pressure: Float32Array | null
  /** Grid shape */
  shape: [number, number, number]
  /** Grid resolution */
  resolution: number
  /** Probe data */
  probeData: {
    probes: Record<string, { data: Float32Array; position?: [number, number, number] }>
    sampleRate: number
    duration: number
  } | null
  /** Get current view state for JSON export */
  getViewState: () => ViewState
  /** Whether in slice view mode */
  isSliceMode?: boolean
  /** Current slice axis */
  sliceAxis?: SliceAxis
  /** Current slice position (0-1) */
  slicePosition?: number
  /** Set slice position for position sweep */
  setSlicePosition?: (position: number) => void
}

type ResolutionOption = 1 | 2 | 4

const RESOLUTION_OPTIONS: { value: ResolutionOption; label: string }[] = [
  { value: 1, label: "1x" },
  { value: 2, label: "2x" },
  { value: 4, label: "4x" },
]

const ANIMATION_RESOLUTION_OPTIONS: { value: 1 | 2; label: string }[] = [
  { value: 1, label: "1x" },
  { value: 2, label: "2x" },
]

const FPS_OPTIONS = [5, 10, 15, 30]

export function ExportPanel({
  exportState,
  exportActions,
  getCanvas,
  totalFrames,
  currentFrame,
  renderFrame,
  pressure,
  shape,
  resolution,
  probeData,
  getViewState,
  isSliceMode = false,
  sliceAxis = "z",
  slicePosition = 0.5,
  setSlicePosition,
}: ExportPanelProps) {
  const [expandedSection, setExpandedSection] = useState<string | null>(null)

  // Screenshot options
  const [screenshotResolution, setScreenshotResolution] = useState<ResolutionOption>(1)

  // Animation options
  const [animationFormat, setAnimationFormat] = useState<"png-sequence" | "gif">("png-sequence")
  const [animationResolution, setAnimationResolution] = useState<1 | 2>(1)
  const [animationFps, setAnimationFps] = useState(15)
  const [frameRangeStart, setFrameRangeStart] = useState(0)
  const [frameRangeEnd, setFrameRangeEnd] = useState(Math.max(0, totalFrames - 1))

  // Slice animation options
  const [sliceAnimationType, setSliceAnimationType] = useState<"time" | "position">("time")
  const [sliceAnimationFormat, setSliceAnimationFormat] = useState<"png-sequence" | "gif">("png-sequence")
  const [sliceAnimationFps, setSliceAnimationFps] = useState(15)
  const [sliceFrameRangeStart, setSliceFrameRangeStart] = useState(0)
  const [sliceFrameRangeEnd, setSliceFrameRangeEnd] = useState(Math.max(0, totalFrames - 1))
  const [slicePositionFrames, setSlicePositionFrames] = useState(50)

  // Update frame range when totalFrames changes
  const maxFrame = Math.max(0, totalFrames - 1)
  useEffect(() => {
    if (frameRangeEnd > maxFrame) {
      setFrameRangeEnd(maxFrame)
    }
    if (sliceFrameRangeEnd > maxFrame) {
      setSliceFrameRangeEnd(maxFrame)
    }
  }, [maxFrame, frameRangeEnd, sliceFrameRangeEnd])

  // Memoize slider value to prevent infinite re-render loops
  const frameRangeValue = useMemo(() => [frameRangeStart, frameRangeEnd], [frameRangeStart, frameRangeEnd])
  const sliceFrameRangeValue = useMemo(() => [sliceFrameRangeStart, sliceFrameRangeEnd], [sliceFrameRangeStart, sliceFrameRangeEnd])

  // Get axis size for position sweep
  const getAxisSize = useCallback(() => {
    switch (sliceAxis) {
      case "x": return shape[0]
      case "y": return shape[1]
      case "z": return shape[2]
    }
  }, [sliceAxis, shape])

  const toggleSection = useCallback((section: string) => {
    setExpandedSection((s) => (s === section ? null : section))
  }, [])

  const handleScreenshot = useCallback(async () => {
    const canvas = getCanvas()
    if (!canvas) {
      console.error("Canvas not available")
      return
    }
    await exportActions.exportScreenshot(canvas, { resolution: screenshotResolution })
  }, [getCanvas, exportActions, screenshotResolution])

  const handleAnimationExport = useCallback(async () => {
    const canvas = getCanvas()
    if (!canvas) {
      console.error("Canvas not available")
      return
    }

    const options = {
      frameRange: [frameRangeStart, frameRangeEnd] as [number, number],
      fps: animationFps,
      resolution: animationResolution,
    }

    if (animationFormat === "png-sequence") {
      await exportActions.exportPngSequence(canvas, totalFrames, renderFrame, options)
    } else {
      await exportActions.exportGif(canvas, totalFrames, renderFrame, options)
    }
  }, [
    getCanvas,
    exportActions,
    animationFormat,
    frameRangeStart,
    frameRangeEnd,
    animationFps,
    animationResolution,
    totalFrames,
    renderFrame,
  ])

  const handleSliceAnimationExport = useCallback(async () => {
    const canvas = getCanvas()
    if (!canvas) {
      console.error("Canvas not available")
      return
    }

    if (sliceAnimationType === "time") {
      // Time animation at fixed slice position
      const options = {
        frameRange: [sliceFrameRangeStart, sliceFrameRangeEnd] as [number, number],
        fps: sliceAnimationFps,
        resolution: 1 as const,
      }

      if (sliceAnimationFormat === "png-sequence") {
        await exportActions.exportPngSequence(canvas, totalFrames, renderFrame, options)
      } else {
        await exportActions.exportGif(canvas, totalFrames, renderFrame, options)
      }
    } else {
      // Position sweep animation at fixed timestep
      if (!setSlicePosition) {
        console.error("setSlicePosition not provided")
        return
      }

      const originalPosition = slicePosition

      // Create a render function that changes slice position
      const renderSliceAtPosition = async (frameIndex: number) => {
        const position = frameIndex / (slicePositionFrames - 1)
        setSlicePosition(position)
        // Wait for render to update
        await new Promise((resolve) => setTimeout(resolve, 50))
      }

      const options = {
        frameRange: [0, slicePositionFrames - 1] as [number, number],
        fps: sliceAnimationFps,
        resolution: 1 as const,
        filename: `slice-sweep-${sliceAxis}-${sliceAnimationFormat === "png-sequence" ? "frames.zip" : "animation.gif"}`,
      }

      try {
        if (sliceAnimationFormat === "png-sequence") {
          await exportActions.exportPngSequence(canvas, slicePositionFrames, renderSliceAtPosition, options)
        } else {
          await exportActions.exportGif(canvas, slicePositionFrames, renderSliceAtPosition, options)
        }
      } finally {
        // Restore original position
        setSlicePosition(originalPosition)
      }
    }
  }, [
    getCanvas,
    exportActions,
    sliceAnimationType,
    sliceAnimationFormat,
    sliceAnimationFps,
    sliceFrameRangeStart,
    sliceFrameRangeEnd,
    slicePositionFrames,
    slicePosition,
    setSlicePosition,
    sliceAxis,
    totalFrames,
    renderFrame,
    getAxisSize,
  ])

  const handleExportPressure = useCallback(() => {
    if (!pressure) return
    exportActions.exportPressureData(pressure, shape, resolution, currentFrame)
  }, [exportActions, pressure, shape, resolution, currentFrame])

  const handleExportProbe = useCallback(() => {
    if (!probeData) return
    exportActions.exportProbeData(probeData.probes, probeData.sampleRate)
  }, [exportActions, probeData])

  const handleExportViewState = useCallback(() => {
    const viewState = getViewState()
    exportActions.exportViewStateJson(viewState)
  }, [exportActions, getViewState])

  const { isExporting, progress, stage, error, exportType } = exportState

  // Show progress indicator when exporting
  if (isExporting) {
    return (
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-medium">Export</h3>
          <Button
            variant="ghost"
            size="sm"
            onClick={exportActions.cancelExport}
            className="h-6 px-2"
          >
            <X className="h-3 w-3" />
          </Button>
        </div>
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-sm">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span className="text-muted-foreground capitalize">
              {exportType}: {stage}...
            </span>
          </div>
          <div className="w-full h-2 bg-secondary rounded-full overflow-hidden">
            <div
              className="h-full bg-primary transition-all duration-200"
              style={{ width: `${progress}%` }}
            />
          </div>
          <div className="text-xs text-muted-foreground text-right">{progress}%</div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      {/* Error display */}
      {error && (
        <div className="flex items-center justify-between bg-destructive/10 text-destructive text-xs p-2 rounded">
          <span>{error}</span>
          <Button
            variant="ghost"
            size="sm"
            onClick={exportActions.resetError}
            className="h-5 w-5 p-0"
          >
            <X className="h-3 w-3" />
          </Button>
        </div>
      )}

      {/* Screenshot Section */}
      <div className="border border-border rounded-md overflow-hidden">
        <button
          className="w-full flex items-center justify-between p-2 hover:bg-accent/50 transition-colors"
          onClick={() => toggleSection("screenshot")}
        >
          <span className="flex items-center gap-2 text-sm">
            <Camera className="h-4 w-4" />
            Screenshot
          </span>
          {expandedSection === "screenshot" ? (
            <ChevronDown className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
        </button>

        {expandedSection === "screenshot" && (
          <div className="p-2 pt-0 space-y-2">
            <div className="space-y-1">
              <span className="text-xs text-muted-foreground">Resolution</span>
              <div className="flex gap-1">
                {RESOLUTION_OPTIONS.map((opt) => (
                  <Badge
                    key={opt.value}
                    variant={screenshotResolution === opt.value ? "default" : "secondary"}
                    className="cursor-pointer text-xs"
                    onClick={() => setScreenshotResolution(opt.value)}
                  >
                    {opt.label}
                  </Badge>
                ))}
              </div>
            </div>
            <Button
              variant="outline"
              size="sm"
              className="w-full gap-2"
              onClick={handleScreenshot}
            >
              <ImageIcon className="h-4 w-4" />
              Export PNG
            </Button>
          </div>
        )}
      </div>

      {/* Animation Section */}
      <div className="border border-border rounded-md overflow-hidden">
        <button
          className="w-full flex items-center justify-between p-2 hover:bg-accent/50 transition-colors"
          onClick={() => toggleSection("animation")}
        >
          <span className="flex items-center gap-2 text-sm">
            <Film className="h-4 w-4" />
            Animation
          </span>
          {expandedSection === "animation" ? (
            <ChevronDown className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
        </button>

        {expandedSection === "animation" && (
          <div className="p-2 pt-0 space-y-3">
            <div className="space-y-1">
              <span className="text-xs text-muted-foreground">Format</span>
              <div className="flex gap-1">
                <Badge
                  variant={animationFormat === "png-sequence" ? "default" : "secondary"}
                  className="cursor-pointer text-xs"
                  onClick={() => setAnimationFormat("png-sequence")}
                >
                  PNG Sequence (ZIP)
                </Badge>
                <Badge
                  variant={animationFormat === "gif" ? "default" : "secondary"}
                  className="cursor-pointer text-xs"
                  onClick={() => setAnimationFormat("gif")}
                >
                  GIF
                </Badge>
              </div>
            </div>

            <div className="space-y-1">
              <span className="text-xs text-muted-foreground">Resolution</span>
              <div className="flex gap-1">
                {ANIMATION_RESOLUTION_OPTIONS.map((opt) => (
                  <Badge
                    key={opt.value}
                    variant={animationResolution === opt.value ? "default" : "secondary"}
                    className="cursor-pointer text-xs"
                    onClick={() => setAnimationResolution(opt.value)}
                  >
                    {opt.label}
                  </Badge>
                ))}
              </div>
            </div>

            <div className="space-y-1">
              <span className="text-xs text-muted-foreground">FPS</span>
              <div className="flex gap-1">
                {FPS_OPTIONS.map((fps) => (
                  <Badge
                    key={fps}
                    variant={animationFps === fps ? "default" : "secondary"}
                    className="cursor-pointer text-xs"
                    onClick={() => setAnimationFps(fps)}
                  >
                    {fps}
                  </Badge>
                ))}
              </div>
            </div>

            <div className="space-y-1">
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Frame Range</span>
                <span>
                  {frameRangeStart} - {frameRangeEnd}
                </span>
              </div>
              <Slider
                value={frameRangeValue}
                min={0}
                max={maxFrame}
                step={1}
                onValueChange={([start, end]) => {
                  setFrameRangeStart(start)
                  setFrameRangeEnd(end)
                }}
                disabled={totalFrames === 0}
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>0</span>
                <span>{frameRangeEnd - frameRangeStart + 1} frames</span>
                <span>{maxFrame}</span>
              </div>
            </div>

            <Button
              variant="outline"
              size="sm"
              className="w-full gap-2"
              onClick={handleAnimationExport}
              disabled={totalFrames === 0}
            >
              <Film className="h-4 w-4" />
              Export {animationFormat === "png-sequence" ? "ZIP" : "GIF"}
            </Button>
          </div>
        )}
      </div>

      {/* Slice Animation Section - Only show when in slice mode */}
      {isSliceMode && (
        <div className="border border-border rounded-md overflow-hidden">
          <button
            className="w-full flex items-center justify-between p-2 hover:bg-accent/50 transition-colors"
            onClick={() => toggleSection("slice-animation")}
          >
            <span className="flex items-center gap-2 text-sm">
              <Layers className="h-4 w-4" />
              Slice Animation
            </span>
            {expandedSection === "slice-animation" ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            )}
          </button>

          {expandedSection === "slice-animation" && (
            <div className="p-2 pt-0 space-y-3">
              <div className="space-y-1">
                <span className="text-xs text-muted-foreground">Animation Type</span>
                <div className="flex gap-1">
                  <Badge
                    variant={sliceAnimationType === "time" ? "default" : "secondary"}
                    className="cursor-pointer text-xs"
                    onClick={() => setSliceAnimationType("time")}
                  >
                    Time Animation
                  </Badge>
                  <Badge
                    variant={sliceAnimationType === "position" ? "default" : "secondary"}
                    className="cursor-pointer text-xs"
                    onClick={() => setSliceAnimationType("position")}
                  >
                    Position Sweep
                  </Badge>
                </div>
                <p className="text-[10px] text-muted-foreground">
                  {sliceAnimationType === "time"
                    ? "Animate through time at fixed slice position"
                    : `Sweep through ${sliceAxis.toUpperCase()}-axis positions at current frame`}
                </p>
              </div>

              <div className="space-y-1">
                <span className="text-xs text-muted-foreground">Format</span>
                <div className="flex gap-1">
                  <Badge
                    variant={sliceAnimationFormat === "png-sequence" ? "default" : "secondary"}
                    className="cursor-pointer text-xs"
                    onClick={() => setSliceAnimationFormat("png-sequence")}
                  >
                    PNG Sequence (ZIP)
                  </Badge>
                  <Badge
                    variant={sliceAnimationFormat === "gif" ? "default" : "secondary"}
                    className="cursor-pointer text-xs"
                    onClick={() => setSliceAnimationFormat("gif")}
                  >
                    GIF
                  </Badge>
                </div>
              </div>

              <div className="space-y-1">
                <span className="text-xs text-muted-foreground">FPS</span>
                <div className="flex gap-1">
                  {FPS_OPTIONS.map((fps) => (
                    <Badge
                      key={fps}
                      variant={sliceAnimationFps === fps ? "default" : "secondary"}
                      className="cursor-pointer text-xs"
                      onClick={() => setSliceAnimationFps(fps)}
                    >
                      {fps}
                    </Badge>
                  ))}
                </div>
              </div>

              {sliceAnimationType === "time" ? (
                <div className="space-y-1">
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Frame Range</span>
                    <span>
                      {sliceFrameRangeStart} - {sliceFrameRangeEnd}
                    </span>
                  </div>
                  <Slider
                    value={sliceFrameRangeValue}
                    min={0}
                    max={maxFrame}
                    step={1}
                    onValueChange={([start, end]) => {
                      setSliceFrameRangeStart(start)
                      setSliceFrameRangeEnd(end)
                    }}
                    disabled={totalFrames === 0}
                  />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>0</span>
                    <span>{sliceFrameRangeEnd - sliceFrameRangeStart + 1} frames</span>
                    <span>{maxFrame}</span>
                  </div>
                </div>
              ) : (
                <div className="space-y-1">
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Position Frames</span>
                    <span>{slicePositionFrames} frames</span>
                  </div>
                  <Slider
                    value={[slicePositionFrames]}
                    min={10}
                    max={Math.min(200, getAxisSize())}
                    step={1}
                    onValueChange={([value]) => setSlicePositionFrames(value)}
                  />
                  <p className="text-[10px] text-muted-foreground">
                    Sweeps through all {getAxisSize()} slices along {sliceAxis.toUpperCase()}-axis
                  </p>
                </div>
              )}

              <Button
                variant="outline"
                size="sm"
                className="w-full gap-2"
                onClick={handleSliceAnimationExport}
                disabled={sliceAnimationType === "time" ? totalFrames === 0 : !setSlicePosition}
              >
                <Layers className="h-4 w-4" />
                Export {sliceAnimationFormat === "png-sequence" ? "ZIP" : "GIF"}
              </Button>
            </div>
          )}
        </div>
      )}

      {/* Data Section */}
      <div className="border border-border rounded-md overflow-hidden">
        <button
          className="w-full flex items-center justify-between p-2 hover:bg-accent/50 transition-colors"
          onClick={() => toggleSection("data")}
        >
          <span className="flex items-center gap-2 text-sm">
            <FileSpreadsheet className="h-4 w-4" />
            Data
          </span>
          {expandedSection === "data" ? (
            <ChevronDown className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
        </button>

        {expandedSection === "data" && (
          <div className="p-2 pt-0 space-y-2">
            <Button
              variant="outline"
              size="sm"
              className="w-full justify-start gap-2"
              onClick={handleExportPressure}
              disabled={!pressure}
            >
              <FileText className="h-4 w-4" />
              Current Frame (CSV)
            </Button>

            <Button
              variant="outline"
              size="sm"
              className="w-full justify-start gap-2"
              onClick={handleExportProbe}
              disabled={!probeData}
            >
              <FileSpreadsheet className="h-4 w-4" />
              Probe Data (CSV)
            </Button>

            <Button
              variant="outline"
              size="sm"
              className="w-full justify-start gap-2"
              onClick={handleExportViewState}
            >
              <FileJson className="h-4 w-4" />
              View State (JSON)
            </Button>
          </div>
        )}
      </div>
    </div>
  )
}
