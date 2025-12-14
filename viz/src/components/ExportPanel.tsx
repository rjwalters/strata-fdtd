/**
 * Export panel component with screenshot, animation, and data export options.
 */

import { useState, useCallback } from "react"
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
} from "lucide-react"
import { type ExportState, type ExportActions } from "@/hooks/useExport"
import type { ViewState } from "@/lib/export"

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

  // Update frame range when totalFrames changes
  const maxFrame = Math.max(0, totalFrames - 1)
  if (frameRangeEnd > maxFrame) {
    setFrameRangeEnd(maxFrame)
  }

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
                value={[frameRangeStart, frameRangeEnd]}
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
