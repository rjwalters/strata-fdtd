import { useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Video, Download, X, Loader2 } from "lucide-react"
import type { VideoExportState, VideoExportActions, VideoExportOptions } from "@/hooks/useVideoExport"

export interface VideoExportButtonProps {
  /** Export state from useVideoExport hook */
  exportState: VideoExportState
  /** Export actions from useVideoExport hook */
  exportActions: VideoExportActions
  /** Canvas to capture from */
  getCanvas: () => HTMLCanvasElement | null
  /** Total frames to export */
  totalFrames: number
  /** Function to render a specific frame */
  renderFrame: (frameIndex: number) => Promise<void>
  /** Whether export is available */
  disabled?: boolean
  /** Export options */
  options?: VideoExportOptions
}

export function VideoExportButton({
  exportState,
  exportActions,
  getCanvas,
  totalFrames,
  renderFrame,
  disabled = false,
  options,
}: VideoExportButtonProps) {
  const { isExporting, progress, stage, error, videoUrl } = exportState
  const { startExport, cancelExport, reset, downloadVideo } = exportActions

  const handleExport = useCallback(async () => {
    const canvas = getCanvas()
    if (!canvas) {
      console.error("Canvas not available")
      return
    }
    await startExport(canvas, totalFrames, renderFrame, options)
  }, [getCanvas, totalFrames, renderFrame, startExport, options])

  const handleDownload = useCallback(() => {
    const filename = options?.filename ?? "simulation.mp4"
    downloadVideo(filename)
  }, [downloadVideo, options?.filename])

  // Export complete - show download button
  if (stage === "complete" && videoUrl) {
    return (
      <div className="flex items-center gap-2">
        <Button
          variant="default"
          size="sm"
          className="gap-2"
          onClick={handleDownload}
        >
          <Download className="h-4 w-4" />
          Download MP4
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={reset}
          title="Dismiss"
        >
          <X className="h-4 w-4" />
        </Button>
      </div>
    )
  }

  // Error state
  if (stage === "error") {
    return (
      <div className="flex items-center gap-2">
        <span className="text-xs text-destructive">{error}</span>
        <Button
          variant="ghost"
          size="sm"
          onClick={reset}
        >
          Dismiss
        </Button>
      </div>
    )
  }

  // Exporting in progress
  if (isExporting) {
    return (
      <div className="flex items-center gap-2">
        <div className="flex items-center gap-2 text-sm">
          <Loader2 className="h-4 w-4 animate-spin" />
          <span className="text-muted-foreground">
            {stage === "capturing" ? "Capturing" : "Encoding"}... {progress}%
          </span>
        </div>
        <div className="w-24 h-2 bg-secondary rounded-full overflow-hidden">
          <div
            className="h-full bg-primary transition-all duration-200"
            style={{ width: `${progress}%` }}
          />
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={cancelExport}
          title="Cancel"
        >
          <X className="h-4 w-4" />
        </Button>
      </div>
    )
  }

  // Idle state - show export button
  return (
    <Button
      variant="outline"
      size="sm"
      className="gap-2"
      onClick={handleExport}
      disabled={disabled || totalFrames === 0}
      title="Export simulation as MP4 video"
    >
      <Video className="h-4 w-4" />
      Export Video
    </Button>
  )
}
