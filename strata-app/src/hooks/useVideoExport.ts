import { useState, useCallback, useRef } from "react"
import { FFmpeg } from "@ffmpeg/ffmpeg"
import { toBlobURL } from "@ffmpeg/util"

export interface VideoExportState {
  /** Whether export is in progress */
  isExporting: boolean
  /** Current export progress (0-100) */
  progress: number
  /** Current stage of export */
  stage: "idle" | "capturing" | "encoding" | "complete" | "error"
  /** Error message if any */
  error: string | null
  /** Resulting video blob URL (available after export) */
  videoUrl: string | null
}

export interface VideoExportOptions {
  /** Frames per second for the video */
  fps?: number
  /** Video width (defaults to canvas width) */
  width?: number
  /** Video height (defaults to canvas height) */
  height?: number
  /** Output filename */
  filename?: string
}

export interface VideoExportActions {
  /** Start exporting video from frame range */
  startExport: (
    canvas: HTMLCanvasElement,
    frameCount: number,
    renderFrame: (frameIndex: number) => Promise<void>,
    options?: VideoExportOptions
  ) => Promise<void>
  /** Cancel ongoing export */
  cancelExport: () => void
  /** Reset state after export */
  reset: () => void
  /** Download the exported video */
  downloadVideo: (filename?: string) => void
}

const DEFAULT_STATE: VideoExportState = {
  isExporting: false,
  progress: 0,
  stage: "idle",
  error: null,
  videoUrl: null,
}

export function useVideoExport(): [VideoExportState, VideoExportActions] {
  const [state, setState] = useState<VideoExportState>(DEFAULT_STATE)
  const ffmpegRef = useRef<FFmpeg | null>(null)
  const cancelledRef = useRef(false)
  const videoUrlRef = useRef<string | null>(null)

  const initFFmpeg = useCallback(async (): Promise<FFmpeg> => {
    if (ffmpegRef.current?.loaded) {
      return ffmpegRef.current
    }

    const ffmpeg = new FFmpeg()

    // Load FFmpeg with CDN-hosted core files
    const baseURL = "https://unpkg.com/@ffmpeg/core@0.12.6/dist/esm"
    await ffmpeg.load({
      coreURL: await toBlobURL(`${baseURL}/ffmpeg-core.js`, "text/javascript"),
      wasmURL: await toBlobURL(`${baseURL}/ffmpeg-core.wasm`, "application/wasm"),
    })

    ffmpegRef.current = ffmpeg
    return ffmpeg
  }, [])

  const startExport = useCallback(async (
    canvas: HTMLCanvasElement,
    frameCount: number,
    renderFrame: (frameIndex: number) => Promise<void>,
    options: VideoExportOptions = {}
  ) => {
    const {
      fps = 15,
      width = canvas.width,
      height = canvas.height,
      filename = "simulation.mp4",
    } = options

    cancelledRef.current = false

    // Clean up previous video URL
    if (videoUrlRef.current) {
      URL.revokeObjectURL(videoUrlRef.current)
      videoUrlRef.current = null
    }

    setState({
      isExporting: true,
      progress: 0,
      stage: "capturing",
      error: null,
      videoUrl: null,
    })

    try {
      // Initialize FFmpeg
      const ffmpeg = await initFFmpeg()

      // Set up progress handler
      ffmpeg.on("progress", ({ progress }) => {
        if (!cancelledRef.current) {
          // Progress is 0-1 during encoding, scale to 50-100 (first 50% is capture)
          setState(s => ({
            ...s,
            progress: 50 + Math.round(progress * 50),
          }))
        }
      })

      // Create offscreen canvas for consistent sizing
      const offscreen = document.createElement("canvas")
      offscreen.width = width
      offscreen.height = height
      const ctx = offscreen.getContext("2d")
      if (!ctx) throw new Error("Failed to get canvas context")

      // Capture frames
      const frames: Uint8Array[] = []
      for (let i = 0; i < frameCount; i++) {
        if (cancelledRef.current) {
          throw new Error("Export cancelled")
        }

        // Render the frame
        await renderFrame(i)

        // Small delay to ensure render completes
        await new Promise(resolve => requestAnimationFrame(resolve))

        // Draw to offscreen canvas (handles scaling)
        ctx.drawImage(canvas, 0, 0, width, height)

        // Get frame as PNG
        const blob = await new Promise<Blob>((resolve, reject) => {
          offscreen.toBlob(
            blob => blob ? resolve(blob) : reject(new Error("Failed to capture frame")),
            "image/png"
          )
        })

        const buffer = await blob.arrayBuffer()
        frames.push(new Uint8Array(buffer))

        // Update progress (0-50% for capture phase)
        setState(s => ({
          ...s,
          progress: Math.round(((i + 1) / frameCount) * 50),
        }))
      }

      if (cancelledRef.current) {
        throw new Error("Export cancelled")
      }

      setState(s => ({ ...s, stage: "encoding" }))

      // Write frames to FFmpeg virtual filesystem
      for (let i = 0; i < frames.length; i++) {
        const paddedIndex = String(i).padStart(5, "0")
        await ffmpeg.writeFile(`frame_${paddedIndex}.png`, frames[i])
      }

      // Encode video with H.264
      // Use yuv420p for compatibility, crf 23 for good quality/size balance
      await ffmpeg.exec([
        "-framerate", String(fps),
        "-i", "frame_%05d.png",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        "-preset", "medium",
        filename,
      ])

      // Read the output file
      const data = await ffmpeg.readFile(filename)
      const videoBlob = new Blob([data as BlobPart], { type: "video/mp4" })
      const videoUrl = URL.createObjectURL(videoBlob)
      videoUrlRef.current = videoUrl

      // Clean up frames from virtual filesystem
      for (let i = 0; i < frames.length; i++) {
        const paddedIndex = String(i).padStart(5, "0")
        await ffmpeg.deleteFile(`frame_${paddedIndex}.png`)
      }
      await ffmpeg.deleteFile(filename)

      setState({
        isExporting: false,
        progress: 100,
        stage: "complete",
        error: null,
        videoUrl,
      })
    } catch (err) {
      const message = err instanceof Error ? err.message : "Export failed"
      setState({
        isExporting: false,
        progress: 0,
        stage: "error",
        error: message,
        videoUrl: null,
      })
    }
  }, [initFFmpeg])

  const cancelExport = useCallback(() => {
    cancelledRef.current = true
    setState(s => ({
      ...s,
      isExporting: false,
      stage: "idle",
      error: "Export cancelled",
    }))
  }, [])

  const reset = useCallback(() => {
    if (videoUrlRef.current) {
      URL.revokeObjectURL(videoUrlRef.current)
      videoUrlRef.current = null
    }
    setState(DEFAULT_STATE)
  }, [])

  const downloadVideo = useCallback((filename = "simulation.mp4") => {
    if (!state.videoUrl) return

    const a = document.createElement("a")
    a.href = state.videoUrl
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }, [state.videoUrl])

  return [
    state,
    { startExport, cancelExport, reset, downloadVideo },
  ]
}
