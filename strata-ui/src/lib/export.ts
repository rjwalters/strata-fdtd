/**
 * Export utilities for screenshot, animation, and data export.
 */

import JSZip from "jszip"
import GIF from "gif.js"

// ============================================================================
// Types
// ============================================================================

export interface ScreenshotOptions {
  /** Resolution multiplier (1, 2, or 4) */
  resolution: 1 | 2 | 4
  /** Include UI overlays */
  includeUI?: boolean
  /** Optional overlay configuration */
  overlay?: {
    timestamp?: boolean
    frameNumber?: boolean
    colorbar?: boolean
  }
  /** Custom filename */
  filename?: string
}

export interface AnimationExportOptions {
  /** Export format */
  format: "png-sequence" | "gif" | "webm"
  /** Frame range [start, end] inclusive */
  frameRange: [number, number]
  /** Frames per second */
  fps: number
  /** Resolution multiplier (1 or 2) */
  resolution: 1 | 2
  /** Custom filename */
  filename?: string
}

export interface DataExportOptions {
  /** Data format */
  format: "pressure-csv" | "probe-csv" | "view-json"
  /** Custom filename */
  filename?: string
}

export interface ViewState {
  /** Camera position */
  cameraPosition: [number, number, number]
  /** Camera target/lookAt point */
  cameraTarget: [number, number, number]
  /** Camera zoom/fov */
  cameraFov: number
  /** View options */
  viewOptions: {
    threshold: number
    displayFill: number
    voxelGeometry: string
    geometryMode: string
    showGrid: boolean
    showAxes: boolean
  }
  /** Simulation info */
  simulation?: {
    currentFrame: number
    totalFrames: number
    shape: [number, number, number]
    resolution: number
  }
  /** Timestamp */
  timestamp: string
}

export interface ExportProgress {
  current: number
  total: number
  stage: string
}

// ============================================================================
// Screenshot Export
// ============================================================================

/**
 * Capture a screenshot from a canvas at the specified resolution.
 */
export async function captureScreenshot(
  canvas: HTMLCanvasElement,
  options: ScreenshotOptions = { resolution: 1 }
): Promise<Blob> {
  const { resolution } = options

  // Get original dimensions
  const originalWidth = canvas.width
  const originalHeight = canvas.height

  // Calculate target dimensions
  const targetWidth = originalWidth * resolution
  const targetHeight = originalHeight * resolution

  // Create offscreen canvas at target resolution
  const offscreen = document.createElement("canvas")
  offscreen.width = targetWidth
  offscreen.height = targetHeight
  const ctx = offscreen.getContext("2d")

  if (!ctx) {
    throw new Error("Failed to get canvas context")
  }

  // Enable image smoothing for better quality upscaling
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = "high"

  // Draw the source canvas scaled up
  ctx.drawImage(canvas, 0, 0, targetWidth, targetHeight)

  // Convert to blob
  return new Promise<Blob>((resolve, reject) => {
    offscreen.toBlob(
      (blob) => {
        if (blob) {
          resolve(blob)
        } else {
          reject(new Error("Failed to create screenshot blob"))
        }
      },
      "image/png",
      1.0
    )
  })
}

/**
 * Download a screenshot from a canvas.
 */
export async function downloadScreenshot(
  canvas: HTMLCanvasElement,
  options: ScreenshotOptions = { resolution: 1 }
): Promise<void> {
  const blob = await captureScreenshot(canvas, options)
  const filename = options.filename ?? `screenshot-${options.resolution}x.png`
  downloadBlob(blob, filename)
}

// ============================================================================
// Animation Export
// ============================================================================

/**
 * Export animation as PNG sequence in a ZIP file.
 */
export async function exportPngSequence(
  canvas: HTMLCanvasElement,
  _frameCount: number,
  renderFrame: (frameIndex: number) => Promise<void>,
  options: Omit<AnimationExportOptions, "format">,
  onProgress?: (progress: ExportProgress) => void
): Promise<Blob> {
  const { frameRange, resolution } = options
  const [startFrame, endFrame] = frameRange
  const totalFrames = endFrame - startFrame + 1

  const zip = new JSZip()
  const folder = zip.folder("frames")

  if (!folder) {
    throw new Error("Failed to create ZIP folder")
  }

  // Calculate target dimensions
  const targetWidth = canvas.width * resolution
  const targetHeight = canvas.height * resolution

  // Create offscreen canvas
  const offscreen = document.createElement("canvas")
  offscreen.width = targetWidth
  offscreen.height = targetHeight
  const ctx = offscreen.getContext("2d")

  if (!ctx) {
    throw new Error("Failed to get canvas context")
  }

  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = "high"

  // Capture each frame
  for (let i = startFrame; i <= endFrame; i++) {
    // Render the frame
    await renderFrame(i)

    // Wait for render to complete
    await new Promise((resolve) => requestAnimationFrame(resolve))

    // Draw to offscreen canvas
    ctx.drawImage(canvas, 0, 0, targetWidth, targetHeight)

    // Get frame as PNG blob
    const blob = await new Promise<Blob>((resolve, reject) => {
      offscreen.toBlob(
        (b) => (b ? resolve(b) : reject(new Error("Failed to capture frame"))),
        "image/png"
      )
    })

    // Add to ZIP
    const frameNumber = String(i - startFrame).padStart(5, "0")
    folder.file(`frame_${frameNumber}.png`, blob)

    // Report progress
    onProgress?.({
      current: i - startFrame + 1,
      total: totalFrames,
      stage: "capturing",
    })
  }

  // Generate ZIP
  onProgress?.({ current: totalFrames, total: totalFrames, stage: "compressing" })

  return await zip.generateAsync({ type: "blob" })
}

/**
 * Export animation as GIF.
 */
export async function exportGif(
  canvas: HTMLCanvasElement,
  _frameCount: number,
  renderFrame: (frameIndex: number) => Promise<void>,
  options: Omit<AnimationExportOptions, "format">,
  onProgress?: (progress: ExportProgress) => void
): Promise<Blob> {
  const { frameRange, fps, resolution } = options
  const [startFrame, endFrame] = frameRange
  const totalFrames = endFrame - startFrame + 1
  const frameDelay = Math.round(1000 / fps)

  // Calculate target dimensions
  const targetWidth = canvas.width * resolution
  const targetHeight = canvas.height * resolution

  // Create offscreen canvas for scaling
  const offscreen = document.createElement("canvas")
  offscreen.width = targetWidth
  offscreen.height = targetHeight
  const ctx = offscreen.getContext("2d")

  if (!ctx) {
    throw new Error("Failed to get canvas context")
  }

  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = "high"

  return new Promise((resolve, reject) => {
    const gif = new GIF({
      workers: 2,
      quality: 10,
      width: targetWidth,
      height: targetHeight,
      workerScript: "/gif.worker.js",
    })

    gif.on("finished", (blob: Blob) => {
      resolve(blob)
    })

    // GIF.js types don't include error event, but it exists at runtime
    ;(gif as unknown as { on: (event: string, cb: (err: Error) => void) => void }).on("error", (err: Error) => {
      reject(err)
    })

    // Capture frames sequentially
    const captureFrames = async () => {
      for (let i = startFrame; i <= endFrame; i++) {
        // Render the frame
        await renderFrame(i)

        // Wait for render to complete
        await new Promise((resolve) => requestAnimationFrame(resolve))

        // Draw to offscreen canvas
        ctx.drawImage(canvas, 0, 0, targetWidth, targetHeight)

        // Add frame to GIF
        gif.addFrame(offscreen, { copy: true, delay: frameDelay })

        // Report progress
        onProgress?.({
          current: i - startFrame + 1,
          total: totalFrames,
          stage: "capturing",
        })
      }

      // Render GIF
      onProgress?.({ current: totalFrames, total: totalFrames, stage: "encoding" })
      gif.render()
    }

    captureFrames().catch(reject)
  })
}

// ============================================================================
// Data Export
// ============================================================================

/**
 * Export pressure data as CSV.
 */
export function exportPressureCsv(
  pressure: Float32Array,
  shape: [number, number, number],
  resolution: number
): Blob {
  const [nx, ny, nz] = shape
  const lines: string[] = []

  // Header
  lines.push("x,y,z,pressure")

  // Data
  const offsetX = ((nx - 1) * resolution) / 2
  const offsetY = ((ny - 1) * resolution) / 2
  const offsetZ = ((nz - 1) * resolution) / 2

  let idx = 0
  for (let x = 0; x < nx; x++) {
    for (let y = 0; y < ny; y++) {
      for (let z = 0; z < nz; z++) {
        const px = x * resolution - offsetX
        const py = y * resolution - offsetY
        const pz = z * resolution - offsetZ
        const value = pressure[idx]
        lines.push(`${px.toFixed(6)},${py.toFixed(6)},${pz.toFixed(6)},${value.toFixed(9)}`)
        idx++
      }
    }
  }

  const content = lines.join("\n")
  return new Blob([content], { type: "text/csv" })
}

/**
 * Export probe data as CSV.
 */
export function exportProbeCsv(
  probes: Record<string, { data: Float32Array; position?: [number, number, number] }>,
  sampleRate: number
): Blob {
  const probeNames = Object.keys(probes)
  if (probeNames.length === 0) {
    throw new Error("No probe data available")
  }

  const sampleCount = probes[probeNames[0]].data.length
  const lines: string[] = []

  // Header
  lines.push(["time", ...probeNames].join(","))

  // Data
  for (let i = 0; i < sampleCount; i++) {
    const time = (i / sampleRate).toFixed(9)
    const values = probeNames.map((name) => probes[name].data[i].toFixed(9))
    lines.push([time, ...values].join(","))
  }

  const content = lines.join("\n")
  return new Blob([content], { type: "text/csv" })
}

/**
 * Export view state as JSON for reproducibility.
 */
export function exportViewState(state: ViewState): Blob {
  const content = JSON.stringify(state, null, 2)
  return new Blob([content], { type: "application/json" })
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Download a blob as a file.
 */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

/**
 * Generate a timestamp string for filenames.
 */
export function generateTimestamp(): string {
  const now = new Date()
  return now.toISOString().replace(/[:.]/g, "-").slice(0, 19)
}
