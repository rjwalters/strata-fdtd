/**
 * Hook for managing export state and actions.
 */

import { useState, useCallback, useRef } from "react"
import {
  captureScreenshot,
  exportPngSequence,
  exportGif,
  exportPressureCsv,
  exportProbeCsv,
  exportViewState,
  downloadBlob,
  generateTimestamp,
  type ScreenshotOptions,
  type AnimationExportOptions,
  type ViewState,
  type ExportProgress,
} from "strata-ui"

export type ExportType =
  | "screenshot"
  | "png-sequence"
  | "gif"
  | "pressure-csv"
  | "probe-csv"
  | "view-json"

export interface ExportState {
  /** Whether an export is in progress */
  isExporting: boolean
  /** Current export type */
  exportType: ExportType | null
  /** Export progress (0-100) */
  progress: number
  /** Current stage description */
  stage: string
  /** Error message if any */
  error: string | null
}

export interface ExportActions {
  /** Export screenshot */
  exportScreenshot: (
    canvas: HTMLCanvasElement,
    options?: Partial<ScreenshotOptions>
  ) => Promise<void>

  /** Export PNG sequence */
  exportPngSequence: (
    canvas: HTMLCanvasElement,
    totalFrames: number,
    renderFrame: (frameIndex: number) => Promise<void>,
    options?: Partial<Omit<AnimationExportOptions, "format">>
  ) => Promise<void>

  /** Export GIF */
  exportGif: (
    canvas: HTMLCanvasElement,
    totalFrames: number,
    renderFrame: (frameIndex: number) => Promise<void>,
    options?: Partial<Omit<AnimationExportOptions, "format">>
  ) => Promise<void>

  /** Export pressure data as CSV */
  exportPressureData: (
    pressure: Float32Array,
    shape: [number, number, number],
    resolution: number,
    frameNumber?: number
  ) => void

  /** Export probe data as CSV */
  exportProbeData: (
    probes: Record<string, { data: Float32Array; position?: [number, number, number] }>,
    sampleRate: number
  ) => void

  /** Export view state as JSON */
  exportViewStateJson: (state: ViewState) => void

  /** Cancel ongoing export */
  cancelExport: () => void

  /** Reset error state */
  resetError: () => void
}

const DEFAULT_STATE: ExportState = {
  isExporting: false,
  exportType: null,
  progress: 0,
  stage: "",
  error: null,
}

export function useExport(): [ExportState, ExportActions] {
  const [state, setState] = useState<ExportState>(DEFAULT_STATE)
  const cancelledRef = useRef(false)

  const setProgress = useCallback((progress: ExportProgress) => {
    if (cancelledRef.current) return
    const percent = Math.round((progress.current / progress.total) * 100)
    setState((s) => ({
      ...s,
      progress: percent,
      stage: progress.stage,
    }))
  }, [])

  const handleExportScreenshot = useCallback(
    async (
      canvas: HTMLCanvasElement,
      options: Partial<ScreenshotOptions> = {}
    ) => {
      const fullOptions: ScreenshotOptions = {
        resolution: options.resolution ?? 1,
        includeUI: options.includeUI ?? false,
        filename: options.filename,
      }

      setState({
        isExporting: true,
        exportType: "screenshot",
        progress: 0,
        stage: "capturing",
        error: null,
      })

      try {
        const blob = await captureScreenshot(canvas, fullOptions)
        const timestamp = generateTimestamp()
        const filename =
          fullOptions.filename ?? `screenshot-${fullOptions.resolution}x-${timestamp}.png`
        downloadBlob(blob, filename)

        setState(DEFAULT_STATE)
      } catch (err) {
        setState({
          isExporting: false,
          exportType: null,
          progress: 0,
          stage: "",
          error: err instanceof Error ? err.message : "Screenshot failed",
        })
      }
    },
    []
  )

  const handleExportPngSequence = useCallback(
    async (
      canvas: HTMLCanvasElement,
      totalFrames: number,
      renderFrame: (frameIndex: number) => Promise<void>,
      options: Partial<Omit<AnimationExportOptions, "format">> = {}
    ) => {
      cancelledRef.current = false

      const fullOptions = {
        frameRange: options.frameRange ?? ([0, totalFrames - 1] as [number, number]),
        fps: options.fps ?? 15,
        resolution: options.resolution ?? 1,
        filename: options.filename,
      }

      setState({
        isExporting: true,
        exportType: "png-sequence",
        progress: 0,
        stage: "preparing",
        error: null,
      })

      try {
        const blob = await exportPngSequence(
          canvas,
          totalFrames,
          renderFrame,
          fullOptions,
          setProgress
        )

        if (cancelledRef.current) {
          setState(DEFAULT_STATE)
          return
        }

        const timestamp = generateTimestamp()
        const filename = fullOptions.filename ?? `frames-${timestamp}.zip`
        downloadBlob(blob, filename)

        setState(DEFAULT_STATE)
      } catch (err) {
        if (!cancelledRef.current) {
          setState({
            isExporting: false,
            exportType: null,
            progress: 0,
            stage: "",
            error: err instanceof Error ? err.message : "PNG sequence export failed",
          })
        }
      }
    },
    [setProgress]
  )

  const handleExportGif = useCallback(
    async (
      canvas: HTMLCanvasElement,
      totalFrames: number,
      renderFrame: (frameIndex: number) => Promise<void>,
      options: Partial<Omit<AnimationExportOptions, "format">> = {}
    ) => {
      cancelledRef.current = false

      const fullOptions = {
        frameRange: options.frameRange ?? ([0, totalFrames - 1] as [number, number]),
        fps: options.fps ?? 10,
        resolution: options.resolution ?? 1,
        filename: options.filename,
      }

      setState({
        isExporting: true,
        exportType: "gif",
        progress: 0,
        stage: "preparing",
        error: null,
      })

      try {
        const blob = await exportGif(
          canvas,
          totalFrames,
          renderFrame,
          fullOptions,
          setProgress
        )

        if (cancelledRef.current) {
          setState(DEFAULT_STATE)
          return
        }

        const timestamp = generateTimestamp()
        const filename = fullOptions.filename ?? `animation-${timestamp}.gif`
        downloadBlob(blob, filename)

        setState(DEFAULT_STATE)
      } catch (err) {
        if (!cancelledRef.current) {
          setState({
            isExporting: false,
            exportType: null,
            progress: 0,
            stage: "",
            error: err instanceof Error ? err.message : "GIF export failed",
          })
        }
      }
    },
    [setProgress]
  )

  const handleExportPressureData = useCallback(
    (
      pressure: Float32Array,
      shape: [number, number, number],
      resolution: number,
      frameNumber?: number
    ) => {
      try {
        const blob = exportPressureCsv(pressure, shape, resolution)
        const timestamp = generateTimestamp()
        const frameSuffix = frameNumber !== undefined ? `-frame${frameNumber}` : ""
        const filename = `pressure${frameSuffix}-${timestamp}.csv`
        downloadBlob(blob, filename)
      } catch (err) {
        setState((s) => ({
          ...s,
          error: err instanceof Error ? err.message : "Pressure export failed",
        }))
      }
    },
    []
  )

  const handleExportProbeData = useCallback(
    (
      probes: Record<string, { data: Float32Array; position?: [number, number, number] }>,
      sampleRate: number
    ) => {
      try {
        const blob = exportProbeCsv(probes, sampleRate)
        const timestamp = generateTimestamp()
        const filename = `probe-data-${timestamp}.csv`
        downloadBlob(blob, filename)
      } catch (err) {
        setState((s) => ({
          ...s,
          error: err instanceof Error ? err.message : "Probe data export failed",
        }))
      }
    },
    []
  )

  const handleExportViewState = useCallback((viewState: ViewState) => {
    try {
      const blob = exportViewState(viewState)
      const timestamp = generateTimestamp()
      const filename = `view-state-${timestamp}.json`
      downloadBlob(blob, filename)
    } catch (err) {
      setState((s) => ({
        ...s,
        error: err instanceof Error ? err.message : "View state export failed",
      }))
    }
  }, [])

  const handleCancelExport = useCallback(() => {
    cancelledRef.current = true
    setState(DEFAULT_STATE)
  }, [])

  const handleResetError = useCallback(() => {
    setState((s) => ({ ...s, error: null }))
  }, [])

  return [
    state,
    {
      exportScreenshot: handleExportScreenshot,
      exportPngSequence: handleExportPngSequence,
      exportGif: handleExportGif,
      exportPressureData: handleExportPressureData,
      exportProbeData: handleExportProbeData,
      exportViewStateJson: handleExportViewState,
      cancelExport: handleCancelExport,
      resetError: handleResetError,
    },
  ]
}
