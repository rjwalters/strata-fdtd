// Shared UI Library for Strata FDTD
// This package contains platform-agnostic components, hooks, stores, and utilities

// =============================================================================
// UI Components
// =============================================================================
export { Button, buttonVariants } from './components/ui/button'
export { Badge, badgeVariants } from './components/ui/badge'
export { Slider } from './components/ui/slider'
export {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectScrollDownButton,
  SelectScrollUpButton,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
} from './components/ui/select'
export {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from './components/ui/tooltip'
export {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogOverlay,
  DialogPortal,
  DialogTitle,
  DialogTrigger,
} from './components/ui/dialog'
export {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from './components/ui/card'

// =============================================================================
// Visualization Components
// =============================================================================
export { BackgroundLoadingIndicator } from './components/visualization/BackgroundLoadingIndicator'
export { ExportPanel } from './components/visualization/ExportPanel'
export { FileUpload } from './components/visualization/FileUpload'
export { FlowParticleRenderer } from './components/visualization/FlowParticleRenderer'
// GeometryOverlay exports utility functions, not a component
export * from './components/visualization/GeometryOverlay'
export { Layout, Panel } from './components/visualization/Layout'
export { OptimizedVoxelRenderer } from './components/visualization/OptimizedVoxelRenderer'
export type { VoxelRendererHandle, OptimizedVoxelRendererProps, ProbeMarkerData, SourceMarkerData } from './components/visualization/OptimizedVoxelRenderer'
export { PerformanceMetrics } from './components/visualization/PerformanceMetrics'
export { PlaybackControls } from './components/visualization/PlaybackControls'
export { ProbesPanel } from './components/visualization/ProbesPanel'
export { SourcesPanel } from './components/visualization/SourcesPanel'
export { ViewerHelpModal } from './components/visualization/ViewerHelpModal'
export { BuilderHelpModal } from './components/visualization/BuilderHelpModal'
export { Spectrogram } from './components/visualization/Spectrogram'
export { SpectrumPlot } from './components/visualization/SpectrumPlot'
export type { SpectrumMode } from './components/visualization/SpectrumPlot'
export { ThreeViewer } from './components/visualization/ThreeViewer'
export { TimeSeriesPlot } from './components/visualization/TimeSeriesPlot'
export type { SourceTimeSeries } from './components/visualization/TimeSeriesPlot'
// VideoExportButton moved to strata-web (requires useVideoExport/FFmpeg)
export { VisualizationModePanel } from './components/visualization/VisualizationModePanel'
export { VoxelRenderer } from './components/visualization/VoxelRenderer'
export { WaterfallPlot } from './components/visualization/WaterfallPlot'
export { SliceRenderer } from './components/visualization/SliceRenderer'
export type { SliceRendererHandle, SliceRendererProps } from './components/visualization/SliceRenderer'
export { SliceControlPanel } from './components/visualization/SliceControlPanel'
export type { SliceControlPanelProps } from './components/visualization/SliceControlPanel'
export { ViewModePanel } from './components/visualization/ViewModePanel'
export type { ViewModePanelProps } from './components/visualization/ViewModePanel'

// =============================================================================
// Hooks
// =============================================================================
export { useDataWorker } from './hooks/useDataWorker'
export { useExport, type ExportState, type ExportActions } from './hooks/useExport'
export { useProbeData as useProbeDataLoader } from './hooks/useProbeData'
export { useSimulation } from './hooks/useSimulation'
export { useStreamingBuffer } from './hooks/useStreamingBuffer'
export { useUrlState } from './hooks/useUrlState'
export { useSliceKeyboardNavigation, type SliceKeyboardCallbacks, type UseSliceKeyboardNavigationOptions } from './hooks/useSliceKeyboardNavigation'
// useVideoExport moved to strata-web (requires @ffmpeg)

// =============================================================================
// Stores
// =============================================================================
export {
  useSimulationStore,
  useCurrentPressure,
  usePlaybackState,
  useViewOptions,
  useGridInfo,
  useLoadingState,
  useProbeData,
  useProbeVisibility,
  useSourceData,
  usePerformanceSettings,
  useBackgroundLoadingState,
} from './stores/simulationStore'
export type {
  SimulationState,
  VoxelGeometry,
  GeometryMode,
  DownsampleMethod,
  VisualizationMode,
  FlowParticleConfig,
  SourceData,
  ViewMode,
  SliceAxis,
} from './stores/simulationStore'

// =============================================================================
// Library Utilities
// =============================================================================
// Colormap
export * from './lib/colormap'

// Downsampling
export {
  lttbDownsample,
  minMaxDownsample,
  decimateDownsample,
  logBinDownsample,
  lttbDownsampleAsync,
  minMaxDownsampleAsync,
  logBinDownsampleAsync,
  terminateDownsampleWorker,
  hasWorkerSupport as downsampleHasWorkerSupport,
  WORKER_THRESHOLD as DOWNSAMPLE_WORKER_THRESHOLD,
} from './lib/downsample'

// FFT
export {
  realFFT,
  computeSpectrum,
  computeSpectrumAsync,
  nextPowerOf2,
  terminateFFTWorker,
  hasWorkerSupport as fftHasWorkerSupport,
} from './lib/fft'

// Export utilities
export * from './lib/export'

// Performance
export * from './lib/performance'

// Slice utilities
export * from './lib/sliceUtils'

// Demo geometry
export * from './lib/demoGeometry'

// Script parsing
export * from './lib/scriptParser'

// Python validation
export * from './lib/pythonValidator'

// Templates
export * from './lib/templates'

// Estimations
export * from './lib/estimations'

// Utils
export * from './lib/utils'

// Loaders
export * from './lib/loaders'
