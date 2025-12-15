// UI Components
export * from './components/ui/button';
export * from './components/ui/select';
export * from './components/ui/slider';
export * from './components/ui/badge';

// Layout components
export { Layout, Panel } from './components/Layout';

// Visualization components
export { ThreeViewer } from './components/ThreeViewer';
export { VoxelRenderer, type VoxelRendererHandle as VoxelRendererHandleBase } from './components/VoxelRenderer';
export {
  OptimizedVoxelRenderer,
  type VoxelRendererHandle,
  type OptimizedVoxelRendererProps,
  type VoxelGeometry,
} from './components/OptimizedVoxelRenderer';
export { generateBoundaryMesh, createGeometryMesh, useGeometryMesh, computeGeometryStats, type GeometryMode } from './components/GeometryOverlay';
export { FlowParticleRenderer } from './components/FlowParticleRenderer';

// Chart components
export { TimeSeriesPlot, type TimeSeriesPlotProps } from './components/TimeSeriesPlot';
export { SpectrumPlot, type SpectrumPlotProps } from './components/SpectrumPlot';
export { Spectrogram, type SpectrogramProps } from './components/Spectrogram';
export { WaterfallPlot, type WaterfallPlotProps } from './components/WaterfallPlot';

// Control components
export { PlaybackControls, type PlaybackControlsProps } from './components/PlaybackControls';
export { ExportPanel } from './components/ExportPanel';
export { VisualizationModePanel, type VisualizationModePanelProps } from './components/VisualizationModePanel';
export { PerformanceMetrics, type PerformanceMetricsProps } from './components/PerformanceMetrics';
export { FileUpload, type FileUploadProps } from './components/FileUpload';
export { ScriptEditor } from './components/ScriptEditor';
export { VideoExportButton } from './components/VideoExportButton';
export { BackgroundLoadingIndicator } from './components/BackgroundLoadingIndicator';
export { ExamplesGallery } from './components/ExamplesGallery';

// Builder components
export { Preview3D } from './components/builder/Preview3D';
export { GridBox } from './components/builder/GridBox';
export { SlicePlane } from './components/builder/SlicePlane';
export { SourceMarker } from './components/builder/SourceMarker';
export { ProbeMarker } from './components/builder/ProbeMarker';
export { MaterialRegion } from './components/builder/MaterialRegion';
export { MeasurementLine } from './components/builder/MeasurementLine';
export { EstimationPanel } from './components/builder/EstimationPanel';
export { ErrorPanel } from './components/builder/ErrorPanel';
export { ExportBar } from './components/builder/ExportBar';
export { TemplateBar } from './components/builder/TemplateBar';

// Stores
export * from './stores';

// Hooks
export { useVideoExport } from './hooks/useVideoExport';
export { useExport, type ExportState, type ExportActions } from './hooks/useExport';
export { useHDF5Simulation, type UseHDF5SimulationResult } from './hooks/useHDF5Simulation';
export { useDataWorker } from './hooks/useDataWorker';
export { useStreamingBuffer } from './hooks/useStreamingBuffer';

// Lib utilities
export * from './lib/colormap';
export { lttbDownsample, minMaxDownsample, lttbDownsampleAsync, minMaxDownsampleAsync, WORKER_THRESHOLD } from './lib/downsample';
export * from './lib/fft';
export * from './lib/export';
export { downsamplePressure, PerformanceTracker, filterByThreshold, type PerformanceMetrics as PerformanceMetricsData, type DownsampleOptions, type DownsampleResult, type FilterResult } from './lib/performance';
export { cn } from './lib/utils';
export * from './lib/estimations';
export * from './lib/templates';
export * from './lib/scriptParser';
export * from './lib/pythonValidator';
export * from './lib/demoGeometry';

// Loaders
export * from './lib/loaders';
