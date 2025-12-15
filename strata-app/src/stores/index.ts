// Re-export simulation store from @strata/ui
export {
  useSimulationStore,
  useCurrentPressure,
  usePlaybackState,
  useViewOptions,
  useGridInfo,
  useLoadingState,
  useProbeData,
  usePerformanceSettings,
  useBackgroundLoadingState,
  type SimulationState,
  type VoxelGeometry,
  type GeometryMode,
  type DownsampleMethod,
  type VisualizationMode,
  type FlowParticleConfig,
} from "@strata/ui";

// Re-export builder store (local to strata-app)
export {
  useBuilderStore,
  type BuilderState,
  type MeasurementPoint,
  type AnimationSpeed,
} from "./builderStore";
