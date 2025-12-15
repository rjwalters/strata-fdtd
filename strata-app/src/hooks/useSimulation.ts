/**
 * Convenience hook for accessing simulation state and actions.
 *
 * Provides a tuple interface [state, actions] for components that prefer
 * this pattern over direct store access.
 */

import { useSimulationStore, usePlaybackState, useLoadingState, useGridInfo } from "strata-ui";
import { useCallback, useMemo } from "react";

/**
 * Simulation state tuple interface
 */
export interface SimulationStateTuple {
  // Data
  manifest: ReturnType<typeof useSimulationStore.getState>["manifest"];
  probeData: ReturnType<typeof useSimulationStore.getState>["probeData"];
  pressure: Float32Array | null;

  // Grid info
  shape: [number, number, number];
  resolution: number;

  // Playback
  totalFrames: number;
  currentFrame: number;
  isPlaying: boolean;

  // Loading
  loading: boolean;
  error: string | null;
}

/**
 * Simulation actions interface
 */
export interface SimulationActions {
  loadSimulation: (path: string) => Promise<void>;
  setFrame: (frame: number) => void;
  setPlaying: (playing: boolean) => void;
}

/**
 * Hook that returns [state, actions] tuple for simulation management.
 *
 * @example
 * ```tsx
 * function MyComponent() {
 *   const [state, actions] = useSimulation();
 *
 *   return (
 *     <button onClick={() => actions.loadSimulation('/data')}>
 *       {state.loading ? 'Loading...' : 'Load'}
 *     </button>
 *   );
 * }
 * ```
 */
export function useSimulation(): [SimulationStateTuple, SimulationActions] {
  // Subscribe to relevant store slices
  const { currentFrame, isPlaying, totalFrames } = usePlaybackState();
  const { isLoading, error } = useLoadingState();
  const { shape, resolution } = useGridInfo();
  const manifest = useSimulationStore((s) => s.manifest);
  const probeData = useSimulationStore((s) => s.probeData);
  const snapshots = useSimulationStore((s) => s.snapshots);

  // Get actions from store
  const loadSimulationAction = useSimulationStore((s) => s.loadSimulation);
  const setCurrentFrame = useSimulationStore((s) => s.setCurrentFrame);
  const play = useSimulationStore((s) => s.play);
  const pause = useSimulationStore((s) => s.pause);

  // Get current pressure data
  const pressure = useMemo(() => {
    return snapshots.get(currentFrame) ?? null;
  }, [snapshots, currentFrame]);

  // Build state object
  const state = useMemo<SimulationStateTuple>(
    () => ({
      manifest,
      probeData,
      pressure,
      shape,
      resolution,
      totalFrames,
      currentFrame,
      isPlaying,
      loading: isLoading,
      error,
    }),
    [manifest, probeData, pressure, shape, resolution, totalFrames, currentFrame, isPlaying, isLoading, error]
  );

  // Build actions object
  const setFrame = useCallback(
    (frame: number) => {
      setCurrentFrame(frame);
    },
    [setCurrentFrame]
  );

  const setPlaying = useCallback(
    (playing: boolean) => {
      if (playing) {
        play();
      } else {
        pause();
      }
    },
    [play, pause]
  );

  const actions = useMemo<SimulationActions>(
    () => ({
      loadSimulation: loadSimulationAction,
      setFrame,
      setPlaying,
    }),
    [loadSimulationAction, setFrame, setPlaying]
  );

  return [state, actions];
}

export default useSimulation;
