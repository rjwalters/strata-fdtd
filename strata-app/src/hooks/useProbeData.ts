import { useState, useCallback } from "react";
import { loadProbeData, type ProbeData } from "strata-ui";

export interface ProbeDataState {
  /** Loaded probe data */
  probeData: ProbeData | null;
  /** Loading state */
  loading: boolean;
  /** Error message */
  error: string | null;
  /** Selected probe name for spectrum view */
  selectedProbe: string | null;
}

export interface ProbeDataActions {
  /** Load probe data from URL */
  loadProbes: (url: string) => Promise<void>;
  /** Select a probe for spectrum analysis */
  selectProbe: (name: string | null) => void;
  /** Reset to initial state */
  reset: () => void;
}

const DEFAULT_STATE: ProbeDataState = {
  probeData: null,
  loading: false,
  error: null,
  selectedProbe: null,
};

export function useProbeData(): [ProbeDataState, ProbeDataActions] {
  const [state, setState] = useState<ProbeDataState>(DEFAULT_STATE);

  const loadProbes = useCallback(async (url: string) => {
    setState((s) => ({ ...s, loading: true, error: null }));

    try {
      const data = await loadProbeData(url);

      // Auto-select first probe
      const probeNames = Object.keys(data.probes);
      const selectedProbe = probeNames.length > 0 ? probeNames[0] : null;

      setState({
        probeData: data,
        loading: false,
        error: null,
        selectedProbe,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to load probe data";
      setState((s) => ({
        ...s,
        loading: false,
        error: message,
      }));
    }
  }, []);

  const selectProbe = useCallback((name: string | null) => {
    setState((s) => ({ ...s, selectedProbe: name }));
  }, []);

  const reset = useCallback(() => {
    setState(DEFAULT_STATE);
  }, []);

  return [state, { loadProbes, selectProbe, reset }];
}

/**
 * Calculate current time in milliseconds from frame index.
 */
export function frameToTime(
  frame: number,
  totalFrames: number,
  duration: number
): number {
  if (totalFrames <= 1) return 0;
  return (frame / (totalFrames - 1)) * duration * 1000;
}

/**
 * Calculate frame index from time in milliseconds.
 */
export function timeToFrame(
  time: number,
  totalFrames: number,
  duration: number
): number {
  if (totalFrames <= 1) return 0;
  const frame = Math.round((time / 1000 / duration) * (totalFrames - 1));
  return Math.max(0, Math.min(totalFrames - 1, frame));
}
