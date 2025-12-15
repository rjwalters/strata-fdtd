/**
 * Hook for synchronizing simulation state with URL query parameters.
 *
 * Enables shareable links by persisting view state in the URL.
 * - Initializes store from URL params on mount
 * - Updates URL when relevant state changes (debounced during playback)
 * - Handles invalid params gracefully with defaults
 * - Uses history.replaceState to avoid polluting browser history
 */
import { useEffect, useRef, useCallback } from "react";
import { useSearchParams } from "react-router-dom";
import {
  useSimulationStore,
  type ColormapType,
  type VoxelGeometry,
} from "strata-ui";

// Valid colormap values for validation
const VALID_COLORMAPS: ColormapType[] = ["diverging", "magnitude", "viridis"];

// Valid voxel geometry values for validation
const VALID_GEOMETRIES: VoxelGeometry[] = ["point", "mesh", "hidden"];

// Debounce delay for URL updates during playback (ms)
const DEBOUNCE_DELAY = 300;

/**
 * URL parameter configurations with parsing and serialization.
 */
const urlParamConfigs = {
  frame: {
    urlKey: "frame",
    parse: (value: string | null): number | null => {
      if (value === null) return null;
      const parsed = parseInt(value, 10);
      return isNaN(parsed) || parsed < 0 ? null : parsed;
    },
    serialize: (value: number) => value.toString(),
    defaultValue: 0,
  },
  speed: {
    urlKey: "speed",
    parse: (value: string | null): number | null => {
      if (value === null) return null;
      const parsed = parseFloat(value);
      return isNaN(parsed) || parsed < 0.25 || parsed > 4 ? null : parsed;
    },
    serialize: (value: number) => value.toString(),
    defaultValue: 1,
  },
  colormap: {
    urlKey: "cmap",
    parse: (value: string | null): ColormapType | null => {
      if (value === null) return null;
      return VALID_COLORMAPS.includes(value as ColormapType)
        ? (value as ColormapType)
        : null;
    },
    serialize: (value: ColormapType) => value,
    defaultValue: "diverging" as ColormapType,
  },
  threshold: {
    urlKey: "thresh",
    parse: (value: string | null): number | null => {
      if (value === null) return null;
      const parsed = parseFloat(value);
      return isNaN(parsed) || parsed < 0 || parsed > 1 ? null : parsed;
    },
    serialize: (value: number) => value.toFixed(2),
    defaultValue: 0,
  },
  probes: {
    urlKey: "probes",
    parse: (value: string | null): string[] | null => {
      if (value === null || value === "") return null;
      return value.split(",").filter((p) => p.length > 0);
    },
    serialize: (value: string[]) => value.join(","),
    defaultValue: [] as string[],
  },
  geometry: {
    urlKey: "geom",
    parse: (value: string | null): VoxelGeometry | null => {
      if (value === null) return null;
      return VALID_GEOMETRIES.includes(value as VoxelGeometry)
        ? (value as VoxelGeometry)
        : null;
    },
    serialize: (value: VoxelGeometry) => value,
    defaultValue: "point" as VoxelGeometry,
  },
};

/**
 * Hook that syncs simulation state with URL query parameters.
 *
 * Must be used within a Router context (e.g., BrowserRouter).
 */
export function useUrlState() {
  const [searchParams, setSearchParams] = useSearchParams();
  const isInitializedRef = useRef(false);
  const debounceTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Get store state and actions
  const store = useSimulationStore;

  /**
   * Initialize store from URL params on mount
   */
  useEffect(() => {
    if (isInitializedRef.current) return;
    isInitializedRef.current = true;

    const storeState = store.getState();

    // Parse frame
    const frame = urlParamConfigs.frame.parse(
      searchParams.get(urlParamConfigs.frame.urlKey)
    );
    if (frame !== null) {
      storeState.setCurrentFrame(frame);
    }

    // Parse speed
    const speed = urlParamConfigs.speed.parse(
      searchParams.get(urlParamConfigs.speed.urlKey)
    );
    if (speed !== null) {
      storeState.setPlaybackSpeed(speed);
    }

    // Parse colormap
    const colormap = urlParamConfigs.colormap.parse(
      searchParams.get(urlParamConfigs.colormap.urlKey)
    );
    if (colormap !== null) {
      storeState.setColormap(colormap);
    }

    // Parse threshold
    const threshold = urlParamConfigs.threshold.parse(
      searchParams.get(urlParamConfigs.threshold.urlKey)
    );
    if (threshold !== null) {
      storeState.setThreshold(threshold);
    }

    // Parse selected probes
    const probes = urlParamConfigs.probes.parse(
      searchParams.get(urlParamConfigs.probes.urlKey)
    );
    if (probes !== null && probes.length > 0) {
      storeState.setSelectedProbes(probes);
    }

    // Parse voxel geometry
    const geometry = urlParamConfigs.geometry.parse(
      searchParams.get(urlParamConfigs.geometry.urlKey)
    );
    if (geometry !== null) {
      storeState.setVoxelGeometry(geometry);
    }
  }, [searchParams, store]);

  /**
   * Update URL params from store state (debounced)
   */
  const updateUrlFromState = useCallback(() => {
    const state = store.getState();

    const newParams: Record<string, string> = {};

    // Only include non-default values to keep URLs short
    if (state.currentFrame !== urlParamConfigs.frame.defaultValue) {
      newParams[urlParamConfigs.frame.urlKey] = urlParamConfigs.frame.serialize(
        state.currentFrame
      );
    }

    if (state.playbackSpeed !== urlParamConfigs.speed.defaultValue) {
      newParams[urlParamConfigs.speed.urlKey] = urlParamConfigs.speed.serialize(
        state.playbackSpeed
      );
    }

    if (state.colormap !== urlParamConfigs.colormap.defaultValue) {
      newParams[urlParamConfigs.colormap.urlKey] =
        urlParamConfigs.colormap.serialize(state.colormap);
    }

    if (state.threshold !== urlParamConfigs.threshold.defaultValue) {
      newParams[urlParamConfigs.threshold.urlKey] =
        urlParamConfigs.threshold.serialize(state.threshold);
    }

    if (
      state.selectedProbes.length > 0 &&
      JSON.stringify(state.selectedProbes) !==
        JSON.stringify(urlParamConfigs.probes.defaultValue)
    ) {
      newParams[urlParamConfigs.probes.urlKey] =
        urlParamConfigs.probes.serialize(state.selectedProbes);
    }

    if (state.voxelGeometry !== urlParamConfigs.geometry.defaultValue) {
      newParams[urlParamConfigs.geometry.urlKey] =
        urlParamConfigs.geometry.serialize(state.voxelGeometry);
    }

    // Update URL with replace to avoid cluttering history
    setSearchParams(newParams, { replace: true });
  }, [store, setSearchParams]);

  /**
   * Debounced URL update for playback
   */
  const debouncedUpdateUrl = useCallback(() => {
    if (debounceTimeoutRef.current) {
      clearTimeout(debounceTimeoutRef.current);
    }
    debounceTimeoutRef.current = setTimeout(() => {
      updateUrlFromState();
    }, DEBOUNCE_DELAY);
  }, [updateUrlFromState]);

  /**
   * Subscribe to store changes
   */
  useEffect(() => {
    // Subscribe to frame changes (debounced during playback)
    const unsubFrame = store.subscribe(
      (state) => state.currentFrame,
      () => {
        const { isPlaying } = store.getState();
        if (isPlaying) {
          // Debounce during playback
          debouncedUpdateUrl();
        } else {
          // Update immediately when paused
          updateUrlFromState();
        }
      }
    );

    // Subscribe to other state changes (immediate updates)
    const unsubOther = store.subscribe(
      (state) => ({
        speed: state.playbackSpeed,
        colormap: state.colormap,
        threshold: state.threshold,
        probes: state.selectedProbes,
        geometry: state.voxelGeometry,
      }),
      updateUrlFromState,
      {
        equalityFn: (a, b) =>
          a.speed === b.speed &&
          a.colormap === b.colormap &&
          a.threshold === b.threshold &&
          JSON.stringify(a.probes) === JSON.stringify(b.probes) &&
          a.geometry === b.geometry,
      }
    );

    return () => {
      unsubFrame();
      unsubOther();
      if (debounceTimeoutRef.current) {
        clearTimeout(debounceTimeoutRef.current);
      }
    };
  }, [store, updateUrlFromState, debouncedUpdateUrl]);

  /**
   * Generate a shareable URL for current state
   */
  const getShareableUrl = useCallback(() => {
    const state = store.getState();
    const params = new URLSearchParams();

    // Always include frame for shareable links
    params.set(
      urlParamConfigs.frame.urlKey,
      urlParamConfigs.frame.serialize(state.currentFrame)
    );

    if (state.playbackSpeed !== urlParamConfigs.speed.defaultValue) {
      params.set(
        urlParamConfigs.speed.urlKey,
        urlParamConfigs.speed.serialize(state.playbackSpeed)
      );
    }

    if (state.colormap !== urlParamConfigs.colormap.defaultValue) {
      params.set(
        urlParamConfigs.colormap.urlKey,
        urlParamConfigs.colormap.serialize(state.colormap)
      );
    }

    if (state.threshold !== urlParamConfigs.threshold.defaultValue) {
      params.set(
        urlParamConfigs.threshold.urlKey,
        urlParamConfigs.threshold.serialize(state.threshold)
      );
    }

    if (state.selectedProbes.length > 0) {
      params.set(
        urlParamConfigs.probes.urlKey,
        urlParamConfigs.probes.serialize(state.selectedProbes)
      );
    }

    if (state.voxelGeometry !== urlParamConfigs.geometry.defaultValue) {
      params.set(
        urlParamConfigs.geometry.urlKey,
        urlParamConfigs.geometry.serialize(state.voxelGeometry)
      );
    }

    const baseUrl = window.location.origin + window.location.pathname;
    return `${baseUrl}?${params.toString()}`;
  }, [store]);

  return {
    getShareableUrl,
  };
}
