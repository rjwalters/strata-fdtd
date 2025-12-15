/**
 * Hook for loading and managing HDF5 simulation data.
 *
 * Integrates with the existing simulation store to provide
 * seamless visualization of HDF5 files.
 */

import { useState, useCallback, useRef } from "react";
import * as h5wasm from "h5wasm";
import {
  loadHDF5File,
  loadHDF5FromURL,
  toSimulationMetadata,
  type HDF5SimulationData,
} from "../lib/loaders/hdf5";
import { useSimulationStore } from "../stores/simulationStore";

export interface UseHDF5SimulationResult {
  /** Whether HDF5 data is loaded */
  isLoaded: boolean;
  /** Whether currently loading */
  isLoading: boolean;
  /** Loading progress (0-1) */
  progress: number;
  /** Error message if loading failed */
  error: string | null;
  /** The loaded HDF5 data */
  hdf5Data: HDF5SimulationData | null;
  /** Load from a File object */
  loadFile: (file: File) => Promise<void>;
  /** Load from a URL */
  loadURL: (url: string) => Promise<void>;
  /** Load a specific timestep */
  loadTimestep: (step: number) => Promise<Float32Array | null>;
  /** Reset/unload the current data */
  reset: () => void;
}

// Module-level state for h5wasm FS
let h5wasmFS: typeof h5wasm.FS | null = null;

async function ensureH5wasmReady(): Promise<void> {
  if (!h5wasmFS) {
    await h5wasm.ready;
    h5wasmFS = h5wasm.FS;
  }
}

/**
 * Hook for loading HDF5 simulation files and integrating with the visualization store.
 */
export function useHDF5Simulation(): UseHDF5SimulationResult {
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [hdf5Data, setHDF5Data] = useState<HDF5SimulationData | null>(null);

  // Keep the raw buffer for timestep loading
  const fileBufferRef = useRef<Uint8Array | null>(null);
  const filenameRef = useRef<string>("simulation.h5");
  const h5fileRef = useRef<h5wasm.File | null>(null);

  // Store actions
  const storeReset = useSimulationStore((s) => s.reset);

  const handleProgress = useCallback((loaded: number, total: number) => {
    setProgress(total > 0 ? loaded / total : 0);
  }, []);

  const initializeStore = useCallback(
    (data: HDF5SimulationData) => {
      // Get store actions
      const store = useSimulationStore.getState();

      // Create a synthetic manifest for compatibility
      const manifest = {
        metadata: "hdf5://metadata",
        probes: "hdf5://probes",
        geometry: "hdf5://geometry",
        snapshots: Array.from({ length: data.numSteps }, (_, i) => ({
          time: i * data.timestep,
          timestep: i,
          shape: data.grid.shape,
          dtype: "float32" as const,
          format: "hdf5",
          downsample: 1,
          file: `hdf5://snapshot/${i}`,
          bytes: 0,
        })),
        basePath: "hdf5://",
      };

      // Convert to SimulationMetadata
      const metadata = toSimulationMetadata(data);

      // Set initial state
      store.manifest = manifest;
      store.metadata = metadata;
      store.shape = data.grid.shape;
      store.resolution = data.grid.resolution;
      store.totalFrames = data.numSteps;
      store.probeData = data.probes;
      store.geometry = data.geometry;
      store.selectedProbes = Object.keys(data.probes.probes);
      store.currentFrame = 0;
      store.isLoading = false;
      store.error = null;
    },
    []
  );

  const loadFile = useCallback(
    async (file: File) => {
      setIsLoading(true);
      setProgress(0);
      setError(null);

      try {
        const data = await loadHDF5File(file, {
          onProgress: handleProgress,
        });

        // Store the file buffer for later timestep loading
        const buffer = await file.arrayBuffer();
        fileBufferRef.current = new Uint8Array(buffer);
        filenameRef.current = file.name;

        // Write to virtual FS and keep file open for timestep loading
        await ensureH5wasmReady();
        if (h5wasmFS) {
          const virtualPath = `/${file.name}`;
          h5wasmFS.writeFile(virtualPath, fileBufferRef.current);
          h5fileRef.current = new h5wasm.File(virtualPath, "r");
        }

        setHDF5Data(data);
        initializeStore(data);
        setProgress(1);
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Failed to load HDF5 file";
        setError(message);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    [handleProgress, initializeStore]
  );

  const loadURL = useCallback(
    async (url: string) => {
      setIsLoading(true);
      setProgress(0);
      setError(null);

      try {
        const data = await loadHDF5FromURL(url, {
          onProgress: handleProgress,
        });

        // For URL loading, we need to fetch the file again for timestep access
        const response = await fetch(url);
        const buffer = await response.arrayBuffer();
        fileBufferRef.current = new Uint8Array(buffer);
        const filename = url.split("/").pop() || "simulation.h5";
        filenameRef.current = filename;

        // Write to virtual FS and keep file open
        await ensureH5wasmReady();
        if (h5wasmFS) {
          const virtualPath = `/${filename}`;
          h5wasmFS.writeFile(virtualPath, fileBufferRef.current);
          h5fileRef.current = new h5wasm.File(virtualPath, "r");
        }

        setHDF5Data(data);
        initializeStore(data);
        setProgress(1);
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Failed to load HDF5 from URL";
        setError(message);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    [handleProgress, initializeStore]
  );

  const loadTimestep = useCallback(
    async (step: number): Promise<Float32Array | null> => {
      if (!h5fileRef.current || !hdf5Data) {
        return null;
      }

      try {
        const pressureEntity = h5fileRef.current.get("/fields/pressure");
        if (!pressureEntity || !(pressureEntity instanceof h5wasm.Dataset)) {
          console.warn("Pressure dataset not found");
          return null;
        }

        const dataset = pressureEntity;
        const datasetShape = dataset.shape;

        if (!datasetShape || step < 0 || step >= datasetShape[0]) {
          console.warn(`Invalid timestep ${step}`);
          return null;
        }

        // Read single timestep slice
        const slice = dataset.slice([[step, step + 1]]);

        let result: Float32Array;
        if (slice instanceof Float32Array) {
          result = slice;
        } else if (slice && typeof slice === "object" && "length" in slice) {
          result = new Float32Array(slice as ArrayLike<number>);
        } else {
          console.warn("Unexpected data type from pressure dataset");
          return null;
        }

        // Update the store's snapshot cache
        const store = useSimulationStore.getState();
        const newSnapshots = new Map(store.snapshots);
        newSnapshots.set(step, result);
        useSimulationStore.setState({ snapshots: newSnapshots });
        return result;
      } catch (err) {
        console.error(`Failed to load timestep ${step}:`, err);
        return null;
      }
    },
    [hdf5Data]
  );

  const reset = useCallback(() => {
    // Close the h5wasm file if open
    if (h5fileRef.current) {
      try {
        h5fileRef.current.close();
      } catch {
        // Ignore close errors
      }
      h5fileRef.current = null;
    }

    // Clean up virtual file
    if (h5wasmFS && filenameRef.current) {
      try {
        h5wasmFS.unlink(`/${filenameRef.current}`);
      } catch {
        // Ignore cleanup errors
      }
    }

    fileBufferRef.current = null;
    setHDF5Data(null);
    setError(null);
    setProgress(0);
    storeReset();
  }, [storeReset]);

  return {
    isLoaded: hdf5Data !== null,
    isLoading,
    progress,
    error,
    hdf5Data,
    loadFile,
    loadURL,
    loadTimestep,
    reset,
  };
}
