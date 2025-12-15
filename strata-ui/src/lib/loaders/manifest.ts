/**
 * Manifest loader for FDTD simulation exports.
 *
 * The manifest file contains paths to all simulation data files:
 * - metadata.json: Solver configuration
 * - probes.json: Probe time series
 * - geometry.bin: Geometry mask
 * - snapshot_NNNN.bin: Pressure field snapshots
 */

import type { SimulationManifest, SimulationMetadata, LoadOptions } from './types';

/**
 * Load simulation manifest from JSON file.
 *
 * @param url - URL to manifest JSON file
 * @param options - Loading options
 * @returns Parsed manifest with file paths
 *
 * @example
 * ```typescript
 * const manifest = await loadManifest('/data/helmholtz_manifest.json');
 * console.log(`Found ${manifest.snapshots.length} snapshots`);
 * ```
 */
export async function loadManifest(
  url: string,
  options: LoadOptions = {}
): Promise<SimulationManifest> {
  const response = await fetch(url, { signal: options.signal });

  if (!response.ok) {
    throw new Error(`Failed to load manifest: ${response.status} ${response.statusText}`);
  }

  const data = await response.json();

  // Validate required fields
  if (!data.metadata || !data.probes || !data.geometry) {
    throw new Error('Invalid manifest: missing required fields (metadata, probes, geometry)');
  }

  // Normalize snapshot info
  const snapshots = (data.snapshots || []).map((snap: Record<string, unknown>) => ({
    time: snap.time as number,
    shape: snap.shape as [number, number, number],
    dtype: snap.dtype as string,
    format: snap.format as string,
    downsample: snap.downsample as number,
    file: snap.file as string,
    bytes: snap.bytes as number,
    valueRange: snap.value_range as [number, number] | undefined,
  }));

  return {
    metadata: data.metadata,
    probes: data.probes,
    geometry: data.geometry,
    snapshots,
  };
}

/**
 * Load simulation metadata from JSON file.
 *
 * @param url - URL to metadata JSON file
 * @param options - Loading options
 * @returns Parsed simulation metadata
 *
 * @example
 * ```typescript
 * const metadata = await loadMetadata('/data/helmholtz_metadata.json');
 * console.log(`Grid size: ${metadata.grid.shape.join('x')}`);
 * console.log(`Speed of sound: ${metadata.physics.c} m/s`);
 * ```
 */
export async function loadMetadata(
  url: string,
  options: LoadOptions = {}
): Promise<SimulationMetadata> {
  const response = await fetch(url, { signal: options.signal });

  if (!response.ok) {
    throw new Error(`Failed to load metadata: ${response.status} ${response.statusText}`);
  }

  const data = await response.json();

  // Convert snake_case keys to camelCase
  return {
    grid: {
      shape: data.grid.shape,
      resolution: data.grid.resolution,
      physicalSize: data.grid.physical_size,
    },
    physics: {
      c: data.physics.c,
      rho: data.physics.rho,
    },
    simulation: {
      dt: data.simulation.dt,
      cflLimit: data.simulation.cfl_limit,
      currentTime: data.simulation.current_time,
      stepCount: data.simulation.step_count,
    },
    probes: data.probes,
    sources: data.sources,
    extra: data.extra,
  };
}
