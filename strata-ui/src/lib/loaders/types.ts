/**
 * TypeScript type definitions for FDTD simulation export formats.
 *
 * These types correspond to the Python export format from metamaterial.io
 */

// =============================================================================
// Core Data Types
// =============================================================================

/**
 * A single pressure field snapshot.
 */
export interface Snapshot {
  /** Simulation time in seconds */
  time: number;
  /** Pressure field data (ready for GPU upload) */
  pressure: Float32Array;
  /** Grid dimensions [nx, ny, nz] */
  shape: [number, number, number];
}

/**
 * A single velocity field snapshot (cell-centered).
 */
export interface VelocitySnapshot {
  /** Simulation time in seconds */
  time: number;
  /** Interleaved velocity data [vx,vy,vz,vx,vy,vz,...] (ready for GPU upload) */
  velocity: Float32Array;
  /** Grid dimensions [nx, ny, nz] */
  shape: [number, number, number];
}

/**
 * Geometry mask for solid/air boundaries.
 */
export interface Geometry {
  /** Boolean mask packed as Uint8Array (1 byte per cell for GPU compat) */
  mask: Uint8Array;
  /** Grid dimensions [nx, ny, nz] */
  shape: [number, number, number];
}

/**
 * Time series data from simulation probes.
 */
export interface ProbeData {
  /** Sample rate in Hz */
  sampleRate: number;
  /** Time step in seconds */
  dt: number;
  /** Total simulation duration in seconds */
  duration: number;
  /** Number of samples recorded */
  nSamples: number;
  /** Probe data indexed by name */
  probes: {
    [name: string]: {
      /** Grid position [i, j, k] */
      position: [number, number, number];
      /** Pressure time series */
      data: Float32Array;
    };
  };
}

// =============================================================================
// Metadata Types
// =============================================================================

/**
 * Information about a single snapshot file.
 */
export interface SnapshotInfo {
  /** Time in seconds */
  time: number;
  /** Timestep index (0-based frame number, optional - may be inferred from array index) */
  timestep?: number;
  /** Grid dimensions after downsampling */
  shape: [number, number, number];
  /** Data type used in file */
  dtype: 'float16' | 'float32' | 'uint8';
  /** Format identifier */
  format: string;
  /** Downsampling factor applied */
  downsample: number;
  /** Path to binary file */
  file: string;
  /** File size in bytes */
  bytes: number;
  /** Value range for uint8 format (for denormalization) */
  valueRange?: [number, number];
}

/**
 * Information about a single velocity snapshot file.
 */
export interface VelocitySnapshotInfo {
  /** Time in seconds */
  time: number;
  /** Grid dimensions after downsampling */
  shape: [number, number, number];
  /** Data type used in file */
  dtype: 'float16' | 'float32';
  /** Format: 'interleaved' or 'separate' */
  format: 'interleaved' | 'separate';
  /** Downsampling factor applied */
  downsample: number;
  /** Number of velocity components (always 3) */
  components: 3;
  /** Path to binary file (for interleaved format) */
  file?: string;
  /** Paths to component files (for separate format) */
  files?: { vx: string; vy: string; vz: string };
  /** File size in bytes */
  bytes: number;
}

/**
 * Information about geometry export.
 */
export interface GeometryInfo {
  /** Grid dimensions */
  shape: [number, number, number];
  /** Format: 'packed_bits' or 'rle_json' */
  format: 'packed_bits' | 'rle_json';
  /** Path to geometry file */
  file: string;
  /** File size in bytes (for binary) */
  bytes?: number;
  /** Number of RLE runs (for json) */
  runs?: number;
}

/**
 * Simulation manifest containing paths to all exported files.
 */
export interface SimulationManifest {
  /** Path to metadata JSON */
  metadata: string;
  /** Path to probe data JSON */
  probes: string;
  /** Path to geometry file */
  geometry: string;
  /** Information about each snapshot */
  snapshots: SnapshotInfo[];
  /** Information about each velocity snapshot (optional) */
  velocitySnapshots?: VelocitySnapshotInfo[];
  /** Base path for resolving relative URLs (set at runtime) */
  basePath?: string;
}

/**
 * Solver configuration metadata.
 */
export interface SimulationMetadata {
  grid: {
    /** Grid dimensions [nx, ny, nz] */
    shape: [number, number, number];
    /** Grid resolution in meters */
    resolution: number;
    /** Physical size in meters [Lx, Ly, Lz] */
    physicalSize: [number, number, number];
  };
  physics: {
    /** Speed of sound in m/s */
    c: number;
    /** Air density in kg/m³ */
    rho: number;
  };
  simulation: {
    /** Time step in seconds */
    dt: number;
    /** CFL stability limit */
    cflLimit: number;
    /** Current simulation time */
    currentTime: number;
    /** Number of steps completed */
    stepCount: number;
  };
  probes: {
    [name: string]: {
      position: [number, number, number];
    };
  };
  sources: Array<{
    type: string;
    position: [number, number, number] | { axis: number; index: number };
    frequency: number;
    bandwidth: number;
  }>;
  extra?: Record<string, unknown>;
}

// =============================================================================
// RLE JSON Format (for geometry)
// =============================================================================

/**
 * Run-length encoded geometry format.
 */
export interface RLEGeometry {
  /** Grid dimensions */
  shape: [number, number, number];
  /** Format identifier */
  format: 'rle';
  /** Run-length encoded data */
  runs: Array<{
    /** Value (true = air, false = solid) */
    v: boolean;
    /** Run length */
    n: number;
  }>;
}

// =============================================================================
// Loading Options
// =============================================================================

/**
 * Options for loading operations.
 */
export interface LoadOptions {
  /** Progress callback (loaded bytes, total bytes) */
  onProgress?: (loaded: number, total: number) => void;
  /** AbortSignal for cancellation */
  signal?: AbortSignal;
}
