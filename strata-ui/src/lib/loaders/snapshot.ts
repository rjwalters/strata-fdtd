/**
 * Snapshot loader for FDTD pressure field exports.
 *
 * Supports loading binary snapshots in various formats:
 * - float16: Half-precision (requires manual unpacking)
 * - float32: Single-precision (native TypedArray)
 * - uint8: Normalized 0-255 (requires denormalization)
 */

import type {
  Snapshot,
  SnapshotInfo,
  VelocitySnapshot,
  VelocitySnapshotInfo,
  LoadOptions,
} from './types';

/**
 * Load a single pressure field snapshot from binary file.
 *
 * @param url - URL to binary snapshot file
 * @param info - Snapshot metadata (from manifest)
 * @param options - Loading options
 * @returns Snapshot with Float32Array pressure data
 *
 * @example
 * ```typescript
 * const manifest = await loadManifest('/data/sim_manifest.json');
 * const snapshot = await loadSnapshot(
 *   manifest.snapshots[0].file,
 *   manifest.snapshots[0]
 * );
 * // Use snapshot.pressure with Three.js DataTexture3D
 * ```
 */
export async function loadSnapshot(
  url: string,
  info: SnapshotInfo,
  options: LoadOptions = {}
): Promise<Snapshot> {
  const response = await fetch(url, { signal: options.signal });

  if (!response.ok) {
    throw new Error(`Failed to load snapshot: ${response.status} ${response.statusText}`);
  }

  // Track progress if callback provided
  const contentLength = response.headers.get('content-length');
  const total = contentLength ? parseInt(contentLength, 10) : info.bytes;

  let buffer: ArrayBuffer;

  if (options.onProgress && response.body) {
    buffer = await readWithProgress(response.body, total, options.onProgress);
  } else {
    buffer = await response.arrayBuffer();
  }

  // Convert to Float32Array based on source format
  const pressure = convertToFloat32(buffer, info);

  return {
    time: info.time,
    pressure,
    shape: info.shape,
  };
}

/**
 * Load multiple snapshots with shared progress tracking.
 *
 * @param snapshots - Array of [url, info] pairs
 * @param options - Loading options (onProgress reports overall progress)
 * @returns Array of loaded snapshots
 */
export async function loadSnapshots(
  snapshots: Array<[string, SnapshotInfo]>,
  options: LoadOptions = {}
): Promise<Snapshot[]> {
  const totalBytes = snapshots.reduce((sum, [, info]) => sum + info.bytes, 0);
  let loadedBytes = 0;

  const results: Snapshot[] = [];

  for (const [url, info] of snapshots) {
    const snapshot = await loadSnapshot(url, info, {
      signal: options.signal,
      onProgress: (loaded) => {
        if (options.onProgress) {
          options.onProgress(loadedBytes + loaded, totalBytes);
        }
      },
    });

    loadedBytes += info.bytes;
    results.push(snapshot);
  }

  return results;
}

/**
 * Convert raw binary data to Float32Array.
 */
function convertToFloat32(buffer: ArrayBuffer, info: SnapshotInfo): Float32Array {
  const [nx, ny, nz] = info.shape;
  const expectedElements = nx * ny * nz;

  switch (info.dtype) {
    case 'float32': {
      const data = new Float32Array(buffer);
      if (data.length !== expectedElements) {
        throw new Error(
          `Snapshot size mismatch: expected ${expectedElements}, got ${data.length}`
        );
      }
      return data;
    }

    case 'float16': {
      // Manual float16 unpacking
      const view = new DataView(buffer);
      const result = new Float32Array(expectedElements);

      for (let i = 0; i < expectedElements; i++) {
        result[i] = float16ToFloat32(view.getUint16(i * 2, true));
      }

      return result;
    }

    case 'uint8': {
      const data = new Uint8Array(buffer);
      if (data.length !== expectedElements) {
        throw new Error(
          `Snapshot size mismatch: expected ${expectedElements}, got ${data.length}`
        );
      }

      const result = new Float32Array(expectedElements);
      const [min, max] = info.valueRange || [0, 1];
      const range = max - min;

      for (let i = 0; i < expectedElements; i++) {
        result[i] = (data[i] / 255) * range + min;
      }

      return result;
    }

    default:
      throw new Error(`Unsupported snapshot format: ${info.dtype}`);
  }
}

/**
 * Convert IEEE 754 float16 to float32.
 *
 * Float16 format:
 * - 1 bit sign
 * - 5 bits exponent (bias 15)
 * - 10 bits mantissa
 */
function float16ToFloat32(h: number): number {
  const sign = (h & 0x8000) >> 15;
  const exponent = (h & 0x7c00) >> 10;
  const mantissa = h & 0x03ff;

  if (exponent === 0) {
    if (mantissa === 0) {
      // Zero (positive or negative)
      return sign ? -0 : 0;
    }
    // Subnormal number
    return (sign ? -1 : 1) * Math.pow(2, -14) * (mantissa / 1024);
  }

  if (exponent === 31) {
    if (mantissa === 0) {
      // Infinity
      return sign ? -Infinity : Infinity;
    }
    // NaN
    return NaN;
  }

  // Normal number
  return (sign ? -1 : 1) * Math.pow(2, exponent - 15) * (1 + mantissa / 1024);
}

/**
 * Read response body with progress tracking.
 */
async function readWithProgress(
  body: ReadableStream<Uint8Array>,
  total: number,
  onProgress: (loaded: number, total: number) => void
): Promise<ArrayBuffer> {
  const reader = body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;

  while (true) {
    const { done, value } = await reader.read();

    if (done) break;

    chunks.push(value);
    loaded += value.length;
    onProgress(loaded, total);
  }

  // Combine chunks into single ArrayBuffer
  const result = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }

  return result.buffer;
}

/**
 * Load a single velocity field snapshot from binary file.
 *
 * Velocity data is stored as interleaved [vx,vy,vz,vx,vy,vz,...] in the file.
 * This is ready for GPU upload as a 3-component texture or for particle advection.
 *
 * @param url - URL to binary velocity snapshot file
 * @param info - Velocity snapshot metadata (from manifest)
 * @param options - Loading options
 * @returns VelocitySnapshot with Float32Array velocity data
 *
 * @example
 * ```typescript
 * const manifest = await loadManifest('/data/sim_manifest.json');
 * if (manifest.velocitySnapshots) {
 *   const velocity = await loadVelocitySnapshot(
 *     manifest.velocitySnapshots[0].file,
 *     manifest.velocitySnapshots[0]
 *   );
 *   // Use velocity.velocity for particle advection
 * }
 * ```
 */
export async function loadVelocitySnapshot(
  url: string,
  info: VelocitySnapshotInfo,
  options: LoadOptions = {}
): Promise<VelocitySnapshot> {
  if (info.format !== 'interleaved') {
    throw new Error(
      `Separate velocity files not yet supported. Got format: ${info.format}`
    );
  }

  const response = await fetch(url, { signal: options.signal });

  if (!response.ok) {
    throw new Error(
      `Failed to load velocity snapshot: ${response.status} ${response.statusText}`
    );
  }

  // Track progress if callback provided
  const contentLength = response.headers.get('content-length');
  const total = contentLength ? parseInt(contentLength, 10) : info.bytes;

  let buffer: ArrayBuffer;

  if (options.onProgress && response.body) {
    buffer = await readWithProgress(response.body, total, options.onProgress);
  } else {
    buffer = await response.arrayBuffer();
  }

  // Convert to Float32Array based on source format
  const velocity = convertVelocityToFloat32(buffer, info);

  return {
    time: info.time,
    velocity,
    shape: info.shape,
  };
}

/**
 * Convert raw velocity binary data to Float32Array.
 * Velocity is interleaved as [vx,vy,vz,vx,vy,vz,...].
 */
function convertVelocityToFloat32(
  buffer: ArrayBuffer,
  info: VelocitySnapshotInfo
): Float32Array {
  const [nx, ny, nz] = info.shape;
  const expectedElements = nx * ny * nz * 3; // 3 components per cell

  switch (info.dtype) {
    case 'float32': {
      const data = new Float32Array(buffer);
      if (data.length !== expectedElements) {
        throw new Error(
          `Velocity size mismatch: expected ${expectedElements}, got ${data.length}`
        );
      }
      return data;
    }

    case 'float16': {
      // Manual float16 unpacking
      const view = new DataView(buffer);
      const result = new Float32Array(expectedElements);

      for (let i = 0; i < expectedElements; i++) {
        result[i] = float16ToFloat32(view.getUint16(i * 2, true));
      }

      return result;
    }

    default:
      throw new Error(`Unsupported velocity format: ${info.dtype}`);
  }
}
