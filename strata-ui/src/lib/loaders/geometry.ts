/**
 * Geometry loader for FDTD simulation boundaries.
 *
 * Supports two formats:
 * - binary: Packed bits (1 bit per cell)
 * - json: Run-length encoded for sparse geometries
 *
 * Output is Uint8Array (1 byte per cell) for GPU texture compatibility.
 */

import type { Geometry, GeometryInfo, RLEGeometry, LoadOptions } from './types';

/**
 * Load geometry mask from binary or JSON file.
 *
 * @param url - URL to geometry file
 * @param info - Geometry metadata (from manifest)
 * @param options - Loading options
 * @returns Geometry with unpacked mask
 *
 * @example
 * ```typescript
 * const geometry = await loadGeometry('/data/sim_geometry.bin', {
 *   shape: [50, 40, 30],
 *   format: 'packed_bits',
 *   file: '/data/sim_geometry.bin',
 *   bytes: 7500,
 * });
 * // geometry.mask[i] is 1 (air) or 0 (solid)
 * ```
 */
export async function loadGeometry(
  url: string,
  info: GeometryInfo,
  options: LoadOptions = {}
): Promise<Geometry> {
  const response = await fetch(url, { signal: options.signal });

  if (!response.ok) {
    throw new Error(`Failed to load geometry: ${response.status} ${response.statusText}`);
  }

  // Track progress for binary format
  if (options.onProgress && info.format === 'packed_bits' && response.body) {
    const total = info.bytes || 0;
    const buffer = await readWithProgress(response.body, total, options.onProgress);
    const mask = unpackBits(new Uint8Array(buffer), info.shape);
    return { mask, shape: info.shape };
  }

  if (info.format === 'packed_bits') {
    const buffer = await response.arrayBuffer();
    const packed = new Uint8Array(buffer);
    const mask = unpackBits(packed, info.shape);
    return { mask, shape: info.shape };
  }

  if (info.format === 'rle_json') {
    const data: RLEGeometry = await response.json();
    const mask = decodeRLE(data);
    return { mask, shape: info.shape };
  }

  throw new Error(`Unsupported geometry format: ${info.format}`);
}

/**
 * Unpack bit-packed geometry to byte array.
 *
 * Input: 1 bit per cell (LSB first in each byte)
 * Output: 1 byte per cell (0 or 1)
 */
function unpackBits(
  packed: Uint8Array,
  shape: [number, number, number]
): Uint8Array {
  const [nx, ny, nz] = shape;
  const totalCells = nx * ny * nz;
  const result = new Uint8Array(totalCells);

  for (let i = 0; i < totalCells; i++) {
    const byteIndex = Math.floor(i / 8);
    const bitIndex = i % 8;
    // numpy.packbits packs MSB first
    result[i] = (packed[byteIndex] >> (7 - bitIndex)) & 1;
  }

  return result;
}

/**
 * Decode run-length encoded geometry.
 *
 * RLE format: Array of {v: boolean, n: count} runs
 */
function decodeRLE(data: RLEGeometry): Uint8Array {
  const [nx, ny, nz] = data.shape;
  const totalCells = nx * ny * nz;
  const result = new Uint8Array(totalCells);

  let offset = 0;
  for (const run of data.runs) {
    const value = run.v ? 1 : 0;
    for (let i = 0; i < run.n && offset < totalCells; i++) {
      result[offset++] = value;
    }
  }

  if (offset !== totalCells) {
    console.warn(
      `RLE geometry size mismatch: expected ${totalCells}, decoded ${offset}`
    );
  }

  return result;
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

  const result = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }

  return result.buffer;
}
