/**
 * Downsampling algorithms for efficient visualization of large datasets.
 */

/**
 * Largest Triangle Three Buckets (LTTB) downsampling algorithm.
 * Preserves visual fidelity while reducing point count.
 *
 * Reference: Sveinn Steinarsson - "Downsampling Time Series for Visual Representation"
 * https://skemman.is/bitstream/1946/15343/3/SS_MSthesis.pdf
 *
 * @param data - Array of [x, y] points or typed array with time series data
 * @param targetPoints - Desired number of output points
 * @param sampleRate - If data is Float32Array, sample rate to compute x values
 * @returns Downsampled array of [x, y] points
 */
export function lttbDownsample(
  data: [number, number][] | Float32Array,
  targetPoints: number,
  sampleRate?: number
): [number, number][] {
  // Convert Float32Array to [x, y] pairs if needed
  let points: [number, number][];
  if (data instanceof Float32Array) {
    if (!sampleRate) {
      throw new Error("sampleRate required for Float32Array input");
    }
    points = Array.from(data, (y, i) => [
      (i / sampleRate) * 1000, // time in ms
      y,
    ]);
  } else {
    points = data;
  }

  const dataLength = points.length;

  // If we have fewer points than target, return all
  if (targetPoints >= dataLength || targetPoints < 3) {
    return points;
  }

  const sampled: [number, number][] = [];

  // Bucket size (leave room for start and end points)
  const bucketSize = (dataLength - 2) / (targetPoints - 2);

  // Always include first point
  sampled.push(points[0]);

  let a = 0; // Point A index (previously selected point)

  for (let i = 0; i < targetPoints - 2; i++) {
    // Calculate bucket boundaries
    const bucketStart = Math.floor((i + 1) * bucketSize) + 1;
    const bucketEnd = Math.floor((i + 2) * bucketSize) + 1;
    const actualBucketEnd = Math.min(bucketEnd, dataLength - 1);

    // Calculate average point in next bucket (point C)
    const nextBucketStart = Math.floor((i + 2) * bucketSize) + 1;
    const nextBucketEnd = Math.floor((i + 3) * bucketSize) + 1;
    const actualNextBucketEnd = Math.min(nextBucketEnd, dataLength);

    let avgX = 0;
    let avgY = 0;
    let nextBucketCount = 0;

    for (let j = nextBucketStart; j < actualNextBucketEnd; j++) {
      avgX += points[j][0];
      avgY += points[j][1];
      nextBucketCount++;
    }

    if (nextBucketCount > 0) {
      avgX /= nextBucketCount;
      avgY /= nextBucketCount;
    } else {
      // Edge case: use last point
      avgX = points[dataLength - 1][0];
      avgY = points[dataLength - 1][1];
    }

    // Point A (previously selected)
    const pointAX = points[a][0];
    const pointAY = points[a][1];

    // Find point in current bucket with largest triangle area
    let maxArea = -1;
    let maxAreaIndex = bucketStart;

    for (let j = bucketStart; j < actualBucketEnd; j++) {
      // Calculate triangle area using cross product method
      // Area = 0.5 * |x_A(y_B - y_C) + x_B(y_C - y_A) + x_C(y_A - y_B)|
      const area = Math.abs(
        (pointAX - avgX) * (points[j][1] - pointAY) -
          (pointAX - points[j][0]) * (avgY - pointAY)
      );

      if (area > maxArea) {
        maxArea = area;
        maxAreaIndex = j;
      }
    }

    // Select point with largest triangle area
    sampled.push(points[maxAreaIndex]);
    a = maxAreaIndex;
  }

  // Always include last point
  sampled.push(points[dataLength - 1]);

  return sampled;
}

/**
 * Min-max downsampling that preserves peaks and troughs.
 * For each bucket, includes both min and max values.
 * Better for waveforms where extremes are important.
 *
 * @param data - Float32Array of sample values
 * @param targetPoints - Desired number of output points (will be ~2x this for min/max pairs)
 * @param sampleRate - Sample rate to compute time values
 * @returns Array of [time, value] points with min/max preserved
 */
export function minMaxDownsample(
  data: Float32Array,
  targetPoints: number,
  sampleRate: number
): [number, number][] {
  const dataLength = data.length;

  if (targetPoints >= dataLength || targetPoints < 2) {
    return Array.from(data, (y, i) => [(i / sampleRate) * 1000, y]);
  }

  const bucketSize = dataLength / targetPoints;
  const result: [number, number][] = [];

  for (let i = 0; i < targetPoints; i++) {
    const bucketStart = Math.floor(i * bucketSize);
    const bucketEnd = Math.min(Math.floor((i + 1) * bucketSize), dataLength);

    let min = data[bucketStart];
    let max = data[bucketStart];
    let minIdx = bucketStart;
    let maxIdx = bucketStart;

    for (let j = bucketStart; j < bucketEnd; j++) {
      if (data[j] < min) {
        min = data[j];
        minIdx = j;
      }
      if (data[j] > max) {
        max = data[j];
        maxIdx = j;
      }
    }

    // Add points in temporal order
    if (minIdx <= maxIdx) {
      result.push([(minIdx / sampleRate) * 1000, min]);
      if (minIdx !== maxIdx) {
        result.push([(maxIdx / sampleRate) * 1000, max]);
      }
    } else {
      result.push([(maxIdx / sampleRate) * 1000, max]);
      result.push([(minIdx / sampleRate) * 1000, min]);
    }
  }

  return result;
}

/**
 * Simple decimation downsampling - takes every Nth point.
 * Fast but may miss peaks.
 *
 * @param data - Float32Array of sample values
 * @param targetPoints - Desired number of output points
 * @param sampleRate - Sample rate to compute time values
 * @returns Array of [time, value] points
 */
export function decimateDownsample(
  data: Float32Array,
  targetPoints: number,
  sampleRate: number
): [number, number][] {
  const dataLength = data.length;

  if (targetPoints >= dataLength) {
    return Array.from(data, (y, i) => [(i / sampleRate) * 1000, y]);
  }

  const step = Math.ceil(dataLength / targetPoints);
  const result: [number, number][] = [];

  for (let i = 0; i < dataLength; i += step) {
    result.push([(i / sampleRate) * 1000, data[i]]);
  }

  return result;
}

/**
 * Downsample frequency spectrum data for visualization.
 * Uses logarithmic binning to match log-frequency display.
 *
 * @param frequencies - Array of frequency values
 * @param magnitude - Array of magnitude values
 * @param targetBins - Number of output bins
 * @param minFreq - Minimum frequency to include
 * @param maxFreq - Maximum frequency to include
 * @returns Downsampled { frequencies, magnitude } arrays
 */
export function logBinDownsample(
  frequencies: Float32Array,
  magnitude: Float32Array,
  targetBins: number,
  minFreq: number,
  maxFreq: number
): { frequencies: Float32Array; magnitude: Float32Array } {
  // Create logarithmically spaced bin edges
  const logMin = Math.log10(Math.max(minFreq, 1));
  const logMax = Math.log10(maxFreq);
  const logStep = (logMax - logMin) / targetBins;

  const outFreq = new Float32Array(targetBins);
  const outMag = new Float32Array(targetBins);

  for (let i = 0; i < targetBins; i++) {
    const binLowFreq = Math.pow(10, logMin + i * logStep);
    const binHighFreq = Math.pow(10, logMin + (i + 1) * logStep);
    const binCenterFreq = Math.sqrt(binLowFreq * binHighFreq); // Geometric mean

    // Find max magnitude in source data that falls within this bin
    let maxMag = 0;
    let found = false;

    for (let j = 0; j < frequencies.length; j++) {
      if (frequencies[j] >= binLowFreq && frequencies[j] < binHighFreq) {
        found = true;
        if (magnitude[j] > maxMag) {
          maxMag = magnitude[j];
        }
      }
    }

    outFreq[i] = binCenterFreq;
    // Use max to preserve peaks
    outMag[i] = found ? maxMag : 0;
  }

  return { frequencies: outFreq, magnitude: outMag };
}

// =============================================================================
// Web Worker Support
// =============================================================================

import type {
  DownsampleWorkerRequest,
  DownsampleWorkerMessage,
  LTTBDownsampleResponse,
  MinMaxDownsampleResponse,
  LogBinDownsampleResponse,
} from "../workers/downsample.worker";

/** Threshold for using worker (number of samples) */
export const WORKER_THRESHOLD = 100000;

let downsampleWorker: Worker | null = null;
let requestId = 0;

type PendingLTTB = {
  type: "lttb";
  resolve: (value: [number, number][]) => void;
  reject: (error: Error) => void;
};

type PendingMinMax = {
  type: "minmax";
  resolve: (value: [number, number][]) => void;
  reject: (error: Error) => void;
};

type PendingLogBin = {
  type: "logbin";
  resolve: (value: { frequencies: Float32Array; magnitude: Float32Array }) => void;
  reject: (error: Error) => void;
};

type PendingRequest = PendingLTTB | PendingMinMax | PendingLogBin;

const pendingRequests = new Map<number, PendingRequest>();

/**
 * Check if Web Workers are available.
 */
export function hasWorkerSupport(): boolean {
  return typeof Worker !== "undefined";
}

/**
 * Get or create the downsample worker instance.
 */
function getDownsampleWorker(): Worker | null {
  if (!hasWorkerSupport()) {
    return null;
  }

  if (!downsampleWorker) {
    try {
      downsampleWorker = new Worker(
        new URL("../workers/downsample.worker.ts", import.meta.url),
        { type: "module" }
      );

      downsampleWorker.onmessage = (event: MessageEvent<DownsampleWorkerMessage>) => {
        const message = event.data;
        const pending = pendingRequests.get(message.id);

        if (pending) {
          pendingRequests.delete(message.id);

          if (message.type === "error") {
            pending.reject(new Error(message.error));
          } else if (message.type === "lttb" && pending.type === "lttb") {
            const response = message as LTTBDownsampleResponse;
            // Convert interleaved Float32Array back to [x, y][] array
            const points: [number, number][] = [];
            for (let i = 0; i < response.length; i++) {
              points.push([response.points[i * 2], response.points[i * 2 + 1]]);
            }
            pending.resolve(points);
          } else if (message.type === "minmax" && pending.type === "minmax") {
            const response = message as MinMaxDownsampleResponse;
            const points: [number, number][] = [];
            for (let i = 0; i < response.length; i++) {
              points.push([response.points[i * 2], response.points[i * 2 + 1]]);
            }
            pending.resolve(points);
          } else if (message.type === "logbin" && pending.type === "logbin") {
            const response = message as LogBinDownsampleResponse;
            pending.resolve({
              frequencies: response.frequencies,
              magnitude: response.magnitude,
            });
          }
        }
      };

      downsampleWorker.onerror = (error) => {
        console.error("Downsample Worker error:", error);
        for (const [id, pending] of pendingRequests) {
          pending.reject(new Error("Worker error"));
          pendingRequests.delete(id);
        }
        downsampleWorker = null;
      };
    } catch {
      console.warn("Failed to create downsample worker, falling back to main thread");
      return null;
    }
  }

  return downsampleWorker;
}

/**
 * LTTB downsample using Web Worker.
 * Falls back to synchronous computation for small datasets or when workers unavailable.
 *
 * @param data - Float32Array of sample values
 * @param targetPoints - Desired number of output points
 * @param sampleRate - Sample rate to compute time values
 * @returns Promise resolving to array of [time, value] points
 */
export async function lttbDownsampleAsync(
  data: Float32Array,
  targetPoints: number,
  sampleRate: number
): Promise<[number, number][]> {
  // Use sync for small datasets
  if (data.length < WORKER_THRESHOLD) {
    return lttbDownsample(data, targetPoints, sampleRate);
  }

  const worker = getDownsampleWorker();
  if (!worker) {
    return lttbDownsample(data, targetPoints, sampleRate);
  }

  const id = requestId++;

  return new Promise((resolve, reject) => {
    pendingRequests.set(id, { type: "lttb", resolve, reject });

    const request: DownsampleWorkerRequest = {
      type: "lttb",
      id,
      data,
      targetPoints,
      sampleRate,
    };

    // Transfer for large datasets
    const dataCopy = new Float32Array(data);
    worker.postMessage(
      { ...request, data: dataCopy },
      { transfer: [dataCopy.buffer] }
    );
  });
}

/**
 * Min-max downsample using Web Worker.
 * Falls back to synchronous computation for small datasets or when workers unavailable.
 *
 * @param data - Float32Array of sample values
 * @param targetPoints - Desired number of output points
 * @param sampleRate - Sample rate to compute time values
 * @returns Promise resolving to array of [time, value] points
 */
export async function minMaxDownsampleAsync(
  data: Float32Array,
  targetPoints: number,
  sampleRate: number
): Promise<[number, number][]> {
  if (data.length < WORKER_THRESHOLD) {
    return minMaxDownsample(data, targetPoints, sampleRate);
  }

  const worker = getDownsampleWorker();
  if (!worker) {
    return minMaxDownsample(data, targetPoints, sampleRate);
  }

  const id = requestId++;

  return new Promise((resolve, reject) => {
    pendingRequests.set(id, { type: "minmax", resolve, reject });

    const request: DownsampleWorkerRequest = {
      type: "minmax",
      id,
      data,
      targetPoints,
      sampleRate,
    };

    const dataCopy = new Float32Array(data);
    worker.postMessage(
      { ...request, data: dataCopy },
      { transfer: [dataCopy.buffer] }
    );
  });
}

/**
 * Log-bin downsample using Web Worker.
 * Falls back to synchronous computation for small datasets or when workers unavailable.
 */
export async function logBinDownsampleAsync(
  frequencies: Float32Array,
  magnitude: Float32Array,
  targetBins: number,
  minFreq: number,
  maxFreq: number
): Promise<{ frequencies: Float32Array; magnitude: Float32Array }> {
  if (frequencies.length < WORKER_THRESHOLD) {
    return logBinDownsample(frequencies, magnitude, targetBins, minFreq, maxFreq);
  }

  const worker = getDownsampleWorker();
  if (!worker) {
    return logBinDownsample(frequencies, magnitude, targetBins, minFreq, maxFreq);
  }

  const id = requestId++;

  return new Promise((resolve, reject) => {
    pendingRequests.set(id, { type: "logbin", resolve, reject });

    const request: DownsampleWorkerRequest = {
      type: "logbin",
      id,
      frequencies: new Float32Array(frequencies),
      magnitude: new Float32Array(magnitude),
      targetBins,
      minFreq,
      maxFreq,
    };

    worker.postMessage(request, {
      transfer: [
        (request as { frequencies: Float32Array }).frequencies.buffer,
        (request as { magnitude: Float32Array }).magnitude.buffer,
      ],
    });
  });
}

/**
 * Terminate the downsample worker and clean up resources.
 */
export function terminateDownsampleWorker(): void {
  if (downsampleWorker) {
    downsampleWorker.terminate();
    downsampleWorker = null;
    pendingRequests.clear();
  }
}
