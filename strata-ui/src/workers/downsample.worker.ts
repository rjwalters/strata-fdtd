/**
 * Web Worker for downsampling operations.
 *
 * Offloads LTTB and min/max downsampling to a background thread
 * to keep the main thread responsive during large dataset processing.
 */

// =============================================================================
// Message Types
// =============================================================================

export interface LTTBDownsampleRequest {
  type: "lttb";
  id: number;
  data: Float32Array;
  targetPoints: number;
  sampleRate: number;
}

export interface MinMaxDownsampleRequest {
  type: "minmax";
  id: number;
  data: Float32Array;
  targetPoints: number;
  sampleRate: number;
}

export interface LogBinDownsampleRequest {
  type: "logbin";
  id: number;
  frequencies: Float32Array;
  magnitude: Float32Array;
  targetBins: number;
  minFreq: number;
  maxFreq: number;
}

export type DownsampleWorkerRequest =
  | LTTBDownsampleRequest
  | MinMaxDownsampleRequest
  | LogBinDownsampleRequest;

export interface LTTBDownsampleResponse {
  type: "lttb";
  id: number;
  // Interleaved [x0, y0, x1, y1, ...] for efficient transfer
  points: Float32Array;
  length: number;
}

export interface MinMaxDownsampleResponse {
  type: "minmax";
  id: number;
  points: Float32Array;
  length: number;
}

export interface LogBinDownsampleResponse {
  type: "logbin";
  id: number;
  frequencies: Float32Array;
  magnitude: Float32Array;
}

export interface DownsampleWorkerError {
  type: "error";
  id: number;
  error: string;
}

export type DownsampleWorkerMessage =
  | LTTBDownsampleResponse
  | MinMaxDownsampleResponse
  | LogBinDownsampleResponse
  | DownsampleWorkerError;

// =============================================================================
// Downsampling Algorithms (duplicated to avoid module import issues in worker)
// =============================================================================

/**
 * LTTB downsampling - preserves visual fidelity
 */
function lttbDownsampleWorker(
  data: Float32Array,
  targetPoints: number,
  sampleRate: number
): Float32Array {
  const dataLength = data.length;

  // Convert to time-value pairs
  const msPerSample = 1000 / sampleRate;

  // If we have fewer points than target, return all
  if (targetPoints >= dataLength || targetPoints < 3) {
    const result = new Float32Array(dataLength * 2);
    for (let i = 0; i < dataLength; i++) {
      result[i * 2] = i * msPerSample;
      result[i * 2 + 1] = data[i];
    }
    return result;
  }

  const sampled: number[] = [];
  const bucketSize = (dataLength - 2) / (targetPoints - 2);

  // Always include first point
  sampled.push(0 * msPerSample, data[0]);

  let a = 0; // Previously selected point index

  for (let i = 0; i < targetPoints - 2; i++) {
    const bucketStart = Math.floor((i + 1) * bucketSize) + 1;
    const bucketEnd = Math.floor((i + 2) * bucketSize) + 1;
    const actualBucketEnd = Math.min(bucketEnd, dataLength - 1);

    // Calculate average point in next bucket
    const nextBucketStart = Math.floor((i + 2) * bucketSize) + 1;
    const nextBucketEnd = Math.floor((i + 3) * bucketSize) + 1;
    const actualNextBucketEnd = Math.min(nextBucketEnd, dataLength);

    let avgX = 0;
    let avgY = 0;
    let nextBucketCount = 0;

    for (let j = nextBucketStart; j < actualNextBucketEnd; j++) {
      avgX += j * msPerSample;
      avgY += data[j];
      nextBucketCount++;
    }

    if (nextBucketCount > 0) {
      avgX /= nextBucketCount;
      avgY /= nextBucketCount;
    } else {
      avgX = (dataLength - 1) * msPerSample;
      avgY = data[dataLength - 1];
    }

    // Point A
    const pointAX = a * msPerSample;
    const pointAY = data[a];

    // Find point with largest triangle area
    let maxArea = -1;
    let maxAreaIndex = bucketStart;

    for (let j = bucketStart; j < actualBucketEnd; j++) {
      const pointBX = j * msPerSample;
      const pointBY = data[j];

      const area = Math.abs(
        (pointAX - avgX) * (pointBY - pointAY) -
          (pointAX - pointBX) * (avgY - pointAY)
      );

      if (area > maxArea) {
        maxArea = area;
        maxAreaIndex = j;
      }
    }

    sampled.push(maxAreaIndex * msPerSample, data[maxAreaIndex]);
    a = maxAreaIndex;
  }

  // Always include last point
  sampled.push((dataLength - 1) * msPerSample, data[dataLength - 1]);

  return new Float32Array(sampled);
}

/**
 * Min-max downsampling - preserves peaks
 */
function minMaxDownsampleWorker(
  data: Float32Array,
  targetPoints: number,
  sampleRate: number
): Float32Array {
  const dataLength = data.length;
  const msPerSample = 1000 / sampleRate;

  if (targetPoints >= dataLength || targetPoints < 2) {
    const result = new Float32Array(dataLength * 2);
    for (let i = 0; i < dataLength; i++) {
      result[i * 2] = i * msPerSample;
      result[i * 2 + 1] = data[i];
    }
    return result;
  }

  const bucketSize = dataLength / targetPoints;
  const result: number[] = [];

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
      result.push(minIdx * msPerSample, min);
      if (minIdx !== maxIdx) {
        result.push(maxIdx * msPerSample, max);
      }
    } else {
      result.push(maxIdx * msPerSample, max);
      result.push(minIdx * msPerSample, min);
    }
  }

  return new Float32Array(result);
}

/**
 * Log-bin downsampling for spectrum data
 */
function logBinDownsampleWorker(
  frequencies: Float32Array,
  magnitude: Float32Array,
  targetBins: number,
  minFreq: number,
  maxFreq: number
): { frequencies: Float32Array; magnitude: Float32Array } {
  const logMin = Math.log10(Math.max(minFreq, 1));
  const logMax = Math.log10(maxFreq);
  const logStep = (logMax - logMin) / targetBins;

  const outFreq = new Float32Array(targetBins);
  const outMag = new Float32Array(targetBins);

  for (let i = 0; i < targetBins; i++) {
    const binLowFreq = Math.pow(10, logMin + i * logStep);
    const binHighFreq = Math.pow(10, logMin + (i + 1) * logStep);
    const binCenterFreq = Math.sqrt(binLowFreq * binHighFreq);

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
    outMag[i] = found ? maxMag : 0;
  }

  return { frequencies: outFreq, magnitude: outMag };
}

// =============================================================================
// Worker Message Handler
// =============================================================================

self.onmessage = (event: MessageEvent<DownsampleWorkerRequest>) => {
  const request = event.data;

  try {
    switch (request.type) {
      case "lttb": {
        const points = lttbDownsampleWorker(
          request.data,
          request.targetPoints,
          request.sampleRate
        );

        const response: LTTBDownsampleResponse = {
          type: "lttb",
          id: request.id,
          points,
          length: points.length / 2,
        };

        self.postMessage(response, { transfer: [points.buffer] });
        break;
      }

      case "minmax": {
        const points = minMaxDownsampleWorker(
          request.data,
          request.targetPoints,
          request.sampleRate
        );

        const response: MinMaxDownsampleResponse = {
          type: "minmax",
          id: request.id,
          points,
          length: points.length / 2,
        };

        self.postMessage(response, { transfer: [points.buffer] });
        break;
      }

      case "logbin": {
        const result = logBinDownsampleWorker(
          request.frequencies,
          request.magnitude,
          request.targetBins,
          request.minFreq,
          request.maxFreq
        );

        const response: LogBinDownsampleResponse = {
          type: "logbin",
          id: request.id,
          frequencies: result.frequencies,
          magnitude: result.magnitude,
        };

        self.postMessage(response, {
          transfer: [result.frequencies.buffer, result.magnitude.buffer],
        });
        break;
      }
    }
  } catch (error) {
    const errorResponse: DownsampleWorkerError = {
      type: "error",
      id: request.id,
      error: error instanceof Error ? error.message : "Unknown error",
    };
    self.postMessage(errorResponse);
  }
};
