/**
 * Web Worker for heavy data processing operations.
 *
 * Handles downsampling and threshold filtering off the main thread
 * to prevent blocking the UI during playback.
 */

import {
  downsamplePressure,
  filterByThreshold,
  type DownsampleOptions,
  type DownsampleResult,
  type FilterResult,
} from "@/lib/performance";

// =============================================================================
// Message Types
// =============================================================================

export type WorkerRequest =
  | {
      type: "downsample";
      id: number;
      data: Float32Array;
      shape: [number, number, number];
      options: Partial<DownsampleOptions>;
    }
  | {
      type: "filter";
      id: number;
      data: Float32Array;
      threshold: number;
      maxPressure: number;
    }
  | {
      type: "downsampleAndFilter";
      id: number;
      data: Float32Array;
      shape: [number, number, number];
      downsampleOptions: Partial<DownsampleOptions>;
      threshold: number;
    };

export type WorkerResponse =
  | {
      type: "downsample";
      id: number;
      result: DownsampleResult;
    }
  | {
      type: "filter";
      id: number;
      result: FilterResult;
    }
  | {
      type: "downsampleAndFilter";
      id: number;
      downsampleResult: DownsampleResult;
      filterResult: FilterResult;
    }
  | {
      type: "error";
      id: number;
      error: string;
    };

// =============================================================================
// Worker Logic
// =============================================================================

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const request = event.data;

  try {
    switch (request.type) {
      case "downsample": {
        const result = downsamplePressure(
          request.data,
          request.shape,
          request.options
        );
        const response: WorkerResponse = {
          type: "downsample",
          id: request.id,
          result,
        };
        self.postMessage(response, { transfer: [result.data.buffer] });
        break;
      }

      case "filter": {
        const result = filterByThreshold(
          request.data,
          request.threshold,
          request.maxPressure
        );
        const response: WorkerResponse = {
          type: "filter",
          id: request.id,
          result,
        };
        self.postMessage(response, {
          transfer: [result.indices.buffer, result.values.buffer],
        });
        break;
      }

      case "downsampleAndFilter": {
        // Combined operation for efficiency
        const downsampleResult = downsamplePressure(
          request.data,
          request.shape,
          request.downsampleOptions
        );

        // Calculate max pressure for threshold
        let maxPressure = 0;
        for (let i = 0; i < downsampleResult.data.length; i++) {
          const abs = Math.abs(downsampleResult.data[i]);
          if (abs > maxPressure) maxPressure = abs;
        }

        const filterResult = filterByThreshold(
          downsampleResult.data,
          request.threshold,
          maxPressure
        );

        const response: WorkerResponse = {
          type: "downsampleAndFilter",
          id: request.id,
          downsampleResult,
          filterResult,
        };

        // Transfer ownership of buffers for zero-copy
        self.postMessage(response, {
          transfer: [
            downsampleResult.data.buffer,
            filterResult.indices.buffer,
            filterResult.values.buffer,
          ],
        });
        break;
      }
    }
  } catch (error) {
    const errorResponse: WorkerResponse = {
      type: "error",
      id: request.id,
      error: error instanceof Error ? error.message : "Unknown error",
    };
    self.postMessage(errorResponse);
  }
};
