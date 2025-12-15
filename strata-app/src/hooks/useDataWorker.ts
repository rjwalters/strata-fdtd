/**
 * Hook for managing Web Worker data processing.
 *
 * Provides async methods for downsampling and filtering that run
 * in a background thread to avoid blocking the main thread.
 */

import { useRef, useEffect, useCallback } from "react";
import type {
  DownsampleOptions,
  DownsampleResult,
  FilterResult,
} from "strata-ui";
import type { WorkerResponse } from "@/workers/dataWorker";

interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
}

export interface UseDataWorkerReturn {
  /** Whether the worker is available */
  isAvailable: boolean;
  /** Downsample pressure data */
  downsample: (
    data: Float32Array,
    shape: [number, number, number],
    options?: Partial<DownsampleOptions>
  ) => Promise<DownsampleResult>;
  /** Filter data by threshold */
  filter: (
    data: Float32Array,
    threshold: number,
    maxPressure: number
  ) => Promise<FilterResult>;
  /** Combined downsample and filter operation */
  downsampleAndFilter: (
    data: Float32Array,
    shape: [number, number, number],
    downsampleOptions: Partial<DownsampleOptions>,
    threshold: number
  ) => Promise<{ downsampleResult: DownsampleResult; filterResult: FilterResult }>;
}

export function useDataWorker(): UseDataWorkerReturn {
  const workerRef = useRef<Worker | null>(null);
  const requestIdRef = useRef(0);
  const pendingRef = useRef<Map<number, PendingRequest>>(new Map());
  const isAvailableRef = useRef(false);

  // Initialize worker
  useEffect(() => {
    try {
      // Create worker using Vite's worker import syntax
      const worker = new Worker(
        new URL("@/workers/dataWorker.ts", import.meta.url),
        { type: "module" }
      );

      worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
        const response = event.data;
        const pending = pendingRef.current.get(response.id);

        if (pending) {
          pendingRef.current.delete(response.id);

          if (response.type === "error") {
            pending.reject(new Error(response.error));
          } else {
            pending.resolve(response);
          }
        }
      };

      worker.onerror = (error) => {
        console.error("Data worker error:", error);
        // Reject all pending requests
        pendingRef.current.forEach((pending) => {
          pending.reject(new Error("Worker error"));
        });
        pendingRef.current.clear();
      };

      workerRef.current = worker;
      isAvailableRef.current = true;
    } catch (error) {
      console.warn("Web Worker not available, falling back to main thread:", error);
      isAvailableRef.current = false;
    }

    return () => {
      workerRef.current?.terminate();
      workerRef.current = null;
      isAvailableRef.current = false;
    };
  }, []);

  // Helper to send messages to worker
  const sendMessage = useCallback(<T,>(message: Record<string, unknown>): Promise<T> => {
    return new Promise((resolve, reject) => {
      if (!workerRef.current) {
        reject(new Error("Worker not available"));
        return;
      }

      const id = requestIdRef.current++;
      pendingRef.current.set(id, {
        resolve: resolve as (value: unknown) => void,
        reject,
      });

      // Clone data to avoid detaching issues
      const messageWithId = { ...message, id };

      // Transfer ArrayBuffer ownership for better performance
      const transfers: ArrayBuffer[] = [];
      if ("data" in messageWithId && messageWithId.data instanceof Float32Array) {
        // Create a copy since we're transferring
        const copy = new Float32Array(messageWithId.data);
        messageWithId.data = copy;
        transfers.push(copy.buffer);
      }

      workerRef.current.postMessage(messageWithId, transfers);
    });
  }, []);

  const downsample = useCallback(
    async (
      data: Float32Array,
      shape: [number, number, number],
      options: Partial<DownsampleOptions> = {}
    ): Promise<DownsampleResult> => {
      if (!isAvailableRef.current) {
        // Fallback to main thread
        const { downsamplePressure } = await import("strata-ui");
        return downsamplePressure(data, shape, options);
      }

      const response = await sendMessage<{ type: "downsample"; result: DownsampleResult }>({
        type: "downsample",
        data,
        shape,
        options,
      });

      return response.result;
    },
    [sendMessage]
  );

  const filter = useCallback(
    async (
      data: Float32Array,
      threshold: number,
      maxPressure: number
    ): Promise<FilterResult> => {
      if (!isAvailableRef.current) {
        // Fallback to main thread
        const { filterByThreshold } = await import("strata-ui");
        return filterByThreshold(data, threshold, maxPressure);
      }

      const response = await sendMessage<{ type: "filter"; result: FilterResult }>({
        type: "filter",
        data,
        threshold,
        maxPressure,
      });

      return response.result;
    },
    [sendMessage]
  );

  const downsampleAndFilter = useCallback(
    async (
      data: Float32Array,
      shape: [number, number, number],
      downsampleOptions: Partial<DownsampleOptions>,
      threshold: number
    ): Promise<{ downsampleResult: DownsampleResult; filterResult: FilterResult }> => {
      if (!isAvailableRef.current) {
        // Fallback to main thread
        const { downsamplePressure, filterByThreshold } = await import("strata-ui");
        const downsampleResult = downsamplePressure(data, shape, downsampleOptions);

        let maxPressure = 0;
        for (let i = 0; i < downsampleResult.data.length; i++) {
          const abs = Math.abs(downsampleResult.data[i]);
          if (abs > maxPressure) maxPressure = abs;
        }

        const filterResult = filterByThreshold(downsampleResult.data, threshold, maxPressure);
        return { downsampleResult, filterResult };
      }

      const response = await sendMessage<{
        type: "downsampleAndFilter";
        downsampleResult: DownsampleResult;
        filterResult: FilterResult;
      }>({
        type: "downsampleAndFilter",
        data,
        shape,
        downsampleOptions,
        threshold,
      });

      return {
        downsampleResult: response.downsampleResult,
        filterResult: response.filterResult,
      };
    },
    [sendMessage]
  );

  return {
    isAvailable: isAvailableRef.current,
    downsample,
    filter,
    downsampleAndFilter,
  };
}
