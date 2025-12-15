import { useRef, useCallback, useMemo } from "react";

export interface StreamingBufferConfig {
  /** Maximum number of points to keep in the render buffer */
  maxBufferSize: number;
  /** Sample rate in Hz */
  sampleRate: number;
}

export interface StreamingBufferResult {
  /** Mark a probe (or all probes) as needing full re-render */
  invalidate: (probeName?: string) => void;
  /** Check if a probe has been invalidated */
  isInvalidated: (probeName: string) => boolean;
  /** Clear invalidation flag for a probe */
  clearInvalidation: (probeName: string) => void;
}

/**
 * Hook for managing streaming buffer invalidation state.
 * Tracks which probes need full re-renders after user interactions
 * (zoom, pan, resize, visibility changes).
 */
export function useStreamingBuffer(
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  _config: StreamingBufferConfig
): StreamingBufferResult {
  // Track if invalidation was requested per probe
  const invalidatedRef = useRef<Set<string>>(new Set());
  // Track all known probe names for "invalidate all" operation
  const knownProbesRef = useRef<Set<string>>(new Set());

  const invalidate = useCallback((probeName?: string) => {
    if (probeName) {
      invalidatedRef.current.add(probeName);
      knownProbesRef.current.add(probeName);
    } else {
      // Invalidate all known probes
      for (const key of knownProbesRef.current) {
        invalidatedRef.current.add(key);
      }
    }
  }, []);

  const isInvalidated = useCallback((probeName: string): boolean => {
    knownProbesRef.current.add(probeName);
    return invalidatedRef.current.has(probeName);
  }, []);

  const clearInvalidation = useCallback((probeName: string) => {
    invalidatedRef.current.delete(probeName);
  }, []);

  return useMemo(
    () => ({
      invalidate,
      isInvalidated,
      clearInvalidation,
    }),
    [invalidate, isInvalidated, clearInvalidation]
  );
}
