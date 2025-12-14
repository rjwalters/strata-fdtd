import { describe, it, expect } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useStreamingBuffer } from "../useStreamingBuffer";

describe("useStreamingBuffer", () => {
  const defaultConfig = {
    maxBufferSize: 50000,
    sampleRate: 44100,
  };

  describe("invalidate", () => {
    it("should invalidate a specific probe", () => {
      const { result } = renderHook(() => useStreamingBuffer(defaultConfig));

      act(() => {
        result.current.invalidate("probe1");
      });

      expect(result.current.isInvalidated("probe1")).toBe(true);
      expect(result.current.isInvalidated("probe2")).toBe(false);
    });

    it("should invalidate all known probes when called without argument", () => {
      const { result } = renderHook(() => useStreamingBuffer(defaultConfig));

      // First, make probes known by checking them
      act(() => {
        result.current.isInvalidated("probe1");
        result.current.isInvalidated("probe2");
        result.current.isInvalidated("probe3");
      });

      // Clear any invalidation from the checks
      act(() => {
        result.current.clearInvalidation("probe1");
        result.current.clearInvalidation("probe2");
        result.current.clearInvalidation("probe3");
      });

      // Invalidate all
      act(() => {
        result.current.invalidate();
      });

      expect(result.current.isInvalidated("probe1")).toBe(true);
      expect(result.current.isInvalidated("probe2")).toBe(true);
      expect(result.current.isInvalidated("probe3")).toBe(true);
    });

    it("should handle invalidate-all with no known probes", () => {
      const { result } = renderHook(() => useStreamingBuffer(defaultConfig));

      // Should not throw when no probes are known
      act(() => {
        result.current.invalidate();
      });

      // New probe should not be invalidated
      expect(result.current.isInvalidated("newProbe")).toBe(false);
    });
  });

  describe("isInvalidated", () => {
    it("should return false for unknown probes", () => {
      const { result } = renderHook(() => useStreamingBuffer(defaultConfig));

      expect(result.current.isInvalidated("unknownProbe")).toBe(false);
    });

    it("should return true after invalidation", () => {
      const { result } = renderHook(() => useStreamingBuffer(defaultConfig));

      expect(result.current.isInvalidated("probe1")).toBe(false);

      act(() => {
        result.current.invalidate("probe1");
      });

      expect(result.current.isInvalidated("probe1")).toBe(true);
    });

    it("should register probes as known when checked", () => {
      const { result } = renderHook(() => useStreamingBuffer(defaultConfig));

      // Check probe (makes it known)
      result.current.isInvalidated("probe1");

      // Now invalidate all - should include probe1
      act(() => {
        result.current.invalidate();
      });

      expect(result.current.isInvalidated("probe1")).toBe(true);
    });
  });

  describe("clearInvalidation", () => {
    it("should clear invalidation for a specific probe", () => {
      const { result } = renderHook(() => useStreamingBuffer(defaultConfig));

      act(() => {
        result.current.invalidate("probe1");
        result.current.invalidate("probe2");
      });

      expect(result.current.isInvalidated("probe1")).toBe(true);
      expect(result.current.isInvalidated("probe2")).toBe(true);

      act(() => {
        result.current.clearInvalidation("probe1");
      });

      expect(result.current.isInvalidated("probe1")).toBe(false);
      expect(result.current.isInvalidated("probe2")).toBe(true);
    });

    it("should handle clearing non-invalidated probe", () => {
      const { result } = renderHook(() => useStreamingBuffer(defaultConfig));

      // Should not throw
      act(() => {
        result.current.clearInvalidation("unknownProbe");
      });

      expect(result.current.isInvalidated("unknownProbe")).toBe(false);
    });
  });

  describe("stability", () => {
    it("should return stable function references", () => {
      const { result, rerender } = renderHook(() =>
        useStreamingBuffer(defaultConfig)
      );

      const { invalidate, isInvalidated, clearInvalidation } = result.current;

      rerender();

      expect(result.current.invalidate).toBe(invalidate);
      expect(result.current.isInvalidated).toBe(isInvalidated);
      expect(result.current.clearInvalidation).toBe(clearInvalidation);
    });
  });

  describe("integration scenarios", () => {
    it("should handle zoom invalidation workflow", () => {
      const { result } = renderHook(() => useStreamingBuffer(defaultConfig));

      // Simulate initial render registering probes
      result.current.isInvalidated("pressure");
      result.current.isInvalidated("velocity");

      // User zooms - invalidate all
      act(() => {
        result.current.invalidate();
      });

      expect(result.current.isInvalidated("pressure")).toBe(true);
      expect(result.current.isInvalidated("velocity")).toBe(true);

      // Full render happens, clear invalidation
      act(() => {
        result.current.clearInvalidation("pressure");
        result.current.clearInvalidation("velocity");
      });

      expect(result.current.isInvalidated("pressure")).toBe(false);
      expect(result.current.isInvalidated("velocity")).toBe(false);
    });

    it("should handle visibility toggle workflow", () => {
      const { result } = renderHook(() => useStreamingBuffer(defaultConfig));

      // Register probes
      result.current.isInvalidated("probe1");
      result.current.isInvalidated("probe2");

      // Toggle probe1 visibility - invalidate only that probe
      act(() => {
        result.current.invalidate("probe1");
      });

      expect(result.current.isInvalidated("probe1")).toBe(true);
      expect(result.current.isInvalidated("probe2")).toBe(false);
    });
  });
});
