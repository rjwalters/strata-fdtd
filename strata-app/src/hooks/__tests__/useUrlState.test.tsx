/**
 * Tests for useUrlState hook - URL state synchronization.
 */
import { describe, it, expect, beforeEach } from "vitest";
import { renderHook } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { useUrlState } from "../useUrlState";
import { useSimulationStore } from "strata-ui";
import type { ReactNode } from "react";

// Helper to create wrapper with router
// Uses InitialEntry format for proper search params handling in react-router v7
function createWrapper(path: string = "/") {
  return function Wrapper({ children }: { children: ReactNode }) {
    // Parse path into pathname and search
    const [pathname, search] = path.includes("?")
      ? [path.split("?")[0], `?${path.split("?")[1]}`]
      : [path, ""];
    return (
      <MemoryRouter initialEntries={[{ pathname, search }]}>
        {children}
      </MemoryRouter>
    );
  };
}

describe("useUrlState", () => {
  beforeEach(() => {
    // Reset store to default state
    useSimulationStore.getState().reset();
  });

  describe("URL parsing on mount", () => {
    // Note: URL parsing tests are skipped due to react-router v7 MemoryRouter
    // behavior in tests. The feature works correctly in production.
    // These tests pass individually but fail when run together due to
    // searchParams initialization timing in React 19 strict mode.

    it("parses speed from URL", () => {
      const wrapper = createWrapper("/?speed=2");
      renderHook(() => useUrlState(), { wrapper });

      const state = useSimulationStore.getState();
      expect(state.playbackSpeed).toBe(2);
    });

    it("parses colormap from URL", () => {
      const wrapper = createWrapper("/?cmap=viridis");
      renderHook(() => useUrlState(), { wrapper });

      const state = useSimulationStore.getState();
      expect(state.colormap).toBe("viridis");
    });

    it("parses threshold from URL", () => {
      const wrapper = createWrapper("/?thresh=0.5");
      renderHook(() => useUrlState(), { wrapper });

      const state = useSimulationStore.getState();
      expect(state.threshold).toBe(0.5);
    });

    it("parses selected probes from URL", () => {
      const wrapper = createWrapper("/?probes=upstream,cavity");
      renderHook(() => useUrlState(), { wrapper });

      const state = useSimulationStore.getState();
      expect(state.selectedProbes).toEqual(["upstream", "cavity"]);
    });

    it("parses voxel geometry from URL", () => {
      const wrapper = createWrapper("/?geom=mesh");
      renderHook(() => useUrlState(), { wrapper });

      const state = useSimulationStore.getState();
      expect(state.voxelGeometry).toBe("mesh");
    });
  });

  describe("invalid URL params", () => {
    it("ignores invalid frame (negative)", () => {
      const wrapper = createWrapper("/?frame=-1");
      renderHook(() => useUrlState(), { wrapper });

      const state = useSimulationStore.getState();
      expect(state.currentFrame).toBe(0); // Default
    });

    it("ignores invalid frame (non-numeric)", () => {
      const wrapper = createWrapper("/?frame=abc");
      renderHook(() => useUrlState(), { wrapper });

      const state = useSimulationStore.getState();
      expect(state.currentFrame).toBe(0); // Default
    });

    it("ignores invalid speed (out of range)", () => {
      const wrapper = createWrapper("/?speed=10");
      renderHook(() => useUrlState(), { wrapper });

      const state = useSimulationStore.getState();
      expect(state.playbackSpeed).toBe(1); // Default
    });

    it("ignores invalid colormap", () => {
      const wrapper = createWrapper("/?cmap=invalid");
      renderHook(() => useUrlState(), { wrapper });

      const state = useSimulationStore.getState();
      expect(state.colormap).toBe("diverging"); // Default
    });

    it("ignores invalid threshold (out of range)", () => {
      const wrapper = createWrapper("/?thresh=2");
      renderHook(() => useUrlState(), { wrapper });

      const state = useSimulationStore.getState();
      expect(state.threshold).toBe(0); // Default
    });

    it("ignores invalid geometry", () => {
      const wrapper = createWrapper("/?geom=invalid");
      renderHook(() => useUrlState(), { wrapper });

      const state = useSimulationStore.getState();
      expect(state.voxelGeometry).toBe("point"); // Default
    });
  });

  describe("getShareableUrl", () => {
    it("includes frame in URL", () => {
      // Default frame of 0 should be included
      const wrapper = createWrapper("/");
      const { result } = renderHook(() => useUrlState(), { wrapper });

      const url = result.current.getShareableUrl();
      expect(url).toContain("frame=");
    });

    it("includes only non-default values", () => {
      // Initialize with defaults via empty URL
      const wrapper = createWrapper("/");
      const { result } = renderHook(() => useUrlState(), { wrapper });

      const url = result.current.getShareableUrl();
      // Should include frame (always) but not other defaults
      expect(url).toContain("frame=0");
      expect(url).not.toContain("speed=");
      expect(url).not.toContain("cmap=");
    });

    it("includes non-default colormap", () => {
      // Initialize with URL params
      const wrapper = createWrapper("/?cmap=viridis");
      const { result } = renderHook(() => useUrlState(), { wrapper });

      const url = result.current.getShareableUrl();
      expect(url).toContain("cmap=viridis");
    });

    it("includes non-default voxel geometry", () => {
      // Initialize with URL params
      const wrapper = createWrapper("/?geom=mesh");
      const { result } = renderHook(() => useUrlState(), { wrapper });

      const url = result.current.getShareableUrl();
      expect(url).toContain("geom=mesh");
    });

    it("includes selected probes", () => {
      // Initialize with URL params
      const wrapper = createWrapper("/?probes=upstream,cavity");
      const { result } = renderHook(() => useUrlState(), { wrapper });

      const url = result.current.getShareableUrl();
      // Comma is URL-encoded as %2C
      expect(url).toContain("probes=upstream%2Ccavity");
    });
  });
});
