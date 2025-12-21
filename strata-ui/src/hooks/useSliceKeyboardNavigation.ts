/**
 * Hook for keyboard navigation in slice view mode.
 *
 * Provides keyboard shortcuts for:
 * - Arrow keys: Move slice position
 * - Number/letter keys: Switch axis
 * - Home/End: Jump to start/end
 */

import { useEffect, useCallback } from "react";
import type { SliceAxis } from "../stores/simulationStore";
import { getAxisSize } from "../lib/sliceUtils";

export interface SliceKeyboardCallbacks {
  /** Called when position should change (normalized 0-1) */
  onPositionChange: (position: number) => void;
  /** Called when axis should change */
  onAxisChange: (axis: SliceAxis) => void;
}

export interface UseSliceKeyboardNavigationOptions {
  /** Whether keyboard navigation is enabled (typically viewMode === "slice") */
  enabled: boolean;
  /** Current slice position (normalized 0-1) */
  position: number;
  /** Current slice axis */
  axis: SliceAxis;
  /** Grid dimensions [nx, ny, nz] */
  shape: [number, number, number];
  /** Callbacks for state changes */
  callbacks: SliceKeyboardCallbacks;
}

/**
 * Hook to handle keyboard navigation for slice view.
 *
 * Shortcuts:
 * - ↑/↓: Move slice position by 1 step
 * - Shift + ↑/↓: Move slice position by 10 steps
 * - 1/X: Switch to X axis (YZ plane)
 * - 2/Y: Switch to Y axis (XZ plane)
 * - 3/Z: Switch to Z axis (XY plane)
 * - Home: Jump to position 0%
 * - End: Jump to position 100%
 */
export function useSliceKeyboardNavigation({
  enabled,
  position,
  axis,
  shape,
  callbacks,
}: UseSliceKeyboardNavigationOptions): void {
  const { onPositionChange, onAxisChange } = callbacks;

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (!enabled) return;

      // Don't handle if user is typing in an input
      const target = event.target as HTMLElement;
      if (
        target.tagName === "INPUT" ||
        target.tagName === "TEXTAREA" ||
        target.isContentEditable
      ) {
        return;
      }

      const axisSize = getAxisSize(shape, axis);
      // Calculate step size as normalized value (1 cell / total cells)
      const singleStep = 1 / (axisSize - 1);
      const largeStep = singleStep * 10;

      switch (event.key) {
        case "ArrowUp":
        case "ArrowRight": {
          event.preventDefault();
          const step = event.shiftKey ? largeStep : singleStep;
          const newPosition = Math.min(1, position + step);
          onPositionChange(newPosition);
          break;
        }

        case "ArrowDown":
        case "ArrowLeft": {
          event.preventDefault();
          const step = event.shiftKey ? largeStep : singleStep;
          const newPosition = Math.max(0, position - step);
          onPositionChange(newPosition);
          break;
        }

        case "Home": {
          event.preventDefault();
          onPositionChange(0);
          break;
        }

        case "End": {
          event.preventDefault();
          onPositionChange(1);
          break;
        }

        case "1":
        case "x":
        case "X": {
          // Don't switch if modifier keys are pressed (except shift for X)
          if (event.ctrlKey || event.metaKey || event.altKey) return;
          event.preventDefault();
          onAxisChange("x");
          break;
        }

        case "2":
        case "y":
        case "Y": {
          if (event.ctrlKey || event.metaKey || event.altKey) return;
          event.preventDefault();
          onAxisChange("y");
          break;
        }

        case "3":
        case "z":
        case "Z": {
          if (event.ctrlKey || event.metaKey || event.altKey) return;
          event.preventDefault();
          onAxisChange("z");
          break;
        }
      }
    },
    [enabled, position, axis, shape, onPositionChange, onAxisChange]
  );

  useEffect(() => {
    if (!enabled) return;

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [enabled, handleKeyDown]);
}
