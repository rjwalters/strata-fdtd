/**
 * SliceControlPanel - UI controls for configuring 2D slice visualization.
 *
 * Provides axis selection and slice position controls.
 */

import { useMemo } from "react";
import { Panel } from "./Layout";
import { Badge } from "../ui/badge";
import { Slider } from "../ui/slider";
import { getSliceIndex, getAxisSize, getSlicePlaneLabel } from "../../lib/sliceUtils";
import type { SliceAxis } from "../../stores/simulationStore";

export interface SliceControlPanelProps {
  /** Currently selected axis */
  axis: SliceAxis;
  /** Current position (0-1 normalized) */
  position: number;
  /** Grid dimensions [nx, ny, nz] */
  shape: [number, number, number];
  /** Grid resolution in meters */
  resolution: number;
  /** Callback when axis changes */
  onAxisChange: (axis: SliceAxis) => void;
  /** Callback when position changes */
  onPositionChange: (position: number) => void;
  /** Whether geometry overlay is shown */
  showGeometry?: boolean;
  /** Callback when geometry visibility changes */
  onShowGeometryChange?: (show: boolean) => void;
  /** Whether geometry data is available */
  hasGeometry?: boolean;
}

const AXIS_OPTIONS: { value: SliceAxis; label: string; description: string }[] = [
  { value: "x", label: "X", description: "YZ plane" },
  { value: "y", label: "Y", description: "XZ plane" },
  { value: "z", label: "Z", description: "XY plane" },
];

export function SliceControlPanel({
  axis,
  position,
  shape,
  resolution,
  onAxisChange,
  onPositionChange,
  showGeometry = false,
  onShowGeometryChange,
  hasGeometry = false,
}: SliceControlPanelProps) {
  // Memoize slider value to prevent infinite re-render loops
  const sliderValue = useMemo(() => [position * 100], [position]);

  // Calculate current slice info
  const axisSize = getAxisSize(shape, axis);
  const sliceIndex = getSliceIndex(position, axisSize);
  const physicalPosition = sliceIndex * resolution;

  return (
    <Panel title="Slice View">
      <div className="space-y-4">
        {/* Axis selector */}
        <div>
          <div className="text-xs text-muted-foreground mb-2">
            Slice Axis
          </div>
          <div className="flex gap-1">
            {AXIS_OPTIONS.map((opt) => (
              <Badge
                key={opt.value}
                variant={axis === opt.value ? "default" : "secondary"}
                className="cursor-pointer flex-1 justify-center"
                onClick={() => onAxisChange(opt.value)}
                title={opt.description}
              >
                {opt.label}
              </Badge>
            ))}
          </div>
          <div className="text-xs text-muted-foreground mt-1">
            {getSlicePlaneLabel(axis)}
          </div>
        </div>

        {/* Position slider */}
        <div>
          <div className="flex justify-between text-xs text-muted-foreground mb-2">
            <span>Position</span>
            <span>{(position * 100).toFixed(0)}%</span>
          </div>
          <Slider
            value={sliderValue}
            min={0}
            max={100}
            step={1}
            onValueChange={([v]) => onPositionChange(v / 100)}
          />
        </div>

        {/* Current slice info */}
        <div className="p-2 bg-secondary/30 rounded-md space-y-1 text-xs">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Slice index</span>
            <span>{sliceIndex} / {axisSize - 1}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Position</span>
            <span>{(physicalPosition * 100).toFixed(1)} cm</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">{axis.toUpperCase()} range</span>
            <span>0 - {((axisSize - 1) * resolution * 100).toFixed(1)} cm</span>
          </div>
        </div>

        {/* Geometry overlay toggle */}
        {hasGeometry && onShowGeometryChange && (
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={showGeometry}
              onChange={(e) => onShowGeometryChange(e.target.checked)}
              className="rounded"
            />
            Show Geometry
          </label>
        )}
      </div>
    </Panel>
  );
}
