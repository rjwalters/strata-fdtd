/**
 * ViewModePanel - Toggle between 3D voxel and 2D slice visualization modes.
 */

import { Panel } from "./Layout";
import { Badge } from "../ui/badge";
import { Box, Layers } from "lucide-react";
import type { ViewMode } from "../../stores/simulationStore";

export interface ViewModePanelProps {
  /** Current view mode */
  mode: ViewMode;
  /** Callback when mode changes */
  onModeChange: (mode: ViewMode) => void;
}

const VIEW_MODES: {
  value: ViewMode;
  label: string;
  icon: React.ReactNode;
  description: string;
}[] = [
  {
    value: "3d",
    label: "3D",
    icon: <Box className="h-3 w-3" />,
    description: "3D voxel visualization",
  },
  {
    value: "slice",
    label: "Slice",
    icon: <Layers className="h-3 w-3" />,
    description: "2D cross-section view",
  },
];

export function ViewModePanel({ mode, onModeChange }: ViewModePanelProps) {
  return (
    <Panel title="View Mode">
      <div className="flex gap-1">
        {VIEW_MODES.map((opt) => (
          <Badge
            key={opt.value}
            variant={mode === opt.value ? "default" : "secondary"}
            className="cursor-pointer flex items-center gap-1 flex-1 justify-center"
            onClick={() => onModeChange(opt.value)}
            title={opt.description}
          >
            {opt.icon}
            {opt.label}
          </Badge>
        ))}
      </div>
      <p className="text-xs text-muted-foreground mt-2">
        {mode === "3d"
          ? "Viewing 3D pressure voxels"
          : "Viewing 2D pressure slice"}
      </p>
    </Panel>
  );
}
