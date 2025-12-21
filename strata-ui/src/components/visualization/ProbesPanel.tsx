import { useMemo } from "react";
import { Eye, EyeOff, MapPin } from "lucide-react";
import { Button } from "@/components/ui/button";

// d3.schemeCategory10 colors for consistent probe coloring
const PROBE_COLORS = [
  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
];

export interface Probe {
  name: string;
  position: [number, number, number];
}

export interface ProbesPanelProps {
  probes: Probe[];
  hiddenProbes: string[];
  showProbeMarkers: boolean;
  onToggleProbe: (name: string) => void;
  onShowAll: () => void;
  onHideAll: () => void;
  onToggleMarkers: (show: boolean) => void;
}

export function ProbesPanel({
  probes,
  hiddenProbes,
  showProbeMarkers,
  onToggleProbe,
  onShowAll,
  onHideAll,
  onToggleMarkers,
}: ProbesPanelProps) {
  const probeColors = useMemo(
    () => new Map(probes.map((probe, i) => [probe.name, PROBE_COLORS[i % PROBE_COLORS.length]])),
    [probes]
  );

  const hiddenSet = useMemo(() => new Set(hiddenProbes), [hiddenProbes]);
  const visibleCount = probes.length - hiddenSet.size;

  if (probes.length === 0) {
    return (
      <div className="text-xs text-muted-foreground italic">
        No probes in simulation
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Header with show/hide all buttons */}
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">
          {visibleCount}/{probes.length} visible
        </span>
        <div className="flex gap-1">
          <Button
            variant="ghost"
            size="sm"
            className="h-6 px-2 text-xs"
            onClick={onShowAll}
            disabled={visibleCount === probes.length}
          >
            Show All
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="h-6 px-2 text-xs"
            onClick={onHideAll}
            disabled={visibleCount === 0}
          >
            Hide All
          </Button>
        </div>
      </div>

      {/* Probe list */}
      <div className="space-y-1">
        {probes.map((probe) => {
          const isVisible = !hiddenSet.has(probe.name);
          const color = probeColors.get(probe.name);

          return (
            <button
              key={probe.name}
              onClick={() => onToggleProbe(probe.name)}
              className={`
                w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-left
                transition-colors cursor-pointer
                ${isVisible
                  ? "bg-secondary/50 hover:bg-secondary"
                  : "opacity-50 hover:bg-secondary/30"
                }
              `}
            >
              {/* Color indicator */}
              <div
                className="w-3 h-3 rounded-full flex-shrink-0"
                style={{ backgroundColor: color }}
              />

              {/* Probe name */}
              <span className="flex-1 text-sm truncate">
                {probe.name}
              </span>

              {/* Position (small) */}
              <span className="text-[10px] text-muted-foreground hidden sm:block">
                ({probe.position[0].toFixed(2)}, {probe.position[1].toFixed(2)}, {probe.position[2].toFixed(2)})
              </span>

              {/* Visibility icon */}
              {isVisible ? (
                <Eye className="h-3.5 w-3.5 text-muted-foreground flex-shrink-0" />
              ) : (
                <EyeOff className="h-3.5 w-3.5 text-muted-foreground flex-shrink-0" />
              )}
            </button>
          );
        })}
      </div>

      {/* 3D Markers toggle */}
      <div className="pt-2 border-t border-border">
        <label className="flex items-center gap-2 text-sm cursor-pointer">
          <input
            type="checkbox"
            checked={showProbeMarkers}
            onChange={(e) => onToggleMarkers(e.target.checked)}
            className="rounded"
          />
          <MapPin className="h-3.5 w-3.5" />
          Show 3D Markers
        </label>
      </div>
    </div>
  );
}
