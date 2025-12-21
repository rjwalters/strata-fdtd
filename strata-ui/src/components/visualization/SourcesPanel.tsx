import { MapPin } from "lucide-react";
import type { SourceData } from "@/stores/simulationStore";

export interface SourcesPanelProps {
  sources: SourceData[];
  showSourceMarkers: boolean;
  onToggleMarkers: (show: boolean) => void;
}

// Format frequency for display
function formatFrequency(freq: number): string {
  if (freq >= 1000) {
    return `${(freq / 1000).toFixed(1)} kHz`;
  }
  return `${freq.toFixed(0)} Hz`;
}

export function SourcesPanel({
  sources,
  showSourceMarkers,
  onToggleMarkers,
}: SourcesPanelProps) {
  if (sources.length === 0) {
    return (
      <div className="text-xs text-muted-foreground italic">
        No sources in simulation
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Source list */}
      <div className="space-y-1">
        {sources.map((source) => (
          <div
            key={source.name}
            className="flex items-center gap-2 px-2 py-1.5 rounded-md bg-secondary/50 text-sm"
          >
            {/* Source type icon/indicator */}
            <div className="w-3 h-3 rounded-full bg-yellow-500 flex-shrink-0" />

            {/* Source info */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="font-medium truncate">{source.type}</span>
                {source.frequency && (
                  <span className="text-xs text-muted-foreground">
                    {formatFrequency(source.frequency)}
                  </span>
                )}
              </div>
              <div className="text-[10px] text-muted-foreground">
                ({source.position[0].toFixed(3)}, {source.position[1].toFixed(3)}, {source.position[2].toFixed(3)})
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* 3D Markers toggle */}
      <div className="pt-2 border-t border-border">
        <label className="flex items-center gap-2 text-sm cursor-pointer">
          <input
            type="checkbox"
            checked={showSourceMarkers}
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
