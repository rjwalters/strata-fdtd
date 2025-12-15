/**
 * VisualizationModePanel - UI controls for switching between voxel and flow particle visualization.
 */

import { Panel } from "@/components/Layout";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Box, Wind } from "lucide-react";
import type { VisualizationMode, FlowParticleConfig } from "@/stores/simulationStore";

export interface VisualizationModePanelProps {
  /** Current visualization mode */
  visualizationMode: VisualizationMode;
  /** Whether velocity data is available */
  hasVelocityData: boolean;
  /** Flow particle configuration */
  flowParticleConfig: FlowParticleConfig;
  /** Called when visualization mode changes */
  onModeChange: (mode: VisualizationMode) => void;
  /** Called when flow particle config changes */
  onConfigChange: (config: Partial<FlowParticleConfig>) => void;
}

const VISUALIZATION_MODES: {
  value: VisualizationMode;
  label: string;
  icon: React.ReactNode;
  description: string;
}[] = [
  {
    value: "voxels",
    label: "Voxels",
    icon: <Box className="h-3 w-3" />,
    description: "Pressure field voxels",
  },
  {
    value: "flow_particles",
    label: "Flow",
    icon: <Wind className="h-3 w-3" />,
    description: "Particle advection by velocity",
  },
];

export function VisualizationModePanel({
  visualizationMode,
  hasVelocityData,
  flowParticleConfig,
  onModeChange,
  onConfigChange,
}: VisualizationModePanelProps) {
  return (
    <Panel title="Visualization Mode">
      <div className="space-y-3">
        {/* Mode selector */}
        <div className="flex gap-1">
          {VISUALIZATION_MODES.map((mode) => {
            const isDisabled = mode.value === "flow_particles" && !hasVelocityData;
            return (
              <Badge
                key={mode.value}
                variant={visualizationMode === mode.value ? "default" : "secondary"}
                className={`cursor-pointer flex items-center gap-1 ${
                  isDisabled ? "opacity-50 cursor-not-allowed" : ""
                }`}
                onClick={() => !isDisabled && onModeChange(mode.value)}
                title={
                  isDisabled
                    ? "Velocity data not available"
                    : mode.description
                }
              >
                {mode.icon}
                {mode.label}
              </Badge>
            );
          })}
        </div>

        {/* Info about flow mode availability */}
        {!hasVelocityData && (
          <p className="text-xs text-muted-foreground">
            Flow mode requires velocity data. Enable{" "}
            <code className="text-xs">capture_velocity=True</code> when running
            simulation.
          </p>
        )}

        {/* Flow particle settings (visible when flow mode is selected) */}
        {visualizationMode === "flow_particles" && hasVelocityData && (
          <div className="space-y-3 pt-2 border-t border-border">
            {/* Particle count */}
            <div>
              <div className="flex justify-between text-xs text-muted-foreground mb-1">
                <span>Particles</span>
                <span>{(flowParticleConfig.particleCount / 1000).toFixed(0)}K</span>
              </div>
              <Slider
                value={[flowParticleConfig.particleCount]}
                min={10000}
                max={100000}
                step={5000}
                onValueChange={([v]) => onConfigChange({ particleCount: v })}
              />
            </div>

            {/* Time scale */}
            <div>
              <div className="flex justify-between text-xs text-muted-foreground mb-1">
                <span>Slow-motion</span>
                <span>{Math.round(1 / flowParticleConfig.timeScale)}x slower</span>
              </div>
              <Slider
                value={[Math.log10(1 / flowParticleConfig.timeScale)]}
                min={1}
                max={4}
                step={0.1}
                onValueChange={([v]) =>
                  onConfigChange({ timeScale: 1 / Math.pow(10, v) })
                }
              />
            </div>

            {/* Particle size */}
            <div>
              <div className="flex justify-between text-xs text-muted-foreground mb-1">
                <span>Particle size</span>
                <span>{(flowParticleConfig.particleSize * 1000).toFixed(1)}mm</span>
              </div>
              <Slider
                value={[flowParticleConfig.particleSize * 1000]}
                min={0.5}
                max={10}
                step={0.5}
                onValueChange={([v]) => onConfigChange({ particleSize: v / 1000 })}
              />
            </div>

            {/* Trail toggle */}
            <label className="flex items-center gap-2 text-sm cursor-pointer">
              <input
                type="checkbox"
                checked={flowParticleConfig.showTrails}
                onChange={(e) => onConfigChange({ showTrails: e.target.checked })}
                className="rounded"
              />
              Show trails
            </label>

            {/* Trail length slider (visible when trails are enabled) */}
            {flowParticleConfig.showTrails && (
              <div>
                <div className="flex justify-between text-xs text-muted-foreground mb-1">
                  <span>Trail length</span>
                  <span>{flowParticleConfig.trailLength} frames</span>
                </div>
                <Slider
                  value={[flowParticleConfig.trailLength]}
                  min={10}
                  max={100}
                  step={5}
                  onValueChange={([v]) => onConfigChange({ trailLength: v })}
                />
              </div>
            )}
          </div>
        )}
      </div>
    </Panel>
  );
}
