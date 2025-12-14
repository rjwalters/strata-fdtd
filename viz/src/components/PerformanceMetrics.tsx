/**
 * Performance metrics overlay for development/debugging.
 *
 * Shows FPS, rendered voxels, memory usage, and other performance stats.
 */

import { useMemo } from "react";
import type { PerformanceMetrics } from "@/lib/performance";
import { PERFORMANCE_THRESHOLDS } from "@/lib/performance";

export interface PerformanceMetricsProps {
  /** Current performance metrics */
  metrics: PerformanceMetrics | null;
  /** Whether to show the metrics panel */
  visible?: boolean;
  /** Compact mode (single line) */
  compact?: boolean;
}

function formatNumber(n: number): string {
  if (n >= 1000000) {
    return `${(n / 1000000).toFixed(1)}M`;
  }
  if (n >= 1000) {
    return `${(n / 1000).toFixed(1)}K`;
  }
  return n.toString();
}

export function PerformanceMetrics({
  metrics,
  visible = true,
  compact = false,
}: PerformanceMetricsProps) {
  // Hooks must be called before early returns
  const fpsColor = useMemo(() => {
    if (!metrics) return "text-gray-400";
    if (metrics.fps >= PERFORMANCE_THRESHOLDS.targetFPS) return "text-green-400";
    if (metrics.fps >= 20) return "text-yellow-400";
    return "text-red-400";
  }, [metrics]);

  const voxelColor = useMemo(() => {
    if (!metrics) return "text-gray-400";
    if (metrics.renderedVoxels < PERFORMANCE_THRESHOLDS.warnVoxelCount)
      return "text-green-400";
    if (metrics.renderedVoxels < PERFORMANCE_THRESHOLDS.maxVoxelsForRealtime)
      return "text-yellow-400";
    return "text-red-400";
  }, [metrics]);

  if (!visible || !metrics) return null;

  if (compact) {
    return (
      <div className="bg-black/70 text-white text-xs px-2 py-1 rounded font-mono flex gap-3">
        <span className={fpsColor}>{metrics.fps} FPS</span>
        <span className={voxelColor}>
          {formatNumber(metrics.renderedVoxels)} voxels
        </span>
        {metrics.downsampleFactor > 1 && (
          <span className="text-blue-400">
            {metrics.downsampleFactor}x DS
          </span>
        )}
      </div>
    );
  }

  return (
    <div className="bg-black/80 text-white text-xs px-3 py-2 rounded font-mono space-y-1">
      <div className="text-gray-400 border-b border-gray-600 pb-1 mb-1">
        Performance
      </div>

      <div className="flex justify-between gap-4">
        <span className="text-gray-400">FPS:</span>
        <span className={fpsColor}>{metrics.fps}</span>
      </div>

      <div className="flex justify-between gap-4">
        <span className="text-gray-400">Frame:</span>
        <span>{metrics.frameTime}ms</span>
      </div>

      <div className="flex justify-between gap-4">
        <span className="text-gray-400">Rendered:</span>
        <span className={voxelColor}>
          {formatNumber(metrics.renderedVoxels)}
        </span>
      </div>

      <div className="flex justify-between gap-4">
        <span className="text-gray-400">Total:</span>
        <span>{formatNumber(metrics.totalVoxels)}</span>
      </div>

      {metrics.downsampleFactor > 1 && (
        <div className="flex justify-between gap-4">
          <span className="text-gray-400">Downsample:</span>
          <span className="text-blue-400">{metrics.downsampleFactor}x</span>
        </div>
      )}

      <div className="flex justify-between gap-4">
        <span className="text-gray-400">Memory:</span>
        <span>{metrics.memoryMB}MB</span>
      </div>

      {/* Quality suggestion */}
      {metrics.fps > 0 && metrics.fps < PERFORMANCE_THRESHOLDS.targetFPS && (
        <div className="mt-2 pt-1 border-t border-gray-600 text-yellow-400 text-[10px]">
          Low FPS detected. Consider increasing threshold or enabling
          downsampling.
        </div>
      )}
    </div>
  );
}
