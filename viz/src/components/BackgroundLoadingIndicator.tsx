import { useBackgroundLoadingState } from "@/stores/simulationStore"

/**
 * Displays a subtle progress indicator for background snapshot preloading.
 * Shows cached frame count and progress when loading is active.
 */
export function BackgroundLoadingIndicator() {
  const {
    isBackgroundLoading,
    backgroundLoadingProgress,
    cachedFrameCount,
    totalFrames,
  } = useBackgroundLoadingState()

  // Don't show anything if no simulation loaded or all frames cached
  if (totalFrames === 0) return null

  const progressPercent = Math.round(backgroundLoadingProgress * 100)
  const isComplete = progressPercent >= 100

  return (
    <div className="flex items-center gap-2 text-xs text-muted-foreground">
      {/* Progress bar */}
      <div className="w-16 h-1.5 bg-muted rounded-full overflow-hidden">
        <div
          className={`h-full transition-all duration-300 ${
            isComplete
              ? "bg-green-500/60"
              : isBackgroundLoading
                ? "bg-blue-500/60"
                : "bg-muted-foreground/40"
          }`}
          style={{ width: `${progressPercent}%` }}
        />
      </div>

      {/* Status text */}
      <span className="tabular-nums whitespace-nowrap">
        {isComplete ? (
          <span className="text-green-600/80">Cached</span>
        ) : (
          <>
            {cachedFrameCount}/{totalFrames}
            {isBackgroundLoading && (
              <span className="ml-1 animate-pulse">...</span>
            )}
          </>
        )}
      </span>
    </div>
  )
}
