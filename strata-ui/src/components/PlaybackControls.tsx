import { useState, useEffect, useCallback, useRef } from "react"
import { Play, Pause, SkipBack, SkipForward, ChevronLeft, ChevronRight, Repeat } from "lucide-react"
import { Button } from "./ui/button"
import { Slider } from "./ui/slider"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select"

export interface SnapshotInfo {
  time: number
  timestep?: number
}

export interface PlaybackControlsProps {
  /** Total number of frames */
  totalFrames?: number
  /** Current frame index */
  currentFrame?: number
  /** Callback when frame changes */
  onFrameChange?: (frame: number) => void
  /** Base playback speed in frames per second */
  fps?: number
  /** Whether playback is currently active */
  isPlaying?: boolean
  /** Callback when play state changes */
  onPlayingChange?: (playing: boolean) => void
  /** Whether controls are disabled */
  disabled?: boolean
  /** Snapshot info for time display */
  snapshots?: SnapshotInfo[]
  /** Callback for frame preloading */
  onPreloadFrames?: (frames: number[]) => void
}

const SPEED_OPTIONS = [
  { value: "0.25", label: "0.25x" },
  { value: "0.5", label: "0.5x" },
  { value: "1", label: "1x" },
  { value: "2", label: "2x" },
  { value: "4", label: "4x" },
]

export function PlaybackControls({
  totalFrames: externalTotal,
  currentFrame: externalFrame,
  onFrameChange,
  fps = 30,
  isPlaying: externalPlaying,
  onPlayingChange,
  disabled = false,
  snapshots,
  onPreloadFrames,
}: PlaybackControlsProps = {}) {
  // Use internal state if no external control
  const [internalFrame, setInternalFrame] = useState(0)
  const [internalPlaying, setInternalPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [loop, setLoop] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  const totalFrames = externalTotal ?? 100
  const currentFrame = externalFrame ?? internalFrame
  const isPlaying = externalPlaying ?? internalPlaying

  const setCurrentFrame = useCallback(
    (frame: number) => {
      const clampedFrame = Math.max(0, Math.min(totalFrames - 1, frame))
      if (onFrameChange) {
        onFrameChange(clampedFrame)
      } else {
        setInternalFrame(clampedFrame)
      }
    },
    [onFrameChange, totalFrames]
  )

  const setIsPlaying = useCallback(
    (playing: boolean) => {
      if (onPlayingChange) {
        onPlayingChange(playing)
      } else {
        setInternalPlaying(playing)
      }
    },
    [onPlayingChange]
  )

  // Animation loop using requestAnimationFrame
  useEffect(() => {
    if (!isPlaying || disabled || totalFrames === 0) return

    const interval = (1000 / fps) / speed
    let lastTime = performance.now()
    let animationId: number

    const animate = (time: number) => {
      if (time - lastTime >= interval) {
        lastTime = time

        if (currentFrame >= totalFrames - 1) {
          if (loop) {
            setCurrentFrame(0)
          } else {
            setIsPlaying(false)
            return
          }
        } else {
          setCurrentFrame(currentFrame + 1)
        }
      }
      animationId = requestAnimationFrame(animate)
    }

    animationId = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(animationId)
  }, [isPlaying, fps, speed, currentFrame, totalFrames, setCurrentFrame, setIsPlaying, disabled, loop])

  // Preload adjacent frames
  useEffect(() => {
    if (!onPreloadFrames || totalFrames === 0) return

    const framesToPreload: number[] = []
    const preloadCount = 5

    for (let i = 1; i <= preloadCount; i++) {
      if (currentFrame + i < totalFrames) {
        framesToPreload.push(currentFrame + i)
      }
      if (currentFrame - i >= 0) {
        framesToPreload.push(currentFrame - i)
      }
    }

    if (framesToPreload.length > 0) {
      onPreloadFrames(framesToPreload)
    }
  }, [currentFrame, totalFrames, onPreloadFrames])

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Only handle if focused on container or no other element is focused
      if (disabled) return

      const target = e.target as HTMLElement
      if (target.tagName === "INPUT" || target.tagName === "TEXTAREA") return

      switch (e.code) {
        case "Space":
          e.preventDefault()
          setIsPlaying(!isPlaying)
          break
        case "ArrowLeft":
          e.preventDefault()
          setCurrentFrame(currentFrame - 1)
          break
        case "ArrowRight":
          e.preventDefault()
          setCurrentFrame(currentFrame + 1)
          break
        case "Home":
          e.preventDefault()
          setCurrentFrame(0)
          break
        case "End":
          e.preventDefault()
          setCurrentFrame(totalFrames - 1)
          break
        case "Equal":
        case "NumpadAdd":
          e.preventDefault()
          setSpeed((s) => Math.min(4, s * 2))
          break
        case "Minus":
        case "NumpadSubtract":
          e.preventDefault()
          setSpeed((s) => Math.max(0.25, s / 2))
          break
      }
    }

    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [disabled, isPlaying, currentFrame, totalFrames, setIsPlaying, setCurrentFrame])

  // Format time display
  const formatTime = (seconds: number): string => {
    if (seconds < 0.001) {
      return `${(seconds * 1e6).toFixed(1)} μs`
    } else if (seconds < 1) {
      return `${(seconds * 1000).toFixed(2)} ms`
    } else {
      return `${seconds.toFixed(3)} s`
    }
  }

  const currentTime = snapshots?.[currentFrame]?.time
  const timeDisplay = currentTime !== undefined ? `t = ${formatTime(currentTime)}` : null

  return (
    <div
      ref={containerRef}
      className="flex items-center gap-3"
      tabIndex={0}
    >
      {/* Transport controls */}
      <div className="flex items-center gap-1">
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setCurrentFrame(0)}
          title="Jump to start (Home)"
          disabled={disabled}
        >
          <SkipBack className="h-4 w-4" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setCurrentFrame(currentFrame - 1)}
          title="Previous frame (Left Arrow)"
          disabled={disabled}
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>
        <Button
          variant="default"
          size="icon"
          onClick={() => setIsPlaying(!isPlaying)}
          title={isPlaying ? "Pause (Space)" : "Play (Space)"}
          disabled={disabled}
        >
          {isPlaying ? (
            <Pause className="h-4 w-4" />
          ) : (
            <Play className="h-4 w-4" />
          )}
        </Button>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setCurrentFrame(currentFrame + 1)}
          title="Next frame (Right Arrow)"
          disabled={disabled}
        >
          <ChevronRight className="h-4 w-4" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setCurrentFrame(totalFrames - 1)}
          title="Jump to end (End)"
          disabled={disabled}
        >
          <SkipForward className="h-4 w-4" />
        </Button>
      </div>

      {/* Scrubber slider */}
      <div className="flex-1 px-2">
        <Slider
          value={[currentFrame]}
          max={Math.max(0, totalFrames - 1)}
          step={1}
          onValueChange={([value]) => setCurrentFrame(value)}
          disabled={disabled}
        />
      </div>

      {/* Frame counter */}
      <div className="text-sm text-muted-foreground tabular-nums w-20 text-right">
        {totalFrames > 0 ? `${currentFrame + 1} / ${totalFrames}` : "0 / 0"}
      </div>

      {/* Time display */}
      {timeDisplay && (
        <div className="text-sm text-muted-foreground tabular-nums w-24 text-right">
          {timeDisplay}
        </div>
      )}

      {/* Loop toggle */}
      <Button
        variant={loop ? "secondary" : "ghost"}
        size="icon"
        onClick={() => setLoop(!loop)}
        title={loop ? "Loop enabled" : "Loop disabled"}
        disabled={disabled}
      >
        <Repeat className="h-4 w-4" />
      </Button>

      {/* Speed selector */}
      <Select
        value={speed.toString()}
        onValueChange={(value) => setSpeed(parseFloat(value))}
        disabled={disabled}
      >
        <SelectTrigger className="w-20">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {SPEED_OPTIONS.map((opt) => (
            <SelectItem key={opt.value} value={opt.value}>
              {opt.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}
