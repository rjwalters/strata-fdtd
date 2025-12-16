import { useState } from "react"
import { HelpCircle, Keyboard, Mouse, Video, Download } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Badge } from "@/components/ui/badge"

type HelpTab = "shortcuts" | "camera" | "playback" | "export"

interface ShortcutItem {
  keys: string[]
  description: string
}

const KEYBOARD_SHORTCUTS: ShortcutItem[] = [
  { keys: ["Space"], description: "Play / Pause" },
  { keys: ["←"], description: "Previous frame" },
  { keys: ["→"], description: "Next frame" },
  { keys: ["Home"], description: "Jump to start" },
  { keys: ["End"], description: "Jump to end" },
  { keys: ["+"], description: "Increase playback speed" },
  { keys: ["-"], description: "Decrease playback speed" },
]

const CAMERA_CONTROLS: ShortcutItem[] = [
  { keys: ["Left drag"], description: "Rotate camera" },
  { keys: ["Right drag"], description: "Pan camera" },
  { keys: ["Scroll"], description: "Zoom in/out" },
  { keys: ["Double-click"], description: "Reset camera" },
]

const PLAYBACK_TIPS = [
  "Use the scrubber to jump to specific frames",
  "Click on the time series plot to jump to that time",
  "Enable looping for continuous playback",
  "Adjust playback speed for detailed analysis",
]

const EXPORT_TIPS = [
  "PNG: Export current frame as an image",
  "GIF: Create animated visualization",
  "Video: Export high-quality MP4 (requires FFmpeg)",
  "CSV: Export probe data for external analysis",
  "JSON: Export complete view state",
]

export interface ViewerHelpModalProps {
  defaultOpen?: boolean
}

export function ViewerHelpModal({ defaultOpen = false }: ViewerHelpModalProps) {
  const [activeTab, setActiveTab] = useState<HelpTab>("shortcuts")

  return (
    <Dialog defaultOpen={defaultOpen}>
      <DialogTrigger asChild>
        <Button variant="ghost" size="icon" title="Help">
          <HelpCircle className="h-4 w-4" />
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Viewer Help</DialogTitle>
          <DialogDescription>
            Learn how to use the simulation viewer
          </DialogDescription>
        </DialogHeader>

        {/* Tab buttons */}
        <div className="flex gap-1 border-b pb-2">
          <TabButton
            active={activeTab === "shortcuts"}
            onClick={() => setActiveTab("shortcuts")}
            icon={<Keyboard className="h-3 w-3" />}
            label="Shortcuts"
          />
          <TabButton
            active={activeTab === "camera"}
            onClick={() => setActiveTab("camera")}
            icon={<Mouse className="h-3 w-3" />}
            label="Camera"
          />
          <TabButton
            active={activeTab === "playback"}
            onClick={() => setActiveTab("playback")}
            icon={<Video className="h-3 w-3" />}
            label="Playback"
          />
          <TabButton
            active={activeTab === "export"}
            onClick={() => setActiveTab("export")}
            icon={<Download className="h-3 w-3" />}
            label="Export"
          />
        </div>

        {/* Tab content */}
        <div className="min-h-[200px]">
          {activeTab === "shortcuts" && (
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground mb-3">
                Keyboard shortcuts for playback control
              </p>
              {KEYBOARD_SHORTCUTS.map((shortcut, i) => (
                <ShortcutRow key={i} {...shortcut} />
              ))}
            </div>
          )}

          {activeTab === "camera" && (
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground mb-3">
                Mouse controls for 3D camera navigation
              </p>
              {CAMERA_CONTROLS.map((control, i) => (
                <ShortcutRow key={i} {...control} />
              ))}
            </div>
          )}

          {activeTab === "playback" && (
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground mb-3">
                Tips for playback and analysis
              </p>
              <ul className="space-y-2">
                {PLAYBACK_TIPS.map((tip, i) => (
                  <li key={i} className="text-sm flex items-start gap-2">
                    <span className="text-primary mt-0.5">•</span>
                    {tip}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {activeTab === "export" && (
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground mb-3">
                Available export formats
              </p>
              <ul className="space-y-2">
                {EXPORT_TIPS.map((tip, i) => (
                  <li key={i} className="text-sm flex items-start gap-2">
                    <span className="text-primary mt-0.5">•</span>
                    {tip}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t pt-3 mt-2">
          <p className="text-xs text-muted-foreground text-center">
            Press <kbd className="px-1 py-0.5 bg-secondary rounded text-xs">?</kbd> anytime to open this help
          </p>
        </div>
      </DialogContent>
    </Dialog>
  )
}

interface TabButtonProps {
  active: boolean
  onClick: () => void
  icon: React.ReactNode
  label: string
}

function TabButton({ active, onClick, icon, label }: TabButtonProps) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
        active
          ? "bg-primary text-primary-foreground"
          : "text-muted-foreground hover:bg-secondary"
      }`}
    >
      {icon}
      {label}
    </button>
  )
}

function ShortcutRow({ keys, description }: ShortcutItem) {
  return (
    <div className="flex items-center justify-between py-1">
      <span className="text-sm">{description}</span>
      <div className="flex gap-1">
        {keys.map((key, i) => (
          <Badge key={i} variant="secondary" className="text-xs font-mono">
            {key}
          </Badge>
        ))}
      </div>
    </div>
  )
}
