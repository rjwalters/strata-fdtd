import { useState } from "react"
import { HelpCircle, Keyboard, Code2, Box, Lightbulb } from "lucide-react"
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

type HelpTab = "shortcuts" | "templates" | "syntax" | "tips"

interface ShortcutItem {
  keys: string[]
  description: string
}

const KEYBOARD_SHORTCUTS: ShortcutItem[] = [
  { keys: ["Ctrl", "S"], description: "Save / Download script" },
  { keys: ["Ctrl", "Z"], description: "Undo" },
  { keys: ["Ctrl", "Shift", "Z"], description: "Redo" },
  { keys: ["Ctrl", "/"], description: "Toggle comment" },
  { keys: ["Ctrl", "D"], description: "Duplicate line" },
  { keys: ["Ctrl", "F"], description: "Find" },
  { keys: ["Ctrl", "H"], description: "Find and replace" },
  { keys: ["Tab"], description: "Indent" },
  { keys: ["Shift", "Tab"], description: "Outdent" },
]

const TEMPLATES = [
  { name: "Rectangle", shortcut: "None", description: "Add rectangular solid material region" },
  { name: "Sphere", shortcut: "None", description: "Add spherical material region" },
  { name: "Pulse", shortcut: "None", description: "Add Gaussian pulse excitation source" },
  { name: "CW", shortcut: "None", description: "Add continuous wave source" },
  { name: "Probe", shortcut: "None", description: "Add pressure recording probe" },
]

const SYNTAX_TIPS = [
  "Use `sim.grid(extent, resolution)` to define the simulation domain",
  "Materials are defined with `sim.material(name, density, speed_of_sound)`",
  "Positions use meters: `[x, y, z]` or tuples `(x, y, z)`",
  "Sources need `position`, `frequency`, and optionally `amplitude`",
  "All Python comments (#) are preserved in the output",
]

const GENERAL_TIPS = [
  "The 3D preview updates in real-time as you type",
  "Red underlines indicate syntax errors - hover for details",
  "Use the Estimation panel to check memory and runtime estimates",
  "Export your simulation as a Python script or JSON config",
  "The slice plane tool helps visualize internal structures",
]

export interface BuilderHelpModalProps {
  defaultOpen?: boolean
}

export function BuilderHelpModal({ defaultOpen = false }: BuilderHelpModalProps) {
  const [activeTab, setActiveTab] = useState<HelpTab>("shortcuts")

  return (
    <Dialog defaultOpen={defaultOpen}>
      <DialogTrigger asChild>
        <Button variant="ghost" size="sm" title="Help">
          <HelpCircle className="h-4 w-4" />
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Builder Help</DialogTitle>
          <DialogDescription>
            Learn how to use the simulation builder
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
            active={activeTab === "templates"}
            onClick={() => setActiveTab("templates")}
            icon={<Box className="h-3 w-3" />}
            label="Templates"
          />
          <TabButton
            active={activeTab === "syntax"}
            onClick={() => setActiveTab("syntax")}
            icon={<Code2 className="h-3 w-3" />}
            label="Syntax"
          />
          <TabButton
            active={activeTab === "tips"}
            onClick={() => setActiveTab("tips")}
            icon={<Lightbulb className="h-3 w-3" />}
            label="Tips"
          />
        </div>

        {/* Tab content */}
        <div className="min-h-[200px]">
          {activeTab === "shortcuts" && (
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground mb-3">
                Editor keyboard shortcuts
              </p>
              {KEYBOARD_SHORTCUTS.map((shortcut, i) => (
                <ShortcutRow key={i} {...shortcut} />
              ))}
            </div>
          )}

          {activeTab === "templates" && (
            <div className="space-y-3">
              <p className="text-sm text-muted-foreground mb-3">
                Quick-insert simulation elements
              </p>
              {TEMPLATES.map((template, i) => (
                <div key={i} className="flex items-center justify-between py-1">
                  <div>
                    <span className="text-sm font-medium">{template.name}</span>
                    <p className="text-xs text-muted-foreground">{template.description}</p>
                  </div>
                </div>
              ))}
            </div>
          )}

          {activeTab === "syntax" && (
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground mb-3">
                Python syntax reference
              </p>
              <ul className="space-y-2">
                {SYNTAX_TIPS.map((tip, i) => (
                  <li key={i} className="text-sm flex items-start gap-2">
                    <span className="text-primary mt-0.5">•</span>
                    <code className="text-xs bg-secondary px-1 py-0.5 rounded">{tip}</code>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {activeTab === "tips" && (
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground mb-3">
                Useful tips and tricks
              </p>
              <ul className="space-y-2">
                {GENERAL_TIPS.map((tip, i) => (
                  <li key={i} className="text-sm flex items-start gap-2">
                    <span className="text-primary mt-0.5">•</span>
                    {tip}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* Footer with link to docs */}
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
