import type { ReactNode } from "react"
import { cn } from "@/lib/utils"

interface LayoutProps {
  sidebar: ReactNode
  main: ReactNode
  bottom: ReactNode
}

export function Layout({ sidebar, main, bottom }: LayoutProps) {
  return (
    <div className="flex h-screen w-full bg-background text-foreground">
      {/* Sidebar */}
      <aside className="w-72 flex-shrink-0 border-r border-border bg-card p-4 overflow-y-auto">
        {sidebar}
      </aside>

      {/* Main content area */}
      <div className="flex flex-1 flex-col min-w-0">
        {/* 3D Viewer */}
        <main className="flex-1 min-h-0 bg-background">
          {main}
        </main>

        {/* Bottom panel for time series */}
        <div className="h-64 flex-shrink-0 border-t border-border bg-card p-4">
          {bottom}
        </div>
      </div>
    </div>
  )
}

interface PanelProps {
  title: string
  children: ReactNode
  className?: string
}

export function Panel({ title, children, className }: PanelProps) {
  return (
    <div className={cn("space-y-3", className)}>
      <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
        {title}
      </h2>
      {children}
    </div>
  )
}
