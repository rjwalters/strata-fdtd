import { useState, useEffect, useCallback } from "react"
import { listen } from "@tauri-apps/api/event"
import { invoke } from "@tauri-apps/api/core"
import { Download, X, RefreshCw, AlertCircle, CheckCircle } from "lucide-react"

interface UpdateInfo {
  version: string
  current_version: string
  body?: string
}

type UpdateState = "idle" | "available" | "downloading" | "ready" | "error"

export function UpdateNotification() {
  const [updateInfo, setUpdateInfo] = useState<UpdateInfo | null>(null)
  const [updateState, setUpdateState] = useState<UpdateState>("idle")
  const [error, setError] = useState<string | null>(null)
  const [dismissed, setDismissed] = useState(false)

  // Listen for update-available event from backend
  useEffect(() => {
    const unlisten = listen<UpdateInfo>("update-available", (event) => {
      setUpdateInfo(event.payload)
      setUpdateState("available")
      setDismissed(false)
    })

    return () => {
      unlisten.then((fn) => fn())
    }
  }, [])

  // Manual check for updates
  const checkForUpdates = useCallback(async () => {
    try {
      setUpdateState("idle")
      setError(null)
      const result = await invoke<UpdateInfo | null>("check_for_updates")
      if (result) {
        setUpdateInfo(result)
        setUpdateState("available")
        setDismissed(false)
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setUpdateState("error")
    }
  }, [])

  // Install update
  const installUpdate = useCallback(async () => {
    try {
      setUpdateState("downloading")
      setError(null)
      await invoke("install_update")
      setUpdateState("ready")
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setUpdateState("error")
    }
  }, [])

  const dismiss = useCallback(() => {
    setDismissed(true)
  }, [])

  // Don't render if dismissed or no update available
  if (dismissed || updateState === "idle") {
    return null
  }

  return (
    <div className="fixed bottom-4 right-4 z-50 max-w-sm">
      <div className="bg-card border border-border rounded-lg shadow-lg p-4">
        <div className="flex items-start gap-3">
          {updateState === "error" ? (
            <AlertCircle className="h-5 w-5 text-destructive flex-shrink-0 mt-0.5" />
          ) : updateState === "ready" ? (
            <CheckCircle className="h-5 w-5 text-green-500 flex-shrink-0 mt-0.5" />
          ) : (
            <Download className="h-5 w-5 text-primary flex-shrink-0 mt-0.5" />
          )}

          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between gap-2">
              <h3 className="font-semibold text-sm text-foreground">
                {updateState === "error"
                  ? "Update Error"
                  : updateState === "ready"
                    ? "Update Ready"
                    : updateState === "downloading"
                      ? "Downloading Update..."
                      : "Update Available"}
              </h3>
              <button
                onClick={dismiss}
                className="text-muted-foreground hover:text-foreground transition-colors"
                aria-label="Dismiss"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            {updateState === "error" && error && (
              <p className="text-xs text-destructive mt-1">{error}</p>
            )}

            {updateState === "available" && updateInfo && (
              <>
                <p className="text-xs text-muted-foreground mt-1">
                  Version {updateInfo.version} is available (current:{" "}
                  {updateInfo.current_version})
                </p>
                {updateInfo.body && (
                  <p className="text-xs text-muted-foreground mt-2 line-clamp-3">
                    {updateInfo.body}
                  </p>
                )}
              </>
            )}

            {updateState === "downloading" && (
              <p className="text-xs text-muted-foreground mt-1">
                Downloading update, please wait...
              </p>
            )}

            {updateState === "ready" && (
              <p className="text-xs text-muted-foreground mt-1">
                The update has been downloaded. Restart to apply.
              </p>
            )}

            <div className="flex gap-2 mt-3">
              {updateState === "available" && (
                <button
                  onClick={installUpdate}
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-primary text-primary-foreground text-xs font-medium rounded-md hover:bg-primary/90 transition-colors"
                >
                  <Download className="h-3 w-3" />
                  Install Update
                </button>
              )}

              {updateState === "downloading" && (
                <button
                  disabled
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-muted text-muted-foreground text-xs font-medium rounded-md cursor-not-allowed"
                >
                  <RefreshCw className="h-3 w-3 animate-spin" />
                  Downloading...
                </button>
              )}

              {updateState === "error" && (
                <button
                  onClick={checkForUpdates}
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-secondary text-secondary-foreground text-xs font-medium rounded-md hover:bg-secondary/80 transition-colors"
                >
                  <RefreshCw className="h-3 w-3" />
                  Retry
                </button>
              )}

              {(updateState === "available" || updateState === "error") && (
                <button
                  onClick={dismiss}
                  className="px-3 py-1.5 text-muted-foreground text-xs font-medium rounded-md hover:text-foreground transition-colors"
                >
                  Later
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// Hook for manual update checking (can be used in settings)
export function useUpdateChecker() {
  const [isChecking, setIsChecking] = useState(false)
  const [updateInfo, setUpdateInfo] = useState<UpdateInfo | null>(null)
  const [error, setError] = useState<string | null>(null)

  const checkForUpdates = useCallback(async () => {
    setIsChecking(true)
    setError(null)
    try {
      const result = await invoke<UpdateInfo | null>("check_for_updates")
      setUpdateInfo(result)
      return result
    } catch (e) {
      const errorMessage = e instanceof Error ? e.message : String(e)
      setError(errorMessage)
      return null
    } finally {
      setIsChecking(false)
    }
  }, [])

  return { isChecking, updateInfo, error, checkForUpdates }
}
