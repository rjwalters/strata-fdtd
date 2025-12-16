/**
 * FileUpload component for loading HDF5 simulation files.
 *
 * Supports:
 * - Drag and drop file upload
 * - File picker button
 * - URL input for remote files
 * - Loading progress indicator
 * - Error display
 * - Quick-load sample files
 */

import { useState, useCallback, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Upload, Link2, Loader2, FileWarning, HardDrive, Beaker } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * Available sample files for quick loading.
 */
const SAMPLE_FILES = [
  {
    id: "simple-pulse",
    name: "Simple Pulse",
    description: "Gaussian pulse propagation in air",
    url: "/samples/simple-pulse.h5",
    size: "~0.7 MB",
  },
] as const;

export interface FileUploadProps {
  /** Called when a file is selected or URL is submitted */
  onFile: (file: File) => Promise<void>;
  /** Called when a URL is submitted */
  onURL: (url: string) => Promise<void>;
  /** Called when a file is selected via native picker (path-based, Tauri only) */
  onPath?: (path: string) => Promise<void>;
  /** Whether running in Tauri environment */
  isTauri?: boolean;
  /** Function to open native file dialog (Tauri only) */
  openNativeDialog?: () => Promise<string | null>;
  /** Current loading state */
  isLoading?: boolean;
  /** Loading progress (0-1) */
  progress?: number;
  /** Error message to display */
  error?: string | null;
  /** Max file size in bytes (default 2GB) */
  maxFileSize?: number;
  /** Accepted file extensions */
  accept?: string;
  /** Custom class name */
  className?: string;
}

export function FileUpload({
  onFile,
  onURL,
  onPath,
  isTauri = false,
  openNativeDialog,
  isLoading = false,
  progress = 0,
  error = null,
  maxFileSize = 2 * 1024 * 1024 * 1024, // 2GB
  accept = ".h5,.hdf5,.hdf",
  className,
}: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [urlInput, setUrlInput] = useState("");
  const [showUrlInput, setShowUrlInput] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);

      const files = e.dataTransfer.files;
      if (files.length > 0) {
        const file = files[0];
        if (file.size > maxFileSize) {
          return; // Let parent handle error via onFile rejection
        }
        await onFile(file);
      }
    },
    [onFile, maxFileSize]
  );

  const handleFileSelect = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (files && files.length > 0) {
        await onFile(files[0]);
      }
      // Reset input so same file can be selected again
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    },
    [onFile]
  );

  const handleURLSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      if (urlInput.trim()) {
        await onURL(urlInput.trim());
      }
    },
    [urlInput, onURL]
  );

  const handleBrowseClick = useCallback(async () => {
    // Use native dialog in Tauri mode if available
    if (isTauri && openNativeDialog && onPath) {
      const path = await openNativeDialog();
      if (path) {
        await onPath(path);
      }
    } else {
      // Fall back to web file input
      fileInputRef.current?.click();
    }
  }, [isTauri, openNativeDialog, onPath]);

  const formatBytes = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024)
      return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  };

  return (
    <div className={cn("flex flex-col items-center justify-center", className)}>
      {/* Main upload area */}
      <div
        className={cn(
          "w-full max-w-xl p-8 border-2 border-dashed rounded-lg transition-colors",
          "bg-background/50 backdrop-blur",
          isDragging
            ? "border-primary bg-primary/5"
            : "border-muted-foreground/25 hover:border-muted-foreground/50",
          isLoading && "pointer-events-none opacity-75"
        )}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {isLoading ? (
          // Loading state
          <div className="flex flex-col items-center gap-4">
            <Loader2 className="h-12 w-12 text-primary animate-spin" />
            <div className="text-center">
              <p className="text-lg font-medium">Loading simulation...</p>
              {progress > 0 && (
                <div className="mt-2 w-48 h-2 bg-secondary rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary transition-all duration-300"
                    style={{ width: `${progress * 100}%` }}
                  />
                </div>
              )}
              {progress > 0 && (
                <p className="text-sm text-muted-foreground mt-1">
                  {Math.round(progress * 100)}%
                </p>
              )}
            </div>
          </div>
        ) : (
          // Upload state
          <div className="flex flex-col items-center gap-4">
            <div className="p-4 rounded-full bg-secondary">
              <Upload className="h-8 w-8 text-muted-foreground" />
            </div>

            <div className="text-center">
              <p className="text-lg font-medium">
                Drop HDF5 file here or{" "}
                <button
                  type="button"
                  className="text-primary hover:underline"
                  onClick={handleBrowseClick}
                >
                  browse
                </button>
              </p>
              <p className="text-sm text-muted-foreground mt-1">
                Supports .h5, .hdf5 files up to {formatBytes(maxFileSize)}
              </p>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept={accept}
              onChange={handleFileSelect}
              className="hidden"
            />

            {/* OR divider */}
            <div className="flex items-center gap-4 w-full max-w-xs">
              <div className="flex-1 h-px bg-border" />
              <span className="text-xs text-muted-foreground">OR</span>
              <div className="flex-1 h-px bg-border" />
            </div>

            {/* URL input toggle */}
            {!showUrlInput ? (
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  onClick={() => setShowUrlInput(true)}
                  className="gap-2"
                >
                  <Link2 className="h-4 w-4" />
                  Load from URL
                </Button>
              </div>
            ) : (
              <form
                onSubmit={handleURLSubmit}
                className="flex gap-2 w-full max-w-md"
              >
                <input
                  type="url"
                  value={urlInput}
                  onChange={(e) => setUrlInput(e.target.value)}
                  placeholder="https://example.com/simulation.h5"
                  className="flex-1 px-3 py-2 text-sm bg-background border border-input rounded-md focus:outline-none focus:ring-1 focus:ring-ring"
                  autoFocus
                />
                <Button type="submit" disabled={!urlInput.trim()}>
                  Load
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  onClick={() => {
                    setShowUrlInput(false);
                    setUrlInput("");
                  }}
                >
                  Cancel
                </Button>
              </form>
            )}

            {/* Sample files section */}
            {SAMPLE_FILES.length > 0 && (
              <>
                <div className="flex items-center gap-4 w-full max-w-xs">
                  <div className="flex-1 h-px bg-border" />
                  <span className="text-xs text-muted-foreground">OR TRY A SAMPLE</span>
                  <div className="flex-1 h-px bg-border" />
                </div>

                <div className="flex flex-wrap gap-2 justify-center">
                  {SAMPLE_FILES.map((sample) => (
                    <Button
                      key={sample.id}
                      variant="secondary"
                      size="sm"
                      className="gap-2"
                      onClick={() => onURL(sample.url)}
                    >
                      <Beaker className="h-3 w-3" />
                      {sample.name}
                      <span className="text-xs text-muted-foreground">
                        ({sample.size})
                      </span>
                    </Button>
                  ))}
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {/* Error display */}
      {error && (
        <div className="mt-4 p-4 bg-destructive/10 border border-destructive/20 rounded-lg flex items-start gap-3 max-w-xl w-full">
          <FileWarning className="h-5 w-5 text-destructive flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-medium text-destructive">Failed to load file</p>
            <p className="text-sm text-destructive/80 mt-1">{error}</p>
          </div>
        </div>
      )}

      {/* Info section */}
      <div className="mt-8 text-center text-sm text-muted-foreground max-w-md">
        <div className="flex items-center justify-center gap-2 mb-2">
          <HardDrive className="h-4 w-4" />
          <span className="font-medium">HDF5 Simulation Format</span>
        </div>
        <p>
          Load simulation results generated by the FDTD solver. Files contain
          pressure fields, probe data, geometry, and metadata.
        </p>
      </div>
    </div>
  );
}
