import { useState, useEffect, useCallback } from 'react';
import { useTauriEvent } from '../hooks/useTauri';
import { FileUp, AlertCircle } from 'lucide-react';

interface DragEnterPayload {
  fileCount: number;
  h5FileCount: number;
}

interface MultipleFilesPayload {
  files: string[];
}

interface DragDropOverlayProps {
  onMultipleFiles?: (files: string[]) => void;
}

export function DragDropOverlay({ onMultipleFiles }: DragDropOverlayProps) {
  const isTauri = typeof window !== 'undefined' && '__TAURI__' in window;
  const [isDragging, setIsDragging] = useState(false);
  const [dragInfo, setDragInfo] = useState<DragEnterPayload | null>(null);

  // Listen for drag-enter from Tauri backend
  useTauriEvent<DragEnterPayload>('drag-enter', useCallback((payload) => {
    setIsDragging(true);
    setDragInfo(payload);
  }, []));

  // Listen for drag-leave from Tauri backend
  useTauriEvent<Record<string, never>>('drag-leave', useCallback(() => {
    setIsDragging(false);
    setDragInfo(null);
  }, []));

  // Listen for multiple files dropped
  useTauriEvent<MultipleFilesPayload>('multiple-files-dropped', useCallback((payload) => {
    onMultipleFiles?.(payload.files);
  }, [onMultipleFiles]));

  // Reset drag state on escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setIsDragging(false);
        setDragInfo(null);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Don't render if not in Tauri or not dragging
  if (!isTauri || !isDragging) return null;

  const hasH5Files = dragInfo && dragInfo.h5FileCount > 0;
  const hasOtherFiles = dragInfo && dragInfo.fileCount > dragInfo.h5FileCount;

  return (
    <div className="fixed inset-0 z-50 pointer-events-none">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-background/80 backdrop-blur-sm" />

      {/* Border indicator */}
      <div className="absolute inset-4 border-4 border-dashed rounded-2xl transition-colors duration-200"
        style={{
          borderColor: hasH5Files ? 'hsl(var(--primary))' : 'hsl(var(--muted-foreground))'
        }}
      />

      {/* Center content */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="bg-card rounded-xl shadow-2xl p-8 max-w-md text-center border border-border">
          {hasH5Files ? (
            <>
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-primary/10 flex items-center justify-center">
                <FileUp className="w-8 h-8 text-primary" />
              </div>
              <h2 className="text-xl font-semibold text-foreground mb-2">
                Drop HDF5 File
              </h2>
              <p className="text-muted-foreground">
                {dragInfo.h5FileCount === 1
                  ? 'Release to open in Viewer'
                  : `${dragInfo.h5FileCount} HDF5 files - release to choose`}
              </p>
              {hasOtherFiles && (
                <p className="text-sm text-muted-foreground mt-2">
                  ({dragInfo.fileCount - dragInfo.h5FileCount} non-HDF5 file{dragInfo.fileCount - dragInfo.h5FileCount !== 1 ? 's' : ''} will be ignored)
                </p>
              )}
            </>
          ) : (
            <>
              <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-muted flex items-center justify-center">
                <AlertCircle className="w-8 h-8 text-muted-foreground" />
              </div>
              <h2 className="text-xl font-semibold text-foreground mb-2">
                No HDF5 Files
              </h2>
              <p className="text-muted-foreground">
                Drop .h5 or .hdf5 files to open them
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
