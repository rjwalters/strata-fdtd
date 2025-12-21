import { useCallback } from 'react';
import { FileText, X } from 'lucide-react';

interface FilePickerModalProps {
  files: string[];
  isOpen: boolean;
  onClose: () => void;
  onSelectFile: (path: string) => void;
}

export function FilePickerModal({ files, isOpen, onClose, onSelectFile }: FilePickerModalProps) {
  const getFileName = useCallback((path: string) => {
    return path.split('/').pop() || path.split('\\').pop() || path;
  }, []);

  const getDirectory = useCallback((path: string) => {
    const parts = path.split('/');
    if (parts.length > 1) {
      parts.pop();
      return parts.join('/');
    }
    const winParts = path.split('\\');
    if (winParts.length > 1) {
      winParts.pop();
      return winParts.join('\\');
    }
    return '';
  }, []);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-background/80 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative bg-card rounded-xl shadow-2xl p-6 max-w-lg w-full mx-4 border border-border">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-foreground">
            Choose File to Open
          </h2>
          <button
            onClick={onClose}
            className="p-1 rounded-md hover:bg-muted transition-colors"
            aria-label="Close"
          >
            <X className="w-5 h-5 text-muted-foreground" />
          </button>
        </div>

        <p className="text-sm text-muted-foreground mb-4">
          Multiple HDF5 files were dropped. Select one to open:
        </p>

        {/* File list */}
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {files.map((file, index) => (
            <button
              key={index}
              onClick={() => onSelectFile(file)}
              className="w-full text-left p-3 rounded-lg border border-border hover:border-primary hover:bg-primary/5 transition-colors group"
            >
              <div className="flex items-start gap-3">
                <FileText className="w-5 h-5 text-muted-foreground group-hover:text-primary flex-shrink-0 mt-0.5" />
                <div className="min-w-0 flex-1">
                  <p className="font-medium text-foreground truncate">
                    {getFileName(file)}
                  </p>
                  <p className="text-xs text-muted-foreground truncate">
                    {getDirectory(file)}
                  </p>
                </div>
              </div>
            </button>
          ))}
        </div>

        {/* Footer */}
        <div className="mt-4 pt-4 border-t border-border flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
