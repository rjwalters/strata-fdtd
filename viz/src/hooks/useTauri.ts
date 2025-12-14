import { useEffect, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';

export interface TauriContext {
  isTauri: boolean;
  openFileDialog: () => Promise<string | null>;
  openRecentFile: (path: string) => Promise<void>;
}

/**
 * Hook for integrating with Tauri desktop app features
 * Provides file dialogs, event listeners, and navigation for desktop builds
 */
export function useTauri(): TauriContext {
  // Check if running in Tauri environment
  const isTauri = typeof window !== 'undefined' && '__TAURI__' in window;

  /**
   * Open native file picker for HDF5 files
   * Falls back to null in web environment
   */
  const openFileDialog = useCallback(async (): Promise<string | null> => {
    if (!isTauri) {
      console.warn('File dialog not available in web environment');
      return null;
    }

    try {
      const path = await invoke<string | null>('open_file_dialog');
      return path;
    } catch (error) {
      console.error('Failed to open file dialog:', error);
      return null;
    }
  }, [isTauri]);

  /**
   * Open a recent file in the viewer
   * This triggers navigation to the viewer with the specified file
   */
  const openRecentFile = useCallback(
    async (path: string): Promise<void> => {
      if (!isTauri) {
        console.warn('Recent files not available in web environment');
        return;
      }

      try {
        await invoke('open_recent_file', { path });
      } catch (error) {
        console.error('Failed to open recent file:', error);
      }
    },
    [isTauri]
  );

  return {
    isTauri,
    openFileDialog,
    openRecentFile,
  };
}

/**
 * Hook for listening to Tauri events
 * Automatically unsubscribes on unmount
 */
export function useTauriEvent<T>(
  event: string,
  handler: (payload: T) => void
): void {
  const isTauri = typeof window !== 'undefined' && '__TAURI__' in window;

  useEffect(() => {
    if (!isTauri) return;

    let unlisten: UnlistenFn | undefined;

    (async () => {
      unlisten = await listen<T>(event, (event) => {
        handler(event.payload);
      });
    })();

    return () => {
      unlisten?.();
    };
  }, [isTauri, event, handler]);
}

/**
 * Hook for listening to file open events from the Tauri backend
 * Used when files are opened via system tray or deep links
 */
export function useTauriFileOpen(
  onFileOpen: (path: string) => void
): void {
  useTauriEvent<{ path: string }>('open-file', (payload) => {
    onFileOpen(payload.path);
  });
}

/**
 * Hook for listening to navigation events from the Tauri backend
 * Used when navigating via system tray menu
 */
export function useTauriNavigate(
  onNavigate: (route: string) => void
): void {
  useTauriEvent<{ route: string }>('navigate', (payload) => {
    onNavigate(payload.route);
  });
}
