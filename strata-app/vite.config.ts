/// <reference types="vitest" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  // Tauri expects a fixed port
  server: {
    port: 1420,
    strictPort: true,
  },
  // Environment variables
  envPrefix: ['VITE_', 'TAURI_'],
  build: {
    // Tauri uses Chromium, optimize for it
    target: process.env.TAURI_PLATFORM === 'windows' ? 'chrome105' : 'safari13',
    // Don't minify for debug builds
    minify: !process.env.TAURI_DEBUG ? 'esbuild' : false,
    // Create source maps for debug builds
    sourcemap: !!process.env.TAURI_DEBUG,
  },
  test: {
    environment: 'happy-dom',
    globals: true,
  },
})
