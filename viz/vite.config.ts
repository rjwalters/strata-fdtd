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
  // Headers required for FFmpeg.wasm SharedArrayBuffer support
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  // Optimize FFmpeg.wasm deps
  optimizeDeps: {
    exclude: ['@ffmpeg/ffmpeg', '@ffmpeg/util'],
  },
  // Multi-page app: include benchmark page
  build: {
    rollupOptions: {
      input: {
        main: path.resolve(__dirname, 'index.html'),
        benchmark: path.resolve(__dirname, 'benchmark.html'),
      },
      output: {
        manualChunks: {
          // Split Three.js into its own chunk (largest dependency)
          'three': ['three'],
          // Split React vendor libraries
          'react-vendor': ['react', 'react-dom'],
          // Split D3 (used for charts)
          'd3': ['d3'],
        },
      },
    },
  },
  test: {
    environment: 'happy-dom',
    globals: true,
  },
})
