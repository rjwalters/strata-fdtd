/// <reference types="vitest" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'
import { execSync } from 'child_process'

// Get git commit hash for version display
const getGitHash = () => {
  try {
    return execSync('git rev-parse --short HEAD').toString().trim()
  } catch {
    return 'dev'
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  define: {
    __GIT_HASH__: JSON.stringify(getGitHash()),
    __BUILD_TIME__: JSON.stringify(new Date().toISOString()),
  },
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
  build: {
    rollupOptions: {
      output: {
        manualChunks: (id) => {
          // React core
          if (id.includes('node_modules/react/') || id.includes('node_modules/react-dom/')) {
            return 'react-vendor';
          }
          // Three.js ecosystem (large 3D library)
          if (id.includes('node_modules/three/') ||
              id.includes('node_modules/@react-three/')) {
            return 'three-vendor';
          }
          // D3 (charting)
          if (id.includes('node_modules/d3')) {
            return 'd3-vendor';
          }
          // Skulpt (Python interpreter - very large)
          if (id.includes('node_modules/skulpt/')) {
            return 'skulpt-vendor';
          }
          // HDF5 WebAssembly
          if (id.includes('node_modules/h5wasm/')) {
            return 'h5wasm-vendor';
          }
          // Export utilities (gif.js, jszip)
          if (id.includes('node_modules/gif.js/') ||
              id.includes('node_modules/jszip/')) {
            return 'export-vendor';
          }
          // Radix UI components
          if (id.includes('node_modules/@radix-ui/')) {
            return 'radix-vendor';
          }
          // Zustand state management
          if (id.includes('node_modules/zustand/')) {
            return 'zustand-vendor';
          }
          // Lucide icons
          if (id.includes('node_modules/lucide-react/')) {
            return 'icons-vendor';
          }
        },
      },
    },
  },
  test: {
    environment: 'happy-dom',
    globals: true,
  },
})
