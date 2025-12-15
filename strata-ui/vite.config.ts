/// <reference types="vitest" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import dts from 'vite-plugin-dts'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    dts({
      insertTypesEntry: true,
      include: ['src'],
      exclude: ['src/**/__tests__/**/*'],
    }),
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  worker: {
    // Inline workers to avoid import.meta.url issues in library mode
    format: 'es',
  },
  build: {
    lib: {
      entry: path.resolve(__dirname, 'src/index.ts'),
      name: 'StrataUI',
      formats: ['es'],
      fileName: 'index',
    },
    rollupOptions: {
      // Externalize peer dependencies
      external: [
        'react',
        'react-dom',
        'react/jsx-runtime',
        'react-router-dom',
      ],
      output: {
        globals: {
          react: 'React',
          'react-dom': 'ReactDOM',
          'react/jsx-runtime': 'jsxRuntime',
          'react-router-dom': 'ReactRouterDOM',
        },
      },
    },
    sourcemap: true,
  },
  test: {
    environment: 'happy-dom',
    globals: true,
  },
})
