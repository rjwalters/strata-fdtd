import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { installDebugLogger, perfStart, perfEnd } from './lib/debugLogger'
import { ErrorBoundary } from './components/ErrorBoundary'
import './index.css'

// Install debug logger FIRST - before any other code runs
installDebugLogger({
  autoWriteOnError: true,
  persistToLocalStorage: true,
})

console.log('[main] Debug logger installed')
console.log('[main] Starting application...')
console.log('[main] Environment:', {
  mode: import.meta.env.MODE,
  dev: import.meta.env.DEV,
  prod: import.meta.env.PROD,
  base: import.meta.env.BASE_URL,
})

perfStart('app-init')

// Dynamically import App to catch any top-level import errors
console.log('[main] Importing App component...')

import('./App.tsx')
  .then(({ default: App }) => {
    console.log('[main] App component imported successfully')
    perfEnd('app-init')

    perfStart('react-render')

    const rootElement = document.getElementById('root')
    if (!rootElement) {
      console.error('[main] FATAL: Root element not found!')
      throw new Error('Root element #root not found in document')
    }

    console.log('[main] Root element found, creating React root...')

    const root = createRoot(rootElement)

    console.log('[main] Rendering React app...')

    root.render(
      <StrictMode>
        <ErrorBoundary
          onError={(error, errorInfo) => {
            console.error('[main] React error boundary caught error:', error.message)
            console.error('[main] Component stack:', errorInfo.componentStack)
          }}
        >
          <BrowserRouter>
            <App />
          </BrowserRouter>
        </ErrorBoundary>
      </StrictMode>
    )

    // Log when React has finished initial render
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        perfEnd('react-render')
        console.log('[main] React render complete')
      })
    })
  })
  .catch((error) => {
    console.error('[main] FATAL: Failed to import App:', error)
    perfEnd('app-init')

    // Show error in DOM
    const rootElement = document.getElementById('root')
    if (rootElement) {
      rootElement.innerHTML = `
        <div style="
          height: 100vh;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          background: #0a0a0a;
          color: #fff;
          font-family: system-ui, sans-serif;
          padding: 2rem;
        ">
          <h1 style="color: #f87171; margin-bottom: 1rem;">Failed to Load Application</h1>
          <pre style="
            background: #1f1f1f;
            padding: 1rem;
            border-radius: 0.5rem;
            max-width: 80%;
            overflow: auto;
            font-size: 0.875rem;
          ">${error.stack || error.message}</pre>
          <p style="margin-top: 1rem; color: #888;">
            Check the console for more details. You can access debug logs with:
            <code style="background: #333; padding: 0.25rem 0.5rem; border-radius: 0.25rem;">
              debugLogger.getLogs()
            </code>
          </p>
          <button
            onclick="debugLogger.downloadLogs()"
            style="
              margin-top: 1rem;
              padding: 0.5rem 1rem;
              background: #3b82f6;
              border: none;
              border-radius: 0.5rem;
              color: white;
              cursor: pointer;
            "
          >
            Download Debug Logs
          </button>
        </div>
      `
    }
  })
