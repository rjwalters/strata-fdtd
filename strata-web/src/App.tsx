import { lazy, Suspense } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'

// Lazy load pages to enable code splitting
// ViewerPage loads heavy deps: Three.js, h5wasm, D3, etc.
const ViewerPage = lazy(() => import('./pages/ViewerPage'))
const ExamplesGallery = lazy(() => import('./pages/ExamplesGallery').then(m => ({ default: m.ExamplesGallery })))

function PageLoader() {
  return (
    <div className="h-screen w-screen flex items-center justify-center bg-background">
      <div className="text-center space-y-4">
        <div className="h-8 w-8 mx-auto border-2 border-primary border-t-transparent rounded-full animate-spin" />
        <p className="text-sm text-muted-foreground">Loading...</p>
      </div>
    </div>
  )
}

/**
 * Strata Web - FDTD Simulation Viewer
 *
 * Web-only version for viewing simulation results on Cloudflare Pages.
 * For simulation building, use the desktop app (strata-app).
 */
export default function App() {
  return (
    <Suspense fallback={<PageLoader />}>
      <Routes>
        {/* Default: Examples gallery */}
        <Route path="/" element={<ExamplesGallery />} />

        {/* Viewer: Load simulation from URL or file upload */}
        <Route path="/viewer" element={<ViewerPage />} />
        <Route path="/viewer/:demoId" element={<ViewerPage />} />

        {/* Examples gallery */}
        <Route path="/examples" element={<ExamplesGallery />} />

        {/* Fallback */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Suspense>
  )
}
