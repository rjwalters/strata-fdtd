import { Routes, Route, Navigate } from 'react-router-dom'
import { ExamplesGallery } from './pages/ExamplesGallery'
import ViewerPage from './pages/ViewerPage'

/**
 * Strata Web - FDTD Simulation Viewer
 *
 * Web-only version for viewing simulation results on Cloudflare Pages.
 * For simulation building, use the desktop app (strata-app).
 */
export default function App() {
  return (
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
  )
}
