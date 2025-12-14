/**
 * Simulation Builder page - Script editor with live 3D preview
 */

import { ArrowLeft, HelpCircle, Eye, EyeOff, Ruler } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { ScriptEditor } from '@/components/ScriptEditor'
import { Preview3D } from '@/components/builder/Preview3D'
import { TemplateBar } from '@/components/builder/TemplateBar'
import { EstimationPanel } from '@/components/builder/EstimationPanel'
import { ExportBar } from '@/components/builder/ExportBar'
import { ErrorPanel } from '@/components/builder/ErrorPanel'
import { useBuilderStore } from '@/stores/builderStore'

interface SimulationBuilderProps {
  onBack: () => void
}

export function SimulationBuilder({ onBack }: SimulationBuilderProps) {
  const script = useBuilderStore((s) => s.script)
  const ast = useBuilderStore((s) => s.ast)
  const viewOptions = useBuilderStore((s) => s.viewOptions)
  const measurementPoints = useBuilderStore((s) => s.measurementPoints)
  const validationErrors = useBuilderStore((s) => s.validationErrors)
  const setScript = useBuilderStore((s) => s.setScript)
  const setValidationErrors = useBuilderStore((s) => s.setValidationErrors)
  const toggleViewOption = useBuilderStore((s) => s.toggleViewOption)
  const resetToDefault = useBuilderStore((s) => s.resetToDefault)
  const setSliceAxis = useBuilderStore((s) => s.setSliceAxis)
  const setSlicePosition = useBuilderStore((s) => s.setSlicePosition)
  const setMeasurementMode = useBuilderStore((s) => s.setMeasurementMode)

  const handleSave = () => {
    // Trigger download on Ctrl+S
    const hashPrefix = useBuilderStore.getState().scriptHash?.slice(0, 8) ?? 'unknown'
    const filename = `simulation_${hashPrefix}.py`
    const blob = new Blob([script], { type: 'text/x-python' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = filename
    link.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="h-screen flex flex-col bg-gray-950 text-gray-100">
      {/* Header */}
      <header className="flex items-center gap-4 px-4 py-3 bg-gray-900 border-b border-gray-800">
        <Button
          variant="ghost"
          size="sm"
          onClick={onBack}
          className="gap-1.5"
        >
          <ArrowLeft className="h-4 w-4" />
          Back
        </Button>

        <div className="flex-1">
          <h1 className="text-lg font-bold text-gray-100">🔨 Simulation Builder</h1>
          <p className="text-xs text-gray-400">Build FDTD simulations visually</p>
        </div>

        <Button
          variant="outline"
          size="sm"
          onClick={resetToDefault}
        >
          Reset to Default
        </Button>

        <Button
          variant="ghost"
          size="sm"
          title="Help"
        >
          <HelpCircle className="h-4 w-4" />
        </Button>
      </header>

      {/* Main content - Split pane */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left pane - Script editor */}
        <div className="flex-1 flex flex-col border-r border-gray-800">
          <div className="flex-shrink-0 px-4 py-2 bg-gray-900 border-b border-gray-800">
            <h2 className="text-sm font-medium text-gray-300">Script Editor</h2>
          </div>

          <div className="flex-1 overflow-hidden">
            <ScriptEditor
              value={script}
              onChange={setScript}
              onSave={handleSave}
              onValidationChange={setValidationErrors}
            />
          </div>

          {/* Error panel */}
          {validationErrors.length > 0 && (
            <div className="flex-shrink-0">
              <ErrorPanel errors={validationErrors} />
            </div>
          )}

          {/* Template bar */}
          <div className="flex-shrink-0 p-2 bg-gray-900 border-t border-gray-800">
            <TemplateBar />
          </div>
        </div>

        {/* Right pane - Live preview */}
        <div className="flex-1 flex flex-col">
          <div className="flex-shrink-0 px-4 py-2 bg-gray-900 border-b border-gray-800">
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-sm font-medium text-gray-300">Live Preview</h2>

              {/* View options */}
              <div className="flex gap-2">
                <Button
                  variant={viewOptions.showGrid ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => toggleViewOption('showGrid')}
                  className="gap-1.5 text-xs h-7"
                >
                  {viewOptions.showGrid ? <Eye className="h-3 w-3" /> : <EyeOff className="h-3 w-3" />}
                  Grid
                </Button>

                <Button
                  variant={viewOptions.showMaterials ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => toggleViewOption('showMaterials')}
                  className="gap-1.5 text-xs h-7"
                >
                  {viewOptions.showMaterials ? <Eye className="h-3 w-3" /> : <EyeOff className="h-3 w-3" />}
                  Materials
                </Button>

                <Button
                  variant={viewOptions.showSources ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => toggleViewOption('showSources')}
                  className="gap-1.5 text-xs h-7"
                >
                  {viewOptions.showSources ? <Eye className="h-3 w-3" /> : <EyeOff className="h-3 w-3" />}
                  Sources
                </Button>

                <Button
                  variant={viewOptions.showProbes ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => toggleViewOption('showProbes')}
                  className="gap-1.5 text-xs h-7"
                >
                  {viewOptions.showProbes ? <Eye className="h-3 w-3" /> : <EyeOff className="h-3 w-3" />}
                  Probes
                </Button>
              </div>
            </div>

            {/* Slice plane controls */}
            <div className="flex items-center gap-3">
              <span className="text-xs text-gray-400">Slice Plane:</span>
              <Select
                value={viewOptions.sliceAxis}
                onValueChange={(value) => setSliceAxis(value as 'none' | 'xy' | 'xz' | 'yz')}
              >
                <SelectTrigger className="w-24 h-7 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">None</SelectItem>
                  <SelectItem value="xy">XY</SelectItem>
                  <SelectItem value="xz">XZ</SelectItem>
                  <SelectItem value="yz">YZ</SelectItem>
                </SelectContent>
              </Select>

              {viewOptions.sliceAxis !== 'none' && (
                <>
                  <span className="text-xs text-gray-400">Position:</span>
                  <Slider
                    value={[viewOptions.slicePosition * 100]}
                    onValueChange={(value) => setSlicePosition(value[0] / 100)}
                    min={0}
                    max={100}
                    step={1}
                    className="w-32"
                  />
                  <span className="text-xs text-gray-400 w-12">
                    {Math.round(viewOptions.slicePosition * 100)}%
                  </span>

                  {/* Measurement tool */}
                  <Button
                    variant={viewOptions.measurementMode ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setMeasurementMode(!viewOptions.measurementMode)}
                    className="gap-1.5 text-xs h-7"
                  >
                    <Ruler className="h-3 w-3" />
                    Measure
                  </Button>
                </>
              )}
            </div>
          </div>

          {/* 3D preview */}
          <div className="flex-1 overflow-hidden">
            <Preview3D
              ast={ast}
              showGrid={viewOptions.showGrid}
              showMaterials={viewOptions.showMaterials}
              showSources={viewOptions.showSources}
              showProbes={viewOptions.showProbes}
              sliceAxis={viewOptions.sliceAxis}
              slicePosition={viewOptions.slicePosition}
              measurementMode={viewOptions.measurementMode}
              measurementPoints={measurementPoints}
            />
          </div>

          {/* Estimation panel */}
          <div className="flex-shrink-0 p-3 bg-gray-900 border-t border-gray-800">
            <EstimationPanel />
          </div>
        </div>
      </div>

      {/* Footer - Export bar */}
      <footer className="flex-shrink-0 p-2 bg-gray-900 border-t border-gray-800">
        <ExportBar />
      </footer>
    </div>
  )
}

export default SimulationBuilder
