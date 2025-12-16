/**
 * Simulation Builder page - Script editor with live 3D preview
 */

import { useEffect, useMemo, useRef } from 'react'
import { ArrowLeft, Eye, EyeOff, Ruler, Layers, Play, Pause, Grid3x3 } from 'lucide-react'
import { Button, Select, SelectContent, SelectItem, SelectTrigger, SelectValue, Slider, BuilderHelpModal } from '@strata/ui'
import { ScriptEditor, type ScriptEditorHandle } from '../components/ScriptEditor'
import { Preview3D } from '../components/builder/Preview3D'
import { TemplateBar } from '../components/builder/TemplateBar'
import { EstimationPanel } from '../components/builder/EstimationPanel'
import { ExportBar } from '../components/builder/ExportBar'
import { ErrorPanel } from '../components/builder/ErrorPanel'
import { useBuilderStore, type AnimationSpeed } from '../stores/builderStore'

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
  const setDualSliceMode = useBuilderStore((s) => s.setDualSliceMode)
  const setSlice1Position = useBuilderStore((s) => s.setSlice1Position)
  const setSlice2Position = useBuilderStore((s) => s.setSlice2Position)
  const setIsAnimating = useBuilderStore((s) => s.setIsAnimating)
  const setAnimationSpeed = useBuilderStore((s) => s.setAnimationSpeed)

  // Refs
  const scriptEditorRef = useRef<ScriptEditorHandle>(null)
  const animationRef = useRef<number | null>(null)
  const lastTimeRef = useRef<number>(0)

  // Handle error click to jump to line in editor
  const handleErrorClick = (line: number, column: number) => {
    scriptEditorRef.current?.scrollToLine(line, column)
  }

  useEffect(() => {
    if (!viewOptions.isAnimating || viewOptions.sliceAxis === 'none') {
      if (animationRef.current !== null) {
        cancelAnimationFrame(animationRef.current)
        animationRef.current = null
      }
      return
    }

    // Speed: positions per second (0 to 1 range)
    const speedMap: Record<AnimationSpeed, number> = {
      slow: 0.1,    // 10 seconds for full sweep
      normal: 0.25, // 4 seconds for full sweep
      fast: 0.5,    // 2 seconds for full sweep
    }
    const speed = speedMap[viewOptions.animationSpeed]

    const animate = (time: number) => {
      if (lastTimeRef.current === 0) {
        lastTimeRef.current = time
      }

      const deltaTime = (time - lastTimeRef.current) / 1000 // seconds
      lastTimeRef.current = time

      const currentPosition = useBuilderStore.getState().viewOptions.slicePosition
      let newPosition = currentPosition + speed * deltaTime

      // Loop back to start when reaching end
      if (newPosition > 1) {
        newPosition = 0
      }

      setSlicePosition(newPosition)
      animationRef.current = requestAnimationFrame(animate)
    }

    lastTimeRef.current = 0
    animationRef.current = requestAnimationFrame(animate)

    return () => {
      if (animationRef.current !== null) {
        cancelAnimationFrame(animationRef.current)
        animationRef.current = null
      }
    }
  }, [viewOptions.isAnimating, viewOptions.sliceAxis, viewOptions.animationSpeed, setSlicePosition])

  // Calculate absolute position in meters for display
  const slicePositionInMeters = useMemo(() => {
    if (!ast?.grid || viewOptions.sliceAxis === 'none') return null

    const axisIndex = viewOptions.sliceAxis === 'yz' ? 0
                    : viewOptions.sliceAxis === 'xz' ? 1
                    : 2

    return ast.grid.extent[axisIndex] * viewOptions.slicePosition
  }, [ast, viewOptions.sliceAxis, viewOptions.slicePosition])

  // Calculate absolute positions for dual slice mode
  const slice1PositionInMeters = useMemo(() => {
    if (!ast?.grid || viewOptions.sliceAxis === 'none') return null

    const axisIndex = viewOptions.sliceAxis === 'yz' ? 0
                    : viewOptions.sliceAxis === 'xz' ? 1
                    : 2

    return ast.grid.extent[axisIndex] * viewOptions.slice1Position
  }, [ast, viewOptions.sliceAxis, viewOptions.slice1Position])

  const slice2PositionInMeters = useMemo(() => {
    if (!ast?.grid || viewOptions.sliceAxis === 'none') return null

    const axisIndex = viewOptions.sliceAxis === 'yz' ? 0
                    : viewOptions.sliceAxis === 'xz' ? 1
                    : 2

    return ast.grid.extent[axisIndex] * viewOptions.slice2Position
  }, [ast, viewOptions.sliceAxis, viewOptions.slice2Position])

  // Auto-pause animation when user manually adjusts slider
  const handleSliderChange = (value: number[]) => {
    if (viewOptions.isAnimating) {
      setIsAnimating(false)
    }
    setSlicePosition(value[0] / 100)
  }

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

        <BuilderHelpModal />
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
              ref={scriptEditorRef}
              value={script}
              onChange={setScript}
              onSave={handleSave}
              onValidationChange={setValidationErrors}
            />
          </div>

          {/* Error panel */}
          {validationErrors.length > 0 && (
            <div className="flex-shrink-0">
              <ErrorPanel errors={validationErrors} onErrorClick={handleErrorClick} />
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
            <div className="flex items-center gap-3 flex-wrap">
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
                  {/* Dual slice toggle */}
                  <Button
                    variant={viewOptions.dualSliceMode ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setDualSliceMode(!viewOptions.dualSliceMode)}
                    className="gap-1.5 text-xs h-7"
                  >
                    <Layers className="h-3 w-3" />
                    Dual Slice
                  </Button>

                  {/* Slice grid toggle */}
                  <Button
                    variant={viewOptions.showSliceGrid ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => toggleViewOption('showSliceGrid')}
                    className="gap-1.5 text-xs h-7"
                  >
                    <Grid3x3 className="h-3 w-3" />
                    Slice Grid
                  </Button>

                  {viewOptions.dualSliceMode ? (
                    <>
                      {/* Slice 1 position */}
                      <span className="text-xs text-blue-400">Slice 1:</span>
                      <Slider
                        value={[viewOptions.slice1Position * 100]}
                        onValueChange={(value) => setSlice1Position(value[0] / 100)}
                        min={0}
                        max={100}
                        step={1}
                        className="w-24"
                      />
                      <span className="text-xs text-blue-400 w-28">
                        {Math.round(viewOptions.slice1Position * 100)}%
                        {slice1PositionInMeters !== null && (
                          <span className="text-blue-500"> ({slice1PositionInMeters.toFixed(3)} m)</span>
                        )}
                      </span>

                      {/* Slice 2 position */}
                      <span className="text-xs text-pink-400">Slice 2:</span>
                      <Slider
                        value={[viewOptions.slice2Position * 100]}
                        onValueChange={(value) => setSlice2Position(value[0] / 100)}
                        min={0}
                        max={100}
                        step={1}
                        className="w-24"
                      />
                      <span className="text-xs text-pink-400 w-28">
                        {Math.round(viewOptions.slice2Position * 100)}%
                        {slice2PositionInMeters !== null && (
                          <span className="text-pink-500"> ({slice2PositionInMeters.toFixed(3)} m)</span>
                        )}
                      </span>
                    </>
                  ) : (
                    <>
                      {/* Single slice position */}
                      <span className="text-xs text-gray-400">Position:</span>
                      <Slider
                        value={[viewOptions.slicePosition * 100]}
                        onValueChange={handleSliderChange}
                        min={0}
                        max={100}
                        step={1}
                        className="w-32"
                      />
                      <span className="text-xs text-gray-400 w-28">
                        {Math.round(viewOptions.slicePosition * 100)}%
                        {slicePositionInMeters !== null && (
                          <span className="text-gray-500"> ({slicePositionInMeters.toFixed(3)} m)</span>
                        )}
                      </span>

                      {/* Animation controls */}
                      <Button
                        variant={viewOptions.isAnimating ? 'default' : 'outline'}
                        size="sm"
                        onClick={() => setIsAnimating(!viewOptions.isAnimating)}
                        className="gap-1.5 text-xs h-7"
                        title={viewOptions.isAnimating ? 'Pause animation' : 'Animate slice'}
                      >
                        {viewOptions.isAnimating ? <Pause className="h-3 w-3" /> : <Play className="h-3 w-3" />}
                      </Button>

                      <span className="text-xs text-gray-400">Speed:</span>
                      <Select
                        value={viewOptions.animationSpeed}
                        onValueChange={(value) => setAnimationSpeed(value as AnimationSpeed)}
                      >
                        <SelectTrigger className="w-20 h-7 text-xs">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="slow">Slow</SelectItem>
                          <SelectItem value="normal">Normal</SelectItem>
                          <SelectItem value="fast">Fast</SelectItem>
                        </SelectContent>
                      </Select>
                    </>
                  )}

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
              dualSliceMode={viewOptions.dualSliceMode}
              slice1Position={viewOptions.slice1Position}
              slice2Position={viewOptions.slice2Position}
              showSliceGrid={viewOptions.showSliceGrid}
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
