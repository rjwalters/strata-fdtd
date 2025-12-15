/**
 * Template insertion toolbar component
 */

import { Button, templates } from '@strata/ui'
import { Box, Circle, Zap, Radio, MapPin } from 'lucide-react'
import { useBuilderStore } from '../../stores/builderStore'

export function TemplateBar() {
  const ast = useBuilderStore((s) => s.ast)
  const insertTemplate = useBuilderStore((s) => s.insertTemplate)

  const handleInsertTemplate = (templateKey: keyof typeof templates) => {
    const template = templates[templateKey]
    const code = template.generate(ast?.grid ?? null)
    insertTemplate(code)
  }

  return (
    <div className="flex gap-2 items-center bg-gray-800 p-2 rounded-md">
      <span className="text-sm text-gray-400 font-medium mr-2">Templates:</span>

      <Button
        variant="outline"
        size="sm"
        className="gap-1.5"
        onClick={() => handleInsertTemplate('rectangle')}
        title="Add rectangular material region"
      >
        <Box className="h-3.5 w-3.5" />
        Rectangle
      </Button>

      <Button
        variant="outline"
        size="sm"
        className="gap-1.5"
        onClick={() => handleInsertTemplate('sphere')}
        title="Add spherical material region"
      >
        <Circle className="h-3.5 w-3.5" />
        Sphere
      </Button>

      <div className="w-px h-6 bg-gray-600 mx-1" />

      <Button
        variant="outline"
        size="sm"
        className="gap-1.5"
        onClick={() => handleInsertTemplate('gaussianPulse')}
        title="Add Gaussian pulse source"
      >
        <Zap className="h-3.5 w-3.5" />
        Pulse
      </Button>

      <Button
        variant="outline"
        size="sm"
        className="gap-1.5"
        onClick={() => handleInsertTemplate('continuousWave')}
        title="Add continuous wave source"
      >
        <Radio className="h-3.5 w-3.5" />
        CW
      </Button>

      <div className="w-px h-6 bg-gray-600 mx-1" />

      <Button
        variant="outline"
        size="sm"
        className="gap-1.5"
        onClick={() => handleInsertTemplate('probe')}
        title="Add pressure probe"
      >
        <MapPin className="h-3.5 w-3.5" />
        Probe
      </Button>
    </div>
  )
}
