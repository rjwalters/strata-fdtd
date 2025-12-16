/**
 * Template insertion toolbar component
 */

import { Button, templates, Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@strata/ui'
import { Box, Circle, Zap, Radio, MapPin } from 'lucide-react'
import { useBuilderStore } from '../../stores/builderStore'

interface TemplateButtonProps {
  onClick: () => void
  icon: React.ReactNode
  label: string
  tooltip: string
}

function TemplateButton({ onClick, icon, label, tooltip }: TemplateButtonProps) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          variant="outline"
          size="sm"
          className="gap-1.5"
          onClick={onClick}
        >
          {icon}
          {label}
        </Button>
      </TooltipTrigger>
      <TooltipContent>{tooltip}</TooltipContent>
    </Tooltip>
  )
}

export function TemplateBar() {
  const ast = useBuilderStore((s) => s.ast)
  const insertTemplate = useBuilderStore((s) => s.insertTemplate)

  const handleInsertTemplate = (templateKey: keyof typeof templates) => {
    const template = templates[templateKey]
    const code = template.generate(ast?.grid ?? null)
    insertTemplate(code)
  }

  return (
    <TooltipProvider delayDuration={300}>
      <div className="flex gap-2 items-center bg-gray-800 p-2 rounded-md">
        <span className="text-sm text-gray-400 font-medium mr-2">Templates:</span>

        <TemplateButton
          onClick={() => handleInsertTemplate('rectangle')}
          icon={<Box className="h-3.5 w-3.5" />}
          label="Rectangle"
          tooltip="Insert rectangular solid material region"
        />

        <TemplateButton
          onClick={() => handleInsertTemplate('sphere')}
          icon={<Circle className="h-3.5 w-3.5" />}
          label="Sphere"
          tooltip="Insert spherical material region"
        />

        <div className="w-px h-6 bg-gray-600 mx-1" />

        <TemplateButton
          onClick={() => handleInsertTemplate('gaussianPulse')}
          icon={<Zap className="h-3.5 w-3.5" />}
          label="Pulse"
          tooltip="Insert Gaussian pulse excitation source"
        />

        <TemplateButton
          onClick={() => handleInsertTemplate('continuousWave')}
          icon={<Radio className="h-3.5 w-3.5" />}
          label="CW"
          tooltip="Insert continuous wave source"
        />

        <div className="w-px h-6 bg-gray-600 mx-1" />

        <TemplateButton
          onClick={() => handleInsertTemplate('probe')}
          icon={<MapPin className="h-3.5 w-3.5" />}
          label="Probe"
          tooltip="Insert pressure recording probe"
        />
      </div>
    </TooltipProvider>
  )
}
