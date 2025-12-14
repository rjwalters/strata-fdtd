/**
 * Export toolbar component for downloading scripts and copying commands
 */

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Download, Copy, Check } from 'lucide-react'
import { useBuilderStore } from '@/stores/builderStore'

export function ExportBar() {
  const script = useBuilderStore((s) => s.script)
  const scriptHash = useBuilderStore((s) => s.scriptHash)
  const validationErrors = useBuilderStore((s) => s.validationErrors)
  const [copied, setCopied] = useState(false)

  // Check if there are any errors (warnings are ok)
  const hasErrors = validationErrors.some(e => e.severity === 'error')

  const handleDownload = () => {
    const hashPrefix = scriptHash?.slice(0, 8) ?? 'unknown'
    const filename = `simulation_${hashPrefix}.py`

    const blob = new Blob([script], { type: 'text/x-python' })
    const url = URL.createObjectURL(blob)

    const link = document.createElement('a')
    link.href = url
    link.download = filename
    link.click()

    URL.revokeObjectURL(url)
  }

  const handleCopyCommand = async () => {
    const hashPrefix = scriptHash?.slice(0, 8) ?? 'unknown'
    const filename = `simulation_${hashPrefix}.py`
    const command = `fdtd-compute ${filename}`

    try {
      await navigator.clipboard.writeText(command)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy command:', err)
    }
  }

  const hashPrefix = scriptHash?.slice(0, 8) ?? '...'

  return (
    <div className="flex gap-2 items-center bg-gray-800 p-2 rounded-md">
      <span className="text-xs text-gray-400 font-mono">
        Hash: <span className="text-gray-300">{hashPrefix}</span>
      </span>

      <div className="flex-1" />

      <Button
        variant="outline"
        size="sm"
        className="gap-1.5"
        onClick={handleDownload}
        disabled={hasErrors}
        title={
          hasErrors
            ? 'Fix syntax errors before downloading'
            : 'Download Python script'
        }
      >
        <Download className="h-3.5 w-3.5" />
        Download Script
      </Button>

      <Button
        variant="outline"
        size="sm"
        className="gap-1.5"
        onClick={handleCopyCommand}
        disabled={hasErrors}
        title={
          hasErrors
            ? 'Fix syntax errors before copying command'
            : 'Copy CLI command to clipboard'
        }
      >
        {copied ? (
          <>
            <Check className="h-3.5 w-3.5" />
            Copied!
          </>
        ) : (
          <>
            <Copy className="h-3.5 w-3.5" />
            Copy Command
          </>
        )}
      </Button>
    </div>
  )
}
