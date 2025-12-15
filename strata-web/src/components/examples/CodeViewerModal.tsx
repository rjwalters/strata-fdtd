import { useState, useEffect } from 'react'
import { Button } from '@strata/ui'
import { X, Copy, Check, Download, Code2 } from 'lucide-react'

interface CodeViewerModalProps {
  isOpen: boolean
  onClose: () => void
  title: string
  filename: string
}

export function CodeViewerModal({
  isOpen,
  onClose,
  title,
  filename,
}: CodeViewerModalProps) {
  const [code, setCode] = useState<string>('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    if (!isOpen || !filename) return

    const loadCode = async () => {
      setIsLoading(true)
      setError(null)
      try {
        const response = await fetch(`/examples/${filename}`)
        if (!response.ok) {
          throw new Error(`Failed to load ${filename}`)
        }
        const text = await response.text()
        setCode(text)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load code')
      } finally {
        setIsLoading(false)
      }
    }

    loadCode()
  }, [isOpen, filename])

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      console.error('Failed to copy to clipboard')
    }
  }

  const handleDownload = () => {
    const blob = new Blob([code], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative bg-card border border-border rounded-lg shadow-2xl w-full max-w-4xl max-h-[85vh] mx-4 flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-border">
          <div className="flex items-center gap-3">
            <Code2 className="h-5 w-5 text-primary" />
            <div>
              <h2 className="text-lg font-semibold text-foreground">{title}</h2>
              <p className="text-sm text-muted-foreground">{filename}</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleCopy}
              disabled={isLoading || !!error}
              className="gap-2"
            >
              {copied ? (
                <>
                  <Check className="h-4 w-4" />
                  Copied
                </>
              ) : (
                <>
                  <Copy className="h-4 w-4" />
                  Copy
                </>
              )}
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={handleDownload}
              disabled={isLoading || !!error}
              className="gap-2"
            >
              <Download className="h-4 w-4" />
              Download
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={onClose}
              className="h-8 w-8 p-0"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-4">
          {isLoading ? (
            <div className="flex items-center justify-center h-64">
              <div className="animate-pulse text-muted-foreground">
                Loading code...
              </div>
            </div>
          ) : error ? (
            <div className="flex items-center justify-center h-64">
              <div className="text-destructive">{error}</div>
            </div>
          ) : (
            <pre className="bg-secondary/30 rounded-lg p-4 overflow-x-auto">
              <code className="text-sm font-mono text-foreground whitespace-pre">
                {code}
              </code>
            </pre>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-border bg-secondary/20">
          <p className="text-xs text-muted-foreground">
            Run this script with:{' '}
            <code className="bg-secondary px-1.5 py-0.5 rounded">
              python {filename}
            </code>
          </p>
        </div>
      </div>
    </div>
  )
}
