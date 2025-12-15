/**
 * Error panel component for displaying Python validation errors
 */

import { AlertCircle, AlertTriangle, X } from 'lucide-react'
import type { ValidationError } from '../../lib/pythonValidator'

interface ErrorPanelProps {
  errors: ValidationError[]
  onClose?: () => void
}

export function ErrorPanel({ errors, onClose }: ErrorPanelProps) {
  if (errors.length === 0) {
    return null
  }

  const errorCount = errors.filter(e => e.severity === 'error').length
  const warningCount = errors.filter(e => e.severity === 'warning').length

  return (
    <div className="border-t border-gray-700 bg-gray-900">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-gray-700 bg-gray-800 px-4 py-2">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-semibold text-white">Validation Issues</h3>
          <div className="flex gap-2 text-xs">
            {errorCount > 0 && (
              <span className="flex items-center gap-1 text-red-400">
                <AlertCircle className="h-3 w-3" />
                {errorCount} {errorCount === 1 ? 'error' : 'errors'}
              </span>
            )}
            {warningCount > 0 && (
              <span className="flex items-center gap-1 text-yellow-400">
                <AlertTriangle className="h-3 w-3" />
                {warningCount} {warningCount === 1 ? 'warning' : 'warnings'}
              </span>
            )}
          </div>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
            title="Close error panel"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      {/* Error list */}
      <div className="max-h-48 overflow-y-auto">
        {errors.map((error, index) => (
          <div
            key={index}
            className={`
              flex items-start gap-3 border-b border-gray-700 px-4 py-3
              hover:bg-gray-800 transition-colors cursor-pointer
              ${error.severity === 'error' ? 'bg-red-950/20' : 'bg-yellow-950/20'}
            `}
          >
            {/* Icon */}
            <div className="flex-shrink-0 pt-0.5">
              {error.severity === 'error' ? (
                <AlertCircle className="h-4 w-4 text-red-400" />
              ) : (
                <AlertTriangle className="h-4 w-4 text-yellow-400" />
              )}
            </div>

            {/* Error details */}
            <div className="flex-1 min-w-0">
              <div className="flex items-baseline gap-2">
                <span className="text-xs font-mono text-gray-400">
                  Line {error.line}:{error.column}
                </span>
                <span
                  className={`
                    text-xs font-semibold uppercase
                    ${error.severity === 'error' ? 'text-red-400' : 'text-yellow-400'}
                  `}
                >
                  {error.severity}
                </span>
              </div>
              <p className="mt-1 text-sm text-gray-200">{error.message}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
