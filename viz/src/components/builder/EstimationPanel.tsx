/**
 * Simulation estimation panel component
 */

import { useMemo } from 'react'
import { AlertTriangle } from 'lucide-react'
import { calculateEstimates } from '@/lib/estimations'
import { useBuilderStore } from '@/stores/builderStore'

export function EstimationPanel() {
  const ast = useBuilderStore((s) => s.ast)

  const estimates = useMemo(() => {
    return calculateEstimates(ast?.grid ?? null)
  }, [ast?.grid])

  if (!estimates) {
    return (
      <div className="bg-gray-800 rounded-md p-3">
        <h3 className="text-sm font-medium text-gray-400 mb-2">Estimated Resources</h3>
        <p className="text-xs text-gray-500">Define a grid to see estimates</p>
      </div>
    )
  }

  return (
    <div className="bg-gray-800 rounded-md p-3">
      <h3 className="text-sm font-medium text-gray-300 mb-2">Estimated Resources</h3>

      <div className="space-y-1.5 text-xs">
        <div className="flex justify-between">
          <span className="text-gray-400">Memory:</span>
          <span className="text-gray-200 font-mono">{estimates.memory.formatted}</span>
        </div>

        <div className="flex justify-between">
          <span className="text-gray-400">Timesteps:</span>
          <span className="text-gray-200 font-mono">~{estimates.timesteps.toLocaleString()}</span>
        </div>

        <div className="flex justify-between">
          <span className="text-gray-400">Runtime:</span>
          <span className="text-gray-200 font-mono">~{estimates.runtime.formatted}</span>
        </div>
      </div>

      {estimates.warnings.length > 0 && (
        <div className="mt-3 pt-3 border-t border-gray-700">
          {estimates.warnings.map((warning, i) => (
            <div key={i} className="flex gap-2 text-xs text-yellow-400">
              <AlertTriangle className="h-3.5 w-3.5 flex-shrink-0 mt-0.5" />
              <span>{warning}</span>
            </div>
          ))}
        </div>
      )}

      <p className="mt-2 pt-2 border-t border-gray-700 text-xs text-gray-500">
        Estimates assume Python backend. Native backend will be ~10-20x faster.
      </p>
    </div>
  )
}
