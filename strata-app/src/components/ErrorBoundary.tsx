/**
 * Error Boundary - Catches React errors and displays helpful debug info
 */

import { Component, type ReactNode } from 'react'
import { debugLogger } from '../lib/debugLogger'

interface Props {
  children: ReactNode
  fallback?: ReactNode
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void
}

interface State {
  hasError: boolean
  error: Error | null
  errorInfo: React.ErrorInfo | null
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, error: null, errorInfo: null }
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    // Log to our debug logger
    debugLogger.error('ErrorBoundary', 'React error caught:', error.message)
    debugLogger.error('ErrorBoundary', 'Component stack:', errorInfo.componentStack)

    this.setState({ errorInfo })
    this.props.onError?.(error, errorInfo)

    // Auto-write logs on React errors
    debugLogger.writeLogsToFile().catch(() => {
      // Silently fail
    })
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null, errorInfo: null })
  }

  handleDownloadLogs = () => {
    debugLogger.downloadLogs()
  }

  handleWriteLogs = async () => {
    try {
      const filename = await debugLogger.writeLogsToFile()
      alert(`Logs written to Desktop/${filename}`)
    } catch (err) {
      alert(`Failed to write logs: ${err}`)
    }
  }

  handleCopyError = () => {
    const { error, errorInfo } = this.state
    const text = [
      '=== ERROR REPORT ===',
      `Error: ${error?.name}: ${error?.message}`,
      '',
      '=== STACK TRACE ===',
      error?.stack || 'No stack trace',
      '',
      '=== COMPONENT STACK ===',
      errorInfo?.componentStack || 'No component stack',
      '',
      '=== RECENT LOGS ===',
      debugLogger.getLogsAsText({ levels: ['error', 'warn'] }),
    ].join('\n')

    navigator.clipboard.writeText(text).then(() => {
      alert('Error details copied to clipboard')
    })
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }

      const { error, errorInfo } = this.state

      return (
        <div className="h-screen w-screen bg-gray-950 text-white p-8 overflow-auto">
          <div className="max-w-4xl mx-auto">
            <h1 className="text-2xl font-bold text-red-400 mb-4">
              Something went wrong
            </h1>

            <div className="bg-red-950/30 border border-red-800 rounded-lg p-4 mb-6">
              <h2 className="text-lg font-semibold text-red-300 mb-2">
                {error?.name}: {error?.message}
              </h2>
              <pre className="text-sm text-red-200 whitespace-pre-wrap font-mono overflow-x-auto">
                {error?.stack}
              </pre>
            </div>

            {errorInfo?.componentStack && (
              <div className="bg-gray-900 border border-gray-700 rounded-lg p-4 mb-6">
                <h3 className="text-sm font-semibold text-gray-300 mb-2">
                  Component Stack
                </h3>
                <pre className="text-xs text-gray-400 whitespace-pre-wrap font-mono">
                  {errorInfo.componentStack}
                </pre>
              </div>
            )}

            <div className="flex gap-3 mb-6">
              <button
                onClick={this.handleRetry}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg font-medium transition-colors"
              >
                Try Again
              </button>
              <button
                onClick={this.handleCopyError}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg font-medium transition-colors"
              >
                Copy Error Details
              </button>
              <button
                onClick={this.handleDownloadLogs}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg font-medium transition-colors"
              >
                Download Logs
              </button>
              <button
                onClick={this.handleWriteLogs}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg font-medium transition-colors"
              >
                Save to Desktop
              </button>
            </div>

            <div className="bg-gray-900 border border-gray-700 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-gray-300 mb-2">
                Debug Info
              </h3>
              <div className="text-xs text-gray-400 space-y-1">
                <p>Time: {new Date().toISOString()}</p>
                <p>URL: {window.location.href}</p>
                <p>User Agent: {navigator.userAgent}</p>
              </div>
            </div>

            <p className="mt-6 text-sm text-gray-500">
              Debug logs have been automatically saved. You can access them from the DevTools console
              using <code className="bg-gray-800 px-1 rounded">debugLogger.getLogs()</code>
            </p>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}
