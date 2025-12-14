import { useState, useEffect } from 'react'

interface Example {
  id: string
  title: string
  description: string
  file: string
  thumbnail: string
  category: string
  difficulty: string
  tags: string[]
  estimatedRuntime: string
  estimatedSize: string
  gridSize: string
  features: string[]
}

interface ExamplesGalleryProps {
  onLoad: (script: string) => void
}

export function ExamplesGallery({ onLoad }: ExamplesGalleryProps) {
  const [examples, setExamples] = useState<Example[]>([])
  const [selectedCategory, setSelectedCategory] = useState<string>('All')
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  const [viewingCode, setViewingCode] = useState<Example | null>(null)
  const [codeContent, setCodeContent] = useState<string>('')
  const [loadError, setLoadError] = useState<string | null>(null)
  const [codeLoading, setCodeLoading] = useState<boolean>(false)

  useEffect(() => {
    // Load examples metadata
    fetch('/examples/index.json')
      .then(res => {
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`)
        }
        return res.json()
      })
      .then(data => {
        setExamples(data.examples)
        setLoading(false)
      })
      .catch(err => {
        console.error('Error loading examples:', err)
        setError('Failed to load examples gallery')
        setLoading(false)
      })
  }, [])

  const categories = ['All', ...new Set(examples.map(e => e.category))]

  const filtered = selectedCategory === 'All'
    ? examples
    : examples.filter(e => e.category === selectedCategory)

  // Shared function to fetch example code
  const fetchExampleCode = async (example: Example): Promise<string> => {
    const response = await fetch(`/examples/${example.file}`)
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    return response.text()
  }

  const handleLoad = async (example: Example) => {
    try {
      setLoadError(null)
      const script = await fetchExampleCode(example)
      onLoad(script)
    } catch (err) {
      console.error('Error loading example script:', err)
      setLoadError('Failed to load example script. Please try again.')
    }
  }

  const handleViewCode = async (example: Example) => {
    try {
      setLoadError(null)
      setCodeLoading(true)
      const code = await fetchExampleCode(example)
      setCodeContent(code)
      setViewingCode(example)
    } catch (err) {
      console.error('Error loading code:', err)
      setLoadError('Failed to load code. Please try again.')
    } finally {
      setCodeLoading(false)
    }
  }

  const closeCodeModal = () => {
    setViewingCode(null)
    setCodeContent('')
  }

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'Beginner':
        return 'bg-green-100 text-green-800 border-green-300'
      case 'Intermediate':
        return 'bg-yellow-100 text-yellow-800 border-yellow-300'
      case 'Advanced':
        return 'bg-red-100 text-red-800 border-red-300'
      default:
        return 'bg-gray-100 text-gray-800 border-gray-300'
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-lg text-gray-600">Loading examples...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-lg text-red-600">{error}</div>
      </div>
    )
  }

  return (
    <div className="w-full space-y-6 p-6">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-3xl font-bold text-gray-900">📚 Example Simulations</h1>
        <p className="text-gray-600">
          Explore curated examples to learn FDTD simulation concepts and get started quickly
        </p>
      </div>

      {/* Error notification */}
      {loadError && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
          <span className="block sm:inline">{loadError}</span>
          <button
            onClick={() => setLoadError(null)}
            className="absolute top-0 bottom-0 right-0 px-4 py-3 hover:text-red-900"
            aria-label="Close error notification"
          >
            <span className="text-2xl">&times;</span>
          </button>
        </div>
      )}

      {/* Category filters */}
      <div className="flex flex-wrap gap-2">
        {categories.map(cat => (
          <button
            key={cat}
            onClick={() => setSelectedCategory(cat)}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              selectedCategory === cat
                ? 'bg-blue-600 text-white shadow-md'
                : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50'
            }`}
          >
            {cat}
          </button>
        ))}
      </div>

      {/* Examples grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filtered.map(example => (
          <div
            key={example.id}
            className="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition-shadow border border-gray-200"
          >
            {/* Thumbnail */}
            <div className="relative h-48 bg-gray-100">
              <img
                src={`/examples/${example.thumbnail}`}
                alt={example.title}
                className="w-full h-full object-cover"
                onError={(e) => {
                  // Fallback if image fails to load
                  const target = e.target as HTMLImageElement
                  target.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="400" height="300"%3E%3Crect fill="%23f0f0f0" width="400" height="300"/%3E%3Ctext x="200" y="150" font-family="Arial" font-size="20" text-anchor="middle" fill="%23999"%3ENo preview%3C/text%3E%3C/svg%3E'
                }}
              />
            </div>

            {/* Content */}
            <div className="p-5 space-y-3">
              {/* Title */}
              <h3 className="text-xl font-bold text-gray-900">{example.title}</h3>

              {/* Description */}
              <p className="text-sm text-gray-600 line-clamp-3">{example.description}</p>

              {/* Badges */}
              <div className="flex flex-wrap gap-2">
                <span className="px-2 py-1 text-xs font-medium rounded-md bg-blue-100 text-blue-800 border border-blue-300">
                  {example.category}
                </span>
                <span className={`px-2 py-1 text-xs font-medium rounded-md border ${getDifficultyColor(example.difficulty)}`}>
                  {example.difficulty}
                </span>
              </div>

              {/* Stats */}
              <div className="space-y-1 text-sm text-gray-600">
                <div className="flex items-center">
                  <span className="w-4">⏱</span>
                  <span>{example.estimatedRuntime}</span>
                </div>
                <div className="flex items-center">
                  <span className="w-4">💾</span>
                  <span>{example.estimatedSize}</span>
                </div>
                <div className="flex items-center">
                  <span className="w-4">📐</span>
                  <span>{example.gridSize}</span>
                </div>
              </div>

              {/* Features */}
              <div className="text-xs">
                <strong className="text-gray-700">Features:</strong>
                <ul className="list-disc list-inside mt-1 space-y-0.5 text-gray-600">
                  {example.features.slice(0, 3).map((f, i) => (
                    <li key={i} className="line-clamp-1">{f}</li>
                  ))}
                </ul>
              </div>

              {/* Actions */}
              <div className="flex gap-2 pt-2">
                <button
                  onClick={() => handleViewCode(example)}
                  className="flex-1 px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  View Code
                </button>
                <button
                  onClick={() => handleLoad(example)}
                  className="flex-1 px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Load
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Code viewing modal */}
      {viewingCode && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
          onClick={closeCodeModal}
          role="dialog"
          aria-modal="true"
          aria-labelledby="modal-title"
          onKeyDown={(e) => {
            if (e.key === 'Escape') closeCodeModal()
          }}
        >
          <div
            className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal header */}
            <div className="flex items-center justify-between p-6 border-b border-gray-200">
              <h2 id="modal-title" className="text-2xl font-bold text-gray-900">{viewingCode.title}</h2>
              <button
                onClick={closeCodeModal}
                className="text-gray-500 hover:text-gray-700 text-2xl font-bold"
                aria-label="Close modal"
              >
                ×
              </button>
            </div>

            {/* Code content */}
            <div className="p-6 overflow-auto max-h-[70vh]">
              {codeLoading ? (
                <div className="flex items-center justify-center py-12">
                  <div className="text-lg text-gray-600">Loading code...</div>
                </div>
              ) : (
                <pre className="bg-gray-50 p-4 rounded-lg overflow-x-auto">
                  <code className="text-sm font-mono text-gray-800">{codeContent}</code>
                </pre>
              )}
            </div>

            {/* Modal footer */}
            <div className="flex justify-end gap-3 p-6 border-t border-gray-200 bg-gray-50">
              <button
                onClick={closeCodeModal}
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50"
              >
                Close
              </button>
              <button
                onClick={() => {
                  handleLoad(viewingCode)
                  closeCodeModal()
                }}
                className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700"
              >
                Load Example
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Empty state */}
      {filtered.length === 0 && (
        <div className="text-center py-12">
          <p className="text-lg text-gray-600">No examples found in this category</p>
        </div>
      )}
    </div>
  )
}
