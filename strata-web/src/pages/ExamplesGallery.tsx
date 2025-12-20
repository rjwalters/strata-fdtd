import { useState, useEffect } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { Button, Badge } from '@strata/ui'
import {
  ArrowRight,
  Waves,
  Volume2,
  Code2,
  BookOpen,
  Zap,
  Target,
  Radio,
  Grid3x3,
  Eye,
} from 'lucide-react'
import { CodeViewerModal } from '../components/examples'

// =============================================================================
// Types
// =============================================================================

interface Demo {
  id: string
  title: string
  description: string
  category: string
}

interface ExampleScript {
  id: string
  title: string
  description: string
  file: string
  thumbnail: string
  category: string
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced'
  tags: string[]
  estimatedRuntime: string
  estimatedSize: string
  gridSize: string
  features: string[]
}

// =============================================================================
// Data
// =============================================================================

const DEMOS: Demo[] = [
  {
    id: 'open-pipe',
    title: 'Open Pipe',
    description:
      'Acoustic simulation of an open-ended pipe showing standing wave patterns and harmonics.',
    category: 'Acoustics',
  },
  {
    id: 'closed-pipe',
    title: 'Closed Pipe',
    description:
      'Acoustic simulation of a closed pipe demonstrating odd harmonic patterns.',
    category: 'Acoustics',
  },
  {
    id: 'half-open-pipe',
    title: 'Half-Open Pipe',
    description:
      'Acoustic simulation showing the behavior of a pipe with one open and one closed end.',
    category: 'Acoustics',
  },
]

const CATEGORY_ICONS: Record<string, React.ReactNode> = {
  Acoustics: <Volume2 className="h-4 w-4" />,
  Basics: <BookOpen className="h-4 w-4" />,
  Physics: <Zap className="h-4 w-4" />,
  Transducers: <Radio className="h-4 w-4" />,
  Metamaterials: <Target className="h-4 w-4" />,
  Imaging: <Eye className="h-4 w-4" />,
  Advanced: <Grid3x3 className="h-4 w-4" />,
}

const DIFFICULTY_COLORS: Record<string, string> = {
  Beginner: 'bg-green-500/20 text-green-400 border-green-500/30',
  Intermediate: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
  Advanced: 'bg-red-500/20 text-red-400 border-red-500/30',
}

// =============================================================================
// Component
// =============================================================================

export function ExamplesGallery() {
  const navigate = useNavigate()
  const [exampleScripts, setExampleScripts] = useState<ExampleScript[]>([])
  const [codeModal, setCodeModal] = useState<{
    isOpen: boolean
    title: string
    filename: string
  }>({ isOpen: false, title: '', filename: '' })

  // Load example scripts from index.json
  useEffect(() => {
    const loadExamples = async () => {
      try {
        const response = await fetch('/examples/index.json')
        if (response.ok) {
          const data = await response.json()
          setExampleScripts(data.examples || [])
        }
      } catch (err) {
        console.error('Failed to load example scripts:', err)
      }
    }
    loadExamples()
  }, [])

  const handleViewDemo = (demo: Demo) => {
    navigate(`/viewer/${demo.id}`)
  }

  const handleViewCode = (script: ExampleScript) => {
    setCodeModal({
      isOpen: true,
      title: script.title,
      filename: script.file,
    })
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Code Viewer Modal */}
      <CodeViewerModal
        isOpen={codeModal.isOpen}
        onClose={() => setCodeModal({ isOpen: false, title: '', filename: '' })}
        title={codeModal.title}
        filename={codeModal.filename}
      />

      {/* Navigation Header */}
      <nav className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-6 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <Link
                to="/"
                className="flex items-center gap-2 text-foreground hover:text-primary transition-colors"
              >
                <Waves className="h-6 w-6 text-primary" />
                <span className="font-bold">Strata FDTD</span>
              </Link>
              <div className="hidden sm:flex items-center gap-1">
                <Link
                  to="/examples"
                  className="px-3 py-1.5 text-sm font-medium text-foreground bg-primary/10 rounded-md"
                >
                  Demos
                </Link>
                <Link
                  to="/viewer"
                  className="px-3 py-1.5 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
                >
                  Viewer
                </Link>
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Header */}
      <header className="border-b border-border">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <h1 className="text-3xl font-bold text-foreground">
            Strata FDTD Demos
          </h1>
          <p className="mt-2 text-muted-foreground">
            Explore interactive acoustic simulations and learn to build your own
          </p>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 space-y-12">
        {/* Live Demos Section */}
        <section>
          <div className="mb-6">
            <h2 className="text-2xl font-bold text-foreground">Interactive Demos</h2>
            <p className="mt-1 text-muted-foreground">
              Experience real-time FDTD simulations of acoustic wave behavior in pipes
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {DEMOS.map((demo) => (
              <div
                key={demo.id}
                className="bg-card border border-border rounded-lg overflow-hidden hover:shadow-lg transition-shadow"
              >
                {/* Thumbnail placeholder */}
                <div className="h-48 bg-secondary/30 flex items-center justify-center">
                  <Volume2 className="h-12 w-12 text-muted-foreground" />
                </div>

                {/* Content */}
                <div className="p-5 space-y-4">
                  <div>
                    <h3 className="text-xl font-bold text-foreground">
                      {demo.title}
                    </h3>
                    <p className="mt-1 text-sm text-muted-foreground line-clamp-2">
                      {demo.description}
                    </p>
                  </div>

                  <div className="flex items-center justify-between">
                    <Badge variant="outline">{demo.category}</Badge>
                    <Button
                      onClick={() => handleViewDemo(demo)}
                      className="gap-2"
                    >
                      View <ArrowRight className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Example Scripts Section */}
        <section>
          <div className="mb-6">
            <h2 className="text-2xl font-bold text-foreground">Example Scripts</h2>
            <p className="mt-1 text-muted-foreground">
              Learn to program your own simulations with these Python examples
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {exampleScripts.map((script) => (
              <div
                key={script.id}
                className="bg-card border border-border rounded-lg overflow-hidden hover:shadow-lg transition-shadow"
              >
                {/* Thumbnail */}
                <div className="h-48 bg-secondary/30 flex items-center justify-center overflow-hidden">
                  <img
                    src={`/examples/${script.thumbnail}`}
                    alt={script.title}
                    className="w-full h-full object-contain p-4"
                  />
                </div>

                {/* Content */}
                <div className="p-5 space-y-4">
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <h3 className="text-lg font-bold text-foreground">
                        {script.title}
                      </h3>
                      <Badge
                        variant="outline"
                        className={DIFFICULTY_COLORS[script.difficulty]}
                      >
                        {script.difficulty}
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground line-clamp-2">
                      {script.description}
                    </p>
                  </div>

                  {/* Features */}
                  <div className="flex flex-wrap gap-1">
                    {script.features.slice(0, 2).map((feature) => (
                      <span
                        key={feature}
                        className="text-xs bg-secondary/50 px-2 py-0.5 rounded text-muted-foreground"
                      >
                        {feature}
                      </span>
                    ))}
                    {script.features.length > 2 && (
                      <span className="text-xs text-muted-foreground">
                        +{script.features.length - 2} more
                      </span>
                    )}
                  </div>

                  {/* Meta info */}
                  <div className="flex items-center gap-4 text-xs text-muted-foreground">
                    <span>{script.gridSize}</span>
                    <span>{script.estimatedRuntime}</span>
                  </div>

                  <div className="flex items-center justify-between">
                    <Badge variant="outline" className="gap-1">
                      {CATEGORY_ICONS[script.category]}
                      {script.category}
                    </Badge>
                    <Button
                      variant="outline"
                      onClick={() => handleViewCode(script)}
                      className="gap-2"
                    >
                      <Code2 className="h-4 w-4" />
                      View Code
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-border mt-12">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-muted-foreground">
            <div>
              <a
                href="https://github.com/rjwalters/strata-fdtd"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-foreground transition-colors"
              >
                Strata FDTD
              </a>
              {' '}— Open source acoustic simulation
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}
