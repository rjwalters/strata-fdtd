import { useState, useEffect } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import {
  Button,
  Badge,
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@strata/ui'
import { Waves, Volume2 } from 'lucide-react'
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

type Difficulty = 'Beginner' | 'Intermediate' | 'Advanced'

const DIFFICULTY_VARIANT: Record<Difficulty, 'default' | 'secondary' | 'destructive'> = {
  Beginner: 'default',
  Intermediate: 'secondary',
  Advanced: 'destructive',
}

// =============================================================================
// Sub-components
// =============================================================================

// Pipe geometry SVG illustrations
function OpenPipeSvg() {
  return (
    <svg viewBox="0 0 200 80" className="w-48 h-auto">
      {/* Pipe walls */}
      <rect x="20" y="10" width="160" height="8" fill="currentColor" className="text-muted-foreground" />
      <rect x="20" y="62" width="160" height="8" fill="currentColor" className="text-muted-foreground" />
      {/* Wave inside */}
      <path
        d="M 30 40 Q 50 25, 70 40 T 110 40 T 150 40 T 190 40"
        fill="none"
        stroke="currentColor"
        strokeWidth="3"
        className="text-blue-400"
      />
      {/* Open ends indicated by gaps */}
    </svg>
  )
}

function ClosedPipeSvg() {
  return (
    <svg viewBox="0 0 200 80" className="w-48 h-auto">
      {/* Pipe walls */}
      <rect x="20" y="10" width="160" height="8" fill="currentColor" className="text-muted-foreground" />
      <rect x="20" y="62" width="160" height="8" fill="currentColor" className="text-muted-foreground" />
      {/* Closed ends */}
      <rect x="12" y="10" width="8" height="60" fill="currentColor" className="text-muted-foreground" />
      <rect x="180" y="10" width="8" height="60" fill="currentColor" className="text-muted-foreground" />
      {/* Wave inside */}
      <path
        d="M 30 40 Q 50 25, 70 40 T 110 40 T 150 40"
        fill="none"
        stroke="currentColor"
        strokeWidth="3"
        className="text-blue-400"
      />
    </svg>
  )
}

function HalfOpenPipeSvg() {
  return (
    <svg viewBox="0 0 200 80" className="w-48 h-auto">
      {/* Pipe walls */}
      <rect x="20" y="10" width="160" height="8" fill="currentColor" className="text-muted-foreground" />
      <rect x="20" y="62" width="160" height="8" fill="currentColor" className="text-muted-foreground" />
      {/* Closed left end */}
      <rect x="12" y="10" width="8" height="60" fill="currentColor" className="text-muted-foreground" />
      {/* Wave inside */}
      <path
        d="M 30 40 Q 60 20, 90 40 T 150 40"
        fill="none"
        stroke="currentColor"
        strokeWidth="3"
        className="text-blue-400"
      />
      {/* Open right end indicated by gap */}
    </svg>
  )
}

const PIPE_ILLUSTRATIONS: Record<string, React.ReactNode> = {
  'open-pipe': <OpenPipeSvg />,
  'closed-pipe': <ClosedPipeSvg />,
  'half-open-pipe': <HalfOpenPipeSvg />,
}

interface DemoCardProps {
  demo: Demo
  onView: () => void
}

function DemoCard({ demo, onView }: DemoCardProps) {
  return (
    <Card className="overflow-hidden hover:shadow-lg transition-shadow h-full">
      {/* Thumbnail */}
      <div className="h-48 bg-secondary/30 flex items-center justify-center -mt-6">
        {PIPE_ILLUSTRATIONS[demo.id] || <Volume2 className="h-12 w-12 text-muted-foreground" />}
      </div>

      <CardHeader className="flex-1">
        <CardTitle className="text-xl">{demo.title}</CardTitle>
        <CardDescription className="line-clamp-2">
          {demo.description}
        </CardDescription>
      </CardHeader>

      <CardFooter className="justify-between mt-auto">
        <Badge variant="secondary">{demo.category}</Badge>
        <Button
          onClick={onView}
          variant="outline"
          size="sm"
          className="gap-2 px-4 whitespace-nowrap shrink-0 hover:bg-primary hover:text-primary-foreground hover:border-primary transition-colors"
        >
          View&nbsp;Demo
        </Button>
      </CardFooter>
    </Card>
  )
}

interface ViewCodeButtonProps {
  onClick: () => void
}

function ViewCodeButton({ onClick }: ViewCodeButtonProps) {
  return (
    <Button
      variant="outline"
      size="sm"
      onClick={onClick}
      className="gap-2 px-4 whitespace-nowrap shrink-0 hover:bg-primary hover:text-primary-foreground hover:border-primary transition-colors"
    >
View&nbsp;Code
    </Button>
  )
}

interface ScriptCardProps {
  script: ExampleScript
  onViewCode: () => void
}

function ScriptCard({ script, onViewCode }: ScriptCardProps) {
  return (
    <Card className="overflow-hidden hover:shadow-lg transition-shadow h-full">
      {/* Thumbnail */}
      <div className="h-48 bg-secondary/30 flex items-center justify-center overflow-hidden -mt-6">
        <img
          src={`/examples/${script.thumbnail}`}
          alt={script.title}
          className="w-full h-full object-contain p-4"
        />
      </div>

      <CardHeader className="pb-2">
        <div className="flex items-center gap-2 flex-wrap">
          <CardTitle className="text-lg">{script.title}</CardTitle>
          <Badge variant={DIFFICULTY_VARIANT[script.difficulty]}>
            {script.difficulty}
          </Badge>
        </div>
        <CardDescription className="line-clamp-2">
          {script.description}
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-3 flex-1">
        {/* Features */}
        <div className="flex flex-wrap gap-1">
          {script.features.slice(0, 2).map((feature) => (
            <Badge key={feature} variant="outline" className="text-xs font-normal">
              {feature}
            </Badge>
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
      </CardContent>

      <CardFooter className="justify-between mt-auto">
        <Badge variant="secondary">{script.category}</Badge>
        <ViewCodeButton onClick={onViewCode} />
      </CardFooter>
    </Card>
  )
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
              <DemoCard
                key={demo.id}
                demo={demo}
                onView={() => handleViewDemo(demo)}
              />
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
              <ScriptCard
                key={script.id}
                script={script}
                onViewCode={() => handleViewCode(script)}
              />
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
