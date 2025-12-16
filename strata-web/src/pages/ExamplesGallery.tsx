import { useState, useEffect } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { Button, Badge } from '@strata/ui'
import {
  ArrowRight,
  Waves,
  Volume2,
  Code2,
  Award,
  Play,
  Eye,
  BookOpen,
  Zap,
  Target,
  Radio,
  Grid3x3,
} from 'lucide-react'
import { LearningPathSelector } from '../components/tutorial'
import { useLearningPath } from '../hooks/useLearningPath'
import { CodeViewerModal } from '../components/examples'
import type { LearningPath } from '../config/learning-paths'

// =============================================================================
// Types
// =============================================================================

interface Demo {
  id: string
  title: string
  description: string
  thumbnail?: string
  category: string
  path: string
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
    path: '/demos/organ-pipes/open-pipe',
  },
  {
    id: 'closed-pipe',
    title: 'Closed Pipe',
    description:
      'Acoustic simulation of a closed pipe demonstrating odd harmonic patterns.',
    category: 'Acoustics',
    path: '/demos/organ-pipes/closed-pipe',
  },
  {
    id: 'half-open-pipe',
    title: 'Half-Open Pipe',
    description:
      'Acoustic simulation showing the behavior of a pipe with one open and one closed end.',
    category: 'Acoustics',
    path: '/demos/organ-pipes/half-open-pipe',
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
  const [activeTab, setActiveTab] = useState<'demos' | 'scripts'>('demos')
  const [selectedCategory, setSelectedCategory] = useState<string>('All')
  const [showLearningPaths, setShowLearningPaths] = useState(true)
  const [exampleScripts, setExampleScripts] = useState<ExampleScript[]>([])
  const [codeModal, setCodeModal] = useState<{
    isOpen: boolean
    title: string
    filename: string
  }>({ isOpen: false, title: '', filename: '' })

  const { paths, progress, startPath, newBadges, clearNewBadges } =
    useLearningPath()

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

  // Get categories based on active tab
  const demoCategories = ['All', ...new Set(DEMOS.map((d) => d.category))]
  const scriptCategories = [
    'All',
    ...new Set(exampleScripts.map((s) => s.category)),
  ]
  const categories = activeTab === 'demos' ? demoCategories : scriptCategories

  // Filter based on category
  const filteredDemos =
    selectedCategory === 'All'
      ? DEMOS
      : DEMOS.filter((d) => d.category === selectedCategory)

  const filteredScripts =
    selectedCategory === 'All'
      ? exampleScripts
      : exampleScripts.filter((s) => s.category === selectedCategory)

  // Reset category when switching tabs
  const handleTabChange = (tab: 'demos' | 'scripts') => {
    setActiveTab(tab)
    setSelectedCategory('All')
  }

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

  const handleSelectPath = (path: LearningPath) => {
    startPath(path.id)
  }

  const handleBrowseAll = () => {
    setShowLearningPaths(false)
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

      {/* New badge notification */}
      {newBadges.length > 0 && (
        <div className="fixed top-4 right-4 z-50 animate-in slide-in-from-top-2">
          <div className="bg-card border border-border rounded-lg p-4 shadow-lg max-w-sm">
            <div className="flex items-center gap-3">
              <div className="text-3xl">{newBadges[0].icon}</div>
              <div className="flex-1">
                <p className="font-semibold text-foreground">Badge Earned!</p>
                <p className="text-sm text-muted-foreground">
                  {newBadges[0].name}
                </p>
              </div>
              <Button variant="ghost" size="sm" onClick={clearNewBadges}>
                Dismiss
              </Button>
            </div>
          </div>
        </div>
      )}

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
                  Examples
                </Link>
                <Link
                  to="/viewer"
                  className="px-3 py-1.5 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
                >
                  Viewer
                </Link>
              </div>
            </div>
            {progress.badges.length > 0 && (
              <div className="flex items-center gap-2">
                <Award className="h-5 w-5 text-yellow-500" />
                <span className="text-sm text-muted-foreground">
                  {progress.badges.length} badge
                  {progress.badges.length !== 1 ? 's' : ''}
                </span>
              </div>
            )}
          </div>
        </div>
      </nav>

      {/* Header */}
      <header className="border-b border-border">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <h1 className="text-3xl font-bold text-foreground">
            Example Simulations
          </h1>
          <p className="mt-2 text-muted-foreground">
            Explore pre-computed demos or browse example Python scripts
          </p>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Learning path selector */}
        {showLearningPaths && activeTab === 'demos' && (
          <LearningPathSelector
            paths={paths}
            progress={progress}
            onSelectPath={handleSelectPath}
            onBrowseAll={handleBrowseAll}
          />
        )}

        {/* Show learning paths toggle if hidden */}
        {!showLearningPaths && activeTab === 'demos' && (
          <Button
            variant="outline"
            onClick={() => setShowLearningPaths(true)}
            className="mb-6"
          >
            Show Learning Paths
          </Button>
        )}

        {/* Tab Switcher */}
        <div className="flex items-center gap-2 mb-6">
          <Button
            variant={activeTab === 'demos' ? 'default' : 'outline'}
            onClick={() => handleTabChange('demos')}
            className="gap-2"
          >
            <Play className="h-4 w-4" />
            Live Demos ({DEMOS.length})
          </Button>
          <Button
            variant={activeTab === 'scripts' ? 'default' : 'outline'}
            onClick={() => handleTabChange('scripts')}
            className="gap-2"
          >
            <Code2 className="h-4 w-4" />
            Example Scripts ({exampleScripts.length})
          </Button>
        </div>

        {/* Category filters */}
        <div className="flex flex-wrap gap-2 mb-8">
          {categories.map((cat) => (
            <Badge
              key={cat}
              variant={selectedCategory === cat ? 'default' : 'secondary'}
              className="cursor-pointer px-4 py-2"
              onClick={() => setSelectedCategory(cat)}
            >
              {cat !== 'All' && CATEGORY_ICONS[cat]}
              {cat}
            </Badge>
          ))}
        </div>

        {/* Demos Grid */}
        {activeTab === 'demos' && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredDemos.map((demo) => (
              <div
                key={demo.id}
                className="bg-card border border-border rounded-lg overflow-hidden hover:shadow-lg transition-shadow"
              >
                {/* Thumbnail placeholder */}
                <div className="h-48 bg-secondary/30 flex items-center justify-center">
                  <div className="text-4xl">
                    {CATEGORY_ICONS[demo.category] || (
                      <Waves className="h-12 w-12 text-muted-foreground" />
                    )}
                  </div>
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

            {filteredDemos.length === 0 && (
              <div className="col-span-full text-center py-12">
                <p className="text-lg text-muted-foreground">
                  No demos found in this category
                </p>
              </div>
            )}
          </div>
        )}

        {/* Scripts Grid */}
        {activeTab === 'scripts' && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredScripts.map((script) => (
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

            {filteredScripts.length === 0 && (
              <div className="col-span-full text-center py-12">
                <p className="text-lg text-muted-foreground">
                  No scripts found in this category
                </p>
              </div>
            )}
          </div>
        )}
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
            <div className="font-mono text-xs">
              Build {__GIT_HASH__}
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}
