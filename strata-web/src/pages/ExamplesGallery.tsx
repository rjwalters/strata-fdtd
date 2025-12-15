import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Button, Badge } from '@strata/ui'
import { ArrowRight, Waves, Volume2, Music } from 'lucide-react'

interface Demo {
  id: string
  title: string
  description: string
  thumbnail?: string
  category: string
  path: string
}

const DEMOS: Demo[] = [
  {
    id: 'open-pipe',
    title: 'Open Pipe',
    description: 'Acoustic simulation of an open-ended pipe showing standing wave patterns and harmonics.',
    category: 'Acoustics',
    path: '/demos/organ-pipes/open-pipe',
  },
  {
    id: 'closed-pipe',
    title: 'Closed Pipe',
    description: 'Acoustic simulation of a closed pipe demonstrating odd harmonic patterns.',
    category: 'Acoustics',
    path: '/demos/organ-pipes/closed-pipe',
  },
  {
    id: 'half-open-pipe',
    title: 'Half-Open Pipe',
    description: 'Acoustic simulation showing the behavior of a pipe with one open and one closed end.',
    category: 'Acoustics',
    path: '/demos/organ-pipes/half-open-pipe',
  },
]

const CATEGORY_ICONS: Record<string, React.ReactNode> = {
  Acoustics: <Volume2 className="h-4 w-4" />,
  Waves: <Waves className="h-4 w-4" />,
  Music: <Music className="h-4 w-4" />,
}

export function ExamplesGallery() {
  const navigate = useNavigate()
  const [selectedCategory, setSelectedCategory] = useState<string>('All')

  const categories = ['All', ...new Set(DEMOS.map(d => d.category))]

  const filtered = selectedCategory === 'All'
    ? DEMOS
    : DEMOS.filter(d => d.category === selectedCategory)

  const handleViewDemo = (demo: Demo) => {
    navigate(`/viewer/${demo.id}`)
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <h1 className="text-3xl font-bold text-foreground">Example Simulations</h1>
          <p className="mt-2 text-muted-foreground">
            Explore pre-computed FDTD simulation results
          </p>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Category filters */}
        <div className="flex flex-wrap gap-2 mb-8">
          {categories.map(cat => (
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

        {/* Demos grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filtered.map(demo => (
            <div
              key={demo.id}
              className="bg-card border border-border rounded-lg overflow-hidden hover:shadow-lg transition-shadow"
            >
              {/* Thumbnail placeholder */}
              <div className="h-48 bg-secondary/30 flex items-center justify-center">
                <div className="text-4xl">
                  {CATEGORY_ICONS[demo.category] || <Waves className="h-12 w-12 text-muted-foreground" />}
                </div>
              </div>

              {/* Content */}
              <div className="p-5 space-y-4">
                <div>
                  <h3 className="text-xl font-bold text-foreground">{demo.title}</h3>
                  <p className="mt-1 text-sm text-muted-foreground line-clamp-2">{demo.description}</p>
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

        {/* Empty state */}
        {filtered.length === 0 && (
          <div className="text-center py-12">
            <p className="text-lg text-muted-foreground">No demos found in this category</p>
          </div>
        )}
      </main>
    </div>
  )
}
