import { Badge, Button } from '@strata/ui'
import { GraduationCap, Clock, ArrowRight, CheckCircle } from 'lucide-react'
import type { LearningPath, UserProgress } from '../../config/learning-paths'

interface LearningPathSelectorProps {
  paths: LearningPath[]
  progress: UserProgress
  onSelectPath: (path: LearningPath) => void
  onBrowseAll: () => void
}

function getDifficultyColor(difficulty: LearningPath['difficulty']): string {
  switch (difficulty) {
    case 'beginner':
      return 'bg-green-500/20 text-green-400 border-green-500/30'
    case 'intermediate':
      return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
    case 'advanced':
      return 'bg-red-500/20 text-red-400 border-red-500/30'
  }
}

function getPathProgress(path: LearningPath, progress: UserProgress): number {
  const completedSteps = progress.completedSteps[path.id] || []
  return Math.round((completedSteps.length / path.steps.length) * 100)
}

function isPathCompleted(path: LearningPath, progress: UserProgress): boolean {
  return progress.completedPaths.includes(path.id)
}

function isPathInProgress(path: LearningPath, progress: UserProgress): boolean {
  const completed = progress.completedSteps[path.id] || []
  return completed.length > 0 && completed.length < path.steps.length
}

export function LearningPathSelector({
  paths,
  progress,
  onSelectPath,
  onBrowseAll,
}: LearningPathSelectorProps) {
  return (
    <div className="bg-card border border-border rounded-lg p-6 mb-8">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 bg-primary/10 rounded-lg">
          <GraduationCap className="h-6 w-6 text-primary" />
        </div>
        <div>
          <h2 className="text-xl font-bold text-foreground">
            Choose Your Learning Path
          </h2>
          <p className="text-sm text-muted-foreground">
            Follow guided tutorials to learn FDTD simulation concepts
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-4">
        {paths.map((path) => {
          const completed = isPathCompleted(path, progress)
          const inProgress = isPathInProgress(path, progress)
          const progressPercent = getPathProgress(path, progress)

          return (
            <div
              key={path.id}
              className={`
                relative bg-secondary/30 border rounded-lg p-4
                hover:bg-secondary/50 transition-colors cursor-pointer
                ${completed ? 'border-green-500/50' : 'border-border'}
              `}
              onClick={() => onSelectPath(path)}
            >
              {completed && (
                <div className="absolute top-3 right-3">
                  <CheckCircle className="h-5 w-5 text-green-500" />
                </div>
              )}

              <div className="text-2xl mb-2">{path.icon}</div>

              <h3 className="font-semibold text-foreground mb-1">
                {path.title}
              </h3>

              <p className="text-sm text-muted-foreground mb-3 line-clamp-2">
                {path.description}
              </p>

              <div className="flex items-center gap-2 flex-wrap mb-3">
                <Badge
                  variant="outline"
                  className={getDifficultyColor(path.difficulty)}
                >
                  {path.difficulty}
                </Badge>
                <div className="flex items-center gap-1 text-xs text-muted-foreground">
                  <Clock className="h-3 w-3" />
                  {path.estimatedMinutes} min
                </div>
                <div className="text-xs text-muted-foreground">
                  {path.steps.length} examples
                </div>
              </div>

              {inProgress && (
                <div className="mb-3">
                  <div className="flex items-center justify-between text-xs text-muted-foreground mb-1">
                    <span>Progress</span>
                    <span>{progressPercent}%</span>
                  </div>
                  <div className="h-1.5 bg-secondary rounded-full overflow-hidden">
                    <div
                      className="h-full bg-primary transition-all"
                      style={{ width: `${progressPercent}%` }}
                    />
                  </div>
                </div>
              )}

              <Button
                variant="outline"
                size="sm"
                className="w-full gap-2"
                onClick={(e: React.MouseEvent) => {
                  e.stopPropagation()
                  onSelectPath(path)
                }}
              >
                {completed
                  ? 'Review Path'
                  : inProgress
                    ? 'Continue'
                    : 'Start Path'}
                <ArrowRight className="h-4 w-4" />
              </Button>
            </div>
          )
        })}
      </div>

      <div className="flex justify-center">
        <Button variant="ghost" onClick={onBrowseAll} className="text-sm">
          Or browse all examples without guidance
        </Button>
      </div>
    </div>
  )
}
