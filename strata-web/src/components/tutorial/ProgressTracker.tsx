import { Button } from '@strata/ui'
import {
  CheckCircle,
  Circle,
  ChevronLeft,
  ChevronRight,
  X,
} from 'lucide-react'
import type { LearningPath, UserProgress } from '../../config/learning-paths'

interface ProgressTrackerProps {
  path: LearningPath
  currentStepIndex: number
  progress: UserProgress
  onStepSelect: (stepIndex: number) => void
  onNext: () => void
  onPrevious: () => void
  onExit: () => void
}

export function ProgressTracker({
  path,
  currentStepIndex,
  progress,
  onStepSelect,
  onNext,
  onPrevious,
  onExit,
}: ProgressTrackerProps) {
  const completedSteps = progress.completedSteps[path.id] || []
  const progressPercent = Math.round(
    (completedSteps.length / path.steps.length) * 100
  )
  const isFirstStep = currentStepIndex === 0
  const isLastStep = currentStepIndex === path.steps.length - 1
  const currentStep = path.steps[currentStepIndex]

  return (
    <div className="bg-card border border-border rounded-lg overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-border flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-lg">{path.icon}</span>
          <div>
            <h3 className="font-semibold text-sm text-foreground">
              {path.title}
            </h3>
            <p className="text-xs text-muted-foreground">
              Step {currentStepIndex + 1} of {path.steps.length}
            </p>
          </div>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={onExit}
          className="h-8 w-8 p-0"
          title="Exit learning path"
        >
          <X className="h-4 w-4" />
        </Button>
      </div>

      {/* Progress bar */}
      <div className="px-4 py-2 border-b border-border bg-secondary/20">
        <div className="flex items-center justify-between text-xs text-muted-foreground mb-1">
          <span>
            {completedSteps.length} of {path.steps.length} completed
          </span>
          <span>{progressPercent}%</span>
        </div>
        <div className="h-1.5 bg-secondary rounded-full overflow-hidden">
          <div
            className="h-full bg-primary transition-all"
            style={{ width: `${progressPercent}%` }}
          />
        </div>
      </div>

      {/* Step list */}
      <div className="p-2 max-h-48 overflow-y-auto">
        {path.steps.map((step, index) => {
          const isCompleted = completedSteps.includes(step.exampleId)
          const isCurrent = index === currentStepIndex

          return (
            <button
              key={step.exampleId}
              onClick={() => onStepSelect(index)}
              className={`
                w-full flex items-center gap-3 px-3 py-2 rounded-md text-left
                transition-colors
                ${isCurrent ? 'bg-primary/10 text-foreground' : 'text-muted-foreground hover:bg-secondary/50'}
              `}
            >
              {isCompleted ? (
                <CheckCircle className="h-4 w-4 text-green-500 flex-shrink-0" />
              ) : isCurrent ? (
                <div className="h-4 w-4 rounded-full border-2 border-primary flex-shrink-0" />
              ) : (
                <Circle className="h-4 w-4 flex-shrink-0" />
              )}
              <div className="flex-1 min-w-0">
                <p
                  className={`text-sm truncate ${isCurrent ? 'font-medium' : ''}`}
                >
                  {index + 1}. {step.exampleId.replace(/-/g, ' ')}
                </p>
                {isCurrent && (
                  <p className="text-xs text-muted-foreground truncate">
                    {step.why}
                  </p>
                )}
              </div>
            </button>
          )
        })}
      </div>

      {/* Navigation */}
      <div className="px-4 py-3 border-t border-border flex items-center justify-between">
        <Button
          variant="outline"
          size="sm"
          onClick={onPrevious}
          disabled={isFirstStep}
          className="gap-1"
        >
          <ChevronLeft className="h-4 w-4" />
          Previous
        </Button>

        {currentStep && (
          <span className="text-xs text-muted-foreground">
            {currentStep.beforeRunning.estimatedTime}
          </span>
        )}

        <Button
          variant={isLastStep ? 'default' : 'outline'}
          size="sm"
          onClick={onNext}
          className="gap-1"
        >
          {isLastStep ? 'Complete Path' : 'Next'}
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>
    </div>
  )
}
