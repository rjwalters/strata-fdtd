import { useState } from 'react'
import { Badge, Button } from '@strata/ui'
import {
  Lightbulb,
  Eye,
  Sparkles,
  BookOpen,
  ChevronDown,
  ChevronUp,
  CheckCircle,
} from 'lucide-react'
import type { LearningStep } from '../../config/learning-paths'

type HelpMode = 'before' | 'after'

interface ContextualHelpProps {
  step: LearningStep
  mode: HelpMode
  onMarkComplete: () => void
  isCompleted: boolean
}

export function ContextualHelp({
  step,
  mode,
  onMarkComplete,
  isCompleted,
}: ContextualHelpProps) {
  const [isExpanded, setIsExpanded] = useState(true)
  const [showExercises, setShowExercises] = useState(false)

  const displayName = step.exampleId
    .split('-')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')

  return (
    <div className="bg-card border border-border rounded-lg overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-4 py-3 flex items-center justify-between hover:bg-secondary/30 transition-colors"
      >
        <div className="flex items-center gap-2">
          {mode === 'before' ? (
            <Lightbulb className="h-5 w-5 text-yellow-500" />
          ) : (
            <Sparkles className="h-5 w-5 text-purple-500" />
          )}
          <div className="text-left">
            <h3 className="font-semibold text-sm text-foreground">
              {mode === 'before' ? 'Before You Start' : 'After Exploring'}
            </h3>
            <p className="text-xs text-muted-foreground">{displayName}</p>
          </div>
        </div>
        {isExpanded ? (
          <ChevronUp className="h-4 w-4 text-muted-foreground" />
        ) : (
          <ChevronDown className="h-4 w-4 text-muted-foreground" />
        )}
      </button>

      {isExpanded && (
        <div className="border-t border-border">
          {mode === 'before' ? (
            <>
              {/* Why this example */}
              <div className="px-4 py-3 border-b border-border bg-primary/5">
                <h4 className="text-xs font-semibold text-primary uppercase tracking-wide mb-1">
                  Why This Example?
                </h4>
                <p className="text-sm text-foreground">{step.why}</p>
              </div>

              {/* Key concepts */}
              <div className="px-4 py-3 border-b border-border">
                <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2 flex items-center gap-1">
                  <BookOpen className="h-3 w-3" />
                  Key Concepts
                </h4>
                <div className="flex flex-wrap gap-1.5">
                  {step.concepts.map((concept) => (
                    <Badge key={concept} variant="secondary" className="text-xs">
                      {concept}
                    </Badge>
                  ))}
                </div>
              </div>

              {/* Tips */}
              <div className="px-4 py-3 border-b border-border">
                <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2 flex items-center gap-1">
                  <Lightbulb className="h-3 w-3" />
                  Tips
                </h4>
                <ul className="space-y-1.5">
                  {step.beforeRunning.tips.map((tip, i) => (
                    <li
                      key={i}
                      className="text-sm text-muted-foreground flex items-start gap-2"
                    >
                      <span className="text-primary mt-0.5">•</span>
                      {tip}
                    </li>
                  ))}
                </ul>
              </div>

              {/* What to look for */}
              <div className="px-4 py-3">
                <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2 flex items-center gap-1">
                  <Eye className="h-3 w-3" />
                  What to Look For
                </h4>
                <ul className="space-y-1.5">
                  {step.beforeRunning.lookFor.map((item, i) => (
                    <li
                      key={i}
                      className="text-sm text-muted-foreground flex items-start gap-2"
                    >
                      <span className="text-green-500 mt-0.5">→</span>
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            </>
          ) : (
            <>
              {/* Observations */}
              <div className="px-4 py-3 border-b border-border">
                <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2 flex items-center gap-1">
                  <Sparkles className="h-3 w-3" />
                  Key Observations
                </h4>
                <ul className="space-y-1.5">
                  {step.afterRunning.observations.map((obs, i) => (
                    <li
                      key={i}
                      className="text-sm text-muted-foreground flex items-start gap-2"
                    >
                      <span className="text-purple-500 mt-0.5">✓</span>
                      {obs}
                    </li>
                  ))}
                </ul>
              </div>

              {/* Exercises (collapsible) */}
              <div className="px-4 py-3 border-b border-border">
                <button
                  onClick={() => setShowExercises(!showExercises)}
                  className="w-full flex items-center justify-between text-xs font-semibold text-muted-foreground uppercase tracking-wide"
                >
                  <span className="flex items-center gap-1">
                    <BookOpen className="h-3 w-3" />
                    Try These Exercises ({step.afterRunning.exercises.length})
                  </span>
                  {showExercises ? (
                    <ChevronUp className="h-3 w-3" />
                  ) : (
                    <ChevronDown className="h-3 w-3" />
                  )}
                </button>
                {showExercises && (
                  <ul className="mt-2 space-y-1.5">
                    {step.afterRunning.exercises.map((ex, i) => (
                      <li
                        key={i}
                        className="text-sm text-muted-foreground flex items-start gap-2"
                      >
                        <span className="text-blue-500 font-medium mt-0.5">
                          {i + 1}.
                        </span>
                        {ex}
                      </li>
                    ))}
                  </ul>
                )}
              </div>

              {/* Mark complete */}
              <div className="px-4 py-3">
                {isCompleted ? (
                  <div className="flex items-center gap-2 text-green-500">
                    <CheckCircle className="h-5 w-5" />
                    <span className="text-sm font-medium">
                      Step completed!
                    </span>
                  </div>
                ) : (
                  <Button
                    onClick={onMarkComplete}
                    className="w-full gap-2"
                    variant="default"
                  >
                    <CheckCircle className="h-4 w-4" />
                    Mark as Complete
                  </Button>
                )}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}
