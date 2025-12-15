/**
 * Hook for managing learning path state with localStorage persistence.
 *
 * Tracks user progress through learning paths, handles step navigation,
 * and manages badge awards.
 */

import { useState, useCallback, useEffect, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  type LearningPath,
  type UserProgress,
  type Badge,
  LEARNING_PATHS,
  loadProgress,
  saveProgress,
  getLearningPath,
  checkForNewBadges,
  DEFAULT_PROGRESS,
} from '../config/learning-paths'

export interface UseLearningPathResult {
  /** All available learning paths */
  paths: LearningPath[]
  /** Current user progress */
  progress: UserProgress
  /** Currently active learning path (null if in browse mode) */
  activePath: LearningPath | null
  /** Current step index within active path */
  currentStepIndex: number
  /** Whether tutorial mode is active */
  isTutorialMode: boolean
  /** Start a learning path */
  startPath: (pathId: string) => void
  /** Exit current learning path */
  exitPath: () => void
  /** Go to specific step */
  goToStep: (stepIndex: number) => void
  /** Go to next step */
  nextStep: () => void
  /** Go to previous step */
  previousStep: () => void
  /** Mark current step as complete */
  completeCurrentStep: () => void
  /** Check if a step is completed */
  isStepCompleted: (stepExampleId: string) => boolean
  /** Newly earned badges (cleared after display) */
  newBadges: Badge[]
  /** Clear new badges notification */
  clearNewBadges: () => void
  /** Reset all progress */
  resetProgress: () => void
}

export function useLearningPath(): UseLearningPathResult {
  const navigate = useNavigate()
  const [progress, setProgress] = useState<UserProgress>(() => loadProgress())
  const [activePath, setActivePath] = useState<LearningPath | null>(null)
  const [currentStepIndex, setCurrentStepIndex] = useState(0)
  const [newBadges, setNewBadges] = useState<Badge[]>([])

  // Load active path from progress on mount
  useEffect(() => {
    if (progress.currentPath) {
      const path = getLearningPath(progress.currentPath)
      if (path) {
        setActivePath(path)
        setCurrentStepIndex(progress.currentStep)
      }
    }
  }, [])

  // Persist progress changes
  const updateProgress = useCallback((updater: (prev: UserProgress) => UserProgress) => {
    setProgress((prev) => {
      const updated = updater(prev)
      saveProgress(updated)

      // Check for new badges
      const earned = checkForNewBadges(updated)
      if (earned.length > 0) {
        // Add new badges to progress
        updated.badges = [...updated.badges, ...earned]
        saveProgress(updated)
        setNewBadges(earned)
      }

      return updated
    })
  }, [])

  const startPath = useCallback(
    (pathId: string) => {
      const path = getLearningPath(pathId)
      if (!path) return

      setActivePath(path)

      // Find first incomplete step or start at beginning
      const completedSteps = progress.completedSteps[pathId] || []
      let startIndex = 0
      for (let i = 0; i < path.steps.length; i++) {
        if (!completedSteps.includes(path.steps[i].exampleId)) {
          startIndex = i
          break
        }
      }
      setCurrentStepIndex(startIndex)

      // Update progress
      updateProgress((prev) => ({
        ...prev,
        currentPath: pathId,
        currentStep: startIndex,
      }))

      // Navigate to the first example
      const firstStep = path.steps[startIndex]
      navigate(`/viewer/${firstStep.exampleId}`)
    },
    [navigate, progress.completedSteps, updateProgress]
  )

  const exitPath = useCallback(() => {
    setActivePath(null)
    setCurrentStepIndex(0)
    updateProgress((prev) => ({
      ...prev,
      currentPath: null,
      currentStep: 0,
    }))
    navigate('/examples')
  }, [navigate, updateProgress])

  const goToStep = useCallback(
    (stepIndex: number) => {
      if (!activePath) return
      if (stepIndex < 0 || stepIndex >= activePath.steps.length) return

      setCurrentStepIndex(stepIndex)
      updateProgress((prev) => ({
        ...prev,
        currentStep: stepIndex,
      }))

      const step = activePath.steps[stepIndex]
      navigate(`/viewer/${step.exampleId}`)
    },
    [activePath, navigate, updateProgress]
  )

  const nextStep = useCallback(() => {
    if (!activePath) return

    if (currentStepIndex < activePath.steps.length - 1) {
      goToStep(currentStepIndex + 1)
    } else {
      // Completed the path
      updateProgress((prev) => {
        const completedPaths = prev.completedPaths.includes(activePath.id)
          ? prev.completedPaths
          : [...prev.completedPaths, activePath.id]
        return {
          ...prev,
          completedPaths,
          currentPath: null,
          currentStep: 0,
        }
      })
      setActivePath(null)
      setCurrentStepIndex(0)
      navigate('/examples')
    }
  }, [activePath, currentStepIndex, goToStep, navigate, updateProgress])

  const previousStep = useCallback(() => {
    if (!activePath || currentStepIndex <= 0) return
    goToStep(currentStepIndex - 1)
  }, [activePath, currentStepIndex, goToStep])

  const completeCurrentStep = useCallback(() => {
    if (!activePath) return

    const currentStep = activePath.steps[currentStepIndex]
    if (!currentStep) return

    updateProgress((prev) => {
      const pathSteps = prev.completedSteps[activePath.id] || []
      if (pathSteps.includes(currentStep.exampleId)) {
        return prev // Already completed
      }
      return {
        ...prev,
        completedSteps: {
          ...prev.completedSteps,
          [activePath.id]: [...pathSteps, currentStep.exampleId],
        },
      }
    })
  }, [activePath, currentStepIndex, updateProgress])

  const isStepCompleted = useCallback(
    (stepExampleId: string) => {
      if (!activePath) return false
      const completedSteps = progress.completedSteps[activePath.id] || []
      return completedSteps.includes(stepExampleId)
    },
    [activePath, progress.completedSteps]
  )

  const clearNewBadges = useCallback(() => {
    setNewBadges([])
  }, [])

  const resetProgress = useCallback(() => {
    setProgress({ ...DEFAULT_PROGRESS })
    saveProgress({ ...DEFAULT_PROGRESS })
    setActivePath(null)
    setCurrentStepIndex(0)
    setNewBadges([])
  }, [])

  const isTutorialMode = useMemo(() => activePath !== null, [activePath])

  return {
    paths: LEARNING_PATHS,
    progress,
    activePath,
    currentStepIndex,
    isTutorialMode,
    startPath,
    exitPath,
    goToStep,
    nextStep,
    previousStep,
    completeCurrentStep,
    isStepCompleted,
    newBadges,
    clearNewBadges,
    resetProgress,
  }
}
