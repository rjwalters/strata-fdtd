/**
 * Learning path definitions for the interactive tutorial system.
 * Guides users through examples in a logical progression.
 */

export interface LearningStep {
  /** ID of the demo/example to load */
  exampleId: string
  /** Why this example is included in the path */
  why: string
  /** Key concepts demonstrated */
  concepts: string[]
  /** Guidance before running the simulation */
  beforeRunning: {
    tips: string[]
    lookFor: string[]
    estimatedTime: string
  }
  /** Analysis after running */
  afterRunning: {
    observations: string[]
    exercises: string[]
  }
}

export interface LearningPath {
  id: string
  title: string
  description: string
  difficulty: 'beginner' | 'intermediate' | 'advanced'
  estimatedMinutes: number
  icon: string
  steps: LearningStep[]
}

export interface UserProgress {
  completedPaths: string[]
  completedSteps: Record<string, string[]> // pathId -> stepIds
  currentPath: string | null
  currentStep: number
  badges: Badge[]
  lastUpdated: number
}

export interface Badge {
  id: string
  name: string
  description: string
  icon: string
  earnedAt?: number
}

export const BADGES: Badge[] = [
  {
    id: 'first-simulation',
    name: 'First Simulation',
    description: 'Viewed your first FDTD simulation',
    icon: '🎯',
  },
  {
    id: 'acoustics-basics',
    name: 'Acoustics Basics',
    description: 'Completed the Pipe Acoustics learning path',
    icon: '🎵',
  },
  {
    id: 'wave-explorer',
    name: 'Wave Explorer',
    description: 'Completed all learning paths',
    icon: '🌊',
  },
  {
    id: 'curious-mind',
    name: 'Curious Mind',
    description: 'Explored 5 different simulations',
    icon: '🔬',
  },
]

export const LEARNING_PATHS: LearningPath[] = [
  {
    id: 'pipe-acoustics',
    title: 'Pipe Acoustics Fundamentals',
    description:
      'Learn how boundary conditions affect acoustic behavior in pipes. Start with the basics of standing waves and progress to understanding harmonic patterns.',
    difficulty: 'beginner',
    estimatedMinutes: 15,
    icon: '🎵',
    steps: [
      {
        exampleId: 'open-pipe',
        why: 'Start with the simplest case - an open-ended pipe allows pressure to equalize at both ends',
        concepts: [
          'Open boundary conditions',
          'Standing wave patterns',
          'Fundamental frequency',
          'Harmonics series',
        ],
        beforeRunning: {
          tips: [
            'Notice how pressure nodes form at the open ends',
            'Watch for the characteristic wavelength pattern',
            'The colormap shows pressure variations - red is high, blue is low',
          ],
          lookFor: [
            'Pressure antinodes (maximum pressure) in the middle',
            'Pressure nodes (zero pressure) at the ends',
            'Regular spacing of wave pattern',
          ],
          estimatedTime: '~3 min',
        },
        afterRunning: {
          observations: [
            'Open pipes support all harmonics (1st, 2nd, 3rd...)',
            'The fundamental wavelength is twice the pipe length',
            'Energy flows freely out of open ends',
          ],
          exercises: [
            'Try pausing at different timesteps to see the standing wave pattern',
            'Use the probe data to see how pressure varies over time',
            'Adjust the threshold slider to see different pressure levels',
          ],
        },
      },
      {
        exampleId: 'closed-pipe',
        why: 'Compare with a fully closed pipe to understand how rigid boundaries change the acoustic behavior',
        concepts: [
          'Rigid boundary conditions',
          'Pressure reflection',
          'Mode shapes',
          'Resonant frequencies',
        ],
        beforeRunning: {
          tips: [
            'Closed ends act as rigid reflectors',
            'Pressure cannot equalize at closed ends',
            'Compare the harmonic pattern to the open pipe',
          ],
          lookFor: [
            'Pressure antinodes at both closed ends',
            'Different standing wave pattern than open pipe',
            'All harmonics present (different from half-open)',
          ],
          estimatedTime: '~3 min',
        },
        afterRunning: {
          observations: [
            'Closed pipes also support all harmonics',
            'The pressure pattern is shifted compared to open pipes',
            'Maximum pressure occurs at the closed ends',
          ],
          exercises: [
            'Compare the probe data to the open pipe - notice the phase difference',
            'Look at how the pressure distribution differs at the boundaries',
            'Try switching between diverging and magnitude colormaps',
          ],
        },
      },
      {
        exampleId: 'half-open-pipe',
        why: 'See how mixing boundary conditions creates unique acoustic properties - this is how many wind instruments work!',
        concepts: [
          'Mixed boundary conditions',
          'Odd harmonics only',
          'Quarter-wave resonators',
          'Musical acoustics',
        ],
        beforeRunning: {
          tips: [
            'One end is open (pressure node), one is closed (pressure antinode)',
            "This configuration only supports odd harmonics - that's musically important!",
            'Think about clarinet vs flute - they have different overtone series',
          ],
          lookFor: [
            'Asymmetric standing wave pattern',
            'Pressure antinode at closed end, node at open end',
            'Missing even harmonics in the frequency spectrum',
          ],
          estimatedTime: '~4 min',
        },
        afterRunning: {
          observations: [
            'Half-open pipes only support odd harmonics (1st, 3rd, 5th...)',
            'The fundamental wavelength is four times the pipe length',
            'This is why clarinets sound different from flutes at the same pitch',
          ],
          exercises: [
            'Check the spectrum plot - notice missing even harmonics',
            'Compare the wavelength to the open and closed pipe cases',
            'Think about why organ pipes of different types produce different timbres',
          ],
        },
      },
    ],
  },
]

/**
 * Default progress state for new users
 */
export const DEFAULT_PROGRESS: UserProgress = {
  completedPaths: [],
  completedSteps: {},
  currentPath: null,
  currentStep: 0,
  badges: [],
  lastUpdated: Date.now(),
}

/**
 * LocalStorage key for persisting user progress
 */
export const PROGRESS_STORAGE_KEY = 'strata-learning-progress'

/**
 * Load user progress from localStorage
 */
export function loadProgress(): UserProgress {
  try {
    const stored = localStorage.getItem(PROGRESS_STORAGE_KEY)
    if (stored) {
      return JSON.parse(stored) as UserProgress
    }
  } catch {
    console.warn('Failed to load learning progress from localStorage')
  }
  return { ...DEFAULT_PROGRESS }
}

/**
 * Save user progress to localStorage
 */
export function saveProgress(progress: UserProgress): void {
  try {
    progress.lastUpdated = Date.now()
    localStorage.setItem(PROGRESS_STORAGE_KEY, JSON.stringify(progress))
  } catch {
    console.warn('Failed to save learning progress to localStorage')
  }
}

/**
 * Get a learning path by ID
 */
export function getLearningPath(pathId: string): LearningPath | undefined {
  return LEARNING_PATHS.find((p) => p.id === pathId)
}

/**
 * Get the current step in a learning path
 */
export function getCurrentStep(
  pathId: string,
  stepIndex: number
): LearningStep | undefined {
  const path = getLearningPath(pathId)
  return path?.steps[stepIndex]
}

/**
 * Check if a badge should be awarded based on current progress
 */
export function checkForNewBadges(progress: UserProgress): Badge[] {
  const newBadges: Badge[] = []
  const earnedIds = new Set(progress.badges.map((b) => b.id))

  // First simulation badge
  const totalCompleted = Object.values(progress.completedSteps).flat().length
  if (totalCompleted >= 1 && !earnedIds.has('first-simulation')) {
    const badge = BADGES.find((b) => b.id === 'first-simulation')
    if (badge) newBadges.push({ ...badge, earnedAt: Date.now() })
  }

  // Acoustics basics badge
  if (
    progress.completedPaths.includes('pipe-acoustics') &&
    !earnedIds.has('acoustics-basics')
  ) {
    const badge = BADGES.find((b) => b.id === 'acoustics-basics')
    if (badge) newBadges.push({ ...badge, earnedAt: Date.now() })
  }

  // Wave explorer badge (all paths completed)
  if (
    progress.completedPaths.length === LEARNING_PATHS.length &&
    !earnedIds.has('wave-explorer')
  ) {
    const badge = BADGES.find((b) => b.id === 'wave-explorer')
    if (badge) newBadges.push({ ...badge, earnedAt: Date.now() })
  }

  // Curious mind badge (5+ simulations explored)
  if (totalCompleted >= 5 && !earnedIds.has('curious-mind')) {
    const badge = BADGES.find((b) => b.id === 'curious-mind')
    if (badge) newBadges.push({ ...badge, earnedAt: Date.now() })
  }

  return newBadges
}
