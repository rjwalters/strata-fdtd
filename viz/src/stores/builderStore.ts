/**
 * Zustand store for Simulation Builder state
 */

import { create } from 'zustand'
import type { SimulationAST } from '@/lib/scriptParser'
import { parseScript, getDefaultScript } from '@/lib/scriptParser'
import type { ValidationError } from '@/lib/pythonValidator'

export type AnimationSpeed = 'slow' | 'normal' | 'fast'

interface ViewOptions {
  showGrid: boolean
  showMaterials: boolean
  showSources: boolean
  showProbes: boolean
  sliceAxis: 'none' | 'xy' | 'xz' | 'yz'
  slicePosition: number
  measurementMode: boolean
  dualSliceMode: boolean
  slice1Position: number
  slice2Position: number
  isAnimating: boolean
  animationSpeed: AnimationSpeed
}

export interface MeasurementPoint {
  x: number
  y: number
  z: number
}

export interface BuilderState {
  // Script state
  script: string
  ast: SimulationAST | null
  scriptHash: string | null
  validationErrors: ValidationError[]

  // View options
  viewOptions: ViewOptions

  // Measurement state
  measurementPoints: MeasurementPoint[]

  // UI state
  isParsing: boolean

  // Actions
  setScript: (script: string) => void
  setValidationErrors: (errors: ValidationError[]) => void
  insertTemplate: (template: string) => void
  resetToDefault: () => void
  toggleViewOption: (key: keyof ViewOptions) => void
  setSliceAxis: (axis: 'none' | 'xy' | 'xz' | 'yz') => void
  setSlicePosition: (position: number) => void
  addMeasurementPoint: (point: MeasurementPoint) => void
  clearMeasurements: () => void
  setMeasurementMode: (enabled: boolean) => void
  setDualSliceMode: (enabled: boolean) => void
  setSlice1Position: (position: number) => void
  setSlice2Position: (position: number) => void
  setIsAnimating: (animating: boolean) => void
  setAnimationSpeed: (speed: AnimationSpeed) => void
  toggleAnimation: () => void
}

/**
 * Generate SHA-256 hash of script content (for reproducible filenames)
 */
async function hashScript(script: string): Promise<string> {
  const encoder = new TextEncoder()
  const data = encoder.encode(script.trim())
  const hashBuffer = await crypto.subtle.digest('SHA-256', data)
  const hashArray = Array.from(new Uint8Array(hashBuffer))
  const hashHex = hashArray.map((b) => b.toString(16).padStart(2, '0')).join('')
  return hashHex
}

/**
 * Parse script with debouncing
 */
let parseTimeout: NodeJS.Timeout | null = null
function debouncedParse(script: string, set: (fn: (state: BuilderState) => Partial<BuilderState>) => void) {
  if (parseTimeout) {
    clearTimeout(parseTimeout)
  }

  set(() => ({ isParsing: true }))

  parseTimeout = setTimeout(async () => {
    const ast = parseScript(script)
    const scriptHash = await hashScript(script)

    set(() => ({
      ast,
      scriptHash,
      isParsing: false,
    }))
  }, 500) // 500ms debounce
}

/**
 * Builder store
 */
export const useBuilderStore = create<BuilderState>((set, get) => ({
  // Initial state
  script: getDefaultScript(),
  ast: parseScript(getDefaultScript()),
  scriptHash: null,
  validationErrors: [],

  viewOptions: {
    showGrid: true,
    showMaterials: true,
    showSources: true,
    showProbes: true,
    sliceAxis: 'none',
    slicePosition: 0.5,
    measurementMode: false,
    dualSliceMode: false,
    slice1Position: 0.33,
    slice2Position: 0.67,
    isAnimating: false,
    animationSpeed: 'normal',
  },

  measurementPoints: [],

  isParsing: false,

  // Actions
  setScript: (script: string) => {
    set({ script })
    debouncedParse(script, set)
  },

  setValidationErrors: (errors: ValidationError[]) => {
    set({ validationErrors: errors })
  },

  insertTemplate: (template: string) => {
    const { script } = get()
    const newScript = script + '\n' + template
    set({ script: newScript })
    debouncedParse(newScript, set)
  },

  resetToDefault: () => {
    const defaultScript = getDefaultScript()
    set({ script: defaultScript })
    debouncedParse(defaultScript, set)
  },

  toggleViewOption: (key: keyof ViewOptions) => {
    set((state) => ({
      viewOptions: {
        ...state.viewOptions,
        [key]: !state.viewOptions[key],
      },
    }))
  },

  setSliceAxis: (axis: 'none' | 'xy' | 'xz' | 'yz') => {
    set((state) => ({
      viewOptions: {
        ...state.viewOptions,
        sliceAxis: axis,
        // Disable measurement mode, dual slice mode, and animation when disabling slice
        measurementMode: axis === 'none' ? false : state.viewOptions.measurementMode,
        dualSliceMode: axis === 'none' ? false : state.viewOptions.dualSliceMode,
        isAnimating: axis === 'none' ? false : state.viewOptions.isAnimating,
      },
      // Clear measurements when switching slice axis
      measurementPoints: [],
    }))
  },

  setSlicePosition: (position: number) => {
    set((state) => ({
      viewOptions: {
        ...state.viewOptions,
        slicePosition: position,
      },
    }))
  },

  addMeasurementPoint: (point: MeasurementPoint) => {
    set((state) => {
      const points = [...state.measurementPoints, point]
      // Keep only the last 2 points for a single measurement
      if (points.length > 2) {
        return { measurementPoints: [points[points.length - 1]] }
      }
      return { measurementPoints: points }
    })
  },

  clearMeasurements: () => {
    set({ measurementPoints: [] })
  },

  setMeasurementMode: (enabled: boolean) => {
    set((state) => ({
      viewOptions: {
        ...state.viewOptions,
        measurementMode: enabled,
      },
      // Clear measurements when disabling mode
      measurementPoints: enabled ? state.measurementPoints : [],
    }))
  },

  setDualSliceMode: (enabled: boolean) => {
    set((state) => ({
      viewOptions: {
        ...state.viewOptions,
        dualSliceMode: enabled,
      },
    }))
  },

  setSlice1Position: (position: number) => {
    set((state) => ({
      viewOptions: {
        ...state.viewOptions,
        slice1Position: position,
      },
    }))
  },

  setSlice2Position: (position: number) => {
    set((state) => ({
      viewOptions: {
        ...state.viewOptions,
        slice2Position: position,
      },
    }))
  },

  setIsAnimating: (animating: boolean) => {
    set((state) => ({
      viewOptions: {
        ...state.viewOptions,
        isAnimating: animating,
      },
    }))
  },

  setAnimationSpeed: (speed: AnimationSpeed) => {
    set((state) => ({
      viewOptions: {
        ...state.viewOptions,
        animationSpeed: speed,
      },
    }))
  },

  toggleAnimation: () => {
    set((state) => ({
      viewOptions: {
        ...state.viewOptions,
        isAnimating: !state.viewOptions.isAnimating,
      },
    }))
  },
}))

// Initialize hash for default script
hashScript(getDefaultScript()).then((hash) => {
  useBuilderStore.setState({ scriptHash: hash })
})
