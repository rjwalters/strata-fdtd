/**
 * Template code snippets for common FDTD simulation patterns
 */

import type { GridInfo } from './scriptParser'

export interface Template {
  name: string
  description: string
  generate: (grid: GridInfo | null) => string
  cursorOffset?: number  // Where to place cursor after insertion
}

/**
 * Generate rectangle template
 */
export const rectangleTemplate: Template = {
  name: 'Rectangle',
  description: 'Add a rectangular material region',
  generate: (grid) => {
    if (!grid) {
      return `
scene.add_rectangle(
    center=(0.05, 0.05, 0.05),
    size=(0.01, 0.01, 0.01),
    material="pzt5"
)`
    }

    const cx = grid.extent[0] / 2
    const cy = grid.extent[1] / 2
    const cz = grid.extent[2] / 2
    const size = grid.resolution * 10

    return `
scene.add_rectangle(
    center=(${cx.toExponential(3)}, ${cy.toExponential(3)}, ${cz.toExponential(3)}),
    size=(${size.toExponential(3)}, ${size.toExponential(3)}, ${size.toExponential(3)}),
    material="pzt5"
)`
  },
}

/**
 * Generate sphere template
 */
export const sphereTemplate: Template = {
  name: 'Sphere',
  description: 'Add a spherical material region',
  generate: (grid) => {
    if (!grid) {
      return `
scene.add_sphere(
    center=(0.05, 0.05, 0.05),
    radius=0.005,
    material="water"
)`
    }

    const cx = grid.extent[0] / 2
    const cy = grid.extent[1] / 2
    const cz = grid.extent[2] / 2
    const radius = grid.resolution * 5

    return `
scene.add_sphere(
    center=(${cx.toExponential(3)}, ${cy.toExponential(3)}, ${cz.toExponential(3)}),
    radius=${radius.toExponential(3)},
    material="water"
)`
  },
}

/**
 * Generate Gaussian pulse source template
 */
export const gaussianPulseTemplate: Template = {
  name: 'Gaussian Pulse',
  description: 'Add a Gaussian pulse source',
  generate: (grid) => {
    if (!grid) {
      return `
scene.add_source(
    GaussianPulse(
        frequency=40e3,  # 40 kHz
        position=(0.025, 0.05, 0.05)
    )
)`
    }

    const x = grid.extent[0] * 0.25
    const y = grid.extent[1] / 2
    const z = grid.extent[2] / 2

    return `
scene.add_source(
    GaussianPulse(
        frequency=40e3,  # 40 kHz
        position=(${x.toExponential(3)}, ${y.toExponential(3)}, ${z.toExponential(3)})
    )
)`
  },
}

/**
 * Generate continuous wave source template
 */
export const continuousWaveTemplate: Template = {
  name: 'Continuous Wave',
  description: 'Add a continuous wave source',
  generate: (grid) => {
    if (!grid) {
      return `
scene.add_source(
    ContinuousWave(
        frequency=40e3,  # 40 kHz
        position=(0.025, 0.05, 0.05),
        amplitude=1.0
    )
)`
    }

    const x = grid.extent[0] * 0.25
    const y = grid.extent[1] / 2
    const z = grid.extent[2] / 2

    return `
scene.add_source(
    ContinuousWave(
        frequency=40e3,  # 40 kHz
        position=(${x.toExponential(3)}, ${y.toExponential(3)}, ${z.toExponential(3)}),
        amplitude=1.0
    )
)`
  },
}

/**
 * Generate probe template
 */
export const probeTemplate: Template = {
  name: 'Probe',
  description: 'Add a pressure probe',
  generate: (grid) => {
    if (!grid) {
      return `
scene.add_probe(
    position=(0.075, 0.05, 0.05),
    name="probe1"
)`
    }

    const x = grid.extent[0] * 0.75
    const y = grid.extent[1] / 2
    const z = grid.extent[2] / 2

    return `
scene.add_probe(
    position=(${x.toExponential(3)}, ${y.toExponential(3)}, ${z.toExponential(3)}),
    name="probe1"
)`
  },
}

/**
 * All available templates
 */
export const templates = {
  rectangle: rectangleTemplate,
  sphere: sphereTemplate,
  gaussianPulse: gaussianPulseTemplate,
  continuousWave: continuousWaveTemplate,
  probe: probeTemplate,
}

/**
 * Template categories for UI organization
 */
export const templateCategories = [
  {
    name: 'Materials',
    templates: [rectangleTemplate, sphereTemplate],
  },
  {
    name: 'Sources',
    templates: [gaussianPulseTemplate, continuousWaveTemplate],
  },
  {
    name: 'Probes',
    templates: [probeTemplate],
  },
]
