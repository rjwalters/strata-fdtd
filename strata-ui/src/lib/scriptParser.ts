/**
 * Script parser for FDTD simulation Python scripts
 *
 * Extracts simulation parameters, materials, sources, and probes from Python code
 * using regex-based parsing (simple approach for MVP).
 */

export interface GridInfo {
  shape: [number, number, number]
  resolution: number
  extent: [number, number, number]
  type: 'uniform' | 'nonuniform'
}

export interface MaterialRegion {
  id: string
  type: 'rectangle' | 'sphere'
  center: [number, number, number]
  size?: [number, number, number]  // for rectangle
  radius?: number  // for sphere
  material: string
}

export interface SourceInfo {
  id: string
  type: string  // e.g., 'GaussianPulse', 'ContinuousWave'
  position: [number, number, number]
  frequency?: number
  amplitude?: number
}

export interface ProbeInfo {
  id: string
  position: [number, number, number]
  name?: string
}

export interface SimulationAST {
  grid: GridInfo | null
  materials: MaterialRegion[]
  sources: SourceInfo[]
  probes: ProbeInfo[]
  hasValidGrid: boolean
  errors: string[]
}

/**
 * Parse Python script to extract simulation configuration
 */
export function parseScript(script: string): SimulationAST {
  const errors: string[] = []

  try {
    // Extract grid configuration
    const grid = parseGrid(script, errors)

    // Extract materials
    const materials = parseMaterials(script, grid, errors)

    // Extract sources
    const sources = parseSources(script, errors)

    // Extract probes
    const probes = parseProbes(script, errors)

    return {
      grid,
      materials,
      sources,
      probes,
      hasValidGrid: grid !== null,
      errors,
    }
  } catch (error) {
    errors.push(`Parse error: ${error instanceof Error ? error.message : String(error)}`)
    return {
      grid: null,
      materials: [],
      sources: [],
      probes: [],
      hasValidGrid: false,
      errors,
    }
  }
}

/**
 * Parse grid configuration from script
 */
function parseGrid(script: string, errors: string[]): GridInfo | null {
  // Try UniformGrid pattern
  const uniformMatch = script.match(
    /UniformGrid\s*\(\s*shape\s*=\s*\((\d+),\s*(\d+),\s*(\d+)\)\s*,\s*resolution\s*=\s*([\d.e-]+)\s*\)/i
  )

  if (uniformMatch) {
    const [, nx, ny, nz, res] = uniformMatch
    const shape: [number, number, number] = [+nx, +ny, +nz]
    const resolution = +res
    const extent: [number, number, number] = [
      shape[0] * resolution,
      shape[1] * resolution,
      shape[2] * resolution,
    ]

    return { shape, resolution, extent, type: 'uniform' }
  }

  // Try alternative pattern: grid = UniformGrid(...)
  const altUniformMatch = script.match(
    /grid\s*=\s*UniformGrid\s*\(\s*shape\s*=\s*\((\d+),\s*(\d+),\s*(\d+)\)\s*,\s*resolution\s*=\s*([\d.e-]+)\s*\)/i
  )

  if (altUniformMatch) {
    const [, nx, ny, nz, res] = altUniformMatch
    const shape: [number, number, number] = [+nx, +ny, +nz]
    const resolution = +res
    const extent: [number, number, number] = [
      shape[0] * resolution,
      shape[1] * resolution,
      shape[2] * resolution,
    ]

    return { shape, resolution, extent, type: 'uniform' }
  }

  // Try NonuniformGrid pattern (simpler parsing - just detect it exists)
  if (script.match(/NonuniformGrid/i)) {
    errors.push('NonuniformGrid detected but not fully supported in preview yet')
    return null
  }

  errors.push('No valid grid configuration found')
  return null
}

/**
 * Parse material regions from script
 */
function parseMaterials(
  script: string,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  _grid: GridInfo | null,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  _errors: string[]
): MaterialRegion[] {
  const materials: MaterialRegion[] = []

  // Parse rectangles: scene.add_rectangle(center=(x,y,z), size=(w,h,d), material="name")
  const rectRegex = /add_rectangle\s*\(\s*center\s*=\s*\(([\d.e-]+),\s*([\d.e-]+),\s*([\d.e-]+)\)\s*,\s*size\s*=\s*\(([\d.e-]+),\s*([\d.e-]+),\s*([\d.e-]+)\)\s*,\s*material\s*=\s*["']([^"']+)["']\s*\)/gi

  let match: RegExpExecArray | null
  let rectId = 0
  while ((match = rectRegex.exec(script)) !== null) {
    const [, cx, cy, cz, sx, sy, sz, material] = match
    materials.push({
      id: `rect-${rectId++}`,
      type: 'rectangle',
      center: [+cx, +cy, +cz],
      size: [+sx, +sy, +sz],
      material,
    })
  }

  // Parse spheres: scene.add_sphere(center=(x,y,z), radius=r, material="name")
  const sphereRegex = /add_sphere\s*\(\s*center\s*=\s*\(([\d.e-]+),\s*([\d.e-]+),\s*([\d.e-]+)\)\s*,\s*radius\s*=\s*([\d.e-]+)\s*,\s*material\s*=\s*["']([^"']+)["']\s*\)/gi

  let sphereId = 0
  while ((match = sphereRegex.exec(script)) !== null) {
    const [, cx, cy, cz, radius, material] = match
    materials.push({
      id: `sphere-${sphereId++}`,
      type: 'sphere',
      center: [+cx, +cy, +cz],
      radius: +radius,
      material,
    })
  }

  return materials
}

/**
 * Parse sources from script
 */
function parseSources(
  script: string,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  _errors: string[]
): SourceInfo[] {
  const sources: SourceInfo[] = []

  // Parse GaussianPulse sources
  // add_source(GaussianPulse(frequency=f, position=(x,y,z)))
  const gaussianRegex = /add_source\s*\(\s*GaussianPulse\s*\(\s*frequency\s*=\s*([\d.e+-]+)\s*,\s*position\s*=\s*\(([\d.e-]+),\s*([\d.e-]+),\s*([\d.e-]+)\)\s*\)\s*\)/gi

  let match: RegExpExecArray | null
  let sourceId = 0
  while ((match = gaussianRegex.exec(script)) !== null) {
    const [, freq, x, y, z] = match
    sources.push({
      id: `source-${sourceId++}`,
      type: 'GaussianPulse',
      position: [+x, +y, +z],
      frequency: +freq,
    })
  }

  // Parse ContinuousWave sources
  const cwRegex = /add_source\s*\(\s*ContinuousWave\s*\(\s*frequency\s*=\s*([\d.e+-]+)\s*,\s*position\s*=\s*\(([\d.e-]+),\s*([\d.e-]+),\s*([\d.e-]+)\)\s*(?:,\s*amplitude\s*=\s*([\d.e+-]+))?\s*\)\s*\)/gi

  while ((match = cwRegex.exec(script)) !== null) {
    const [, freq, x, y, z, amp] = match
    sources.push({
      id: `source-${sourceId++}`,
      type: 'ContinuousWave',
      position: [+x, +y, +z],
      frequency: +freq,
      amplitude: amp ? +amp : undefined,
    })
  }

  return sources
}

/**
 * Parse probes from script
 */
function parseProbes(
  script: string,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  _errors: string[]
): ProbeInfo[] {
  const probes: ProbeInfo[] = []

  // Parse probes: add_probe(position=(x,y,z), name="name")
  const probeRegex = /add_probe\s*\(\s*position\s*=\s*\(([\d.e-]+),\s*([\d.e-]+),\s*([\d.e-]+)\)\s*(?:,\s*name\s*=\s*["']([^"']+)["'])?\s*\)/gi

  let match: RegExpExecArray | null
  let probeId = 0
  while ((match = probeRegex.exec(script)) !== null) {
    const [, x, y, z, name] = match
    probes.push({
      id: `probe-${probeId++}`,
      position: [+x, +y, +z],
      name: name || undefined,
    })
  }

  return probes
}

/**
 * Get default template script
 */
export function getDefaultScript(): string {
  return `from metamaterial import UniformGrid, Scene, GaussianPulse

# Create a uniform grid
grid = UniformGrid(
    shape=(100, 100, 100),
    resolution=1e-3  # 1mm resolution
)

# Create a scene
scene = Scene(grid)

# Add a rectangular material region
scene.add_rectangle(
    center=(0.05, 0.05, 0.05),
    size=(0.02, 0.02, 0.02),
    material="pzt5"
)

# Add a Gaussian pulse source
scene.add_source(
    GaussianPulse(
        frequency=40e3,  # 40 kHz
        position=(0.025, 0.05, 0.05)
    )
)

# Add a probe to record pressure
scene.add_probe(
    position=(0.075, 0.05, 0.05),
    name="receiver"
)
`
}
