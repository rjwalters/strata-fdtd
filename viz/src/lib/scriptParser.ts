/**
 * Python script parser for simulation builder
 * Parses Python scripts to extract simulation configuration
 */

export interface GridInfo {
  extent: [number, number, number]
  resolution: number
  shape: [number, number, number]
}

export interface MaterialRegion {
  id: string
  type: 'rectangle' | 'sphere'
  center: [number, number, number]
  size?: [number, number, number]
  radius?: number
  material: string
}

export interface SourceInfo {
  id: string
  type: string
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
 * Default simulation script template
 */
export function getDefaultScript(): string {
  return `# Strata FDTD Simulation Configuration
# Define your simulation setup here

from strata import UniformGrid, Material, Source, Probe

# Define the simulation grid
# Resolution: spatial step size (in meters)
# Shape: number of cells in each dimension (x, y, z)
grid = UniformGrid(
    resolution=1e-3,  # 1mm resolution
    shape=(100, 100, 50)  # 100x100x50 cells
)

# Define materials
# Material.rectangle(center, size, material_name)
pzt_transducer = Material.rectangle(
    center=(0.05, 0.05, 0.025),
    size=(0.02, 0.02, 0.005),
    material="pzt5"
)

# Define acoustic source
# Source.gaussian_pulse(position, frequency, amplitude)
source = Source.gaussian_pulse(
    position=(0.05, 0.05, 0.025),
    frequency=1e6,  # 1 MHz
    amplitude=1.0
)

# Define measurement probes
# Probe.point(position, name)
probe1 = Probe.point(
    position=(0.05, 0.05, 0.04),
    name="far_field"
)
`
}

/**
 * Check if a line is inside a comment or should be skipped
 */
function isCommentLine(line: string): boolean {
  const trimmed = line.trim()
  return trimmed.startsWith('#') || trimmed.length === 0
}

/**
 * Parse a tuple/list from Python code
 */
function parseTuple(str: string): number[] | null {
  // Match (x, y, z) or [x, y, z]
  const match = str.match(/[\[(]([^)\]]+)[\])]/)
  if (!match) return null

  const values = match[1].split(',').map(s => {
    const trimmed = s.trim()
    // Handle scientific notation
    const num = parseFloat(trimmed)
    return isNaN(num) ? null : num
  })

  if (values.some(v => v === null)) return null
  return values as number[]
}

/**
 * Extract value from a keyword argument
 */
function extractKwarg(content: string, kwarg: string): string | null {
  // Match kwarg=value pattern, handling various value types
  // Order matters: check tuples/lists first, then simple values
  const pattern = new RegExp(`${kwarg}\\s*=\\s*(\\([^)]+\\)|\\[[^\\]]+\\]|"[^"]*"|'[^']*'|[^,)\\]]+)`)
  const match = content.match(pattern)
  return match ? match[1].trim() : null
}

/**
 * Parse UniformGrid definition
 */
function parseGrid(script: string): GridInfo | null {
  const lines = script.split('\n')

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]

    if (isCommentLine(line)) continue

    // Look for UniformGrid constructor call
    if (line.includes('UniformGrid')) {
      // Collect the full call (may span multiple lines)
      let fullCall = ''
      let depth = 0
      let started = false

      for (let j = i; j < lines.length; j++) {
        const currentLine = lines[j]

        // Skip comments at the end of lines but keep the code
        const codeOnly = currentLine.split('#')[0]

        for (const char of codeOnly) {
          if (char === '(') {
            started = true
            depth++
          } else if (char === ')') {
            depth--
          }

          if (started) {
            fullCall += char
          }

          if (started && depth === 0) break
        }

        if (started && depth === 0) break
        if (started) fullCall += ' '
      }

      // Parse resolution
      const resolutionStr = extractKwarg(fullCall, 'resolution')
      const resolution = resolutionStr ? parseFloat(resolutionStr) : null

      // Parse shape
      const shapeStr = extractKwarg(fullCall, 'shape')
      const shape = shapeStr ? parseTuple(shapeStr) : null

      if (resolution !== null && shape && shape.length === 3) {
        return {
          resolution,
          shape: shape as [number, number, number],
          extent: [
            resolution * shape[0],
            resolution * shape[1],
            resolution * shape[2],
          ],
        }
      }
    }
  }

  return null
}

/**
 * Parse material definitions
 */
function parseMaterials(script: string): MaterialRegion[] {
  const materials: MaterialRegion[] = []
  const lines = script.split('\n')

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]

    if (isCommentLine(line)) continue

    // Look for Material.rectangle or Material.sphere
    if (line.includes('Material.rectangle') || line.includes('Material.sphere')) {
      // Collect full call
      let fullCall = ''
      let depth = 0
      let started = false

      for (let j = i; j < lines.length; j++) {
        const currentLine = lines[j]
        const codeOnly = currentLine.split('#')[0]

        for (const char of codeOnly) {
          if (char === '(') {
            started = true
            depth++
          } else if (char === ')') {
            depth--
          }

          if (started) {
            fullCall += char
          }

          if (started && depth === 0) break
        }

        if (started && depth === 0) break
        if (started) fullCall += ' '
      }

      const isRectangle = line.includes('Material.rectangle')

      // Parse center
      const centerStr = extractKwarg(fullCall, 'center')
      const center = centerStr ? parseTuple(centerStr) : null

      // Parse material name
      const materialStr = extractKwarg(fullCall, 'material')
      const material = materialStr?.replace(/['"]/g, '') || 'unknown'

      if (center && center.length === 3) {
        const region: MaterialRegion = {
          id: `material-${materials.length}`,
          type: isRectangle ? 'rectangle' : 'sphere',
          center: center as [number, number, number],
          material,
        }

        if (isRectangle) {
          const sizeStr = extractKwarg(fullCall, 'size')
          const size = sizeStr ? parseTuple(sizeStr) : null
          if (size && size.length === 3) {
            region.size = size as [number, number, number]
          }
        } else {
          const radiusStr = extractKwarg(fullCall, 'radius')
          const radius = radiusStr ? parseFloat(radiusStr) : null
          if (radius !== null) {
            region.radius = radius
          }
        }

        materials.push(region)
      }
    }
  }

  return materials
}

/**
 * Parse source definitions
 */
function parseSources(script: string): SourceInfo[] {
  const sources: SourceInfo[] = []
  const lines = script.split('\n')

  const sourcePatterns = [
    { pattern: 'Source.gaussian_pulse', type: 'GaussianPulse' },
    { pattern: 'Source.continuous_wave', type: 'ContinuousWave' },
    { pattern: 'Source.tone_burst', type: 'ToneBurst' },
  ]

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]

    if (isCommentLine(line)) continue

    for (const { pattern, type } of sourcePatterns) {
      if (line.includes(pattern)) {
        // Collect full call
        let fullCall = ''
        let depth = 0
        let started = false

        for (let j = i; j < lines.length; j++) {
          const currentLine = lines[j]
          const codeOnly = currentLine.split('#')[0]

          for (const char of codeOnly) {
            if (char === '(') {
              started = true
              depth++
            } else if (char === ')') {
              depth--
            }

            if (started) {
              fullCall += char
            }

            if (started && depth === 0) break
          }

          if (started && depth === 0) break
          if (started) fullCall += ' '
        }

        // Parse position
        const positionStr = extractKwarg(fullCall, 'position')
        const position = positionStr ? parseTuple(positionStr) : null

        // Parse optional parameters
        const frequencyStr = extractKwarg(fullCall, 'frequency')
        const frequency = frequencyStr ? parseFloat(frequencyStr) : undefined

        const amplitudeStr = extractKwarg(fullCall, 'amplitude')
        const amplitude = amplitudeStr ? parseFloat(amplitudeStr) : undefined

        if (position && position.length === 3) {
          sources.push({
            id: `source-${sources.length}`,
            type,
            position: position as [number, number, number],
            frequency,
            amplitude,
          })
        }

        break
      }
    }
  }

  return sources
}

/**
 * Parse probe definitions
 */
function parseProbes(script: string): ProbeInfo[] {
  const probes: ProbeInfo[] = []
  const lines = script.split('\n')

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]

    if (isCommentLine(line)) continue

    if (line.includes('Probe.point') || line.includes('Probe.line') || line.includes('Probe.plane')) {
      // Collect full call
      let fullCall = ''
      let depth = 0
      let started = false

      for (let j = i; j < lines.length; j++) {
        const currentLine = lines[j]
        const codeOnly = currentLine.split('#')[0]

        for (const char of codeOnly) {
          if (char === '(') {
            started = true
            depth++
          } else if (char === ')') {
            depth--
          }

          if (started) {
            fullCall += char
          }

          if (started && depth === 0) break
        }

        if (started && depth === 0) break
        if (started) fullCall += ' '
      }

      // Parse position
      const positionStr = extractKwarg(fullCall, 'position')
      const position = positionStr ? parseTuple(positionStr) : null

      // Parse optional name
      const nameStr = extractKwarg(fullCall, 'name')
      const name = nameStr?.replace(/['"]/g, '') || undefined

      if (position && position.length === 3) {
        probes.push({
          id: `probe-${probes.length}`,
          position: position as [number, number, number],
          name,
        })
      }
    }
  }

  return probes
}

/**
 * Parse a Python script and extract simulation configuration
 */
export function parseScript(script: string): SimulationAST {
  const errors: string[] = []

  // Parse grid
  const grid = parseGrid(script)

  if (!grid) {
    errors.push('No valid UniformGrid definition found')
  } else {
    // Validate grid configuration
    if (grid.resolution <= 0) {
      errors.push('Grid resolution must be positive')
    }
    if (grid.shape.some(s => s <= 0)) {
      errors.push('Grid shape dimensions must be positive')
    }
  }

  // Parse materials
  const materials = parseMaterials(script)

  // Parse sources
  const sources = parseSources(script)

  // Parse probes
  const probes = parseProbes(script)

  return {
    grid,
    materials,
    sources,
    probes,
    hasValidGrid: grid !== null && errors.length === 0,
    errors,
  }
}
