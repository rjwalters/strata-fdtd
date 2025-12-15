/**
 * Data loaders for various simulation output formats
 */

export interface SimulationData {
  timeSteps: number
  probes: ProbeData[]
  fields?: FieldData
  metadata: SimulationMetadata
}

export interface ProbeData {
  name: string
  position: [number, number, number]
  times: Float32Array
  values: Float32Array
}

export interface FieldData {
  dimensions: [number, number, number]
  times: number[]
  data: Float32Array[]
}

export interface SimulationMetadata {
  format?: string
  version?: string
  createdAt?: string
  resolution?: number
  duration?: number
  sampleRate?: number
  grid?: {
    shape: [number, number, number]
    resolution: number
  }
}

export interface SimulationManifest {
  basePath?: string
  metadata: string
  probes: string
  geometry: string
  snapshots: SnapshotInfo[]
  velocitySnapshots?: VelocitySnapshotInfo[]
}

export interface SnapshotInfo {
  frame: number
  time: number
  file: string
  format?: string
  shape?: [number, number, number]
}

export interface VelocitySnapshotInfo {
  frame: number
  time: number
  file?: string
  format: 'interleaved' | 'separate'
  files?: { vx: string; vy: string; vz: string }
}

export interface VelocitySnapshot {
  vx: Float32Array
  vy: Float32Array
  vz: Float32Array
}

export interface Geometry {
  shape: [number, number, number]
  data: Uint8Array
}

export interface SnapshotData {
  pressure: Float32Array
}

/**
 * Load simulation data from JSON format
 */
export async function loadJSON(file: File): Promise<SimulationData> {
  const text = await file.text()
  const json = JSON.parse(text)

  const probes: ProbeData[] = []

  if (json.probes) {
    for (const probe of json.probes) {
      probes.push({
        name: probe.name || 'probe',
        position: probe.position || [0, 0, 0],
        times: new Float32Array(probe.times || []),
        values: new Float32Array(probe.values || []),
      })
    }
  }

  return {
    timeSteps: json.timeSteps || probes[0]?.times.length || 0,
    probes,
    metadata: {
      format: 'json',
      version: json.version,
      resolution: json.resolution,
      duration: json.duration,
      sampleRate: json.sampleRate,
    },
  }
}

/**
 * Load simulation data from CSV format
 */
export async function loadCSV(file: File): Promise<SimulationData> {
  const text = await file.text()
  const lines = text.trim().split('\n')

  if (lines.length < 2) {
    throw new Error('CSV file must have at least a header and one data row')
  }

  const headers = lines[0].split(',').map((h) => h.trim())
  const timeIndex = headers.findIndex((h) => h.toLowerCase() === 'time' || h.toLowerCase() === 't')

  const times: number[] = []
  const values: Record<string, number[]> = {}

  for (const header of headers) {
    if (header.toLowerCase() !== 'time' && header.toLowerCase() !== 't') {
      values[header] = []
    }
  }

  for (let i = 1; i < lines.length; i++) {
    const row = lines[i].split(',').map((v) => parseFloat(v.trim()))

    if (timeIndex >= 0) {
      times.push(row[timeIndex])
    } else {
      times.push(i - 1)
    }

    for (let j = 0; j < headers.length; j++) {
      if (j !== timeIndex) {
        values[headers[j]].push(row[j])
      }
    }
  }

  const probes: ProbeData[] = Object.entries(values).map(([name, vals]) => ({
    name,
    position: [0, 0, 0],
    times: new Float32Array(times),
    values: new Float32Array(vals),
  }))

  return {
    timeSteps: times.length,
    probes,
    metadata: {
      format: 'csv',
      sampleRate: times.length > 1 ? 1 / (times[1] - times[0]) : 1,
    },
  }
}

/**
 * Load simulation from File
 */
export async function loadSimulation(file: File): Promise<SimulationData> {
  const extension = file.name.split('.').pop()?.toLowerCase()

  switch (extension) {
    case 'json':
      return loadJSON(file)
    case 'csv':
      return loadCSV(file)
    case 'h5':
    case 'hdf5':
      throw new Error('HDF5 files require the HDF5 loader - use loadHDF5() instead')
    default:
      throw new Error(`Unsupported file format: ${extension}`)
  }
}

/**
 * Get file extension
 */
export function getFileExtension(filename: string): string {
  return filename.split('.').pop()?.toLowerCase() || ''
}

/**
 * Check if file format is supported
 */
export function isSupportedFormat(filename: string): boolean {
  const ext = getFileExtension(filename)
  return ['json', 'csv', 'h5', 'hdf5'].includes(ext)
}

/**
 * Load manifest.json file
 */
export async function loadManifest(url: string): Promise<SimulationManifest> {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Failed to load manifest: ${response.statusText}`)
  }
  return response.json()
}

/**
 * Load metadata file
 */
export async function loadMetadata(url: string): Promise<SimulationMetadata> {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Failed to load metadata: ${response.statusText}`)
  }
  return response.json()
}

/**
 * Load probe data file
 */
export async function loadProbeData(url: string): Promise<ProbeData> {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Failed to load probe data: ${response.statusText}`)
  }
  return response.json()
}

/**
 * Load geometry file
 */
export async function loadGeometry(
  url: string,
  options: { shape: [number, number, number]; format: string; file: string }
): Promise<Geometry> {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Failed to load geometry: ${response.statusText}`)
  }
  const buffer = await response.arrayBuffer()
  return {
    shape: options.shape,
    data: new Uint8Array(buffer),
  }
}

/**
 * Load snapshot file
 */
export async function loadSnapshot(
  url: string,
  _info: SnapshotInfo
): Promise<SnapshotData> {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Failed to load snapshot: ${response.statusText}`)
  }
  const buffer = await response.arrayBuffer()
  return {
    pressure: new Float32Array(buffer),
  }
}

/**
 * Load velocity snapshot file
 */
export async function loadVelocitySnapshot(
  url: string,
  _info: VelocitySnapshotInfo
): Promise<VelocitySnapshot> {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Failed to load velocity snapshot: ${response.statusText}`)
  }
  const buffer = await response.arrayBuffer()
  const data = new Float32Array(buffer)

  // Interleaved format: [vx0, vy0, vz0, vx1, vy1, vz1, ...]
  const len = Math.floor(data.length / 3)
  const vx = new Float32Array(len)
  const vy = new Float32Array(len)
  const vz = new Float32Array(len)

  for (let i = 0; i < len; i++) {
    vx[i] = data[i * 3]
    vy[i] = data[i * 3 + 1]
    vz[i] = data[i * 3 + 2]
  }

  return { vx, vy, vz }
}
