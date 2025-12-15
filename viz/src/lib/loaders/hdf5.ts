/**
 * HDF5 file loader for simulation data
 */

import type { SimulationData, ProbeData, SimulationMetadata } from '../loaders'

// HDF5 library type (loaded dynamically)
interface H5WasmModule {
  File: new (data: ArrayBuffer) => H5File
  ready: Promise<void>
}

interface H5File {
  get(path: string): H5Dataset | H5Group | null
  keys(): string[]
  close(): void
}

interface H5Dataset {
  value: ArrayLike<number> | number | string
  shape: number[]
  dtype: string
}

interface H5Group {
  keys(): string[]
  get(name: string): H5Dataset | H5Group | null
}

let h5wasm: H5WasmModule | null = null

/**
 * Load HDF5 library dynamically
 */
async function loadH5Wasm(): Promise<H5WasmModule> {
  if (h5wasm) {
    return h5wasm
  }

  // @ts-expect-error - dynamic import
  const module = await import('h5wasm')
  await module.ready
  h5wasm = module as unknown as H5WasmModule
  return h5wasm
}

/**
 * Extract probe data from HDF5 file
 */
function extractProbes(file: H5File): ProbeData[] {
  const probes: ProbeData[] = []

  // Try common probe locations
  const probePaths = ['probes', 'Probes', 'data/probes', 'output/probes']

  for (const path of probePaths) {
    const group = file.get(path) as H5Group | null
    if (group && 'keys' in group) {
      for (const name of group.keys()) {
        const probeGroup = group.get(name) as H5Group | null
        if (!probeGroup) continue

        const timeDataset = probeGroup.get('time') as H5Dataset | null
        const valueDataset = probeGroup.get('values') as H5Dataset | null
        const positionDataset = probeGroup.get('position') as H5Dataset | null

        if (timeDataset && valueDataset) {
          const position: [number, number, number] = positionDataset
            ? [
                (positionDataset.value as number[])[0] || 0,
                (positionDataset.value as number[])[1] || 0,
                (positionDataset.value as number[])[2] || 0,
              ]
            : [0, 0, 0]

          probes.push({
            name,
            position,
            times: new Float32Array(timeDataset.value as ArrayLike<number>),
            values: new Float32Array(valueDataset.value as ArrayLike<number>),
          })
        }
      }
      break
    }
  }

  return probes
}

/**
 * Extract metadata from HDF5 file
 */
function extractMetadata(file: H5File): SimulationMetadata {
  const metadata: SimulationMetadata = {
    format: 'hdf5',
  }

  // Try common metadata locations
  const metadataPaths = ['metadata', 'Metadata', 'attrs', 'simulation']

  for (const path of metadataPaths) {
    const group = file.get(path) as H5Group | H5Dataset | null
    if (!group) continue

    if ('keys' in group) {
      // It's a group
      const version = (group as H5Group).get('version') as H5Dataset | null
      if (version) {
        metadata.version = String(version.value)
      }

      const resolution = (group as H5Group).get('resolution') as H5Dataset | null
      if (resolution) {
        metadata.resolution = Number(resolution.value)
      }

      const duration = (group as H5Group).get('duration') as H5Dataset | null
      if (duration) {
        metadata.duration = Number(duration.value)
      }
    }
    break
  }

  return metadata
}

/**
 * Load HDF5 simulation file
 */
export async function loadHDF5(file: File): Promise<SimulationData> {
  const h5 = await loadH5Wasm()
  const buffer = await file.arrayBuffer()
  const h5file = new h5.File(buffer)

  try {
    const probes = extractProbes(h5file)
    const metadata = extractMetadata(h5file)

    const timeSteps = probes.length > 0 ? probes[0].times.length : 0

    return {
      timeSteps,
      probes,
      metadata,
    }
  } finally {
    h5file.close()
  }
}

/**
 * Check if HDF5 support is available
 */
export async function isHDF5Available(): Promise<boolean> {
  try {
    await loadH5Wasm()
    return true
  } catch {
    return false
  }
}
