/**
 * Export utilities for simulation data
 */

export interface ViewState {
  cameraPosition: [number, number, number]
  cameraTarget: [number, number, number]
  cameraFov: number
  viewOptions: {
    threshold: number
    displayFill: boolean
    voxelGeometry: string
    geometryMode: string
    showGrid: boolean
    showAxes: boolean
  }
  simulation: {
    currentFrame: number
    totalFrames: number
    shape: [number, number, number]
    resolution: number
  }
  timestamp: string
}

export interface ExportOptions {
  format: 'csv' | 'json' | 'png' | 'svg'
  filename?: string
  includeMetadata?: boolean
}

export interface TimeSeriesData {
  name: string
  times: Float32Array | number[]
  values: Float32Array | number[]
}

/**
 * Export time series data to CSV
 */
export function exportToCSV(data: TimeSeriesData[], options: ExportOptions): void {
  const headers = ['time', ...data.map((d) => d.name)]
  const rows: string[] = [headers.join(',')]

  const maxLength = Math.max(...data.map((d) => d.times.length))

  for (let i = 0; i < maxLength; i++) {
    const row: (number | string)[] = [data[0].times[i]]
    for (const series of data) {
      row.push(series.values[i] ?? '')
    }
    rows.push(row.join(','))
  }

  const csv = rows.join('\n')
  downloadFile(csv, options.filename || 'data.csv', 'text/csv')
}

/**
 * Export time series data to JSON
 */
export function exportToJSON(
  data: TimeSeriesData[],
  options: ExportOptions & { metadata?: Record<string, unknown> }
): void {
  const jsonData: Record<string, unknown> = {
    probes: data.map((d) => ({
      name: d.name,
      times: Array.from(d.times),
      values: Array.from(d.values),
    })),
  }

  if (options.includeMetadata && options.metadata) {
    jsonData.metadata = options.metadata
  }

  const json = JSON.stringify(jsonData, null, 2)
  downloadFile(json, options.filename || 'data.json', 'application/json')
}

/**
 * Export canvas to PNG
 */
export function exportToPNG(canvas: HTMLCanvasElement, filename: string = 'image.png'): void {
  canvas.toBlob((blob) => {
    if (blob) {
      downloadBlob(blob, filename)
    }
  }, 'image/png')
}

/**
 * Export SVG element to file
 */
export function exportToSVG(svg: SVGSVGElement, filename: string = 'image.svg'): void {
  const serializer = new XMLSerializer()
  const svgString = serializer.serializeToString(svg)
  const blob = new Blob([svgString], { type: 'image/svg+xml' })
  downloadBlob(blob, filename)
}

/**
 * Download a string as a file
 */
function downloadFile(content: string, filename: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType })
  downloadBlob(blob, filename)
}

/**
 * Download a blob as a file
 */
function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

/**
 * Generate filename with timestamp
 */
export function generateFilename(base: string, extension: string): string {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
  return `${base}_${timestamp}.${extension}`
}

/**
 * Export data with automatic format detection
 */
export function exportData(
  data: TimeSeriesData[],
  options: ExportOptions & { metadata?: Record<string, unknown> }
): void {
  switch (options.format) {
    case 'csv':
      exportToCSV(data, options)
      break
    case 'json':
      exportToJSON(data, options)
      break
    default:
      throw new Error(`Unsupported export format: ${options.format}`)
  }
}
