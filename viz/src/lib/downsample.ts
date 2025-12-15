/**
 * Data downsampling utilities for efficient visualization
 */

export interface DownsampleOptions {
  targetPoints: number
  method: 'lttb' | 'minmax' | 'average'
}

export interface DataPoint {
  t: number
  v: number
}

/**
 * Largest Triangle Three Buckets (LTTB) downsampling
 * Preserves visual features while reducing data points
 */
export function lttbDownsample(
  data: DataPoint[],
  targetPoints: number
): DataPoint[] {
  const n = data.length

  if (targetPoints >= n || targetPoints < 3) {
    return data
  }

  const sampled: DataPoint[] = []
  const bucketSize = (n - 2) / (targetPoints - 2)

  // Always keep first point
  sampled.push(data[0])

  for (let i = 0; i < targetPoints - 2; i++) {
    // Calculate bucket boundaries
    const bucketStart = Math.floor((i + 1) * bucketSize) + 1
    const bucketEnd = Math.floor((i + 2) * bucketSize) + 1

    // Calculate average point for next bucket (for triangle calculation)
    let avgX = 0
    let avgY = 0
    const nextBucketStart = Math.floor((i + 2) * bucketSize) + 1
    const nextBucketEnd = Math.min(Math.floor((i + 3) * bucketSize) + 1, n)
    const nextBucketSize = nextBucketEnd - nextBucketStart

    if (nextBucketSize > 0) {
      for (let j = nextBucketStart; j < nextBucketEnd; j++) {
        avgX += data[j].t
        avgY += data[j].v
      }
      avgX /= nextBucketSize
      avgY /= nextBucketSize
    }

    // Find point in current bucket that forms largest triangle
    const prevPoint = sampled[sampled.length - 1]
    let maxArea = -1
    let maxIndex = bucketStart

    for (let j = bucketStart; j < Math.min(bucketEnd, n); j++) {
      const area = Math.abs(
        (prevPoint.t - avgX) * (data[j].v - prevPoint.v) -
          (prevPoint.t - data[j].t) * (avgY - prevPoint.v)
      )

      if (area > maxArea) {
        maxArea = area
        maxIndex = j
      }
    }

    sampled.push(data[maxIndex])
  }

  // Always keep last point
  sampled.push(data[n - 1])

  return sampled
}

/**
 * Min-max downsampling - preserves peaks
 */
export function minmaxDownsample(
  data: DataPoint[],
  targetPoints: number
): DataPoint[] {
  const n = data.length

  if (targetPoints >= n) {
    return data
  }

  const sampled: DataPoint[] = []
  const bucketSize = n / (targetPoints / 2)

  for (let i = 0; i < targetPoints / 2; i++) {
    const start = Math.floor(i * bucketSize)
    const end = Math.min(Math.floor((i + 1) * bucketSize), n)

    let minPoint = data[start]
    let maxPoint = data[start]

    for (let j = start; j < end; j++) {
      if (data[j].v < minPoint.v) {
        minPoint = data[j]
      }
      if (data[j].v > maxPoint.v) {
        maxPoint = data[j]
      }
    }

    // Add min and max in time order
    if (minPoint.t < maxPoint.t) {
      sampled.push(minPoint)
      if (minPoint !== maxPoint) {
        sampled.push(maxPoint)
      }
    } else {
      sampled.push(maxPoint)
      if (minPoint !== maxPoint) {
        sampled.push(minPoint)
      }
    }
  }

  return sampled
}

/**
 * Average downsampling
 */
export function averageDownsample(
  data: DataPoint[],
  targetPoints: number
): DataPoint[] {
  const n = data.length

  if (targetPoints >= n) {
    return data
  }

  const sampled: DataPoint[] = []
  const bucketSize = n / targetPoints

  for (let i = 0; i < targetPoints; i++) {
    const start = Math.floor(i * bucketSize)
    const end = Math.min(Math.floor((i + 1) * bucketSize), n)

    let sumT = 0
    let sumV = 0

    for (let j = start; j < end; j++) {
      sumT += data[j].t
      sumV += data[j].v
    }

    const count = end - start
    sampled.push({
      t: sumT / count,
      v: sumV / count,
    })
  }

  return sampled
}

/**
 * Main downsample function
 */
export function downsample(
  data: DataPoint[],
  options: DownsampleOptions
): DataPoint[] {
  switch (options.method) {
    case 'lttb':
      return lttbDownsample(data, options.targetPoints)
    case 'minmax':
      return minmaxDownsample(data, options.targetPoints)
    case 'average':
      return averageDownsample(data, options.targetPoints)
    default:
      return lttbDownsample(data, options.targetPoints)
  }
}

/**
 * Downsample typed arrays (for worker communication)
 */
export function downsampleArrays(
  times: Float32Array,
  values: Float32Array,
  targetPoints: number,
  method: DownsampleOptions['method'] = 'lttb'
): { times: Float32Array; values: Float32Array } {
  const data: DataPoint[] = []
  for (let i = 0; i < times.length; i++) {
    data.push({ t: times[i], v: values[i] })
  }

  const sampled = downsample(data, { targetPoints, method })

  const newTimes = new Float32Array(sampled.length)
  const newValues = new Float32Array(sampled.length)

  for (let i = 0; i < sampled.length; i++) {
    newTimes[i] = sampled[i].t
    newValues[i] = sampled[i].v
  }

  return { times: newTimes, values: newValues }
}
