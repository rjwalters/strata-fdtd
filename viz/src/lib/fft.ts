/**
 * FFT (Fast Fourier Transform) utilities
 */

export interface FFTResult {
  frequencies: Float32Array
  magnitudes: Float32Array
  phases: Float32Array
}

/**
 * Compute FFT of real-valued signal
 * Uses Cooley-Tukey algorithm
 */
export function computeFFT(signal: Float32Array, sampleRate: number): FFTResult {
  const n = signal.length

  // Pad to power of 2
  const paddedLength = nextPowerOf2(n)
  const real = new Float32Array(paddedLength)
  const imag = new Float32Array(paddedLength)

  // Copy input
  real.set(signal)

  // Bit reversal permutation
  for (let i = 0; i < paddedLength; i++) {
    const j = bitReverse(i, Math.log2(paddedLength))
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]]
      ;[imag[i], imag[j]] = [imag[j], imag[i]]
    }
  }

  // Cooley-Tukey FFT
  for (let size = 2; size <= paddedLength; size *= 2) {
    const halfSize = size / 2
    const angle = -2 * Math.PI / size

    for (let i = 0; i < paddedLength; i += size) {
      for (let j = 0; j < halfSize; j++) {
        const theta = angle * j
        const cosTheta = Math.cos(theta)
        const sinTheta = Math.sin(theta)

        const idx1 = i + j
        const idx2 = i + j + halfSize

        const tReal = real[idx2] * cosTheta - imag[idx2] * sinTheta
        const tImag = real[idx2] * sinTheta + imag[idx2] * cosTheta

        real[idx2] = real[idx1] - tReal
        imag[idx2] = imag[idx1] - tImag
        real[idx1] = real[idx1] + tReal
        imag[idx1] = imag[idx1] + tImag
      }
    }
  }

  // Compute magnitudes and phases
  const halfN = Math.floor(paddedLength / 2)
  const frequencies = new Float32Array(halfN)
  const magnitudes = new Float32Array(halfN)
  const phases = new Float32Array(halfN)

  for (let i = 0; i < halfN; i++) {
    frequencies[i] = (i * sampleRate) / paddedLength
    magnitudes[i] = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]) / paddedLength
    phases[i] = Math.atan2(imag[i], real[i])
  }

  return { frequencies, magnitudes, phases }
}

/**
 * Compute inverse FFT
 */
export function computeIFFT(real: Float32Array, imag: Float32Array): Float32Array {
  const n = real.length

  // Conjugate
  const conjImag = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    conjImag[i] = -imag[i]
  }

  // FFT of conjugated
  const result = computeFFTComplex(real, conjImag)

  // Conjugate and scale
  const output = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    output[i] = result.real[i] / n
  }

  return output
}

/**
 * Compute FFT of complex signal
 */
function computeFFTComplex(
  inputReal: Float32Array,
  inputImag: Float32Array
): { real: Float32Array; imag: Float32Array } {
  const n = inputReal.length
  const real = new Float32Array(inputReal)
  const imag = new Float32Array(inputImag)

  // Bit reversal permutation
  for (let i = 0; i < n; i++) {
    const j = bitReverse(i, Math.log2(n))
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]]
      ;[imag[i], imag[j]] = [imag[j], imag[i]]
    }
  }

  // Cooley-Tukey FFT
  for (let size = 2; size <= n; size *= 2) {
    const halfSize = size / 2
    const angle = -2 * Math.PI / size

    for (let i = 0; i < n; i += size) {
      for (let j = 0; j < halfSize; j++) {
        const theta = angle * j
        const cosTheta = Math.cos(theta)
        const sinTheta = Math.sin(theta)

        const idx1 = i + j
        const idx2 = i + j + halfSize

        const tReal = real[idx2] * cosTheta - imag[idx2] * sinTheta
        const tImag = real[idx2] * sinTheta + imag[idx2] * cosTheta

        real[idx2] = real[idx1] - tReal
        imag[idx2] = imag[idx1] - tImag
        real[idx1] = real[idx1] + tReal
        imag[idx1] = imag[idx1] + tImag
      }
    }
  }

  return { real, imag }
}

/**
 * Bit reversal for FFT
 */
function bitReverse(x: number, bits: number): number {
  let result = 0
  for (let i = 0; i < bits; i++) {
    result = (result << 1) | (x & 1)
    x >>= 1
  }
  return result
}

/**
 * Find next power of 2
 */
export function nextPowerOf2(n: number): number {
  let p = 1
  while (p < n) {
    p *= 2
  }
  return p
}

/**
 * Alias for computeFFT (for backward compatibility)
 */
export const realFFT = computeFFT

/**
 * Apply Hann window to signal
 */
export function applyHannWindow(signal: Float32Array): Float32Array {
  const n = signal.length
  const windowed = new Float32Array(n)

  for (let i = 0; i < n; i++) {
    const w = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)))
    windowed[i] = signal[i] * w
  }

  return windowed
}

/**
 * Compute power spectral density
 */
export function computePSD(signal: Float32Array, sampleRate: number): FFTResult {
  const windowed = applyHannWindow(signal)
  const result = computeFFT(windowed, sampleRate)

  // Square magnitudes for PSD
  for (let i = 0; i < result.magnitudes.length; i++) {
    result.magnitudes[i] = result.magnitudes[i] * result.magnitudes[i]
  }

  return result
}
