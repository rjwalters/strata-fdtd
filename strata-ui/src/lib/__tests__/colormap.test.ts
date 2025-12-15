import { describe, it, expect } from "vitest"
import * as THREE from "three"
import {
  pressureColormap,
  applyPressureColormap,
  magnitudeColormap,
  getRange,
  getSymmetricRange,
} from "../colormap"

describe("pressureColormap", () => {
  it("returns white for zero-centered values at midpoint", () => {
    const color = pressureColormap(0, -1, 1)
    expect(color.r).toBeCloseTo(1, 5)
    expect(color.g).toBeCloseTo(1, 5)
    expect(color.b).toBeCloseTo(1, 5)
  })

  it("returns blue for minimum values", () => {
    const color = pressureColormap(-1, -1, 1)
    // Blue-600 is #2563eb = rgb(37, 99, 235)
    expect(color.r).toBeLessThan(0.5)
    expect(color.b).toBeGreaterThan(0.5)
  })

  it("returns red for maximum values", () => {
    const color = pressureColormap(1, -1, 1)
    // Red-600 is #dc2626 = rgb(220, 38, 38)
    expect(color.r).toBeGreaterThan(0.5)
    expect(color.b).toBeLessThan(0.5)
  })

  it("handles equal min and max", () => {
    const color = pressureColormap(5, 5, 5)
    // Should return white
    expect(color.r).toBeCloseTo(1, 5)
    expect(color.g).toBeCloseTo(1, 5)
    expect(color.b).toBeCloseTo(1, 5)
  })

  it("interpolates smoothly", () => {
    const low = pressureColormap(-0.5, -1, 1)
    const mid = pressureColormap(0, -1, 1)
    const high = pressureColormap(0.5, -1, 1)

    // Low should have less red than mid (which is white)
    expect(low.r).toBeLessThan(mid.r)
    // High should have more red than low
    expect(high.r).toBeGreaterThan(low.r)
    // Low should have more blue than high
    expect(low.b).toBeGreaterThan(high.b)
  })
})

describe("applyPressureColormap", () => {
  it("modifies existing color object", () => {
    const target = new THREE.Color(0, 0, 0)
    applyPressureColormap(0, -1, 1, target)

    expect(target.r).toBeCloseTo(1, 5)
    expect(target.g).toBeCloseTo(1, 5)
    expect(target.b).toBeCloseTo(1, 5)
  })

  it("produces same result as pressureColormap", () => {
    const target = new THREE.Color()
    applyPressureColormap(-0.5, -1, 1, target)
    const color = pressureColormap(-0.5, -1, 1)

    expect(target.r).toBeCloseTo(color.r, 5)
    expect(target.g).toBeCloseTo(color.g, 5)
    expect(target.b).toBeCloseTo(color.b, 5)
  })
})

describe("magnitudeColormap", () => {
  it("returns dark color for minimum values", () => {
    const color = magnitudeColormap(0, 0, 1)
    // Should be dark (low luminance)
    const luminance = 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b
    expect(luminance).toBeLessThan(0.5)
  })

  it("returns lighter color for maximum values", () => {
    const color = magnitudeColormap(1, 0, 1)
    // Should be lighter (higher luminance)
    const luminance = 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b
    expect(luminance).toBeGreaterThan(0.3)
  })

  it("handles equal min and max", () => {
    const color = magnitudeColormap(5, 5, 5)
    // Should return first color in scale (dark)
    expect(color.r).toBeDefined()
    expect(color.g).toBeDefined()
    expect(color.b).toBeDefined()
  })
})

describe("getRange", () => {
  it("finds min and max of Float32Array", () => {
    const data = new Float32Array([-3, 1, 5, -2, 4])
    const [min, max] = getRange(data)
    expect(min).toBe(-3)
    expect(max).toBe(5)
  })

  it("handles single element array", () => {
    const data = new Float32Array([42])
    const [min, max] = getRange(data)
    expect(min).toBe(42)
    expect(max).toBe(42)
  })

  it("skips NaN values", () => {
    const data = new Float32Array([1, NaN, 5, NaN, 2])
    const [min, max] = getRange(data)
    expect(min).toBe(1)
    expect(max).toBe(5)
  })

  it("skips Infinity values", () => {
    const data = new Float32Array([1, Infinity, 5, -Infinity, 2])
    const [min, max] = getRange(data)
    expect(min).toBe(1)
    expect(max).toBe(5)
  })

  it("handles all NaN/Infinity array", () => {
    const data = new Float32Array([NaN, Infinity, -Infinity])
    const [min, max] = getRange(data)
    expect(min).toBe(0)
    expect(max).toBe(0)
  })
})

describe("getSymmetricRange", () => {
  it("returns symmetric range around zero", () => {
    const data = new Float32Array([-3, 1, 5, -2, 4])
    const [min, max] = getSymmetricRange(data)
    expect(min).toBe(-5)
    expect(max).toBe(5)
  })

  it("handles all positive values", () => {
    const data = new Float32Array([1, 2, 3])
    const [min, max] = getSymmetricRange(data)
    expect(min).toBe(-3)
    expect(max).toBe(3)
  })

  it("handles all negative values", () => {
    const data = new Float32Array([-5, -2, -1])
    const [min, max] = getSymmetricRange(data)
    expect(min).toBe(-5)
    expect(max).toBe(5)
  })

  it("skips NaN values", () => {
    const data = new Float32Array([1, NaN, -2])
    const [min, max] = getSymmetricRange(data)
    expect(min).toBe(-2)
    expect(max).toBe(2)
  })
})
