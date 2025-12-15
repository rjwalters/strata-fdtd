/**
 * Tests for scriptParser module
 */

import { describe, it, expect } from 'vitest'
import { parseScript, getDefaultScript } from '../scriptParser'

describe('scriptParser', () => {
  describe('getDefaultScript', () => {
    it('should return a non-empty default script', () => {
      const script = getDefaultScript()
      expect(script.length).toBeGreaterThan(0)
    })

    it('should include UniformGrid definition', () => {
      const script = getDefaultScript()
      expect(script).toContain('UniformGrid')
    })

    it('should include example probe', () => {
      const script = getDefaultScript()
      expect(script).toContain('Probe')
    })
  })

  describe('parseScript - Grid parsing', () => {
    it('should parse a simple grid definition', () => {
      const script = `
grid = UniformGrid(
    resolution=1e-3,
    shape=(100, 100, 50)
)`
      const ast = parseScript(script)
      expect(ast.grid).not.toBeNull()
      expect(ast.grid?.resolution).toBe(0.001)
      expect(ast.grid?.shape).toEqual([100, 100, 50])
      expect(ast.hasValidGrid).toBe(true)
    })

    it('should calculate extent from shape and resolution', () => {
      const script = `grid = UniformGrid(resolution=1e-3, shape=(10, 20, 30))`
      const ast = parseScript(script)
      expect(ast.grid?.extent).toEqual([0.01, 0.02, 0.03])
    })

    it('should return null grid when no UniformGrid is defined', () => {
      const script = `# Just a comment`
      const ast = parseScript(script)
      expect(ast.grid).toBeNull()
      expect(ast.hasValidGrid).toBe(false)
    })

    it('should ignore UniformGrid in comments', () => {
      const script = `# grid = UniformGrid(resolution=1e-3, shape=(10, 10, 10))`
      const ast = parseScript(script)
      expect(ast.grid).toBeNull()
    })

    it('should parse multiline grid definitions', () => {
      const script = `
grid = UniformGrid(
    resolution=2e-3,  # 2mm resolution
    shape=(50, 50, 25)  # 50x50x25 cells
)
`
      const ast = parseScript(script)
      expect(ast.grid?.resolution).toBe(0.002)
      expect(ast.grid?.shape).toEqual([50, 50, 25])
    })
  })

  describe('parseScript - Material parsing', () => {
    it('should parse rectangle materials', () => {
      const script = `
grid = UniformGrid(resolution=1e-3, shape=(10, 10, 10))
mat = Material.rectangle(
    center=(0.05, 0.05, 0.025),
    size=(0.02, 0.02, 0.005),
    material="pzt5"
)
`
      const ast = parseScript(script)
      expect(ast.materials.length).toBe(1)
      expect(ast.materials[0].type).toBe('rectangle')
      expect(ast.materials[0].center).toEqual([0.05, 0.05, 0.025])
      expect(ast.materials[0].size).toEqual([0.02, 0.02, 0.005])
      expect(ast.materials[0].material).toBe('pzt5')
    })

    it('should parse sphere materials', () => {
      const script = `
grid = UniformGrid(resolution=1e-3, shape=(10, 10, 10))
mat = Material.sphere(
    center=(0.05, 0.05, 0.05),
    radius=0.01,
    material="water"
)
`
      const ast = parseScript(script)
      expect(ast.materials.length).toBe(1)
      expect(ast.materials[0].type).toBe('sphere')
      expect(ast.materials[0].center).toEqual([0.05, 0.05, 0.05])
      expect(ast.materials[0].radius).toBe(0.01)
      expect(ast.materials[0].material).toBe('water')
    })

    it('should parse multiple materials', () => {
      const script = `
grid = UniformGrid(resolution=1e-3, shape=(10, 10, 10))
mat1 = Material.rectangle(center=(0.05, 0.05, 0.025), size=(0.02, 0.02, 0.005), material="pzt5")
mat2 = Material.sphere(center=(0.05, 0.05, 0.05), radius=0.01, material="water")
`
      const ast = parseScript(script)
      expect(ast.materials.length).toBe(2)
    })

    it('should ignore materials in comments', () => {
      const script = `
grid = UniformGrid(resolution=1e-3, shape=(10, 10, 10))
# mat = Material.rectangle(center=(0.05, 0.05, 0.025), size=(0.02, 0.02, 0.005), material="pzt5")
`
      const ast = parseScript(script)
      expect(ast.materials.length).toBe(0)
    })
  })

  describe('parseScript - Source parsing', () => {
    it('should parse gaussian pulse source', () => {
      const script = `
grid = UniformGrid(resolution=1e-3, shape=(10, 10, 10))
src = Source.gaussian_pulse(
    position=(0.05, 0.05, 0.025),
    frequency=1e6,
    amplitude=1.0
)
`
      const ast = parseScript(script)
      expect(ast.sources.length).toBe(1)
      expect(ast.sources[0].type).toBe('GaussianPulse')
      expect(ast.sources[0].position).toEqual([0.05, 0.05, 0.025])
      expect(ast.sources[0].frequency).toBe(1e6)
      expect(ast.sources[0].amplitude).toBe(1.0)
    })

    it('should parse continuous wave source', () => {
      const script = `
grid = UniformGrid(resolution=1e-3, shape=(10, 10, 10))
src = Source.continuous_wave(
    position=(0.05, 0.05, 0.025),
    frequency=500e3
)
`
      const ast = parseScript(script)
      expect(ast.sources.length).toBe(1)
      expect(ast.sources[0].type).toBe('ContinuousWave')
    })

    it('should ignore sources in comments', () => {
      const script = `
grid = UniformGrid(resolution=1e-3, shape=(10, 10, 10))
# src = Source.gaussian_pulse(position=(0.05, 0.05, 0.025), frequency=1e6)
`
      const ast = parseScript(script)
      expect(ast.sources.length).toBe(0)
    })
  })

  describe('parseScript - Probe parsing', () => {
    it('should parse point probes', () => {
      const script = `
grid = UniformGrid(resolution=1e-3, shape=(10, 10, 10))
probe = Probe.point(
    position=(0.05, 0.05, 0.04),
    name="far_field"
)
`
      const ast = parseScript(script)
      expect(ast.probes.length).toBe(1)
      expect(ast.probes[0].position).toEqual([0.05, 0.05, 0.04])
      expect(ast.probes[0].name).toBe('far_field')
    })

    it('should ignore probes in comments', () => {
      const script = `
grid = UniformGrid(resolution=1e-3, shape=(10, 10, 10))
# probe = Probe.point(position=(0.05, 0.05, 0.04), name="test")
`
      const ast = parseScript(script)
      expect(ast.probes.length).toBe(0)
    })
  })

  describe('parseScript - Error handling', () => {
    it('should report error for negative resolution', () => {
      const script = `grid = UniformGrid(resolution=-1e-3, shape=(10, 10, 10))`
      const ast = parseScript(script)
      expect(ast.errors.length).toBeGreaterThan(0)
      expect(ast.errors.some(e => e.includes('positive'))).toBe(true)
    })

    it('should report error for invalid shape', () => {
      const script = `grid = UniformGrid(resolution=1e-3, shape=(0, 10, 10))`
      const ast = parseScript(script)
      expect(ast.errors.length).toBeGreaterThan(0)
    })
  })

  describe('parseScript - Default script', () => {
    it('should parse the default script successfully', () => {
      const script = getDefaultScript()
      const ast = parseScript(script)
      expect(ast.hasValidGrid).toBe(true)
      expect(ast.grid).not.toBeNull()
      expect(ast.materials.length).toBeGreaterThan(0)
      expect(ast.sources.length).toBeGreaterThan(0)
      expect(ast.probes.length).toBeGreaterThan(0)
      expect(ast.errors.length).toBe(0)
    })
  })
})
