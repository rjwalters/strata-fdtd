/**
 * FlowParticleRenderer - GPU-accelerated particle visualization for acoustic flow.
 *
 * Renders particles that are advected by the velocity field and colored by
 * local pressure, providing an intuitive representation of acoustic wave propagation.
 *
 * Features:
 * - Particle advection using trilinear interpolation of velocity field
 * - Pressure-based coloring with diverging colormap (blue → white → red)
 * - Automatic respawning of particles that exit domain or enter solids
 * - Configurable particle count, size, and time scale
 */

import { useEffect, useRef, useCallback, useMemo } from "react";
import * as THREE from "three";
import type { VelocitySnapshot } from "../lib/loaders";
import type { FlowParticleConfig } from "../stores/simulationStore";

export interface FlowParticleRendererProps {
  /** Pressure field for coloring particles */
  pressure: Float32Array | null;
  /** Velocity field for advection */
  velocity: VelocitySnapshot | null;
  /** Grid dimensions [nx, ny, nz] */
  shape: [number, number, number];
  /** Grid resolution in meters */
  resolution: number;
  /** Geometry mask (true = air, false = solid) */
  geometryMask?: Uint8Array | null;
  /** Flow particle configuration */
  config: FlowParticleConfig;
  /** Three.js scene to add particles to */
  scene: THREE.Scene | null;
  /** Pressure range for colormap normalization */
  pressureRange: [number, number];
  /** Delta time since last update (in simulation time) */
  deltaTime: number;
}

// Colormap for pressure visualization: blue (rarefaction) → white → red (compression)
const PRESSURE_COLORMAP = {
  negative: new THREE.Color(0x0077ff), // Blue for rarefaction
  neutral: new THREE.Color(0xffffff), // White for neutral
  positive: new THREE.Color(0xff3333), // Red for compression
};

/**
 * Trilinear interpolation of a 3D scalar field at a given position.
 */
function sampleField(
  field: Float32Array,
  shape: [number, number, number],
  x: number,
  y: number,
  z: number
): number {
  const [nx, ny, nz] = shape;

  // Clamp to valid range
  x = Math.max(0, Math.min(nx - 1.001, x));
  y = Math.max(0, Math.min(ny - 1.001, y));
  z = Math.max(0, Math.min(nz - 1.001, z));

  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const z0 = Math.floor(z);
  const x1 = Math.min(x0 + 1, nx - 1);
  const y1 = Math.min(y0 + 1, ny - 1);
  const z1 = Math.min(z0 + 1, nz - 1);

  const xd = x - x0;
  const yd = y - y0;
  const zd = z - z0;

  // Get corner values
  const idx = (i: number, j: number, k: number) => i + j * nx + k * nx * ny;
  const c000 = field[idx(x0, y0, z0)];
  const c100 = field[idx(x1, y0, z0)];
  const c010 = field[idx(x0, y1, z0)];
  const c110 = field[idx(x1, y1, z0)];
  const c001 = field[idx(x0, y0, z1)];
  const c101 = field[idx(x1, y0, z1)];
  const c011 = field[idx(x0, y1, z1)];
  const c111 = field[idx(x1, y1, z1)];

  // Trilinear interpolation
  const c00 = c000 * (1 - xd) + c100 * xd;
  const c01 = c001 * (1 - xd) + c101 * xd;
  const c10 = c010 * (1 - xd) + c110 * xd;
  const c11 = c011 * (1 - xd) + c111 * xd;

  const c0 = c00 * (1 - yd) + c10 * yd;
  const c1 = c01 * (1 - yd) + c11 * yd;

  return c0 * (1 - zd) + c1 * zd;
}

/**
 * Sample velocity (3 components) at a position using trilinear interpolation.
 * Velocity is stored as interleaved [vx,vy,vz,vx,vy,vz,...].
 */
function sampleVelocity(
  velocity: Float32Array,
  shape: [number, number, number],
  x: number,
  y: number,
  z: number
): [number, number, number] {
  const [nx, ny, nz] = shape;

  // Clamp to valid range
  x = Math.max(0, Math.min(nx - 1.001, x));
  y = Math.max(0, Math.min(ny - 1.001, y));
  z = Math.max(0, Math.min(nz - 1.001, z));

  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const z0 = Math.floor(z);
  const x1 = Math.min(x0 + 1, nx - 1);
  const y1 = Math.min(y0 + 1, ny - 1);
  const z1 = Math.min(z0 + 1, nz - 1);

  const xd = x - x0;
  const yd = y - y0;
  const zd = z - z0;

  // Get corner indices (3 values per cell)
  const idx = (i: number, j: number, k: number) => (i + j * nx + k * nx * ny) * 3;

  const result: [number, number, number] = [0, 0, 0];

  // Interpolate each velocity component
  for (let c = 0; c < 3; c++) {
    const c000 = velocity[idx(x0, y0, z0) + c];
    const c100 = velocity[idx(x1, y0, z0) + c];
    const c010 = velocity[idx(x0, y1, z0) + c];
    const c110 = velocity[idx(x1, y1, z0) + c];
    const c001 = velocity[idx(x0, y0, z1) + c];
    const c101 = velocity[idx(x1, y0, z1) + c];
    const c011 = velocity[idx(x0, y1, z1) + c];
    const c111 = velocity[idx(x1, y1, z1) + c];

    const c00 = c000 * (1 - xd) + c100 * xd;
    const c01 = c001 * (1 - xd) + c101 * xd;
    const c10 = c010 * (1 - xd) + c110 * xd;
    const c11 = c011 * (1 - xd) + c111 * xd;

    const c0Val = c00 * (1 - yd) + c10 * yd;
    const c1Val = c01 * (1 - yd) + c11 * yd;

    result[c] = c0Val * (1 - zd) + c1Val * zd;
  }

  return result;
}

/**
 * Check if a position is inside a solid (geometry mask is false at that cell).
 */
function isInSolid(
  geometryMask: Uint8Array | null,
  shape: [number, number, number],
  x: number,
  y: number,
  z: number
): boolean {
  if (!geometryMask) return false;

  const [nx, ny, nz] = shape;
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const iz = Math.floor(z);

  if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) {
    return true; // Outside domain = treat as solid
  }

  const idx = ix + iy * nx + iz * nx * ny;
  return geometryMask[idx] === 0; // 0 = solid
}

/**
 * Get a color from the diverging pressure colormap.
 */
function getPressureColor(
  normalizedPressure: number, // -1 to 1
  color: THREE.Color
): void {
  if (normalizedPressure < 0) {
    // Interpolate blue → white
    color.lerpColors(
      PRESSURE_COLORMAP.negative,
      PRESSURE_COLORMAP.neutral,
      normalizedPressure + 1
    );
  } else {
    // Interpolate white → red
    color.lerpColors(
      PRESSURE_COLORMAP.neutral,
      PRESSURE_COLORMAP.positive,
      normalizedPressure
    );
  }
}

export function FlowParticleRenderer({
  pressure,
  velocity,
  shape,
  resolution,
  geometryMask,
  config,
  scene,
  pressureRange,
  deltaTime,
}: FlowParticleRendererProps) {
  const pointsRef = useRef<THREE.Points | null>(null);
  const trailLinesRef = useRef<THREE.LineSegments | null>(null);
  const particlePositionsRef = useRef<Float32Array | null>(null);
  const particleColorsRef = useRef<Float32Array | null>(null);
  // Trail history: circular buffer of past positions for each particle
  // Shape: [particleCount][trailLength][3] flattened to [particleCount * trailLength * 3]
  const trailHistoryRef = useRef<Float32Array | null>(null);
  // Current write index in circular buffer for each particle
  const trailIndexRef = useRef<Uint16Array | null>(null);
  // Track the particle count used to create the current arrays
  const currentParticleCountRef = useRef<number>(0);
  // Track trail config to detect changes
  const currentTrailLengthRef = useRef<number>(0);

  const [nx, ny, nz] = shape;
  const physicalSize = useMemo(
    () => [nx * resolution, ny * resolution, nz * resolution],
    [nx, ny, nz, resolution]
  );

  // Initialize particles at random positions in air cells
  const initializeParticles = useCallback(() => {
    const positions = new Float32Array(config.particleCount * 3);
    const colors = new Float32Array(config.particleCount * 3);
    // Trail history: each particle has trailLength positions (x,y,z)
    const trailHistory = new Float32Array(config.particleCount * config.trailLength * 3);
    // Trail write index for each particle's circular buffer
    const trailIndex = new Uint16Array(config.particleCount);

    for (let i = 0; i < config.particleCount; i++) {
      // Random position in grid coordinates
      let x, y, z;
      let attempts = 0;
      do {
        x = Math.random() * nx;
        y = Math.random() * ny;
        z = Math.random() * nz;
        attempts++;
      } while (isInSolid(geometryMask ?? null, shape, x, y, z) && attempts < 10);

      // Convert to world coordinates (centered at origin)
      const wx = (x / nx - 0.5) * physicalSize[0];
      const wy = (y / ny - 0.5) * physicalSize[1];
      const wz = (z / nz - 0.5) * physicalSize[2];

      positions[i * 3] = wx;
      positions[i * 3 + 1] = wy;
      positions[i * 3 + 2] = wz;

      // Initial color (neutral white)
      colors[i * 3] = 1;
      colors[i * 3 + 1] = 1;
      colors[i * 3 + 2] = 1;

      // Initialize trail history with current position (all slots same)
      const historyBase = i * config.trailLength * 3;
      for (let t = 0; t < config.trailLength; t++) {
        trailHistory[historyBase + t * 3] = wx;
        trailHistory[historyBase + t * 3 + 1] = wy;
        trailHistory[historyBase + t * 3 + 2] = wz;
      }
      trailIndex[i] = 0;
    }

    particlePositionsRef.current = positions;
    particleColorsRef.current = colors;
    trailHistoryRef.current = trailHistory;
    trailIndexRef.current = trailIndex;
    currentParticleCountRef.current = config.particleCount;
    currentTrailLengthRef.current = config.trailLength;

    return { positions, colors, trailHistory, trailIndex };
  }, [config.particleCount, config.trailLength, nx, ny, nz, physicalSize, geometryMask, shape]);

  // Create Three.js Points object and trail LineSegments
  useEffect(() => {
    if (!scene) return;

    const { positions, colors } = initializeParticles();

    // Create particle points
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: config.particleSize,
      vertexColors: true,
      transparent: true,
      opacity: 0.8,
      sizeAttenuation: true,
      depthWrite: false,
    });

    const points = new THREE.Points(geometry, material);
    points.name = "flowParticles";
    scene.add(points);
    pointsRef.current = points;

    // Create trail line segments
    // Each particle has (trailLength - 1) segments, each segment has 2 vertices
    const numSegments = config.particleCount * (config.trailLength - 1);
    const trailPositions = new Float32Array(numSegments * 2 * 3);
    const trailColors = new Float32Array(numSegments * 2 * 4); // RGBA for opacity fade

    const trailGeometry = new THREE.BufferGeometry();
    trailGeometry.setAttribute("position", new THREE.BufferAttribute(trailPositions, 3));
    trailGeometry.setAttribute("color", new THREE.BufferAttribute(trailColors, 4));

    const trailMaterial = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 1.0,
      depthWrite: false,
      linewidth: 1, // Note: linewidth > 1 not supported in WebGL
    });

    const trailLines = new THREE.LineSegments(trailGeometry, trailMaterial);
    trailLines.name = "flowParticleTrails";
    trailLines.visible = config.showTrails;
    scene.add(trailLines);
    trailLinesRef.current = trailLines;

    return () => {
      scene.remove(points);
      geometry.dispose();
      material.dispose();
      pointsRef.current = null;

      scene.remove(trailLines);
      trailGeometry.dispose();
      trailMaterial.dispose();
      trailLinesRef.current = null;
    };
  }, [scene, initializeParticles, config.particleSize, config.particleCount, config.trailLength]);

  // Update particle positions, colors, and trails
  useEffect(() => {
    if (
      !pointsRef.current ||
      !velocity ||
      !pressure ||
      !particlePositionsRef.current ||
      !particleColorsRef.current ||
      !trailHistoryRef.current ||
      !trailIndexRef.current
    ) {
      return;
    }

    // Guard against particle count mismatch during recreation
    // Skip update if arrays don't match the current config particle count
    if (
      currentParticleCountRef.current !== config.particleCount ||
      currentTrailLengthRef.current !== config.trailLength
    ) {
      return;
    }

    const positions = particlePositionsRef.current;
    const colors = particleColorsRef.current;
    const trailHistory = trailHistoryRef.current;
    const trailIndex = trailIndexRef.current;
    const velocityData = velocity.velocity;
    const tempColor = new THREE.Color();
    const [pMin, pMax] = pressureRange;
    const pRange = pMax - pMin || 1;
    const particleCount = currentParticleCountRef.current;
    const trailLength = currentTrailLengthRef.current;

    // Advect particles
    for (let i = 0; i < particleCount; i++) {
      // Get current world position
      const wx = positions[i * 3];
      const wy = positions[i * 3 + 1];
      const wz = positions[i * 3 + 2];

      // Store current position in trail history BEFORE moving
      const historyBase = i * trailLength * 3;
      const writeIdx = trailIndex[i];
      trailHistory[historyBase + writeIdx * 3] = wx;
      trailHistory[historyBase + writeIdx * 3 + 1] = wy;
      trailHistory[historyBase + writeIdx * 3 + 2] = wz;
      // Advance circular buffer index
      trailIndex[i] = (writeIdx + 1) % trailLength;

      // Convert to grid coordinates
      let gx = (wx / physicalSize[0] + 0.5) * nx;
      let gy = (wy / physicalSize[1] + 0.5) * ny;
      let gz = (wz / physicalSize[2] + 0.5) * nz;

      // Sample velocity at current position
      const [vx, vy, vz] = sampleVelocity(velocityData, shape, gx, gy, gz);

      // Advect particle (scaled by timeScale for slow-motion effect)
      const dt = deltaTime * config.timeScale;
      gx += (vx / resolution) * dt;
      gy += (vy / resolution) * dt;
      gz += (vz / resolution) * dt;

      // Check if particle needs respawning
      const outOfBounds =
        gx < 0 || gx >= nx || gy < 0 || gy >= ny || gz < 0 || gz >= nz;
      const inSolid = isInSolid(geometryMask ?? null, shape, gx, gy, gz);

      let respawned = false;
      if (outOfBounds || inSolid) {
        // Respawn at random air position
        let attempts = 0;
        do {
          gx = Math.random() * nx;
          gy = Math.random() * ny;
          gz = Math.random() * nz;
          attempts++;
        } while (
          isInSolid(geometryMask ?? null, shape, gx, gy, gz) &&
          attempts < 10
        );
        respawned = true;
      }

      // Convert back to world coordinates
      const newWx = (gx / nx - 0.5) * physicalSize[0];
      const newWy = (gy / ny - 0.5) * physicalSize[1];
      const newWz = (gz / nz - 0.5) * physicalSize[2];

      positions[i * 3] = newWx;
      positions[i * 3 + 1] = newWy;
      positions[i * 3 + 2] = newWz;

      // If respawned, clear trail history to new position
      if (respawned) {
        for (let t = 0; t < trailLength; t++) {
          trailHistory[historyBase + t * 3] = newWx;
          trailHistory[historyBase + t * 3 + 1] = newWy;
          trailHistory[historyBase + t * 3 + 2] = newWz;
        }
        trailIndex[i] = 0;
      }

      // Sample pressure for color
      const p = sampleField(pressure, shape, gx, gy, gz);
      const normalizedP = ((p - pMin) / pRange) * 2 - 1; // Map to -1..1
      getPressureColor(Math.max(-1, Math.min(1, normalizedP)), tempColor);

      colors[i * 3] = tempColor.r;
      colors[i * 3 + 1] = tempColor.g;
      colors[i * 3 + 2] = tempColor.b;
    }

    // Update particle geometry buffers
    const geom = pointsRef.current.geometry;
    const positionAttr = geom.getAttribute("position") as THREE.BufferAttribute;
    const colorAttr = geom.getAttribute("color") as THREE.BufferAttribute;

    positionAttr.array.set(positions);
    colorAttr.array.set(colors);

    positionAttr.needsUpdate = true;
    colorAttr.needsUpdate = true;

    // Update trail geometry if trails are visible
    if (config.showTrails && trailLinesRef.current) {
      const trailGeom = trailLinesRef.current.geometry;
      const trailPosAttr = trailGeom.getAttribute("position") as THREE.BufferAttribute;
      const trailColorAttr = trailGeom.getAttribute("color") as THREE.BufferAttribute;
      const trailPositions = trailPosAttr.array as Float32Array;
      const trailColors = trailColorAttr.array as Float32Array;

      // Build line segments from trail history
      // Each particle contributes (trailLength - 1) segments
      const segmentsPerParticle = trailLength - 1;

      for (let i = 0; i < particleCount; i++) {
        const historyBase = i * trailLength * 3;
        const currentIdx = trailIndex[i]; // Points to next write slot (oldest)
        const segmentBase = i * segmentsPerParticle * 2 * 3;
        const colorBase = i * segmentsPerParticle * 2 * 4;

        // Get particle color for trail
        const r = colors[i * 3];
        const g = colors[i * 3 + 1];
        const b = colors[i * 3 + 2];

        for (let s = 0; s < segmentsPerParticle; s++) {
          // Read from circular buffer: oldest to newest
          // currentIdx is where we'll write next, so it's the oldest
          const idx0 = (currentIdx + s) % trailLength;
          const idx1 = (currentIdx + s + 1) % trailLength;

          // Segment start vertex
          trailPositions[segmentBase + s * 6] = trailHistory[historyBase + idx0 * 3];
          trailPositions[segmentBase + s * 6 + 1] = trailHistory[historyBase + idx0 * 3 + 1];
          trailPositions[segmentBase + s * 6 + 2] = trailHistory[historyBase + idx0 * 3 + 2];

          // Segment end vertex
          trailPositions[segmentBase + s * 6 + 3] = trailHistory[historyBase + idx1 * 3];
          trailPositions[segmentBase + s * 6 + 4] = trailHistory[historyBase + idx1 * 3 + 1];
          trailPositions[segmentBase + s * 6 + 5] = trailHistory[historyBase + idx1 * 3 + 2];

          // Opacity fade: oldest (tail) = 0, newest (head) = 1
          const alpha0 = (s + 1) / trailLength;
          const alpha1 = (s + 2) / trailLength;

          // Color for start vertex (RGBA)
          trailColors[colorBase + s * 8] = r;
          trailColors[colorBase + s * 8 + 1] = g;
          trailColors[colorBase + s * 8 + 2] = b;
          trailColors[colorBase + s * 8 + 3] = alpha0;

          // Color for end vertex (RGBA)
          trailColors[colorBase + s * 8 + 4] = r;
          trailColors[colorBase + s * 8 + 5] = g;
          trailColors[colorBase + s * 8 + 6] = b;
          trailColors[colorBase + s * 8 + 7] = alpha1;
        }
      }

      trailPosAttr.needsUpdate = true;
      trailColorAttr.needsUpdate = true;
    }
  }, [
    velocity,
    pressure,
    deltaTime,
    config.particleCount,
    config.trailLength,
    config.timeScale,
    config.showTrails,
    shape,
    resolution,
    physicalSize,
    pressureRange,
    geometryMask,
  ]);

  // Update particle size when config changes
  useEffect(() => {
    if (pointsRef.current) {
      const material = pointsRef.current.material as THREE.PointsMaterial;
      material.size = config.particleSize;
    }
  }, [config.particleSize]);

  // Update trail visibility when showTrails changes
  useEffect(() => {
    if (trailLinesRef.current) {
      trailLinesRef.current.visible = config.showTrails;
    }
  }, [config.showTrails]);

  return null; // This component manages Three.js objects directly
}
