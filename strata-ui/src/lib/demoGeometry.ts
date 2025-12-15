/**
 * Demo geometry generation for testing the GeometryOverlay component.
 *
 * Creates sample FDTD geometries like Helmholtz resonators for visualization testing.
 */

/**
 * Generate a simple Helmholtz resonator geometry.
 *
 * Creates a cavity with a neck opening, centered in the grid.
 *
 * @param shape - Grid dimensions [nx, ny, nz]
 * @returns Uint8Array mask (1=air, 0=solid)
 */
export function createHelmholtzResonator(
  shape: [number, number, number]
): Uint8Array {
  const [nx, ny, nz] = shape;
  const mask = new Uint8Array(nx * ny * nz);

  // Fill with air initially
  mask.fill(1);

  const idx = (x: number, y: number, z: number) => x + y * nx + z * nx * ny;

  // Define resonator parameters (as fractions of grid size)
  const cavityWidth = Math.floor(nx * 0.4);
  const cavityHeight = Math.floor(ny * 0.4);
  const cavityDepth = Math.floor(nz * 0.4);
  const wallThickness = Math.max(2, Math.floor(nx * 0.05));
  const neckWidth = Math.floor(nx * 0.15);
  const neckLength = Math.floor(ny * 0.2);

  // Center position
  const cx = Math.floor(nx / 2);
  const cy = Math.floor(ny / 2);
  const cz = Math.floor(nz / 2);

  // Create outer box (solid)
  const outerW = cavityWidth + 2 * wallThickness;
  const outerH = cavityHeight + 2 * wallThickness + neckLength;
  const outerD = cavityDepth + 2 * wallThickness;

  for (let z = cz - Math.floor(outerD / 2); z < cz + Math.ceil(outerD / 2); z++) {
    for (let y = cy - Math.floor(outerH / 2); y < cy + Math.ceil(outerH / 2); y++) {
      for (let x = cx - Math.floor(outerW / 2); x < cx + Math.ceil(outerW / 2); x++) {
        if (x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz) {
          mask[idx(x, y, z)] = 0; // solid
        }
      }
    }
  }

  // Carve out cavity (air)
  const cavityStartY = cy - Math.floor(outerH / 2) + wallThickness;
  for (let z = cz - Math.floor(cavityDepth / 2); z < cz + Math.ceil(cavityDepth / 2); z++) {
    for (let y = cavityStartY; y < cavityStartY + cavityHeight; y++) {
      for (let x = cx - Math.floor(cavityWidth / 2); x < cx + Math.ceil(cavityWidth / 2); x++) {
        if (x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz) {
          mask[idx(x, y, z)] = 1; // air
        }
      }
    }
  }

  // Carve out neck (air)
  const neckStartY = cavityStartY + cavityHeight;
  for (let z = cz - Math.floor(neckWidth / 2); z < cz + Math.ceil(neckWidth / 2); z++) {
    for (let y = neckStartY; y < neckStartY + neckLength + wallThickness; y++) {
      for (let x = cx - Math.floor(neckWidth / 2); x < cx + Math.ceil(neckWidth / 2); x++) {
        if (x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz) {
          mask[idx(x, y, z)] = 1; // air
        }
      }
    }
  }

  return mask;
}

/**
 * Generate a simple rectangular duct with a constriction.
 *
 * @param shape - Grid dimensions [nx, ny, nz]
 * @returns Uint8Array mask (1=air, 0=solid)
 */
export function createDuctWithConstriction(
  shape: [number, number, number]
): Uint8Array {
  const [nx, ny, nz] = shape;
  const mask = new Uint8Array(nx * ny * nz);

  // Fill with air initially
  mask.fill(1);

  const idx = (x: number, y: number, z: number) => x + y * nx + z * nx * ny;

  const ductWidth = Math.floor(nx * 0.6);
  const wallThickness = Math.max(2, Math.floor(nx * 0.08));
  const constrictionWidth = Math.floor(nx * 0.25);
  const constrictionLength = Math.floor(ny * 0.2);

  const cx = Math.floor(nx / 2);
  const cz = Math.floor(nz / 2);

  // Create duct walls (solid) along Y axis
  for (let y = 0; y < ny; y++) {
    for (let z = cz - Math.floor(ductWidth / 2) - wallThickness; z < cz + Math.ceil(ductWidth / 2) + wallThickness; z++) {
      for (let x = cx - Math.floor(ductWidth / 2) - wallThickness; x < cx + Math.ceil(ductWidth / 2) + wallThickness; x++) {
        if (x >= 0 && x < nx && z >= 0 && z < nz) {
          // Check if this is the wall region
          const inXWall = x < cx - Math.floor(ductWidth / 2) || x >= cx + Math.ceil(ductWidth / 2);
          const inZWall = z < cz - Math.floor(ductWidth / 2) || z >= cz + Math.ceil(ductWidth / 2);

          if (inXWall || inZWall) {
            mask[idx(x, y, z)] = 0; // solid
          }
        }
      }
    }
  }

  // Add constriction in the middle
  const constrictionStart = Math.floor(ny / 2) - Math.floor(constrictionLength / 2);
  for (let y = constrictionStart; y < constrictionStart + constrictionLength; y++) {
    for (let z = cz - Math.floor(ductWidth / 2); z < cz + Math.ceil(ductWidth / 2); z++) {
      for (let x = cx - Math.floor(ductWidth / 2); x < cx + Math.ceil(ductWidth / 2); x++) {
        // Add solid material to narrow the duct
        const inConstrictionX = x < cx - Math.floor(constrictionWidth / 2) || x >= cx + Math.ceil(constrictionWidth / 2);
        const inConstrictionZ = z < cz - Math.floor(constrictionWidth / 2) || z >= cz + Math.ceil(constrictionWidth / 2);

        if (inConstrictionX || inConstrictionZ) {
          if (x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz) {
            mask[idx(x, y, z)] = 0; // solid
          }
        }
      }
    }
  }

  return mask;
}

/**
 * Generate a simple sphere (for testing curved surfaces).
 *
 * @param shape - Grid dimensions [nx, ny, nz]
 * @param radiusFraction - Sphere radius as fraction of smallest dimension
 * @returns Uint8Array mask (1=air, 0=solid)
 */
export function createSphere(
  shape: [number, number, number],
  radiusFraction: number = 0.3
): Uint8Array {
  const [nx, ny, nz] = shape;
  const mask = new Uint8Array(nx * ny * nz);

  // Fill with air initially
  mask.fill(1);

  const idx = (x: number, y: number, z: number) => x + y * nx + z * nx * ny;

  const cx = nx / 2;
  const cy = ny / 2;
  const cz = nz / 2;
  const radius = Math.min(nx, ny, nz) * radiusFraction;

  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const dx = x - cx + 0.5;
        const dy = y - cy + 0.5;
        const dz = z - cz + 0.5;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

        if (dist <= radius) {
          mask[idx(x, y, z)] = 0; // solid
        }
      }
    }
  }

  return mask;
}

/**
 * Available demo geometries.
 */
export type DemoGeometryType = "helmholtz" | "duct" | "sphere";

/**
 * Create demo geometry by type.
 */
export function createDemoGeometry(
  type: DemoGeometryType,
  shape: [number, number, number]
): Uint8Array {
  switch (type) {
    case "helmholtz":
      return createHelmholtzResonator(shape);
    case "duct":
      return createDuctWithConstriction(shape);
    case "sphere":
      return createSphere(shape);
    default:
      throw new Error(`Unknown demo geometry type: ${type}`);
  }
}

/**
 * Generate synthetic pressure data for demo visualization.
 *
 * Creates a standing wave pattern that matches the geometry, making it
 * possible to visualize voxels without loading simulation data.
 *
 * @param type - Demo geometry type
 * @param shape - Grid dimensions [nx, ny, nz]
 * @param time - Animation time parameter (0-1 for one period)
 * @returns Float32Array of pressure values
 */
export function createDemoPressure(
  type: DemoGeometryType,
  shape: [number, number, number],
  time: number = 0
): Float32Array {
  const [nx, ny, nz] = shape;
  const pressure = new Float32Array(nx * ny * nz);
  const mask = createDemoGeometry(type, shape);

  const cx = nx / 2;
  const cy = ny / 2;
  const cz = nz / 2;

  // Time-varying phase for animation
  const phase = time * Math.PI * 2;

  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const idx = x + y * nx + z * nx * ny;

        // Only generate pressure in air cells
        if (mask[idx] === 0) {
          pressure[idx] = 0;
          continue;
        }

        let value: number;

        switch (type) {
          case "helmholtz": {
            // Standing wave in cavity with resonance in neck
            const dy = (y - cy) / ny;
            const radial = Math.sqrt(
              ((x - cx) / nx) ** 2 + ((z - cz) / nz) ** 2
            );
            // Helmholtz mode: pressure maximum in cavity, velocity in neck
            value =
              Math.cos(dy * Math.PI * 2) *
              Math.exp(-radial * 3) *
              Math.cos(phase);
            break;
          }
          case "duct": {
            // Standing wave along the duct (Y axis)
            const ky = (y / ny) * Math.PI * 3; // 1.5 wavelengths
            value = Math.sin(ky) * Math.cos(phase);
            break;
          }
          case "sphere": {
            // Radial standing wave from center
            const r = Math.sqrt(
              ((x - cx) / nx) ** 2 +
                ((y - cy) / ny) ** 2 +
                ((z - cz) / nz) ** 2
            );
            value = Math.sin(r * Math.PI * 6) * Math.cos(phase);
            break;
          }
          default:
            value = 0;
        }

        pressure[idx] = value;
      }
    }
  }

  return pressure;
}
