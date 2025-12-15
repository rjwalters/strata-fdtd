/**
 * Tests for demo geometry generation.
 */

import { describe, it, expect } from "vitest";
import {
  createHelmholtzResonator,
  createDuctWithConstriction,
  createSphere,
  createDemoGeometry,
} from "../demoGeometry";

describe("createHelmholtzResonator", () => {
  it("creates array of correct size", () => {
    const shape: [number, number, number] = [30, 30, 30];
    const geometry = createHelmholtzResonator(shape);

    expect(geometry.length).toBe(27000);
  });

  it("contains both air and solid cells", () => {
    const shape: [number, number, number] = [30, 30, 30];
    const geometry = createHelmholtzResonator(shape);

    let airCount = 0;
    let solidCount = 0;
    for (let i = 0; i < geometry.length; i++) {
      if (geometry[i] === 1) airCount++;
      else if (geometry[i] === 0) solidCount++;
    }

    expect(airCount).toBeGreaterThan(0);
    expect(solidCount).toBeGreaterThan(0);
    expect(airCount + solidCount).toBe(27000);
  });

  it("has solid fraction between 5% and 30%", () => {
    const shape: [number, number, number] = [50, 50, 50];
    const geometry = createHelmholtzResonator(shape);

    let solidCount = 0;
    for (let i = 0; i < geometry.length; i++) {
      if (geometry[i] === 0) solidCount++;
    }

    const solidFraction = solidCount / geometry.length;
    expect(solidFraction).toBeGreaterThan(0.05);
    expect(solidFraction).toBeLessThan(0.3);
  });
});

describe("createDuctWithConstriction", () => {
  it("creates array of correct size", () => {
    const shape: [number, number, number] = [30, 40, 30];
    const geometry = createDuctWithConstriction(shape);

    expect(geometry.length).toBe(36000);
  });

  it("contains both air and solid cells", () => {
    const shape: [number, number, number] = [30, 40, 30];
    const geometry = createDuctWithConstriction(shape);

    let airCount = 0;
    let solidCount = 0;
    for (let i = 0; i < geometry.length; i++) {
      if (geometry[i] === 1) airCount++;
      else if (geometry[i] === 0) solidCount++;
    }

    expect(airCount).toBeGreaterThan(0);
    expect(solidCount).toBeGreaterThan(0);
  });
});

describe("createSphere", () => {
  it("creates array of correct size", () => {
    const shape: [number, number, number] = [20, 20, 20];
    const geometry = createSphere(shape);

    expect(geometry.length).toBe(8000);
  });

  it("creates spherical solid region", () => {
    const shape: [number, number, number] = [30, 30, 30];
    const geometry = createSphere(shape, 0.3);

    // Check center is solid
    const idx = (x: number, y: number, z: number) => x + y * 30 + z * 900;
    expect(geometry[idx(15, 15, 15)]).toBe(0); // solid at center

    // Check corners are air
    expect(geometry[idx(0, 0, 0)]).toBe(1); // air at corner
    expect(geometry[idx(29, 29, 29)]).toBe(1); // air at opposite corner
  });

  it("respects radius fraction parameter", () => {
    const shape: [number, number, number] = [30, 30, 30];
    const smallSphere = createSphere(shape, 0.1);
    const largeSphere = createSphere(shape, 0.4);

    let smallSolidCount = 0;
    let largeSolidCount = 0;
    for (let i = 0; i < smallSphere.length; i++) {
      if (smallSphere[i] === 0) smallSolidCount++;
      if (largeSphere[i] === 0) largeSolidCount++;
    }

    expect(largeSolidCount).toBeGreaterThan(smallSolidCount);
  });
});

describe("createDemoGeometry", () => {
  it("creates helmholtz geometry", () => {
    const shape: [number, number, number] = [30, 30, 30];
    const geometry = createDemoGeometry("helmholtz", shape);

    expect(geometry.length).toBe(27000);
  });

  it("creates duct geometry", () => {
    const shape: [number, number, number] = [30, 30, 30];
    const geometry = createDemoGeometry("duct", shape);

    expect(geometry.length).toBe(27000);
  });

  it("creates sphere geometry", () => {
    const shape: [number, number, number] = [30, 30, 30];
    const geometry = createDemoGeometry("sphere", shape);

    expect(geometry.length).toBe(27000);
  });

  it("throws for unknown geometry type", () => {
    const shape: [number, number, number] = [30, 30, 30];
    expect(() => {
      // @ts-expect-error Testing invalid type
      createDemoGeometry("invalid", shape);
    }).toThrow("Unknown demo geometry type");
  });
});
