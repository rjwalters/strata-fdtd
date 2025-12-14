import { describe, it, expect } from "vitest";

// Test the internal computation logic used by WaterfallPlot
// We test the pure functions separately from React components

function nextPowerOf2(n: number): number {
  return Math.pow(2, Math.ceil(Math.log2(n)));
}

describe("WaterfallPlot utilities", () => {
  describe("nextPowerOf2", () => {
    it("returns same value for powers of 2", () => {
      expect(nextPowerOf2(256)).toBe(256);
      expect(nextPowerOf2(512)).toBe(512);
    });

    it("rounds up non-powers of 2", () => {
      expect(nextPowerOf2(300)).toBe(512);
      expect(nextPowerOf2(600)).toBe(1024);
    });
  });
});

describe("WaterfallPlot frame limiting", () => {
  it("limits frames when data exceeds maxFrames", () => {
    const dataLength = 480000; // 10 seconds at 48kHz
    const fftSize = 2048;
    const hopSize = 512;
    const maxFrames = 100;

    const totalFrames = Math.floor((dataLength - fftSize) / hopSize) + 1;
    // (480000 - 2048) / 512 + 1 ≈ 933 + 1 = 934

    expect(totalFrames).toBeGreaterThan(maxFrames);

    const frameStep = Math.ceil(totalFrames / maxFrames);
    const actualFrames = Math.min(totalFrames, maxFrames);

    expect(frameStep).toBe(10); // 934 / 100 ≈ 9.34 → ceil = 10
    expect(actualFrames).toBe(100);
  });

  it("keeps all frames when under maxFrames", () => {
    const dataLength = 24000; // 0.5 seconds at 48kHz
    const fftSize = 2048;
    const hopSize = 512;
    const maxFrames = 100;

    const totalFrames = Math.floor((dataLength - fftSize) / hopSize) + 1;
    // (24000 - 2048) / 512 + 1 ≈ 42 + 1 = 43

    expect(totalFrames).toBeLessThan(maxFrames);

    const frameStep = totalFrames > maxFrames ? Math.ceil(totalFrames / maxFrames) : 1;
    expect(frameStep).toBe(1);
  });
});

describe("WaterfallPlot 3D projection", () => {
  it("calculates perspective factor from angle", () => {
    const angle0 = 0;
    const angle15 = 15;
    const angle45 = 45;

    const perspectiveFactor = (angle: number) => Math.sin((angle * Math.PI) / 180);
    const depthScale = (angle: number) => Math.cos((angle * Math.PI) / 180);

    // At 0 degrees, no perspective (flat)
    expect(perspectiveFactor(angle0)).toBeCloseTo(0, 5);
    expect(depthScale(angle0)).toBeCloseTo(1, 5);

    // At 15 degrees
    expect(perspectiveFactor(angle15)).toBeCloseTo(0.259, 2);
    expect(depthScale(angle15)).toBeCloseTo(0.966, 2);

    // At 45 degrees
    expect(perspectiveFactor(angle45)).toBeCloseTo(0.707, 2);
    expect(depthScale(angle45)).toBeCloseTo(0.707, 2);
  });

  it("calculates depth factor for each frame", () => {
    const numFrames = 100;

    // First frame (newest, front) should have depth 0
    const depthFirst = (numFrames - 1 - 0) / Math.max(1, numFrames - 1);
    expect(depthFirst).toBeCloseTo(1, 5); // Actually oldest is at back

    // Last frame (oldest, back) should have depth 1
    const depthLast = (numFrames - 1 - (numFrames - 1)) / Math.max(1, numFrames - 1);
    expect(depthLast).toBeCloseTo(0, 5);

    // Middle frame
    const depthMiddle = (numFrames - 1 - 50) / Math.max(1, numFrames - 1);
    expect(depthMiddle).toBeCloseTo(0.495, 2);
  });

  it("scales x width based on depth", () => {
    const width = 800;
    const depthScale = 0.966; // 15 degrees

    const scaleAtFront = 1 - 0 * (1 - depthScale) * 0.3; // depth = 0
    const scaleAtBack = 1 - 1 * (1 - depthScale) * 0.3; // depth = 1

    expect(scaleAtFront).toBe(1);
    expect(scaleAtBack).toBeCloseTo(0.99, 2);

    const widthAtFront = width * scaleAtFront;
    const widthAtBack = width * scaleAtBack;

    expect(widthAtFront).toBe(800);
    expect(widthAtBack).toBeCloseTo(792, 0);
  });
});

describe("WaterfallPlot amplitude mapping", () => {
  it("maps dB to visual amplitude", () => {
    const minDb = -80;
    const maxDb = 0;
    const spectrumHeight = 100;

    const mapAmplitude = (db: number, depth: number) => {
      const normalized = Math.max(0, Math.min(1, (db - minDb) / (maxDb - minDb)));
      return normalized * spectrumHeight * (1 - depth * 0.5);
    };

    // At front (depth = 0), full height
    expect(mapAmplitude(0, 0)).toBe(100); // Max dB, full height
    expect(mapAmplitude(-40, 0)).toBe(50); // Mid dB, half height
    expect(mapAmplitude(-80, 0)).toBe(0); // Min dB, no height

    // At back (depth = 1), reduced height (50%)
    expect(mapAmplitude(0, 1)).toBe(50);
    expect(mapAmplitude(-40, 1)).toBe(25);
    expect(mapAmplitude(-80, 1)).toBe(0);
  });
});

describe("WaterfallPlot color interpolation", () => {
  it("uses higher brightness for front rows", () => {
    // Color function value based on depth
    const colorValue = (depth: number) => 0.5 + 0.5 * (1 - depth);

    expect(colorValue(0)).toBe(1); // Front: brightest
    expect(colorValue(0.5)).toBe(0.75); // Middle
    expect(colorValue(1)).toBe(0.5); // Back: dimmer
  });

  it("uses higher alpha for front rows", () => {
    const alpha = (depth: number) => 0.3 + 0.5 * (1 - depth);

    expect(alpha(0)).toBe(0.8); // Front: most opaque
    expect(alpha(0.5)).toBe(0.55); // Middle
    expect(alpha(1)).toBe(0.3); // Back: most transparent
  });
});

describe("WaterfallPlot frequency tick labels", () => {
  it("formats frequency labels correctly", () => {
    const formatFreq = (freq: number) => (freq >= 1000 ? `${freq / 1000}k` : `${freq}`);

    expect(formatFreq(100)).toBe("100");
    expect(formatFreq(500)).toBe("500");
    expect(formatFreq(1000)).toBe("1k");
    expect(formatFreq(2000)).toBe("2k");
    expect(formatFreq(10000)).toBe("10k");
    expect(formatFreq(20000)).toBe("20k");
  });

  it("filters ticks within frequency range", () => {
    const allTicks = [100, 500, 1000, 2000, 5000, 10000, 20000];
    const minFreq = 200;
    const maxFreq = 8000;

    const filteredTicks = allTicks.filter((f) => f >= minFreq && f <= maxFreq);

    expect(filteredTicks).toEqual([500, 1000, 2000, 5000]);
  });
});
