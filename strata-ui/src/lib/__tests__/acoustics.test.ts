import { describe, it, expect } from "vitest";
import {
  inverseFFT,
  applySpectralWindow,
  computeImpulseResponse,
  schroederIntegration,
  estimateRT60,
  estimateEDT,
  computeClarity,
  computeDefinition,
  computeCentreTime,
  computeAcousticMetrics,
  analyzeTransferFunction,
} from "../acoustics";
import { realFFT } from "../fft";

describe("inverseFFT", () => {
  it("inverts a DC signal correctly", () => {
    // FFT of a constant signal [1, 1, 1, 1, 1, 1, 1, 1]
    // should have DC component = N and all others = 0
    // IFFT of that should give back the constant signal
    const fftResult = new Float32Array(16);
    fftResult[0] = 8; // DC component (real) = N * amplitude

    const ifftResult = inverseFFT(fftResult);

    // All samples should be 1 (constant signal)
    for (let i = 0; i < 8; i++) {
      expect(ifftResult[2 * i]).toBeCloseTo(1, 5);
    }
  });

  it("inverts a sine wave correctly", () => {
    const n = 64;
    const freq = 4; // 4 cycles

    // Create time-domain sine wave
    const original: number[] = [];
    for (let i = 0; i < n; i++) {
      original.push(Math.sin((2 * Math.PI * freq * i) / n));
    }

    // Forward FFT
    const fftResult = realFFT(original, n);

    // Inverse FFT
    const ifftResult = inverseFFT(fftResult);

    // Should recover the original signal
    for (let i = 0; i < n; i++) {
      expect(ifftResult[2 * i]).toBeCloseTo(original[i], 4);
    }
  });

  it("throws for non-power-of-2 input", () => {
    const badInput = new Float32Array(6); // 3 complex numbers, not power of 2
    expect(() => inverseFFT(badInput)).toThrow("power of 2");
  });
});

describe("applySpectralWindow", () => {
  it("applies Hanning window correctly", () => {
    const n = 8;
    const spectrum = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      spectrum[2 * i] = 1; // All ones
      spectrum[2 * i + 1] = 0;
    }

    const windowed = applySpectralWindow(spectrum, "hanning");

    // Hanning window is zero at edges, 1 at center
    expect(windowed[0]).toBeCloseTo(0, 5); // First sample
    expect(windowed[2 * (n - 1)]).toBeCloseTo(0, 5); // Last sample
    // Middle samples should be non-zero
    expect(windowed[2 * Math.floor(n / 2)]).toBeGreaterThan(0);
  });

  it("applies Tukey window correctly", () => {
    const n = 16;
    const spectrum = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      spectrum[2 * i] = 1;
      spectrum[2 * i + 1] = 0;
    }

    const windowed = applySpectralWindow(spectrum, "tukey");

    // Tukey window has tapered edges and flat center
    // Middle section should be close to 1
    expect(windowed[2 * 8]).toBeCloseTo(1, 2);
    // Edges should be reduced
    expect(windowed[0]).toBeLessThan(1);
    expect(windowed[2 * (n - 1)]).toBeLessThan(1);
  });

  it("passes through with no window", () => {
    const n = 4;
    const spectrum = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      spectrum[2 * i] = i + 1;
      spectrum[2 * i + 1] = (i + 1) * 0.5;
    }

    const windowed = applySpectralWindow(spectrum, "none");

    // Should be identical
    for (let i = 0; i < spectrum.length; i++) {
      expect(windowed[i]).toBeCloseTo(spectrum[i], 10);
    }
  });
});

describe("computeImpulseResponse", () => {
  it("computes impulse response from flat transfer function", () => {
    // Flat transfer function = 1 at all frequencies → impulse at t=0
    const n = 32;
    const transferReal = new Float32Array(n);
    const transferImag = new Float32Array(n);

    for (let i = 0; i < n; i++) {
      transferReal[i] = 1;
      transferImag[i] = 0;
    }

    const result = computeImpulseResponse(transferReal, transferImag, 1000, "none");

    // Impulse response should have a peak at or near t=0
    const maxIdx = result.impulseResponse.indexOf(Math.max(...result.impulseResponse));
    expect(maxIdx).toBeLessThan(5); // Peak should be near the start
  });

  it("creates correct time axis", () => {
    const sampleRate = 1000;
    const transferReal = new Float32Array(16).fill(1);
    const transferImag = new Float32Array(16).fill(0);

    const result = computeImpulseResponse(transferReal, transferImag, sampleRate, "none");

    // Time axis should start at 0 and increment by 1/sampleRate
    expect(result.timeAxis[0]).toBe(0);
    expect(result.timeAxis[1]).toBeCloseTo(1 / sampleRate, 6);
    expect(result.sampleRate).toBe(sampleRate);
  });
});

describe("schroederIntegration", () => {
  it("computes energy decay curve", () => {
    const sampleRate = 1000;
    // Create exponentially decaying impulse response
    const n = 1000;
    const ir = new Float32Array(n);
    const decayRate = 0.005; // decay constant

    for (let i = 0; i < n; i++) {
      ir[i] = Math.exp(-decayRate * i);
    }

    const result = schroederIntegration(ir, sampleRate);

    // Decay curve should start at 0 dB
    expect(result.decayCurve[0]).toBeCloseTo(0, 1);

    // Decay curve should decrease monotonically
    for (let i = 1; i < result.decayCurve.length; i++) {
      expect(result.decayCurve[i]).toBeLessThanOrEqual(result.decayCurve[i - 1] + 0.001);
    }

    // Time axis should be correct
    expect(result.timeAxis[0]).toBe(0);
    expect(result.timeAxis[100]).toBeCloseTo(0.1, 5);
  });

  it("handles zero-energy impulse", () => {
    const ir = new Float32Array(100).fill(0);
    const result = schroederIntegration(ir, 1000);

    // Should handle gracefully (all very negative dB values)
    expect(result.decayCurve.every((v) => !isNaN(v))).toBe(true);
  });
});

describe("estimateRT60", () => {
  it("estimates RT60 for exponential decay", () => {
    const sampleRate = 10000;
    const n = 10000;
    const ir = new Float32Array(n);

    // Create an exponential decay with known RT60
    // RT60 = time for 60 dB decay
    // For exponential decay: e^(-t/τ), 60 dB decay = 10^-3 = e^(-RT60/τ)
    // So RT60 = τ * ln(1000) ≈ τ * 6.91
    const tau = 0.1; // time constant in seconds
    const expectedRT60 = tau * Math.log(1000); // ~0.69 seconds

    for (let i = 0; i < n; i++) {
      ir[i] = Math.exp((-i / sampleRate) / tau);
    }

    const { decayCurve } = schroederIntegration(ir, sampleRate);
    const rt60 = estimateRT60(decayCurve, sampleRate, -5, -35);

    // Should be within 10% of expected value
    expect(rt60).toBeGreaterThan(expectedRT60 * 0.9);
    expect(rt60).toBeLessThan(expectedRT60 * 1.1);
  });

  it("returns NaN for insufficient decay", () => {
    // Decay that doesn't reach -35 dB
    const sampleRate = 1000;
    const n = 100;
    const ir = new Float32Array(n);

    for (let i = 0; i < n; i++) {
      ir[i] = Math.exp(-0.001 * i); // Very slow decay
    }

    const { decayCurve } = schroederIntegration(ir, sampleRate);
    const rt60 = estimateRT60(decayCurve, sampleRate, -5, -35);

    expect(isNaN(rt60)).toBe(true);
  });
});

describe("estimateEDT", () => {
  it("estimates EDT for exponential decay", () => {
    const sampleRate = 10000;
    const n = 10000;
    const ir = new Float32Array(n);

    const tau = 0.1;
    const expectedRT60 = tau * Math.log(1000);

    for (let i = 0; i < n; i++) {
      ir[i] = Math.exp((-i / sampleRate) / tau);
    }

    const { decayCurve } = schroederIntegration(ir, sampleRate);
    const edt = estimateEDT(decayCurve, sampleRate);

    // For exponential decay, EDT should equal RT60
    expect(edt).toBeGreaterThan(expectedRT60 * 0.8);
    expect(edt).toBeLessThan(expectedRT60 * 1.2);
  });
});

describe("computeClarity", () => {
  it("computes C80 correctly for impulse", () => {
    const sampleRate = 1000;
    const ir = new Float32Array(1000);

    // Put all energy at t=0 (before 80ms)
    ir[0] = 1;

    const c80 = computeClarity(ir, sampleRate, 80);

    // With all energy early, C80 should be very high
    expect(c80).toBeGreaterThan(60); // Very high dB value
  });

  it("computes C50 correctly for late energy", () => {
    const sampleRate = 1000;
    const ir = new Float32Array(1000);

    // Put all energy after 50ms (sample 50)
    ir[100] = 1;

    const c50 = computeClarity(ir, sampleRate, 50);

    // With all energy late, C50 should be very low (negative dB)
    expect(c50).toBeLessThan(-60);
  });

  it("returns 0 dB for equal early/late energy", () => {
    const sampleRate = 1000;
    const ir = new Float32Array(200);

    // Equal energy before and after 80ms
    ir[0] = 1;
    ir[100] = 1;

    const c80 = computeClarity(ir, sampleRate, 80);

    // Should be close to 0 dB
    expect(Math.abs(c80)).toBeLessThan(0.1);
  });
});

describe("computeDefinition", () => {
  it("returns 1.0 for all early energy", () => {
    const sampleRate = 1000;
    const ir = new Float32Array(1000);

    // All energy before 50ms
    ir[0] = 1;
    ir[10] = 0.5;
    ir[20] = 0.25;

    const d50 = computeDefinition(ir, sampleRate, 50);

    expect(d50).toBeCloseTo(1.0, 5);
  });

  it("returns 0 for all late energy", () => {
    const sampleRate = 1000;
    const ir = new Float32Array(1000);

    // All energy after 80ms
    ir[100] = 1;
    ir[200] = 0.5;

    const d80 = computeDefinition(ir, sampleRate, 80);

    expect(d80).toBeCloseTo(0, 5);
  });

  it("returns 0.5 for equal early/late energy", () => {
    const sampleRate = 1000;
    const ir = new Float32Array(200);

    // Equal energy before and after 50ms
    ir[0] = 1;
    ir[100] = 1;

    const d50 = computeDefinition(ir, sampleRate, 50);

    expect(d50).toBeCloseTo(0.5, 5);
  });
});

describe("computeCentreTime", () => {
  it("returns 0 for impulse at t=0", () => {
    const sampleRate = 1000;
    const ir = new Float32Array(1000);
    ir[0] = 1;

    const ts = computeCentreTime(ir, sampleRate);

    expect(ts).toBeCloseTo(0, 6);
  });

  it("returns correct time for impulse at specific time", () => {
    const sampleRate = 1000;
    const ir = new Float32Array(1000);
    const targetSample = 100; // 100ms
    ir[targetSample] = 1;

    const ts = computeCentreTime(ir, sampleRate);

    expect(ts).toBeCloseTo(targetSample / sampleRate, 6);
  });

  it("returns weighted average for multiple impulses", () => {
    const sampleRate = 1000;
    const ir = new Float32Array(1000);

    // Two equal impulses at 0ms and 200ms → centre at 100ms
    ir[0] = 1;
    ir[200] = 1;

    const ts = computeCentreTime(ir, sampleRate);

    expect(ts).toBeCloseTo(0.1, 5); // 100ms
  });
});

describe("computeAcousticMetrics", () => {
  it("computes all metrics for exponential decay", () => {
    const sampleRate = 10000;
    const n = 20000;
    const ir = new Float32Array(n);

    const tau = 0.15;
    for (let i = 0; i < n; i++) {
      ir[i] = Math.exp((-i / sampleRate) / tau);
    }

    const metrics = computeAcousticMetrics(ir, sampleRate);

    // Check that all metrics are defined
    expect(typeof metrics.t20).toBe("number");
    expect(typeof metrics.t30).toBe("number");
    expect(typeof metrics.edt).toBe("number");
    expect(typeof metrics.c50).toBe("number");
    expect(typeof metrics.c80).toBe("number");
    expect(typeof metrics.d50).toBe("number");
    expect(typeof metrics.d80).toBe("number");
    expect(typeof metrics.ts).toBe("number");

    // D50 and D80 should be percentages (0-100)
    expect(metrics.d50).toBeGreaterThanOrEqual(0);
    expect(metrics.d50).toBeLessThanOrEqual(100);
    expect(metrics.d80).toBeGreaterThanOrEqual(0);
    expect(metrics.d80).toBeLessThanOrEqual(100);

    // Ts should be in ms
    expect(metrics.ts).toBeGreaterThan(0);
  });
});

describe("analyzeTransferFunction", () => {
  it("combines all analysis steps", () => {
    const n = 64;
    const transferReal = new Float32Array(n);
    const transferImag = new Float32Array(n);

    // Simple flat transfer function
    for (let i = 0; i < n; i++) {
      transferReal[i] = 1;
      transferImag[i] = 0;
    }

    const result = analyzeTransferFunction(
      transferReal,
      transferImag,
      1000,
      "tukey"
    );

    // Check that all parts are present
    expect(result.impulseResponse).toBeDefined();
    expect(result.impulseResponse.impulseResponse).toBeInstanceOf(Float32Array);
    expect(result.impulseResponse.timeAxis).toBeInstanceOf(Float32Array);

    expect(result.energyDecay).toBeDefined();
    expect(result.energyDecay.decayCurve).toBeInstanceOf(Float32Array);
    expect(result.energyDecay.timeAxis).toBeInstanceOf(Float32Array);

    expect(result.metrics).toBeDefined();
    expect(typeof result.metrics.c80).toBe("number");
  });
});
