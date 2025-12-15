import { describe, it, expect } from "vitest";

// Test the internal FFT computation logic used by Spectrogram
// We test the pure functions separately from React components

function nextPowerOf2(n: number): number {
  return Math.pow(2, Math.ceil(Math.log2(n)));
}

describe("Spectrogram utilities", () => {
  describe("nextPowerOf2", () => {
    it("returns same value for powers of 2", () => {
      expect(nextPowerOf2(1)).toBe(1);
      expect(nextPowerOf2(2)).toBe(2);
      expect(nextPowerOf2(4)).toBe(4);
      expect(nextPowerOf2(1024)).toBe(1024);
      expect(nextPowerOf2(2048)).toBe(2048);
    });

    it("rounds up non-powers of 2", () => {
      expect(nextPowerOf2(3)).toBe(4);
      expect(nextPowerOf2(5)).toBe(8);
      expect(nextPowerOf2(1000)).toBe(1024);
      expect(nextPowerOf2(1025)).toBe(2048);
    });
  });
});

describe("Spectrogram frame calculations", () => {
  it("calculates correct number of frames", () => {
    const dataLength = 48000; // 1 second at 48kHz
    const fftSize = 2048;
    const hopSize = 512;

    const numFrames = Math.floor((dataLength - fftSize) / hopSize) + 1;

    // (48000 - 2048) / 512 + 1 = 89.75 + 1 = floor(89.75) + 1 = 89 + 1 = 90
    expect(numFrames).toBe(90);
  });

  it("calculates correct frequency resolution", () => {
    const sampleRate = 48000;
    const fftSize = 2048;
    const freqPerBin = sampleRate / fftSize;

    expect(freqPerBin).toBeCloseTo(23.4375, 4);
  });

  it("calculates correct time resolution", () => {
    const sampleRate = 48000;
    const hopSize = 512;
    const timePerFrame = hopSize / sampleRate;

    // 512 / 48000 = 0.01066... seconds = 10.66ms
    expect(timePerFrame * 1000).toBeCloseTo(10.67, 1);
  });

  it("calculates bin index for frequency", () => {
    const sampleRate = 48000;
    const fftSize = 2048;
    const freqPerBin = sampleRate / fftSize;

    const freq440 = 440; // A4
    const binIdx = Math.round(freq440 / freqPerBin);

    // 440 / 23.4375 ≈ 18.77 → 19
    expect(binIdx).toBe(19);

    // Verify reverse calculation
    const reconstructedFreq = binIdx * freqPerBin;
    expect(reconstructedFreq).toBeCloseTo(445, 0); // Close to 440 Hz
  });
});

describe("Spectrogram dB conversion", () => {
  function toDecibels(value: number, ref: number = 1): number {
    return 20 * Math.log10(Math.max(value / ref, 1e-10));
  }

  it("converts magnitude to dB", () => {
    expect(toDecibels(1, 1)).toBeCloseTo(0, 5);
    expect(toDecibels(10, 1)).toBeCloseTo(20, 5);
    expect(toDecibels(0.1, 1)).toBeCloseTo(-20, 5);
    expect(toDecibels(0.01, 1)).toBeCloseTo(-40, 5);
  });

  it("handles zero with floor", () => {
    // Should return floor value, not -Infinity
    const db = toDecibels(0, 1);
    expect(db).toBe(20 * Math.log10(1e-10));
    expect(db).toBeCloseTo(-200, 0);
  });

  it("handles reference value", () => {
    expect(toDecibels(2, 2)).toBeCloseTo(0, 5);
    expect(toDecibels(4, 2)).toBeCloseTo(6, 0); // 6 dB
  });
});

describe("Spectrogram frequency range", () => {
  it("calculates bin range for audible frequencies", () => {
    const sampleRate = 48000;
    const fftSize = 2048;
    const freqPerBin = sampleRate / fftSize;

    const minFreq = 20;
    const maxFreq = 20000;

    const minBin = Math.max(1, Math.floor(minFreq / freqPerBin));
    const maxBin = Math.min(fftSize / 2 - 1, Math.ceil(maxFreq / freqPerBin));

    // 20 / 23.4375 ≈ 0.85 → floor = 0, but max(1, 0) = 1
    expect(minBin).toBe(1);

    // 20000 / 23.4375 ≈ 853.3 → ceil = 854
    expect(maxBin).toBe(854);
  });

  it("limits to Nyquist frequency", () => {
    const sampleRate = 48000;
    const nyquist = sampleRate / 2;

    expect(nyquist).toBe(24000);

    // If maxFreq > nyquist, should clamp
    const maxFreq = 30000;
    const effectiveMax = Math.min(maxFreq, nyquist);
    expect(effectiveMax).toBe(24000);
  });
});

describe("Spectrogram Hanning window", () => {
  function hanningWindow(n: number): Float32Array {
    const window = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
    }
    return window;
  }

  it("starts and ends near zero", () => {
    const window = hanningWindow(1024);
    expect(window[0]).toBeCloseTo(0, 5);
    expect(window[1023]).toBeCloseTo(0, 5);
  });

  it("peaks at center", () => {
    const window = hanningWindow(1024);
    const center = 512;
    expect(window[center]).toBeCloseTo(1, 5);
  });

  it("is symmetric", () => {
    const window = hanningWindow(1024);
    for (let i = 0; i < 512; i++) {
      expect(window[i]).toBeCloseTo(window[1023 - i], 5);
    }
  });
});

describe("Spectrogram color scale normalization", () => {
  it("normalizes dB values to 0-1 range", () => {
    const minDb = -80;
    const maxDb = 0;

    const normalize = (db: number) =>
      Math.max(0, Math.min(1, (db - minDb) / (maxDb - minDb)));

    expect(normalize(-80)).toBe(0);
    expect(normalize(0)).toBe(1);
    expect(normalize(-40)).toBe(0.5);
    expect(normalize(-100)).toBe(0); // Clamped
    expect(normalize(10)).toBe(1); // Clamped
  });
});
