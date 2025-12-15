import { useRef, useEffect, useState, useMemo } from "react";
import * as d3 from "d3";
import { realFFT } from "../lib/fft";

export type WaterfallColorScale = "viridis" | "magma" | "inferno" | "plasma" | "grayscale";

export interface WaterfallPlotProps {
  data: Float32Array;
  sampleRate: number;
  fftSize?: number;
  hopSize?: number;
  maxFrames?: number;
  perspective?: number;
  colorScale?: WaterfallColorScale;
  minFrequency?: number;
  maxFrequency?: number;
}

const MARGIN = { top: 20, right: 60, bottom: 30, left: 50 };

function nextPowerOf2(n: number): number {
  return Math.pow(2, Math.ceil(Math.log2(n)));
}

// Color scale implementations
const colorScales: Record<WaterfallColorScale, (t: number) => string> = {
  viridis: d3.interpolateViridis,
  magma: d3.interpolateMagma,
  inferno: d3.interpolateInferno,
  plasma: d3.interpolatePlasma,
  grayscale: (t: number) => d3.interpolateGreys(1 - t),
};

interface WaterfallData {
  magnitudes: Float32Array[];
  numFrames: number;
  numBins: number;
  freqPerBin: number;
  minDb: number;
  maxDb: number;
}

function computeWaterfallData(
  data: Float32Array,
  sampleRate: number,
  fftSize: number,
  hopSize: number,
  maxFrames: number
): WaterfallData {
  const n = nextPowerOf2(fftSize);
  let numFrames = Math.floor((data.length - n) / hopSize) + 1;

  // Limit number of frames for performance
  const frameStep = numFrames > maxFrames ? Math.ceil(numFrames / maxFrames) : 1;
  numFrames = Math.min(numFrames, maxFrames);

  const numBins = n / 2;
  const magnitudes: Float32Array[] = [];
  let minDb = Infinity;
  let maxDb = -Infinity;

  for (let frameIdx = 0; frameIdx < numFrames; frameIdx++) {
    const frame = frameIdx * frameStep;
    const offset = frame * hopSize;
    const input: number[] = new Array(n).fill(0);

    // Apply Hanning window
    for (let i = 0; i < n && offset + i < data.length; i++) {
      const window = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
      input[i] = data[offset + i] * window;
    }

    // Compute FFT using local implementation
    const out = realFFT(input, n);

    // Compute magnitude in dB
    const frameMag = new Float32Array(numBins);
    for (let i = 0; i < numBins; i++) {
      const re = out[2 * i];
      const im = out[2 * i + 1];
      const mag = Math.sqrt(re * re + im * im) / n;
      const db = 20 * Math.log10(Math.max(mag, 1e-10));
      frameMag[i] = db;
      if (db > maxDb) maxDb = db;
      if (db < minDb && db > -100) minDb = db;
    }
    magnitudes.push(frameMag);
  }

  // Clamp min to reasonable value
  minDb = Math.max(minDb, maxDb - 80);

  return {
    magnitudes,
    numFrames: magnitudes.length,
    numBins,
    freqPerBin: sampleRate / n,
    minDb,
    maxDb,
  };
}

export function WaterfallPlot({
  data,
  sampleRate,
  fftSize = 2048,
  hopSize = 512,
  maxFrames = 100,
  perspective: initialPerspective = 15,
  colorScale: initialColorScale = "viridis",
  minFrequency = 20,
  maxFrequency,
}: WaterfallPlotProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [perspective, setPerspective] = useState(initialPerspective);
  const [colorScale, setColorScale] = useState<WaterfallColorScale>(initialColorScale);

  const effectiveMaxFreq = maxFrequency ?? Math.min(sampleRate / 2, 20000);

  // Compute waterfall data
  const waterfallData = useMemo(() => {
    if (data.length === 0) return null;
    return computeWaterfallData(data, sampleRate, fftSize, hopSize, maxFrames);
  }, [data, sampleRate, fftSize, hopSize, maxFrames]);

  // Track container size
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setDimensions({ width, height });
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  // Render waterfall
  useEffect(() => {
    if (!canvasRef.current || dimensions.width === 0 || !waterfallData) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = dimensions.width - MARGIN.left - MARGIN.right;
    const height = dimensions.height - MARGIN.top - MARGIN.bottom;

    if (width <= 0 || height <= 0) return;

    // Set canvas size
    const dpr = window.devicePixelRatio || 1;
    canvas.width = dimensions.width * dpr;
    canvas.height = dimensions.height * dpr;
    canvas.style.width = `${dimensions.width}px`;
    canvas.style.height = `${dimensions.height}px`;
    ctx.scale(dpr, dpr);

    // Clear canvas
    ctx.fillStyle = "hsl(var(--background))";
    ctx.fillRect(0, 0, dimensions.width, dimensions.height);

    const { magnitudes, numFrames, numBins, freqPerBin, minDb, maxDb } = waterfallData;
    const colorFn = colorScales[colorScale];

    // Find bin indices for frequency range
    const minBin = Math.max(1, Math.floor(minFrequency / freqPerBin));
    const maxBin = Math.min(numBins - 1, Math.ceil(effectiveMaxFreq / freqPerBin));
    const displayBins = maxBin - minBin;

    // 3D projection parameters
    const perspectiveAngle = (perspective * Math.PI) / 180;
    const perspectiveFactor = Math.sin(perspectiveAngle);
    const depthScale = Math.cos(perspectiveAngle);

    // Height allocated per spectrum row
    const rowHeight = (height * 0.7) / numFrames;
    const spectrumHeight = height * 0.3; // Height for amplitude

    // Draw from back to front (oldest to newest)
    for (let frameIdx = 0; frameIdx < numFrames; frameIdx++) {
      const frameMag = magnitudes[frameIdx];
      const depth = (numFrames - 1 - frameIdx) / Math.max(1, numFrames - 1);

      // Calculate y offset based on perspective
      const yOffset = MARGIN.top + depth * height * perspectiveFactor * 0.6;
      const baseY = MARGIN.top + height - rowHeight * (numFrames - 1 - frameIdx);

      // Horizontal scaling based on depth
      const xScale = 1 - depth * (1 - depthScale) * 0.3;
      const xOffset = MARGIN.left + (width * (1 - xScale)) / 2;
      const scaledWidth = width * xScale;

      // Build path for this spectrum
      ctx.beginPath();
      ctx.moveTo(xOffset, baseY - yOffset);

      const points: Array<{ x: number; y: number; color: string }> = [];

      for (let binIdx = minBin; binIdx < maxBin; binIdx++) {
        const x = xOffset + ((binIdx - minBin) / displayBins) * scaledWidth;
        const db = frameMag[binIdx];
        const normalized = Math.max(0, Math.min(1, (db - minDb) / (maxDb - minDb)));
        const amplitude = normalized * spectrumHeight * (1 - depth * 0.5);
        const y = baseY - yOffset - amplitude;

        points.push({ x, y, color: colorFn(normalized) });
        ctx.lineTo(x, y);
      }

      // Close the path to create filled area
      ctx.lineTo(xOffset + scaledWidth, baseY - yOffset);
      ctx.lineTo(xOffset, baseY - yOffset);
      ctx.closePath();

      // Fill with gradient based on depth
      const alpha = 0.3 + 0.5 * (1 - depth);
      ctx.fillStyle = `rgba(30, 30, 30, ${alpha})`;
      ctx.fill();

      // Draw colored line on top
      ctx.beginPath();
      for (let i = 0; i < points.length; i++) {
        const pt = points[i];
        if (i === 0) {
          ctx.moveTo(pt.x, pt.y);
        } else {
          ctx.lineTo(pt.x, pt.y);
        }
      }
      ctx.strokeStyle = colorFn(0.5 + 0.5 * (1 - depth));
      ctx.lineWidth = 1 + (1 - depth);
      ctx.stroke();
    }

    // Draw axes
    ctx.fillStyle = "hsl(var(--muted-foreground))";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "center";

    // X-axis (frequency)
    const freqTicks = [100, 500, 1000, 2000, 5000, 10000, 20000].filter(
      (f) => f >= minFrequency && f <= effectiveMaxFreq
    );

    ctx.strokeStyle = "hsl(var(--border))";
    ctx.lineWidth = 1;

    for (const freq of freqTicks) {
      const binIdx = freq / freqPerBin;
      const x = MARGIN.left + ((binIdx - minBin) / displayBins) * width;

      ctx.beginPath();
      ctx.moveTo(x, MARGIN.top + height);
      ctx.lineTo(x, MARGIN.top + height + 5);
      ctx.stroke();

      const label = freq >= 1000 ? `${freq / 1000}k` : `${freq}`;
      ctx.fillText(label, x, MARGIN.top + height + 15);
    }

    // X-axis label
    ctx.fillText("Frequency (Hz)", MARGIN.left + width / 2, MARGIN.top + height + 28);

    // Time indicator
    ctx.textAlign = "left";
    ctx.fillStyle = "hsl(var(--muted-foreground))";
    ctx.font = "9px sans-serif";
    ctx.fillText("← Time (newest front, oldest back)", MARGIN.left, MARGIN.top - 5);

    // Color bar on the right
    const colorBarWidth = 15;
    const colorBarHeight = height * 0.6;
    const colorBarX = dimensions.width - MARGIN.right + 10;
    const colorBarY = MARGIN.top + (height - colorBarHeight) / 2;

    for (let i = 0; i < colorBarHeight; i++) {
      const t = 1 - i / colorBarHeight;
      ctx.fillStyle = colorFn(t);
      ctx.fillRect(colorBarX, colorBarY + i, colorBarWidth, 1);
    }

    // Color bar labels
    ctx.textAlign = "left";
    ctx.fillStyle = "hsl(var(--muted-foreground))";
    ctx.font = "9px sans-serif";
    ctx.fillText(`${Math.round(maxDb)}dB`, colorBarX + colorBarWidth + 3, colorBarY + 4);
    ctx.fillText(
      `${Math.round(minDb)}dB`,
      colorBarX + colorBarWidth + 3,
      colorBarY + colorBarHeight
    );
  }, [
    dimensions,
    waterfallData,
    perspective,
    colorScale,
    minFrequency,
    effectiveMaxFreq,
  ]);

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-muted-foreground">Waterfall</h3>
        <div className="flex gap-2 items-center">
          <label className="text-xs text-muted-foreground">Angle:</label>
          <input
            type="range"
            min="0"
            max="45"
            value={perspective}
            onChange={(e) => setPerspective(Number(e.target.value))}
            className="w-16 h-4"
          />
          <span className="text-xs text-muted-foreground w-6">{perspective}°</span>
          <select
            value={colorScale}
            onChange={(e) => setColorScale(e.target.value as WaterfallColorScale)}
            className="h-6 text-xs px-1 bg-background border border-border rounded"
          >
            <option value="viridis">Viridis</option>
            <option value="magma">Magma</option>
            <option value="inferno">Inferno</option>
            <option value="plasma">Plasma</option>
            <option value="grayscale">Grayscale</option>
          </select>
        </div>
      </div>
      <div ref={containerRef} className="flex-1 min-h-0 relative">
        <canvas ref={canvasRef} />
      </div>
    </div>
  );
}
