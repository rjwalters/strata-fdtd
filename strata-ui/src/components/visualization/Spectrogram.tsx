import { useRef, useEffect, useState, useMemo, useCallback } from "react";
import * as d3 from "d3";
import { realFFT } from "@/lib/fft";
import { Button } from "@/components/ui/button";

export type ColorScale = "viridis" | "magma" | "inferno" | "plasma" | "grayscale";

export interface SpectrogramProps {
  data: Float32Array;
  sampleRate: number;
  fftSize?: number;
  hopSize?: number;
  colorScale?: ColorScale;
  logFrequency?: boolean;
  minFrequency?: number;
  maxFrequency?: number;
  onRegionSelect?: (timeRange: [number, number], freqRange: [number, number]) => void;
}

const MARGIN = { top: 10, right: 60, bottom: 30, left: 50 };

function nextPowerOf2(n: number): number {
  return Math.pow(2, Math.ceil(Math.log2(n)));
}

// Color scale implementations
const colorScales: Record<ColorScale, (t: number) => string> = {
  viridis: d3.interpolateViridis,
  magma: d3.interpolateMagma,
  inferno: d3.interpolateInferno,
  plasma: d3.interpolatePlasma,
  grayscale: (t: number) => d3.interpolateGreys(1 - t), // Invert so 0=black, 1=white
};

interface SpectrogramData {
  magnitudes: Float32Array[];
  numFrames: number;
  numBins: number;
  timePerFrame: number;
  freqPerBin: number;
  minDb: number;
  maxDb: number;
}

function computeSpectrogram(
  data: Float32Array,
  sampleRate: number,
  fftSize: number,
  hopSize: number
): SpectrogramData {
  const n = nextPowerOf2(fftSize);
  const numFrames = Math.floor((data.length - n) / hopSize) + 1;
  const numBins = n / 2;

  const magnitudes: Float32Array[] = [];
  let minDb = Infinity;
  let maxDb = -Infinity;

  for (let frame = 0; frame < numFrames; frame++) {
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
    numFrames,
    numBins,
    timePerFrame: hopSize / sampleRate,
    freqPerBin: sampleRate / n,
    minDb,
    maxDb,
  };
}

export function Spectrogram({
  data,
  sampleRate,
  fftSize = 2048,
  hopSize = 512,
  colorScale: initialColorScale = "viridis",
  logFrequency: initialLogFrequency = true,
  minFrequency = 20,
  maxFrequency,
  onRegionSelect,
}: SpectrogramProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<SVGSVGElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [colorScale, setColorScale] = useState<ColorScale>(initialColorScale);
  const [logFrequency, setLogFrequency] = useState(initialLogFrequency);
  const [selection, setSelection] = useState<{
    startX: number;
    startY: number;
    endX: number;
    endY: number;
  } | null>(null);
  const [isSelecting, setIsSelecting] = useState(false);

  const effectiveMaxFreq = maxFrequency ?? Math.min(sampleRate / 2, 20000);

  // Compute spectrogram data
  const spectrogramData = useMemo(() => {
    if (data.length === 0) return null;
    return computeSpectrogram(data, sampleRate, fftSize, hopSize);
  }, [data, sampleRate, fftSize, hopSize]);

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

  // Render spectrogram to canvas
  useEffect(() => {
    if (!canvasRef.current || dimensions.width === 0 || !spectrogramData) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = dimensions.width - MARGIN.left - MARGIN.right;
    const height = dimensions.height - MARGIN.top - MARGIN.bottom;

    if (width <= 0 || height <= 0) return;

    // Set canvas size (with pixel ratio for sharp rendering)
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.scale(dpr, dpr);

    const { magnitudes, numFrames, numBins, freqPerBin, minDb, maxDb } = spectrogramData;
    const colorFn = colorScales[colorScale];

    // Find bin indices for frequency range
    const minBin = Math.max(1, Math.floor(minFrequency / freqPerBin));
    const maxBin = Math.min(numBins - 1, Math.ceil(effectiveMaxFreq / freqPerBin));

    // Create frequency scale
    const yScale = logFrequency
      ? d3.scaleLog().domain([minFrequency, effectiveMaxFreq]).range([height, 0]).clamp(true)
      : d3.scaleLinear().domain([minFrequency, effectiveMaxFreq]).range([height, 0]);

    // Create image data
    const imageData = ctx.createImageData(width, height);
    const pixels = imageData.data;

    for (let x = 0; x < width; x++) {
      const frameIdx = Math.floor((x / width) * numFrames);
      if (frameIdx >= magnitudes.length) continue;

      const frameMag = magnitudes[frameIdx];

      for (let y = 0; y < height; y++) {
        // Map y pixel to frequency
        const freq = yScale.invert(y);
        const binIdx = Math.round(freq / freqPerBin);

        if (binIdx < minBin || binIdx >= maxBin) continue;

        const db = frameMag[binIdx];
        const normalized = Math.max(0, Math.min(1, (db - minDb) / (maxDb - minDb)));

        // Get color
        const colorStr = colorFn(normalized);
        const color = d3.color(colorStr);
        if (!color) continue;

        const rgb = color.rgb();
        const idx = (y * width + x) * 4;
        pixels[idx] = rgb.r;
        pixels[idx + 1] = rgb.g;
        pixels[idx + 2] = rgb.b;
        pixels[idx + 3] = 255;
      }
    }

    ctx.putImageData(imageData, 0, 0);
  }, [dimensions, spectrogramData, colorScale, logFrequency, minFrequency, effectiveMaxFreq]);

  // Render axes overlay
  useEffect(() => {
    if (!overlayRef.current || dimensions.width === 0 || !spectrogramData) return;

    const svg = d3.select(overlayRef.current);
    svg.selectAll("*").remove();

    const width = dimensions.width - MARGIN.left - MARGIN.right;
    const height = dimensions.height - MARGIN.top - MARGIN.bottom;

    if (width <= 0 || height <= 0) return;

    const { numFrames, timePerFrame, minDb, maxDb } = spectrogramData;
    const duration = numFrames * timePerFrame * 1000; // ms

    // Create scales
    const xScale = d3.scaleLinear().domain([0, duration]).range([0, width]);

    const yScale = logFrequency
      ? d3.scaleLog().domain([minFrequency, effectiveMaxFreq]).range([height, 0]).clamp(true)
      : d3.scaleLinear().domain([minFrequency, effectiveMaxFreq]).range([height, 0]);

    // Create main group
    const g = svg.append("g").attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);

    // Create axes
    const xAxis = d3
      .axisBottom(xScale)
      .ticks(8)
      .tickFormat((d) => `${d}ms`);

    const yAxis = d3
      .axisLeft(yScale)
      .ticks(logFrequency ? 6 : 8)
      .tickFormat((d) => {
        const val = +d;
        if (val >= 1000) return `${val / 1000}k`;
        return `${val}`;
      });

    g.append("g")
      .attr("transform", `translate(0,${height})`)
      .attr("class", "x-axis")
      .call(xAxis)
      .selectAll("text")
      .style("fill", "hsl(var(--muted-foreground))");

    // X-axis label
    g.append("text")
      .attr("x", width / 2)
      .attr("y", height + 25)
      .attr("text-anchor", "middle")
      .attr("fill", "hsl(var(--muted-foreground))")
      .attr("font-size", "10px")
      .text("Time (ms)");

    g.append("g")
      .attr("class", "y-axis")
      .call(yAxis)
      .selectAll("text")
      .style("fill", "hsl(var(--muted-foreground))");

    // Y-axis label
    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -height / 2)
      .attr("y", -35)
      .attr("text-anchor", "middle")
      .attr("fill", "hsl(var(--muted-foreground))")
      .attr("font-size", "10px")
      .text("Frequency (Hz)");

    // Style axis lines
    g.selectAll(".domain, .tick line").style("stroke", "hsl(var(--border))");

    // Add color bar
    const colorBarWidth = 15;
    const colorBarHeight = height;
    const colorBarX = width + 10;

    const colorBarScale = d3.scaleLinear().domain([minDb, maxDb]).range([colorBarHeight, 0]);

    const colorBarAxis = d3
      .axisRight(colorBarScale)
      .ticks(5)
      .tickFormat((d) => `${Math.round(+d)}dB`);

    // Draw color bar gradient
    const colorFn = colorScales[colorScale];
    for (let i = 0; i < colorBarHeight; i++) {
      const t = 1 - i / colorBarHeight;
      g.append("rect")
        .attr("x", colorBarX)
        .attr("y", i)
        .attr("width", colorBarWidth)
        .attr("height", 1)
        .attr("fill", colorFn(t));
    }

    g.append("g")
      .attr("transform", `translate(${colorBarX + colorBarWidth},0)`)
      .call(colorBarAxis)
      .selectAll("text")
      .style("fill", "hsl(var(--muted-foreground))")
      .attr("font-size", "9px");

    // Draw selection rectangle if selecting
    if (selection) {
      const x1 = Math.min(selection.startX, selection.endX);
      const y1 = Math.min(selection.startY, selection.endY);
      const selWidth = Math.abs(selection.endX - selection.startX);
      const selHeight = Math.abs(selection.endY - selection.startY);

      g.append("rect")
        .attr("x", x1)
        .attr("y", y1)
        .attr("width", selWidth)
        .attr("height", selHeight)
        .attr("fill", "rgba(255, 255, 255, 0.2)")
        .attr("stroke", "white")
        .attr("stroke-width", 1)
        .attr("stroke-dasharray", "4,4");
    }
  }, [
    dimensions,
    spectrogramData,
    logFrequency,
    colorScale,
    minFrequency,
    effectiveMaxFreq,
    selection,
  ]);

  // Selection handling
  const handleMouseDown = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      if (!onRegionSelect) return;

      const rect = overlayRef.current?.getBoundingClientRect();
      if (!rect) return;

      const x = e.clientX - rect.left - MARGIN.left;
      const y = e.clientY - rect.top - MARGIN.top;

      setIsSelecting(true);
      setSelection({ startX: x, startY: y, endX: x, endY: y });
    },
    [onRegionSelect]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      if (!isSelecting || !selection) return;

      const rect = overlayRef.current?.getBoundingClientRect();
      if (!rect) return;

      const x = e.clientX - rect.left - MARGIN.left;
      const y = e.clientY - rect.top - MARGIN.top;

      setSelection((prev) => (prev ? { ...prev, endX: x, endY: y } : null));
    },
    [isSelecting, selection]
  );

  const handleMouseUp = useCallback(() => {
    if (!isSelecting || !selection || !spectrogramData || !onRegionSelect) {
      setIsSelecting(false);
      setSelection(null);
      return;
    }

    const width = dimensions.width - MARGIN.left - MARGIN.right;
    const height = dimensions.height - MARGIN.top - MARGIN.bottom;

    const { numFrames, timePerFrame } = spectrogramData;
    const duration = numFrames * timePerFrame * 1000;

    // Create scales for inversion
    const xScale = d3.scaleLinear().domain([0, duration]).range([0, width]);
    const yScale = logFrequency
      ? d3.scaleLog().domain([minFrequency, effectiveMaxFreq]).range([height, 0]).clamp(true)
      : d3.scaleLinear().domain([minFrequency, effectiveMaxFreq]).range([height, 0]);

    const x1 = Math.min(selection.startX, selection.endX);
    const x2 = Math.max(selection.startX, selection.endX);
    const y1 = Math.min(selection.startY, selection.endY);
    const y2 = Math.max(selection.startY, selection.endY);

    const timeRange: [number, number] = [xScale.invert(x1), xScale.invert(x2)];
    const freqRange: [number, number] = [yScale.invert(y2), yScale.invert(y1)]; // Inverted because y is flipped

    onRegionSelect(timeRange, freqRange);

    setIsSelecting(false);
    setSelection(null);
  }, [
    isSelecting,
    selection,
    spectrogramData,
    onRegionSelect,
    dimensions,
    logFrequency,
    minFrequency,
    effectiveMaxFreq,
  ]);

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-muted-foreground">Spectrogram</h3>
        <div className="flex gap-1">
          <Button
            variant={logFrequency ? "default" : "outline"}
            size="sm"
            className="text-xs h-6 px-2"
            onClick={() => setLogFrequency(true)}
          >
            Log
          </Button>
          <Button
            variant={!logFrequency ? "default" : "outline"}
            size="sm"
            className="text-xs h-6 px-2"
            onClick={() => setLogFrequency(false)}
          >
            Linear
          </Button>
          <select
            value={colorScale}
            onChange={(e) => setColorScale(e.target.value as ColorScale)}
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
        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            left: MARGIN.left,
            top: MARGIN.top,
          }}
        />
        <svg
          ref={overlayRef}
          width={dimensions.width}
          height={dimensions.height}
          style={{ position: "absolute", cursor: onRegionSelect ? "crosshair" : "default" }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        />
      </div>
    </div>
  );
}
