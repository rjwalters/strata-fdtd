import { useRef, useEffect, useState, useCallback, useMemo } from "react";
import * as d3 from "d3";
import { Badge } from "@/components/ui/badge";
import { Loader2 } from "lucide-react";
import {
  lttbDownsample,
  minMaxDownsample,
  lttbDownsampleAsync,
  minMaxDownsampleAsync,
  WORKER_THRESHOLD,
} from "@/lib/downsample";
import { useStreamingBuffer } from "@/hooks/useStreamingBuffer";

export interface ProbeTimeSeries {
  position: [number, number, number];
  data: Float32Array;
}

export interface TimeSeriesPlotProps {
  probes: {
    [name: string]: ProbeTimeSeries;
  };
  sampleRate: number;
  currentTime?: number;
  selectedProbes?: string[];
  onTimeSelect?: (time: number) => void;
  onRangeSelect?: (range: [number, number]) => void;
  yRange?: [number, number];
  /** Render mode: 'auto' uses Canvas for large datasets, 'svg' forces SVG, 'canvas' forces Canvas */
  renderMode?: "auto" | "svg" | "canvas";
  /** Downsampling algorithm: 'lttb' (default), 'minmax', or 'none' */
  downsampleAlgorithm?: "lttb" | "minmax" | "none";
  /**
   * Enable incremental updates for streaming data.
   * When enabled, new data is rendered incrementally using canvas shifting
   * instead of full re-renders. Requires Canvas rendering mode.
   * Note: Incremental updates only work in Canvas mode (renderMode='canvas' or 'auto' with large datasets).
   * In SVG mode, streaming data will still trigger full re-renders.
   */
  streamingMode?: boolean;
  /** Callback when buffer should be flushed (e.g., on zoom) */
  onBufferFlush?: () => void;
  /** Maximum buffer size in streaming mode (default: 50000 points per probe) */
  streamingBufferSize?: number;
}

/** Threshold for switching to Canvas rendering in 'auto' mode */
const CANVAS_THRESHOLD = 10000;

const MARGIN = { top: 10, right: 20, bottom: 30, left: 50 };
const COLORS = d3.schemeCategory10;

/** Default buffer size for streaming mode */
const DEFAULT_STREAMING_BUFFER_SIZE = 50000;

export function TimeSeriesPlot({
  probes,
  sampleRate,
  currentTime,
  selectedProbes,
  onTimeSelect,
  onRangeSelect,
  yRange,
  renderMode = "auto",
  downsampleAlgorithm = "lttb",
  streamingMode = false,
  onBufferFlush,
  streamingBufferSize = DEFAULT_STREAMING_BUFFER_SIZE,
}: TimeSeriesPlotProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [hiddenProbes, setHiddenProbes] = useState<Set<string>>(new Set());
  const [zoomDomain, setZoomDomain] = useState<[number, number] | null>(null);
  const [isDownsampling, setIsDownsampling] = useState(false);
  // Track brush state to suppress tooltip during drag
  const isBrushingRef = useRef(false);
  // Cache for downsampled data per probe
  const [downsampledCache, setDownsampledCache] = useState<
    Map<string, [number, number][]>
  >(new Map());
  // Track last rendered data lengths for streaming mode
  const lastRenderedLengthsRef = useRef<Map<string, number>>(new Map());
  // Track if a full render is needed
  const needsFullRenderRef = useRef(true);

  // Streaming buffer hook for invalidation tracking
  const streamingBuffer = useStreamingBuffer({
    maxBufferSize: streamingBufferSize,
    sampleRate,
  });

  const probeNames = useMemo(() => Object.keys(probes), [probes]);
  const probeColors = useMemo(
    () => new Map(probeNames.map((name, i) => [name, COLORS[i % COLORS.length]])),
    [probeNames]
  );

  // Calculate visible probes based on selectedProbes prop and hidden state
  const visibleProbes = useMemo(() => {
    const base = selectedProbes ?? probeNames;
    return new Set(base.filter((name) => !hiddenProbes.has(name)));
  }, [selectedProbes, probeNames, hiddenProbes]);

  // Track container size
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setDimensions({ width, height });
      // Invalidate streaming buffer on resize
      if (streamingMode) {
        streamingBuffer.invalidate();
        needsFullRenderRef.current = true;
        onBufferFlush?.();
      }
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, [streamingMode, streamingBuffer, onBufferFlush]);

  // Calculate full duration and total samples
  const { fullDuration, totalSamples } = useMemo(() => {
    const firstProbe = probes[probeNames[0]];
    if (!firstProbe) return { fullDuration: 0, totalSamples: 0 };
    return {
      fullDuration: (firstProbe.data.length / sampleRate) * 1000, // ms
      totalSamples: firstProbe.data.length,
    };
  }, [probes, probeNames, sampleRate]);

  // Determine if we should use Canvas rendering
  const useCanvas = useMemo(() => {
    if (renderMode === "svg") return false;
    if (renderMode === "canvas") return true;
    // Auto mode: use Canvas for large datasets
    return totalSamples > CANVAS_THRESHOLD;
  }, [renderMode, totalSamples]);

  // Reset zoom callback - invalidates streaming buffer
  const resetZoom = useCallback(() => {
    setZoomDomain(null);
    if (streamingMode) {
      streamingBuffer.invalidate();
      needsFullRenderRef.current = true;
      onBufferFlush?.();
    }
  }, [streamingMode, streamingBuffer, onBufferFlush]);

  // Pre-compute downsampled data for large datasets using Web Workers
  useEffect(() => {
    // Only use async for very large datasets
    if (totalSamples < WORKER_THRESHOLD) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setDownsampledCache(new Map());
      return;
    }

    let cancelled = false;
    setIsDownsampling(true);

    const downsampleProbes = async () => {
      const newCache = new Map<string, [number, number][]>();
      const width = dimensions.width - MARGIN.left - MARGIN.right;
      const targetPoints = Math.max(Math.min(width * 4, totalSamples), 100);

      // Calculate sample range for current view
      const [xMin, xMax] = zoomDomain ?? [0, fullDuration];
      const startSample = Math.max(0, Math.floor((xMin / 1000) * sampleRate));
      const endSample = Math.min(totalSamples, Math.ceil((xMax / 1000) * sampleRate));

      for (const name of visibleProbes) {
        if (cancelled) return;
        const probe = probes[name];
        if (!probe) continue;

        const viewData = probe.data.subarray(startSample, endSample);
        const timeOffset = (startSample / sampleRate) * 1000;

        try {
          let result: [number, number][];
          if (downsampleAlgorithm === "minmax") {
            result = await minMaxDownsampleAsync(viewData, targetPoints, sampleRate);
          } else {
            result = await lttbDownsampleAsync(viewData, targetPoints, sampleRate);
          }

          // Adjust time offset
          result = result.map(([t, v]) => [t + timeOffset, v]);
          newCache.set(name, result);
        } catch (error) {
          console.error(`Downsampling error for ${name}:`, error);
          // Fallback to sync
          let result: [number, number][];
          if (downsampleAlgorithm === "minmax") {
            result = minMaxDownsample(viewData, targetPoints, sampleRate);
          } else {
            result = lttbDownsample(viewData, targetPoints, sampleRate);
          }
          result = result.map(([t, v]) => [t + timeOffset, v]);
          newCache.set(name, result);
        }
      }

      if (!cancelled) {
        setDownsampledCache(newCache);
        setIsDownsampling(false);
      }
    };

    downsampleProbes();

    return () => {
      cancelled = true;
    };
  }, [
    probes,
    visibleProbes,
    totalSamples,
    dimensions.width,
    zoomDomain,
    fullDuration,
    sampleRate,
    downsampleAlgorithm,
  ]);

  // Render chart (SVG for axes/UI, Canvas for data when useCanvas is true)
  useEffect(() => {
    if (!svgRef.current || dimensions.width === 0 || probeNames.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = dimensions.width - MARGIN.left - MARGIN.right;
    const height = dimensions.height - MARGIN.top - MARGIN.bottom;

    if (width <= 0 || height <= 0) return;

    // Use zoom domain if set, otherwise full duration
    // In streaming mode without zoom, use a sliding window of latest data
    let xMin: number, xMax: number;
    if (zoomDomain) {
      [xMin, xMax] = zoomDomain;
    } else if (streamingMode && fullDuration > 5000) {
      // Sliding window: show last 5 seconds
      xMax = fullDuration;
      xMin = Math.max(0, fullDuration - 5000);
    } else {
      xMin = 0;
      xMax = fullDuration;
    }

    // Set up Canvas if using canvas mode
    const canvas = canvasRef.current;
    let ctx: CanvasRenderingContext2D | null = null;
    if (useCanvas && canvas) {
      // Set canvas size with device pixel ratio for sharp rendering
      const dpr = window.devicePixelRatio || 1;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      canvas.style.left = `${MARGIN.left}px`;
      canvas.style.top = `${MARGIN.top}px`;
      ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, width, height);
      }
    }

    // Calculate pressure domain across visible probes (within zoom range if set)
    let minPressure = Infinity;
    let maxPressure = -Infinity;

    if (yRange) {
      // Use fixed y-range if provided
      [minPressure, maxPressure] = yRange;
    } else {
      // Auto-scale based on visible data
      for (const name of visibleProbes) {
        const probe = probes[name];
        if (!probe) continue;

        // Calculate sample indices for current view
        const startSample = Math.floor((xMin / 1000) * sampleRate);
        const endSample = Math.ceil((xMax / 1000) * sampleRate);

        for (let i = Math.max(0, startSample); i < Math.min(probe.data.length, endSample); i++) {
          const v = probe.data[i];
          if (v < minPressure) minPressure = v;
          if (v > maxPressure) maxPressure = v;
        }
      }

      // Handle edge case of no data
      if (!isFinite(minPressure)) {
        minPressure = -1;
        maxPressure = 1;
      }

      // Add some padding to y-axis
      const yPad = (maxPressure - minPressure) * 0.1 || 0.1;
      minPressure -= yPad;
      maxPressure += yPad;
    }

    // Create scales
    const xScale = d3.scaleLinear().domain([xMin, xMax]).range([0, width]);

    const yScale = d3.scaleLinear().domain([minPressure, maxPressure]).range([height, 0]);

    // Create main group
    const g = svg
      .append("g")
      .attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);

    // Create axes
    const xAxis = d3.axisBottom(xScale).ticks(8).tickFormat((d) => `${d}ms`);

    const yAxis = d3.axisLeft(yScale).ticks(5).tickFormat(d3.format(".2e"));

    g.append("g")
      .attr("transform", `translate(0,${height})`)
      .attr("class", "x-axis")
      .call(xAxis)
      .selectAll("text")
      .style("fill", "hsl(var(--muted-foreground))");

    g.append("g")
      .attr("class", "y-axis")
      .call(yAxis)
      .selectAll("text")
      .style("fill", "hsl(var(--muted-foreground))");

    // Style axis lines
    g.selectAll(".domain, .tick line").style("stroke", "hsl(var(--border))");

    // Add grid lines
    g.append("g")
      .attr("class", "grid")
      .selectAll("line")
      .data(yScale.ticks(5))
      .join("line")
      .attr("x1", 0)
      .attr("x2", width)
      .attr("y1", (d) => yScale(d))
      .attr("y2", (d) => yScale(d))
      .attr("stroke", "hsl(var(--border))")
      .attr("stroke-opacity", 0.3);

    // Downsample data for the current view
    // Uses cached data from Web Worker if available for large datasets
    const downsampleData = (probeName: string, data: Float32Array): [number, number][] => {
      // Check if we have pre-computed data from the worker
      const cached = downsampledCache.get(probeName);
      if (cached && cached.length > 0) {
        return cached;
      }

      // Calculate sample range for current view
      const startSample = Math.max(0, Math.floor((xMin / 1000) * sampleRate));
      const endSample = Math.min(data.length, Math.ceil((xMax / 1000) * sampleRate));
      const viewData = data.subarray(startSample, endSample);

      // Target points based on pixel width (2-4 points per pixel)
      const targetPoints = Math.min(width * 4, viewData.length);

      if (viewData.length <= targetPoints || downsampleAlgorithm === "none") {
        // No downsampling needed
        return Array.from(viewData, (y, i) => [
          ((startSample + i) / sampleRate) * 1000,
          y,
        ]);
      }

      // Apply selected downsampling algorithm (sync for small datasets)
      if (downsampleAlgorithm === "minmax") {
        // Min-max preserves peaks better for waveforms
        const result = minMaxDownsample(viewData, targetPoints, sampleRate);
        // Adjust time offset for the view
        const timeOffset = (startSample / sampleRate) * 1000;
        return result.map(([t, v]) => [t + timeOffset, v]);
      } else {
        // LTTB provides best visual fidelity
        const result = lttbDownsample(viewData, targetPoints, sampleRate);
        // Adjust time offset for the view
        const timeOffset = (startSample / sampleRate) * 1000;
        return result.map(([t, v]) => [t + timeOffset, v]);
      }
    };

    // Create line generator for SVG path
    const createLinePath = (points: [number, number][]) => {
      const line = d3
        .line<[number, number]>()
        .x((d) => xScale(d[0]))
        .y((d) => yScale(d[1]));
      return line(points);
    };

    // Draw lines for each visible probe
    for (const name of visibleProbes) {
      const probe = probes[name];
      if (!probe) continue;

      const downsampledPoints = downsampleData(name, probe.data);
      const color = probeColors.get(name) ?? COLORS[0];

      if (useCanvas && ctx) {
        // Canvas rendering for large datasets
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;

        let started = false;
        for (const [t, v] of downsampledPoints) {
          const x = xScale(t);
          const y = yScale(v);
          if (!isFinite(x) || !isFinite(y)) continue;

          if (!started) {
            ctx.moveTo(x, y);
            started = true;
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.stroke();
      } else {
        // SVG rendering
        const path = createLinePath(downsampledPoints);
        if (!path) continue;

        g.append("path")
          .attr("fill", "none")
          .attr("stroke", color)
          .attr("stroke-width", 1.5)
          .attr("d", path);
      }
    }

    // Draw current time marker
    if (currentTime !== undefined && currentTime >= 0 && currentTime <= fullDuration) {
      g.append("line")
        .attr("x1", xScale(currentTime))
        .attr("x2", xScale(currentTime))
        .attr("y1", 0)
        .attr("y2", height)
        .attr("stroke", "white")
        .attr("stroke-opacity", 0.6)
        .attr("stroke-width", 1)
        .attr("stroke-dasharray", "4,4");
    }

    // Create tooltip group
    const tooltip = g
      .append("g")
      .attr("class", "tooltip")
      .style("display", "none");

    // Tooltip background
    tooltip
      .append("rect")
      .attr("fill", "hsl(var(--popover))")
      .attr("stroke", "hsl(var(--border))")
      .attr("rx", 4)
      .attr("ry", 4);

    // Tooltip text
    const tooltipText = tooltip
      .append("text")
      .attr("fill", "hsl(var(--popover-foreground))")
      .attr("font-size", "11px")
      .attr("font-family", "monospace");

    // Vertical line for hover
    const hoverLine = g
      .append("line")
      .attr("stroke", "hsl(var(--muted-foreground))")
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "2,2")
      .style("display", "none");

    // Helper to find value at time for a probe
    const getValueAtTime = (probeName: string, timeMs: number): number | null => {
      const probe = probes[probeName];
      if (!probe) return null;
      const sampleIndex = Math.round((timeMs / 1000) * sampleRate);
      if (sampleIndex < 0 || sampleIndex >= probe.data.length) return null;
      return probe.data[sampleIndex];
    };

    // Add D3 brush for zoom/range selection
    const brush = d3
      .brushX<unknown>()
      .extent([
        [0, 0],
        [width, height],
      ])
      .on("start", () => {
        // Mark brush as active to suppress tooltip
        isBrushingRef.current = true;
        tooltip.style("display", "none");
        hoverLine.style("display", "none");
      })
      .on("end", (event: d3.D3BrushEvent<unknown>) => {
        // Mark brush as inactive
        isBrushingRef.current = false;

        if (!event.selection) return;

        const [x0, x1] = (event.selection as [number, number]).map(xScale.invert);

        // Only zoom if selection is significant (more than 5 pixels)
        if (Math.abs((event.selection as [number, number])[1] - (event.selection as [number, number])[0]) > 5) {
          // Call range select callback if provided
          if (onRangeSelect) {
            onRangeSelect([x0, x1]);
          }

          // Update zoom domain - invalidates streaming buffer
          setZoomDomain([x0, x1]);
          if (streamingMode) {
            streamingBuffer.invalidate();
            needsFullRenderRef.current = true;
            onBufferFlush?.();
          }
        }

        // Clear the brush selection visually
        g.select<SVGGElement>(".brush").call(brush.move, null);
      });

    g.append("g")
      .attr("class", "brush")
      .call(brush)
      .selectAll(".selection")
      .style("fill", "hsl(var(--primary))")
      .style("fill-opacity", 0.2)
      .style("stroke", "hsl(var(--primary))");

    // Interaction overlay for tooltip and click
    g.append("rect")
      .attr("width", width)
      .attr("height", height)
      .attr("fill", "transparent")
      .style("cursor", "crosshair")
      .style("pointer-events", "all")
      .lower() // Put behind brush
      .on("mousemove", (event) => {
        // Suppress tooltip while brush is active
        if (isBrushingRef.current) {
          return;
        }

        const [x] = d3.pointer(event);
        const timeMs = xScale.invert(x);

        // Update hover line
        hoverLine
          .attr("x1", x)
          .attr("x2", x)
          .attr("y1", 0)
          .attr("y2", height)
          .style("display", null);

        // Build tooltip content
        const lines: string[] = [`t = ${timeMs.toFixed(2)} ms`];
        for (const name of visibleProbes) {
          const value = getValueAtTime(name, timeMs);
          if (value !== null) {
            lines.push(`${name}: ${value.toExponential(2)}`);
          }
        }

        // Update tooltip text
        tooltipText.selectAll("tspan").remove();
        lines.forEach((line, i) => {
          tooltipText
            .append("tspan")
            .attr("x", 8)
            .attr("dy", i === 0 ? 14 : 14)
            .text(line);
        });

        // Size tooltip background
        const textBox = (tooltipText.node() as SVGTextElement).getBBox();
        tooltip
          .select("rect")
          .attr("width", textBox.width + 16)
          .attr("height", textBox.height + 8)
          .attr("y", 2);

        // Position tooltip (flip if near right edge)
        const tooltipWidth = textBox.width + 16;
        const tooltipX = x + 15 + tooltipWidth > width ? x - tooltipWidth - 10 : x + 15;
        const tooltipY = Math.min(10, height - textBox.height - 20);
        tooltip.attr("transform", `translate(${tooltipX},${tooltipY})`).style("display", null);
      })
      .on("mouseleave", () => {
        tooltip.style("display", "none");
        hoverLine.style("display", "none");
      })
      .on("click", (event) => {
        if (onTimeSelect) {
          const [x] = d3.pointer(event);
          const time = xScale.invert(x);
          onTimeSelect(time);
        }
      })
      .on("dblclick", () => {
        // Double-click to reset zoom
        setZoomDomain(null);
        if (streamingMode) {
          streamingBuffer.invalidate();
          needsFullRenderRef.current = true;
          onBufferFlush?.();
        }
      });

    // Mark full render complete in streaming mode
    if (streamingMode) {
      needsFullRenderRef.current = false;
      // Update last rendered lengths
      for (const name of visibleProbes) {
        const probe = probes[name];
        if (probe) {
          lastRenderedLengthsRef.current.set(name, probe.data.length);
        }
      }
    }
  }, [
    dimensions,
    probes,
    sampleRate,
    currentTime,
    visibleProbes,
    onTimeSelect,
    onRangeSelect,
    probeNames,
    probeColors,
    zoomDomain,
    fullDuration,
    yRange,
    downsampleAlgorithm,
    useCanvas,
    downsampledCache,
    streamingMode,
    streamingBuffer,
    onBufferFlush,
  ]);

  // Incremental render effect for streaming mode (Canvas only)
  // Uses canvas content shifting to append new data without full re-render.
  // Falls back to full re-render in SVG mode or when zoomed.
  useEffect(() => {
    if (!streamingMode || !useCanvas || !canvasRef.current) return;
    if (needsFullRenderRef.current) return; // Wait for full render
    if (zoomDomain !== null) return; // Don't do incremental updates when zoomed

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = dimensions.width - MARGIN.left - MARGIN.right;
    const height = dimensions.height - MARGIN.top - MARGIN.bottom;
    if (width <= 0 || height <= 0) return;

    const dpr = window.devicePixelRatio || 1;

    // Check if any probe has new data
    let hasNewData = false;
    for (const name of visibleProbes) {
      const probe = probes[name];
      if (!probe) continue;
      const lastLength = lastRenderedLengthsRef.current.get(name) ?? 0;
      if (probe.data.length > lastLength) {
        hasNewData = true;
        break;
      }
    }

    if (!hasNewData) return;

    // Calculate time window (sliding window showing latest data)
    const windowDuration = fullDuration > 0 ? Math.min(fullDuration, 5000) : 5000; // 5s window or less
    const latestTime = fullDuration;
    const windowStart = Math.max(0, latestTime - windowDuration);

    // Calculate how much to shift the canvas
    const prevLatestTime = Math.max(
      ...Array.from(lastRenderedLengthsRef.current.values()).map(
        (len) => (len / sampleRate) * 1000
      ),
      0
    );
    const timeDelta = latestTime - prevLatestTime;
    const pixelsPerMs = width / windowDuration;
    const shiftPixels = Math.round(timeDelta * pixelsPerMs);

    if (shiftPixels > 0 && shiftPixels < width) {
      // Shift existing content left
      const imageData = ctx.getImageData(
        shiftPixels * dpr,
        0,
        (width - shiftPixels) * dpr,
        height * dpr
      );
      ctx.clearRect(0, 0, width, height);
      ctx.putImageData(imageData, 0, 0);

      // Create scales for the new data region
      const xScale = d3.scaleLinear().domain([windowStart, latestTime]).range([0, width]);

      // Calculate y range for visible data
      let minPressure = Infinity;
      let maxPressure = -Infinity;

      if (yRange) {
        [minPressure, maxPressure] = yRange;
      } else {
        for (const name of visibleProbes) {
          const probe = probes[name];
          if (!probe) continue;
          const startSample = Math.floor((windowStart / 1000) * sampleRate);
          const endSample = Math.min(probe.data.length, Math.ceil((latestTime / 1000) * sampleRate));
          for (let i = Math.max(0, startSample); i < endSample; i++) {
            const v = probe.data[i];
            if (v < minPressure) minPressure = v;
            if (v > maxPressure) maxPressure = v;
          }
        }
        if (!isFinite(minPressure)) {
          minPressure = -1;
          maxPressure = 1;
        }
        const yPad = (maxPressure - minPressure) * 0.1 || 0.1;
        minPressure -= yPad;
        maxPressure += yPad;
      }

      const yScale = d3.scaleLinear().domain([minPressure, maxPressure]).range([height, 0]);

      // Draw only the new portion for each probe
      const newRegionStart = latestTime - timeDelta;
      for (const name of visibleProbes) {
        const probe = probes[name];
        if (!probe) continue;

        const lastLength = lastRenderedLengthsRef.current.get(name) ?? 0;
        if (probe.data.length <= lastLength) continue;

        const color = probeColors.get(name) ?? COLORS[0];

        // Get new samples
        const startSample = Math.max(0, Math.floor((newRegionStart / 1000) * sampleRate) - 1);
        const newPoints: [number, number][] = [];
        for (let i = startSample; i < probe.data.length; i++) {
          const timeMs = (i / sampleRate) * 1000;
          newPoints.push([timeMs, probe.data[i]]);
        }

        if (newPoints.length === 0) continue;

        // Draw new segment
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;

        let started = false;
        for (const [t, v] of newPoints) {
          const x = xScale(t);
          const y = yScale(v);
          if (!isFinite(x) || !isFinite(y)) continue;
          if (x < width - shiftPixels - 5) continue; // Skip points in shifted region (with small overlap)

          if (!started) {
            ctx.moveTo(x, y);
            started = true;
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.stroke();
      }
    } else if (shiftPixels >= width) {
      // Too much new data, need full re-render
      needsFullRenderRef.current = true;
    }

    // Update last rendered lengths
    for (const name of visibleProbes) {
      const probe = probes[name];
      if (probe) {
        lastRenderedLengthsRef.current.set(name, probe.data.length);
      }
    }
  }, [
    streamingMode,
    useCanvas,
    probes,
    visibleProbes,
    dimensions,
    fullDuration,
    sampleRate,
    probeColors,
    yRange,
    zoomDomain,
  ]);

  const toggleProbe = useCallback((name: string) => {
    setHiddenProbes((prev) => {
      const next = new Set(prev);
      if (next.has(name)) {
        next.delete(name);
      } else {
        next.add(name);
      }
      return next;
    });
    // Invalidate streaming buffer when visibility changes
    if (streamingMode) {
      streamingBuffer.invalidate(name);
      needsFullRenderRef.current = true;
      onBufferFlush?.();
    }
  }, [streamingMode, streamingBuffer, onBufferFlush]);

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold text-muted-foreground">Pressure vs Time</h3>
          {streamingMode && (
            <Badge variant="secondary" className="text-xs">
              LIVE
            </Badge>
          )}
          {zoomDomain && (
            <Badge
              variant="outline"
              className="cursor-pointer text-xs"
              onClick={resetZoom}
            >
              {zoomDomain[0].toFixed(1)}–{zoomDomain[1].toFixed(1)} ms ✕
            </Badge>
          )}
        </div>
        <div className="flex gap-1">
          {probeNames.map((name) => (
            <Badge
              key={name}
              variant={visibleProbes.has(name) ? "default" : "outline"}
              className="cursor-pointer text-xs"
              style={{
                backgroundColor: visibleProbes.has(name)
                  ? probeColors.get(name)
                  : undefined,
                borderColor: probeColors.get(name),
              }}
              onClick={() => toggleProbe(name)}
            >
              {name}
            </Badge>
          ))}
        </div>
      </div>
      <div ref={containerRef} className="flex-1 min-h-0 relative">
        <svg ref={svgRef} width={dimensions.width} height={dimensions.height} />
        {useCanvas && (
          <canvas
            ref={canvasRef}
            className="absolute pointer-events-none"
            style={{ position: "absolute" }}
          />
        )}
        {isDownsampling && (
          <div className="absolute inset-0 flex items-center justify-center bg-background/50">
            <div className="flex items-center gap-2 text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span className="text-sm">Processing data...</span>
            </div>
          </div>
        )}
      </div>
      {zoomDomain && (
        <div className="text-xs text-muted-foreground mt-1 text-center">
          Double-click to reset zoom
        </div>
      )}
    </div>
  );
}
