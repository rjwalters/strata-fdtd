import { useRef, useEffect, useState, useMemo, useCallback } from "react";
import * as d3 from "d3";
import {
  analyzeTransferFunctionAsync,
  terminateAcousticsWorker,
  type AcousticMetrics,
  type ImpulseResponseResult,
  type EnergyDecayResult,
  type WindowType,
} from "@/lib/acoustics";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Loader2, TrendingDown, Timer, Waves } from "lucide-react";

/** Display mode for the impulse response view */
export type ImpulseDisplayMode = "impulse" | "decay";

export interface ImpulseResponseViewProps {
  /** Real part of transfer function (positive frequencies) */
  transferReal: Float32Array;
  /** Imaginary part of transfer function (positive frequencies) */
  transferImag: Float32Array;
  /** Sample rate in Hz */
  sampleRate: number;
  /** Name of the reference source for display */
  referenceName?: string;
  /** Name of the probe for display */
  probeName?: string;
}

const MARGIN = { top: 10, right: 20, bottom: 30, left: 50 };

/**
 * Format a value with unit, handling NaN values.
 */
function formatValue(value: number, unit: string, decimals: number = 2): string {
  if (isNaN(value) || !isFinite(value)) return "N/A";
  return `${value.toFixed(decimals)} ${unit}`;
}

/**
 * ImpulseResponseView displays the room impulse response and acoustic metrics.
 *
 * Features:
 * - Time-domain impulse response waveform
 * - Energy decay curve (Schroeder integration)
 * - Acoustic metrics panel (RT60, EDT, C80, D50, etc.)
 */
export function ImpulseResponseView({
  transferReal,
  transferImag,
  sampleRate,
  referenceName,
  probeName,
}: ImpulseResponseViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [displayMode, setDisplayMode] = useState<ImpulseDisplayMode>("impulse");
  const [windowType, setWindowType] = useState<WindowType>("tukey");
  const [isComputing, setIsComputing] = useState(false);
  const [zoomDomain, setZoomDomain] = useState<[number, number] | null>(null);

  // Analysis results
  const [impulseResponse, setImpulseResponse] = useState<ImpulseResponseResult | null>(null);
  const [energyDecay, setEnergyDecay] = useState<EnergyDecayResult | null>(null);
  const [metrics, setMetrics] = useState<AcousticMetrics | null>(null);

  // Track brush state
  const isBrushingRef = useRef(false);

  // Compute analysis when transfer function changes (using Web Worker)
  useEffect(() => {
    if (transferReal.length === 0 || transferImag.length === 0) {
      setImpulseResponse(null);
      setEnergyDecay(null);
      setMetrics(null);
      return;
    }

    setIsComputing(true);
    let cancelled = false;

    // Use async Web Worker for computation
    analyzeTransferFunctionAsync(
      transferReal,
      transferImag,
      sampleRate,
      windowType
    )
      .then((result) => {
        if (cancelled) return;
        setImpulseResponse(result.impulseResponse);
        setEnergyDecay(result.energyDecay);
        setMetrics(result.metrics);
      })
      .catch((error) => {
        if (cancelled) return;
        console.error("Impulse response analysis error:", error);
        setImpulseResponse(null);
        setEnergyDecay(null);
        setMetrics(null);
      })
      .finally(() => {
        if (!cancelled) {
          setIsComputing(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [transferReal, transferImag, sampleRate, windowType]);

  // Cleanup worker on unmount
  useEffect(() => {
    return () => {
      terminateAcousticsWorker();
    };
  }, []);

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

  // Maximum time to display (ms)
  const maxDisplayTimeMs = useMemo(() => {
    if (!impulseResponse) return 500;
    // Auto-scale based on RT60 or default to 500ms
    const rt60 = metrics?.t30 ?? metrics?.t20 ?? 0.5;
    return Math.min(Math.max(rt60 * 1.5 * 1000, 200), 2000); // 200ms to 2000ms
  }, [impulseResponse, metrics]);

  // Render impulse response chart
  useEffect(() => {
    if (!svgRef.current || dimensions.width === 0 || !impulseResponse) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = dimensions.width - MARGIN.left - MARGIN.right;
    const height = dimensions.height - MARGIN.top - MARGIN.bottom;

    if (width <= 0 || height <= 0) return;

    const { impulseResponse: ir, timeAxis } = impulseResponse;

    // Time range (in ms)
    const defaultMaxTime = maxDisplayTimeMs / 1000; // Convert to seconds
    const [minTime, maxTime] = zoomDomain ?? [0, defaultMaxTime];

    // Find sample range
    const startIdx = Math.max(0, Math.floor(minTime * sampleRate));
    const endIdx = Math.min(ir.length - 1, Math.ceil(maxTime * sampleRate));

    // Get data for current display mode
    let displayData: Float32Array;
    let yLabel: string;
    let yMin: number, yMax: number;

    if (displayMode === "impulse") {
      displayData = ir;
      yLabel = "Amplitude";
      // Find amplitude range
      let absMax = 0;
      for (let i = startIdx; i <= endIdx; i++) {
        absMax = Math.max(absMax, Math.abs(ir[i]));
      }
      yMin = -absMax * 1.1;
      yMax = absMax * 1.1;
    } else {
      displayData = energyDecay?.decayCurve ?? new Float32Array(0);
      yLabel = "Level (dB)";
      yMin = -60;
      yMax = 0;
    }

    // Create scales
    const xScale = d3
      .scaleLinear()
      .domain([minTime * 1000, maxTime * 1000]) // Convert to ms for display
      .range([0, width]);

    const yScale = d3.scaleLinear().domain([yMin, yMax]).range([height, 0]);

    // Create main group
    const g = svg
      .append("g")
      .attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);

    // Create axes
    const xAxis = d3
      .axisBottom(xScale)
      .ticks(6)
      .tickFormat((d) => `${d}`);

    const yAxis = d3
      .axisLeft(yScale)
      .ticks(5);

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
      .text(yLabel);

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

    // Add 0 dB reference line for decay mode
    if (displayMode === "decay") {
      g.append("line")
        .attr("x1", 0)
        .attr("x2", width)
        .attr("y1", yScale(0))
        .attr("y2", yScale(0))
        .attr("stroke", "hsl(var(--muted-foreground))")
        .attr("stroke-width", 1)
        .attr("stroke-dasharray", "4,4");
    }

    // Build path data with downsampling for performance
    const targetPoints = Math.min(width * 2, endIdx - startIdx + 1);
    const step = Math.max(1, Math.floor((endIdx - startIdx + 1) / targetPoints));

    const pathData: [number, number][] = [];
    for (let i = startIdx; i <= endIdx; i += step) {
      const x = xScale(timeAxis[i] * 1000);
      const y = yScale(displayData[i]);
      if (isFinite(x) && isFinite(y)) {
        pathData.push([x, y]);
      }
    }

    // Create line generator
    const line = d3.line().x((d) => d[0]).y((d) => d[1]);

    // Draw waveform/decay line
    g.append("path")
      .attr("fill", "none")
      .attr("stroke", displayMode === "impulse" ? "hsl(var(--primary))" : "hsl(var(--accent))")
      .attr("stroke-width", displayMode === "impulse" ? 1 : 2)
      .attr("d", line(pathData));

    // Add RT60 markers in decay mode
    if (displayMode === "decay" && metrics) {
      // Draw -5 dB line
      g.append("line")
        .attr("x1", 0)
        .attr("x2", width)
        .attr("y1", yScale(-5))
        .attr("y2", yScale(-5))
        .attr("stroke", "hsl(var(--chart-1))")
        .attr("stroke-width", 1)
        .attr("stroke-dasharray", "2,2")
        .attr("stroke-opacity", 0.5);

      // Draw -35 dB line (T30 endpoint)
      g.append("line")
        .attr("x1", 0)
        .attr("x2", width)
        .attr("y1", yScale(-35))
        .attr("y2", yScale(-35))
        .attr("stroke", "hsl(var(--chart-2))")
        .attr("stroke-width", 1)
        .attr("stroke-dasharray", "2,2")
        .attr("stroke-opacity", 0.5);

      // Add legend
      const legend = g
        .append("g")
        .attr("transform", `translate(${width - 80}, 10)`);

      legend
        .append("text")
        .attr("x", 0)
        .attr("y", 0)
        .attr("fill", "hsl(var(--chart-1))")
        .attr("font-size", "9px")
        .text("-5 dB (start)");

      legend
        .append("text")
        .attr("x", 0)
        .attr("y", 12)
        .attr("fill", "hsl(var(--chart-2))")
        .attr("font-size", "9px")
        .text("-35 dB (T30)");
    }

    // Add early/late boundary markers in impulse mode
    if (displayMode === "impulse") {
      const boundaries = [
        { ms: 50, label: "50ms", color: "hsl(var(--chart-3))" },
        { ms: 80, label: "80ms", color: "hsl(var(--chart-4))" },
      ];

      for (const { ms, label, color } of boundaries) {
        if (ms >= minTime * 1000 && ms <= maxTime * 1000) {
          const x = xScale(ms);
          g.append("line")
            .attr("x1", x)
            .attr("x2", x)
            .attr("y1", 0)
            .attr("y2", height)
            .attr("stroke", color)
            .attr("stroke-width", 1)
            .attr("stroke-dasharray", "4,4");

          g.append("text")
            .attr("x", x + 4)
            .attr("y", 12)
            .attr("fill", color)
            .attr("font-size", "9px")
            .text(label);
        }
      }
    }

    // Create tooltip
    const tooltip = g
      .append("g")
      .attr("class", "tooltip")
      .style("display", "none");

    tooltip
      .append("rect")
      .attr("fill", "hsl(var(--popover))")
      .attr("stroke", "hsl(var(--border))")
      .attr("rx", 4)
      .attr("ry", 4);

    const tooltipText = tooltip
      .append("text")
      .attr("fill", "hsl(var(--popover-foreground))")
      .attr("font-size", "11px")
      .attr("font-family", "monospace");

    const hoverLine = g
      .append("line")
      .attr("stroke", "hsl(var(--muted-foreground))")
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "2,2")
      .style("display", "none");

    // Brush for zoom
    const brush = d3
      .brushX<unknown>()
      .extent([
        [0, 0],
        [width, height],
      ])
      .on("start", () => {
        isBrushingRef.current = true;
        tooltip.style("display", "none");
        hoverLine.style("display", "none");
      })
      .on("end", (event: d3.D3BrushEvent<unknown>) => {
        isBrushingRef.current = false;
        if (!event.selection) return;

        const [x0, x1] = (event.selection as [number, number]).map((x) => xScale.invert(x) / 1000);

        if (Math.abs((event.selection as [number, number])[1] - (event.selection as [number, number])[0]) > 5) {
          setZoomDomain([x0, x1]);
        }

        g.select<SVGGElement>(".brush").call(brush.move, null);
      });

    g.append("g")
      .attr("class", "brush")
      .call(brush)
      .selectAll(".selection")
      .style("fill", "hsl(var(--primary))")
      .style("fill-opacity", 0.2)
      .style("stroke", "hsl(var(--primary))");

    // Interaction overlay
    g.append("rect")
      .attr("width", width)
      .attr("height", height)
      .attr("fill", "transparent")
      .style("cursor", "crosshair")
      .style("pointer-events", "all")
      .lower()
      .on("mousemove", (event) => {
        if (isBrushingRef.current) return;

        const [x] = d3.pointer(event);
        const timeMs = xScale.invert(x);
        const timeSec = timeMs / 1000;

        if (timeSec < minTime || timeSec > maxTime) {
          tooltip.style("display", "none");
          hoverLine.style("display", "none");
          return;
        }

        hoverLine
          .attr("x1", x)
          .attr("x2", x)
          .attr("y1", 0)
          .attr("y2", height)
          .style("display", null);

        // Find closest sample
        const sampleIdx = Math.round(timeSec * sampleRate);
        if (sampleIdx < 0 || sampleIdx >= displayData.length) return;

        const value = displayData[sampleIdx];
        const timeStr = `${timeMs.toFixed(1)} ms`;
        const valueStr = displayMode === "impulse"
          ? value.toExponential(2)
          : `${value.toFixed(1)} dB`;

        tooltipText.selectAll("tspan").remove();
        [timeStr, valueStr].forEach((line, i) => {
          tooltipText
            .append("tspan")
            .attr("x", 8)
            .attr("dy", i === 0 ? 14 : 14)
            .text(line);
        });

        const textBox = (tooltipText.node() as SVGTextElement).getBBox();
        tooltip
          .select("rect")
          .attr("width", textBox.width + 16)
          .attr("height", textBox.height + 8)
          .attr("y", 2);

        const tooltipWidth = textBox.width + 16;
        const tooltipX = x + 15 + tooltipWidth > width ? x - tooltipWidth - 10 : x + 15;
        const tooltipY = Math.min(10, height - textBox.height - 20);
        tooltip.attr("transform", `translate(${tooltipX},${tooltipY})`).style("display", null);
      })
      .on("mouseleave", () => {
        tooltip.style("display", "none");
        hoverLine.style("display", "none");
      })
      .on("dblclick", () => {
        setZoomDomain(null);
      });
  }, [dimensions, impulseResponse, energyDecay, displayMode, metrics, sampleRate, zoomDomain, maxDisplayTimeMs]);

  // Reset zoom callback
  const resetZoom = useCallback(() => {
    setZoomDomain(null);
  }, []);

  // Format time domain label
  const formatTimeRange = useCallback((domain: [number, number]) => {
    const [min, max] = domain;
    return `${(min * 1000).toFixed(0)}–${(max * 1000).toFixed(0)} ms`;
  }, []);

  return (
    <div className="h-full flex flex-col gap-2">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold text-muted-foreground">
            Room Impulse Response
          </h3>
          {probeName && referenceName && (
            <Badge variant="secondary" className="text-xs">
              {referenceName} {"\u2192"} {probeName}
            </Badge>
          )}
          {zoomDomain && (
            <Badge
              variant="outline"
              className="cursor-pointer text-xs"
              onClick={resetZoom}
            >
              {formatTimeRange(zoomDomain)} ✕
            </Badge>
          )}
        </div>
        <div className="flex gap-1">
          <Button
            variant={displayMode === "impulse" ? "default" : "outline"}
            size="sm"
            className="text-xs h-6 px-2"
            onClick={() => setDisplayMode("impulse")}
          >
            <Waves className="h-3 w-3 mr-1" />
            Impulse
          </Button>
          <Button
            variant={displayMode === "decay" ? "default" : "outline"}
            size="sm"
            className="text-xs h-6 px-2"
            onClick={() => setDisplayMode("decay")}
          >
            <TrendingDown className="h-3 w-3 mr-1" />
            Decay
          </Button>
        </div>
      </div>

      {/* Window type selector */}
      <div className="flex items-center gap-2 text-xs">
        <span className="text-muted-foreground">Window:</span>
        <div className="flex gap-1">
          {(["tukey", "hanning", "none"] as WindowType[]).map((type) => (
            <Badge
              key={type}
              variant={windowType === type ? "default" : "outline"}
              className="cursor-pointer capitalize"
              onClick={() => setWindowType(type)}
            >
              {type}
            </Badge>
          ))}
        </div>
      </div>

      {/* Main plot area */}
      <div ref={containerRef} className="flex-1 min-h-0 relative">
        <svg ref={svgRef} width={dimensions.width} height={dimensions.height} />
        {isComputing && (
          <div className="absolute inset-0 flex items-center justify-center bg-background/50">
            <div className="flex items-center gap-2 text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span className="text-sm">Computing impulse response...</span>
            </div>
          </div>
        )}
        {!isComputing && !impulseResponse && transferReal.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-sm text-muted-foreground">
              No transfer function data available
            </span>
          </div>
        )}
      </div>

      {/* Acoustic Metrics Panel */}
      {metrics && (
        <Card className="flex-none">
          <CardHeader className="py-2 px-3">
            <CardTitle className="text-xs flex items-center gap-1">
              <Timer className="h-3 w-3" />
              Acoustic Metrics
            </CardTitle>
          </CardHeader>
          <CardContent className="py-2 px-3">
            <div
              className="grid gap-x-4 gap-y-1 text-xs"
              style={{
                gridTemplateColumns: "repeat(auto-fit, minmax(100px, 1fr))",
              }}
            >
              <div className="flex justify-between">
                <span className="text-muted-foreground">RT60 (T30):</span>
                <span className="font-mono">{formatValue(metrics.t30, "s")}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">RT60 (T20):</span>
                <span className="font-mono">{formatValue(metrics.t20, "s")}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">EDT:</span>
                <span className="font-mono">{formatValue(metrics.edt, "s")}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">C80:</span>
                <span className="font-mono">{formatValue(metrics.c80, "dB", 1)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">C50:</span>
                <span className="font-mono">{formatValue(metrics.c50, "dB", 1)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">D50:</span>
                <span className="font-mono">{formatValue(metrics.d50, "%", 1)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">D80:</span>
                <span className="font-mono">{formatValue(metrics.d80, "%", 1)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Ts:</span>
                <span className="font-mono">{formatValue(metrics.ts, "ms", 1)}</span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {zoomDomain && (
        <div className="text-xs text-muted-foreground text-center">
          Double-click to reset zoom
        </div>
      )}
    </div>
  );
}
