import { useRef, useEffect, useState, useCallback, useMemo } from "react";
import * as d3 from "d3";
import {
  computeSpectrumAsync,
  computeComplexSpectrum,
  extractTransferPhase,
  unwrapPhase,
  computeGroupDelay,
  computeCoherence,
  type ComplexSpectrum,
  type CoherenceResult,
} from "@/lib/fft";
import { logBinDownsample, WORKER_THRESHOLD } from "@/lib/downsample";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Loader2 } from "lucide-react";

export type PhaseViewMode = "phase" | "groupDelay";

export interface FrequencyMarker {
  frequency: number;
  label: string;
}

/** Analysis mode for the spectrum plot */
export type AnalysisMode = "spectrum" | "coherence";

/** Spectrum display mode */
export type SpectrumMode = "spectrum" | "transfer";

/** Probe data structure for coherence analysis */
export interface ProbeRecord {
  position: [number, number, number];
  data: Float32Array;
}

export interface SpectrumPlotProps {
  /** Single probe data for spectrum mode */
  data: Float32Array;
  /** Sample rate in Hz */
  sampleRate: number;
  /** FFT size (defaults to next power of 2) */
  nfft?: number;
  /** Use logarithmic (dB) scale */
  logScale?: boolean;
  /** Frequency markers to display */
  markers?: FrequencyMarker[];
  /** Analysis mode: spectrum or coherence */
  analysisMode?: AnalysisMode;
  /** Display mode: 'spectrum' for power spectrum, 'transfer' for transfer function */
  mode?: SpectrumMode;
  /** Reference signal for transfer function (source waveform) */
  referenceData?: Float32Array;
  /** Name of the reference source for display */
  referenceName?: string;
  /** All available probes for coherence analysis */
  probes?: Record<string, ProbeRecord>;
  /** Selected reference probe name for coherence */
  referenceProbe?: string | null;
  /** Selected measurement probe name for coherence */
  measurementProbe?: string | null;
  /** Callback when reference probe changes */
  onReferenceProbeChange?: (name: string) => void;
  /** Callback when measurement probe changes */
  onMeasurementProbeChange?: (name: string) => void;
}

const MARGIN = { top: 10, right: 20, bottom: 30, left: 50 };

/**
 * Calculate Q-factor at a peak using the -3dB bandwidth method
 * Q = f₀ / Δf where Δf is the bandwidth at -3dB from peak
 */
function calculateQFactor(
  frequencies: Float32Array,
  magnitude: Float32Array,
  peakIdx: number
): number | null {
  const peakMag = magnitude[peakIdx];
  // -3dB point is where magnitude drops to 1/√2 of peak
  const threshold = peakMag / Math.sqrt(2);

  // Find -3dB points on either side
  let lowIdx = peakIdx;
  let highIdx = peakIdx;

  // Search left for -3dB point
  while (lowIdx > 0 && magnitude[lowIdx] > threshold) {
    lowIdx--;
  }

  // Search right for -3dB point
  while (highIdx < magnitude.length - 1 && magnitude[highIdx] > threshold) {
    highIdx++;
  }

  // If we hit the boundaries, Q-factor is undefined
  if (lowIdx === 0 || highIdx === magnitude.length - 1) {
    return null;
  }

  const bandwidth = frequencies[highIdx] - frequencies[lowIdx];
  if (bandwidth <= 0) return null;

  return frequencies[peakIdx] / bandwidth;
}

interface PeakInfo {
  frequency: number;
  magnitude: number;
  index: number;
  qFactor: number | null;
}


function findPeaks(
  frequencies: Float32Array,
  magnitude: Float32Array,
  threshold: number = 0.1,
  maxPeaks: number = 5
): PeakInfo[] {
  const maxMag = Math.max(...magnitude);
  const peaks: PeakInfo[] = [];

  for (let i = 1; i < magnitude.length - 1; i++) {
    // Local maximum check
    if (magnitude[i] > magnitude[i - 1] && magnitude[i] > magnitude[i + 1]) {
      // Above threshold
      if (magnitude[i] > maxMag * threshold) {
        const qFactor = calculateQFactor(frequencies, magnitude, i);
        peaks.push({
          frequency: frequencies[i],
          magnitude: magnitude[i],
          index: i,
          qFactor,
        });
      }
    }
  }

  // Sort by magnitude and take top N
  peaks.sort((a, b) => b.magnitude - a.magnitude);
  return peaks.slice(0, maxPeaks);
}

function toDecibels(value: number, ref: number = 1): number {
  return 20 * Math.log10(Math.max(value / ref, 1e-10));
}

export function SpectrumPlot({
  data,
  sampleRate,
  nfft,
  logScale: initialLogScale = true,
  markers = [],
  analysisMode = "spectrum",
  mode = "spectrum",
  referenceData,
  referenceName,
  probes,
  referenceProbe,
  measurementProbe,
  onReferenceProbeChange,
  onMeasurementProbeChange,
}: SpectrumPlotProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const phaseSvgRef = useRef<SVGSVGElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [logScale, setLogScale] = useState(initialLogScale);
  const [showPeaks, setShowPeaks] = useState(true);
  const [zoomDomain, setZoomDomain] = useState<[number, number] | null>(null);
  const [spectrum, setSpectrum] = useState<{
    frequencies: Float32Array;
    magnitude: Float32Array;
  } | null>(null);
  const [coherenceResult, setCoherenceResult] = useState<CoherenceResult | null>(null);
  const [complexSpectrum, setComplexSpectrum] = useState<ComplexSpectrum | null>(null);
  const [referenceSpectrum, setReferenceSpectrum] = useState<{
    frequencies: Float32Array;
    magnitude: Float32Array;
  } | null>(null);
  const [referenceComplexSpectrum, setReferenceComplexSpectrum] = useState<ComplexSpectrum | null>(null);
  const [isComputing, setIsComputing] = useState(false);
  // Phase display options
  const [showPhase, setShowPhase] = useState(false);
  const [phaseViewMode, setPhaseViewMode] = useState<PhaseViewMode>("phase");
  const [phaseUnwrap, setPhaseUnwrap] = useState(true);
  // Track brush state to suppress tooltip during drag
  const isBrushingRef = useRef(false);

  // Available probe names for coherence mode
  const probeNames = useMemo(() => probes ? Object.keys(probes) : [], [probes]);

  // Compute spectrum - use async for large datasets (spectrum mode only)
  useEffect(() => {
    if (analysisMode !== "spectrum") {
      return;
    }

    if (data.length === 0) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setSpectrum(null);
      setComplexSpectrum(null);
      return;
    }

    let cancelled = false;

    // Use async for large datasets to avoid blocking main thread
    if (data.length >= WORKER_THRESHOLD) {
      setIsComputing(true);
      computeSpectrumAsync(data, sampleRate, nfft)
        .then((result) => {
          if (!cancelled) {
            setSpectrum(result);
            // Also compute complex spectrum for phase analysis
            setComplexSpectrum(computeComplexSpectrum(data, sampleRate, nfft));
            setIsComputing(false);
          }
        })
        .catch((error) => {
          console.error("FFT computation error:", error);
          if (!cancelled) {
            // Fallback to sync on error
            const complexResult = computeComplexSpectrum(data, sampleRate, nfft);
            setSpectrum({ frequencies: complexResult.frequencies, magnitude: complexResult.magnitude });
            setComplexSpectrum(complexResult);
            setIsComputing(false);
          }
        });
    } else {
      // Sync for small datasets
      const complexResult = computeComplexSpectrum(data, sampleRate, nfft);
      setSpectrum({ frequencies: complexResult.frequencies, magnitude: complexResult.magnitude });
      setComplexSpectrum(complexResult);
    }

    return () => {
      cancelled = true;
    };
  }, [data, sampleRate, nfft, analysisMode]);

  // Compute coherence (coherence mode only)
  useEffect(() => {
    if (analysisMode !== "coherence") {
      setCoherenceResult(null);
      return;
    }

    if (!probes || !referenceProbe || !measurementProbe) {
      setCoherenceResult(null);
      return;
    }

    const refProbe = probes[referenceProbe];
    const measProbe = probes[measurementProbe];

    if (!refProbe || !measProbe || refProbe.data.length === 0 || measProbe.data.length === 0) {
      setCoherenceResult(null);
      return;
    }

    setIsComputing(true);

    // Use requestAnimationFrame to avoid blocking main thread
    const frameId = requestAnimationFrame(() => {
      try {
        const result = computeCoherence(
          refProbe.data,
          measProbe.data,
          sampleRate,
          4096, // segment size
          0.5   // 50% overlap
        );
        setCoherenceResult(result);
      } catch (error) {
        console.error("Coherence computation error:", error);
        setCoherenceResult(null);
      } finally {
        setIsComputing(false);
      }
    });

    return () => {
      cancelAnimationFrame(frameId);
    };
  }, [analysisMode, probes, referenceProbe, measurementProbe, sampleRate]);

  // Compute reference spectrum for transfer function mode
  useEffect(() => {
    if (!referenceData || referenceData.length === 0 || mode !== "transfer") {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setReferenceSpectrum(null);
      setReferenceComplexSpectrum(null);
      return;
    }

    let cancelled = false;

    if (referenceData.length >= WORKER_THRESHOLD) {
      computeSpectrumAsync(referenceData, sampleRate, nfft)
        .then((result) => {
          if (!cancelled) {
            setReferenceSpectrum(result);
            setReferenceComplexSpectrum(computeComplexSpectrum(referenceData, sampleRate, nfft));
          }
        })
        .catch((error) => {
          console.error("Reference FFT computation error:", error);
          if (!cancelled) {
            const complexResult = computeComplexSpectrum(referenceData, sampleRate, nfft);
            setReferenceSpectrum({ frequencies: complexResult.frequencies, magnitude: complexResult.magnitude });
            setReferenceComplexSpectrum(complexResult);
          }
        });
    } else {
      const complexResult = computeComplexSpectrum(referenceData, sampleRate, nfft);
      setReferenceSpectrum({ frequencies: complexResult.frequencies, magnitude: complexResult.magnitude });
      setReferenceComplexSpectrum(complexResult);
    }

    return () => {
      cancelled = true;
    };
  }, [referenceData, sampleRate, nfft, mode]);

  // Find peaks (cached based on spectrum)
  const [peaks, setPeaks] = useState<PeakInfo[]>([]);
  useEffect(() => {
    if (!spectrum) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setPeaks([]);
      return;
    }
    setPeaks(findPeaks(spectrum.frequencies, spectrum.magnitude));
  }, [spectrum]);

  // Compute phase and group delay for transfer function mode
  const phaseData = useMemo(() => {
    if (mode !== "transfer" || !complexSpectrum || !referenceComplexSpectrum) {
      return null;
    }

    // Extract phase from complex transfer function
    const rawPhase = extractTransferPhase(
      referenceComplexSpectrum.real,
      referenceComplexSpectrum.imag,
      complexSpectrum.real,
      complexSpectrum.imag
    );

    // Optionally unwrap phase
    const phase = phaseUnwrap ? unwrapPhase(rawPhase) : rawPhase;

    // Compute group delay from unwrapped phase
    const groupDelay = computeGroupDelay(
      unwrapPhase(rawPhase), // Always use unwrapped for group delay
      complexSpectrum.frequencies
    );

    return {
      frequencies: complexSpectrum.frequencies,
      phase,
      groupDelay,
    };
  }, [mode, complexSpectrum, referenceComplexSpectrum, phaseUnwrap]);

  // Threshold for downsampling (number of frequency bins)
  const DOWNSAMPLE_THRESHOLD = 2000;

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

  // Render chart
  useEffect(() => {
    if (!svgRef.current || dimensions.width === 0 || !spectrum) return;

    // In transfer mode, need reference spectrum to display
    const isTransferMode = mode === "transfer" && referenceSpectrum !== null;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = dimensions.width - MARGIN.left - MARGIN.right;
    const height = dimensions.height - MARGIN.top - MARGIN.bottom;

    if (width <= 0 || height <= 0) return;

    const { frequencies, magnitude } = spectrum;

    // Default frequency range (audible or simulation range)
    const defaultMaxFreq = Math.min(sampleRate / 2, 20000); // Nyquist or 20kHz
    const defaultMinFreq = 20; // 20 Hz lower bound for audio

    // Use zoom domain if set
    const [minFreq, maxFreq] = zoomDomain ?? [defaultMinFreq, defaultMaxFreq];

    // Find indices for frequency range
    let startIdx = 0;
    let endIdx = frequencies.length - 1;
    for (let i = 0; i < frequencies.length; i++) {
      if (frequencies[i] >= minFreq) {
        startIdx = i;
        break;
      }
    }
    for (let i = frequencies.length - 1; i >= 0; i--) {
      if (frequencies[i] <= maxFreq) {
        endIdx = i;
        break;
      }
    }

    // Compute display magnitude based on mode
    let displayMag: Float32Array;
    let refMag: number;
    let yMin: number;
    let yMax: number;

    if (isTransferMode && referenceSpectrum) {
      // Transfer function mode: H(f) = Y(f) / X(f)
      // Y = probe output (magnitude), X = source input (referenceSpectrum.magnitude)
      const transferMag = new Float32Array(magnitude.length);
      const refSpecMag = referenceSpectrum.magnitude;

      // Compute transfer function magnitude (with small epsilon to avoid division by zero)
      const epsilon = 1e-10;
      for (let i = 0; i < magnitude.length; i++) {
        const refVal = i < refSpecMag.length ? refSpecMag[i] : epsilon;
        transferMag[i] = magnitude[i] / Math.max(refVal, epsilon);
      }

      // In transfer function mode, reference is 1 (0 dB = unity gain)
      refMag = 1;
      displayMag = logScale
        ? Float32Array.from(transferMag, (v) => toDecibels(v, 1))
        : transferMag;

      // Transfer function typically shows range around 0 dB
      if (logScale) {
        yMin = -40; // -40 dB
        yMax = 20;  // +20 dB
      } else {
        const maxTransfer = Math.max(...transferMag);
        yMin = 0;
        yMax = Math.max(maxTransfer, 2); // At least show up to 2x gain
      }
    } else {
      // Standard spectrum mode
      refMag = Math.max(...magnitude);
      displayMag = logScale
        ? Float32Array.from(magnitude, (v) => toDecibels(v, refMag))
        : magnitude;
      yMin = logScale ? -80 : 0;
      yMax = logScale ? 0 : refMag;
    }

    // Create scales
    const xScale = d3
      .scaleLog()
      .domain([Math.max(minFreq, frequencies[startIdx]), maxFreq])
      .range([0, width])
      .clamp(true);

    const yScale = d3.scaleLinear().domain([yMin, yMax]).range([height, 0]);

    // Create main group
    const g = svg
      .append("g")
      .attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);

    // Create axes
    const xAxis = d3
      .axisBottom(xScale)
      .ticks(5, ",.0f")
      .tickFormat((d) => {
        const val = +d;
        if (val >= 1000) return `${val / 1000}k`;
        return `${val}`;
      });

    const yAxis = d3
      .axisLeft(yScale)
      .ticks(5)
      .tickFormat((d) => {
        if (logScale) {
          // Show +/- sign for transfer function mode
          const val = +d;
          if (isTransferMode) {
            return val > 0 ? `+${val}dB` : `${val}dB`;
          }
          return `${val}dB`;
        }
        return d3.format(".1e")(+d);
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
      .text("Frequency (Hz)");

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

    // Determine if downsampling is needed
    const visibleBins = endIdx - startIdx + 1;
    const targetBins = Math.min(width * 2, visibleBins); // Max 2 points per pixel

    // Get display data (possibly downsampled)
    let displayFreqs: Float32Array;
    let displayMagValues: Float32Array;

    if (visibleBins > DOWNSAMPLE_THRESHOLD && targetBins < visibleBins) {
      // Downsample using log-binning for logarithmic frequency display
      const result = logBinDownsample(
        frequencies.subarray(startIdx, endIdx + 1),
        magnitude.subarray(startIdx, endIdx + 1),
        targetBins,
        minFreq,
        maxFreq
      );
      displayFreqs = result.frequencies;
      // Convert downsampled magnitude to dB if needed
      displayMagValues = logScale
        ? Float32Array.from(result.magnitude, (v) => toDecibels(v, refMag))
        : result.magnitude;
    } else {
      // Use original data for visible range
      displayFreqs = frequencies.subarray(startIdx, endIdx + 1);
      displayMagValues = displayMag.subarray(startIdx, endIdx + 1);
    }

    // Build path data from (possibly downsampled) display data
    const pathData: [number, number][] = [];
    for (let i = 0; i < displayFreqs.length; i++) {
      const x = xScale(displayFreqs[i]);
      const y = yScale(displayMagValues[i]);
      if (isFinite(x) && isFinite(y)) {
        pathData.push([x, y]);
      }
    }

    // Create line generator
    const line = d3.line().x((d) => d[0]).y((d) => d[1]);

    // Draw spectrum line
    g.append("path")
      .attr("fill", "none")
      .attr("stroke", "hsl(var(--primary))")
      .attr("stroke-width", 1.5)
      .attr("d", line(pathData));

    // Draw markers
    for (const marker of markers) {
      if (marker.frequency >= minFreq && marker.frequency <= maxFreq) {
        const x = xScale(marker.frequency);
        g.append("line")
          .attr("x1", x)
          .attr("x2", x)
          .attr("y1", 0)
          .attr("y2", height)
          .attr("stroke", "hsl(var(--destructive))")
          .attr("stroke-width", 1)
          .attr("stroke-dasharray", "4,4");

        g.append("text")
          .attr("x", x + 4)
          .attr("y", 12)
          .attr("fill", "hsl(var(--destructive))")
          .attr("font-size", "10px")
          .text(marker.label);
      }
    }

    // Draw peaks with Q-factor
    if (showPeaks) {
      for (const peak of peaks) {
        if (peak.frequency >= minFreq && peak.frequency <= maxFreq) {
          const x = xScale(peak.frequency);
          const y = yScale(logScale ? toDecibels(peak.magnitude, refMag) : peak.magnitude);

          g.append("circle")
            .attr("cx", x)
            .attr("cy", y)
            .attr("r", 4)
            .attr("fill", "hsl(var(--accent))")
            .attr("stroke", "white")
            .attr("stroke-width", 1);

          // Frequency label
          g.append("text")
            .attr("x", x)
            .attr("y", y - 16)
            .attr("text-anchor", "middle")
            .attr("fill", "hsl(var(--accent-foreground))")
            .attr("font-size", "9px")
            .text(`${Math.round(peak.frequency)} Hz`);

          // Q-factor label (if available)
          if (peak.qFactor !== null) {
            g.append("text")
              .attr("x", x)
              .attr("y", y - 6)
              .attr("text-anchor", "middle")
              .attr("fill", "hsl(var(--muted-foreground))")
              .attr("font-size", "8px")
              .text(`Q=${peak.qFactor.toFixed(1)}`);
          }
        }
      }
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

    // Helper to find magnitude at frequency
    const getMagnitudeAtFreq = (freq: number): { mag: number; displayMag: number } | null => {
      // Find closest frequency bin
      let closestIdx = startIdx;
      let minDiff = Math.abs(frequencies[startIdx] - freq);
      for (let i = startIdx; i <= endIdx; i++) {
        const diff = Math.abs(frequencies[i] - freq);
        if (diff < minDiff) {
          minDiff = diff;
          closestIdx = i;
        }
      }
      return {
        mag: magnitude[closestIdx],
        displayMag: displayMag[closestIdx],
      };
    };

    // Add brush for frequency zoom
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

        // Only zoom if selection is significant
        if (Math.abs((event.selection as [number, number])[1] - (event.selection as [number, number])[0]) > 5) {
          setZoomDomain([x0, x1]);
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

    // Interaction overlay for tooltip
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
        const freq = xScale.invert(x);

        // Clamp to valid range
        if (freq < minFreq || freq > maxFreq) {
          tooltip.style("display", "none");
          hoverLine.style("display", "none");
          return;
        }

        // Update hover line
        hoverLine
          .attr("x1", x)
          .attr("x2", x)
          .attr("y1", 0)
          .attr("y2", height)
          .style("display", null);

        // Get magnitude at frequency
        const result = getMagnitudeAtFreq(freq);
        if (!result) return;

        // Build tooltip content
        const freqStr = freq >= 1000 ? `${(freq / 1000).toFixed(2)} kHz` : `${freq.toFixed(1)} Hz`;
        const magStr = logScale
          ? `${result.displayMag.toFixed(1)} dB`
          : result.mag.toExponential(2);

        const lines = [freqStr, magStr];

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
      .on("dblclick", () => {
        setZoomDomain(null);
      });
  }, [dimensions, spectrum, logScale, markers, peaks, showPeaks, sampleRate, zoomDomain, mode, referenceSpectrum, analysisMode]);

  // Render coherence chart (coherence mode)
  useEffect(() => {
    if (analysisMode !== "coherence") return;
    if (!svgRef.current || dimensions.width === 0 || !coherenceResult) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = dimensions.width - MARGIN.left - MARGIN.right;
    const height = dimensions.height - MARGIN.top - MARGIN.bottom;

    if (width <= 0 || height <= 0) return;

    const { frequencies, coherence, transferMagnitude } = coherenceResult;

    // Default frequency range (audible or simulation range)
    const defaultMaxFreq = Math.min(sampleRate / 2, 20000);
    const defaultMinFreq = 20;
    const [minFreq, maxFreq] = zoomDomain ?? [defaultMinFreq, defaultMaxFreq];

    // Find indices for frequency range
    let startIdx = 0;
    let endIdx = frequencies.length - 1;
    for (let i = 0; i < frequencies.length; i++) {
      if (frequencies[i] >= minFreq) {
        startIdx = i;
        break;
      }
    }
    for (let i = frequencies.length - 1; i >= 0; i--) {
      if (frequencies[i] <= maxFreq) {
        endIdx = i;
        break;
      }
    }

    // Create scales
    const xScale = d3
      .scaleLog()
      .domain([Math.max(minFreq, frequencies[startIdx] || 1), maxFreq])
      .range([0, width])
      .clamp(true);

    // Coherence scale: 0-1 (left axis)
    const yScaleCoherence = d3.scaleLinear().domain([0, 1]).range([height, 0]);

    // Transfer function scale: dB (right axis)
    const refMag = Math.max(...transferMagnitude.subarray(startIdx, endIdx + 1));
    const transferDb = Float32Array.from(transferMagnitude, (v) =>
      20 * Math.log10(Math.max(v / refMag, 1e-10))
    );
    const yScaleTransfer = d3.scaleLinear().domain([-40, 10]).range([height, 0]);

    // Create main group
    const g = svg
      .append("g")
      .attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);

    // X-axis
    const xAxis = d3
      .axisBottom(xScale)
      .ticks(5, ",.0f")
      .tickFormat((d) => {
        const val = +d;
        if (val >= 1000) return `${val / 1000}k`;
        return `${val}`;
      });

    g.append("g")
      .attr("transform", `translate(0,${height})`)
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
      .text("Frequency (Hz)");

    // Left Y-axis (Coherence)
    const yAxisCoherence = d3
      .axisLeft(yScaleCoherence)
      .ticks(5)
      .tickFormat((d) => `${(+d * 100).toFixed(0)}%`);

    g.append("g")
      .call(yAxisCoherence)
      .selectAll("text")
      .style("fill", "hsl(var(--primary))");

    // Right Y-axis (Transfer function)
    const yAxisTransfer = d3
      .axisRight(yScaleTransfer)
      .ticks(5)
      .tickFormat((d) => `${d}dB`);

    g.append("g")
      .attr("transform", `translate(${width},0)`)
      .call(yAxisTransfer)
      .selectAll("text")
      .style("fill", "hsl(var(--accent-foreground))");

    // Style axis lines
    g.selectAll(".domain, .tick line").style("stroke", "hsl(var(--border))");

    // Add grid lines (based on coherence scale)
    g.append("g")
      .attr("class", "grid")
      .selectAll("line")
      .data(yScaleCoherence.ticks(5))
      .join("line")
      .attr("x1", 0)
      .attr("x2", width)
      .attr("y1", (d) => yScaleCoherence(d))
      .attr("y2", (d) => yScaleCoherence(d))
      .attr("stroke", "hsl(var(--border))")
      .attr("stroke-opacity", 0.3);

    // Build coherence path data
    const coherencePathData: [number, number][] = [];
    for (let i = startIdx; i <= endIdx; i++) {
      const x = xScale(frequencies[i]);
      const y = yScaleCoherence(coherence[i]);
      if (isFinite(x) && isFinite(y)) {
        coherencePathData.push([x, y]);
      }
    }

    // Build transfer function path data
    const transferPathData: [number, number][] = [];
    for (let i = startIdx; i <= endIdx; i++) {
      const x = xScale(frequencies[i]);
      const y = yScaleTransfer(transferDb[i]);
      if (isFinite(x) && isFinite(y)) {
        transferPathData.push([x, y]);
      }
    }

    const line = d3.line().x((d) => d[0]).y((d) => d[1]);

    // Draw transfer function line (secondary, behind coherence)
    g.append("path")
      .attr("fill", "none")
      .attr("stroke", "hsl(var(--accent))")
      .attr("stroke-width", 1)
      .attr("stroke-opacity", 0.6)
      .attr("d", line(transferPathData));

    // Draw coherence line (primary)
    g.append("path")
      .attr("fill", "none")
      .attr("stroke", "hsl(var(--primary))")
      .attr("stroke-width", 2)
      .attr("d", line(coherencePathData));

    // Add threshold line at 0.5 coherence
    g.append("line")
      .attr("x1", 0)
      .attr("x2", width)
      .attr("y1", yScaleCoherence(0.5))
      .attr("y2", yScaleCoherence(0.5))
      .attr("stroke", "hsl(var(--warning, var(--destructive)))")
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "4,4")
      .attr("stroke-opacity", 0.5);

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

    // Helper to find values at frequency
    const getValuesAtFreq = (freq: number) => {
      let closestIdx = startIdx;
      let minDiff = Math.abs(frequencies[startIdx] - freq);
      for (let i = startIdx; i <= endIdx; i++) {
        const diff = Math.abs(frequencies[i] - freq);
        if (diff < minDiff) {
          minDiff = diff;
          closestIdx = i;
        }
      }
      return {
        coherence: coherence[closestIdx],
        transferDb: transferDb[closestIdx],
      };
    };

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
        const [x0, x1] = (event.selection as [number, number]).map(xScale.invert);
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
        const freq = xScale.invert(x);

        if (freq < minFreq || freq > maxFreq) {
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

        const values = getValuesAtFreq(freq);
        const freqStr = freq >= 1000 ? `${(freq / 1000).toFixed(2)} kHz` : `${freq.toFixed(1)} Hz`;
        const coherenceStr = `γ²: ${(values.coherence * 100).toFixed(1)}%`;
        const transferStr = `H: ${values.transferDb.toFixed(1)} dB`;

        tooltipText.selectAll("tspan").remove();
        [freqStr, coherenceStr, transferStr].forEach((line, i) => {
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

    // Legend
    const legend = g
      .append("g")
      .attr("class", "legend")
      .attr("transform", `translate(${width - 100}, 10)`);

    legend
      .append("line")
      .attr("x1", 0)
      .attr("x2", 20)
      .attr("y1", 0)
      .attr("y2", 0)
      .attr("stroke", "hsl(var(--primary))")
      .attr("stroke-width", 2);

    legend
      .append("text")
      .attr("x", 25)
      .attr("y", 4)
      .attr("fill", "hsl(var(--foreground))")
      .attr("font-size", "10px")
      .text("Coherence");

    legend
      .append("line")
      .attr("x1", 0)
      .attr("x2", 20)
      .attr("y1", 15)
      .attr("y2", 15)
      .attr("stroke", "hsl(var(--accent))")
      .attr("stroke-width", 1)
      .attr("stroke-opacity", 0.6);

    legend
      .append("text")
      .attr("x", 25)
      .attr("y", 19)
      .attr("fill", "hsl(var(--foreground))")
      .attr("font-size", "10px")
      .text("Transfer Fn");
  }, [dimensions, coherenceResult, sampleRate, zoomDomain, analysisMode]);

  // Render phase plot
  useEffect(() => {
    if (!phaseSvgRef.current || !showPhase || !phaseData || dimensions.width === 0) return;

    const PHASE_MARGIN = { top: 5, right: 20, bottom: 25, left: 50 };
    const phaseHeight = 120; // Fixed height for phase plot
    const svg = d3.select(phaseSvgRef.current);
    svg.selectAll("*").remove();

    const width = dimensions.width - PHASE_MARGIN.left - PHASE_MARGIN.right;
    const height = phaseHeight - PHASE_MARGIN.top - PHASE_MARGIN.bottom;

    if (width <= 0 || height <= 0) return;

    const { frequencies, phase, groupDelay } = phaseData;

    // Default frequency range
    const defaultMaxFreq = Math.min(sampleRate / 2, 20000);
    const defaultMinFreq = 20;
    const [minFreq, maxFreq] = zoomDomain ?? [defaultMinFreq, defaultMaxFreq];

    // Find indices for frequency range
    let startIdx = 0;
    let endIdx = frequencies.length - 1;
    for (let i = 0; i < frequencies.length; i++) {
      if (frequencies[i] >= minFreq) {
        startIdx = i;
        break;
      }
    }
    for (let i = frequencies.length - 1; i >= 0; i--) {
      if (frequencies[i] <= maxFreq) {
        endIdx = i;
        break;
      }
    }

    // Select data based on view mode
    const displayData = phaseViewMode === "phase" ? phase : groupDelay;
    const yLabel = phaseViewMode === "phase"
      ? (phaseUnwrap ? "Phase (rad)" : "Phase (°)")
      : "Group Delay (ms)";

    // Convert phase to degrees if not unwrapped
    const convertedData = phaseViewMode === "phase" && !phaseUnwrap
      ? Float32Array.from(displayData, v => v * (180 / Math.PI))
      : phaseViewMode === "groupDelay"
        ? Float32Array.from(displayData, v => v * 1000) // Convert to milliseconds
        : displayData;

    // Calculate y-axis range
    let yMin: number, yMax: number;
    const visibleData = convertedData.subarray(startIdx, endIdx + 1);
    const dataMin = Math.min(...visibleData);
    const dataMax = Math.max(...visibleData);
    const padding = (dataMax - dataMin) * 0.1 || 1;
    yMin = dataMin - padding;
    yMax = dataMax + padding;

    // Create scales
    const xScale = d3
      .scaleLog()
      .domain([Math.max(minFreq, frequencies[startIdx]), maxFreq])
      .range([0, width])
      .clamp(true);

    const yScale = d3.scaleLinear().domain([yMin, yMax]).range([height, 0]);

    // Create main group
    const g = svg
      .append("g")
      .attr("transform", `translate(${PHASE_MARGIN.left},${PHASE_MARGIN.top})`);

    // Create axes
    const xAxis = d3
      .axisBottom(xScale)
      .ticks(5, ",.0f")
      .tickFormat((d) => {
        const val = +d;
        if (val >= 1000) return `${val / 1000}k`;
        return `${val}`;
      });

    const yAxis = d3
      .axisLeft(yScale)
      .ticks(4)
      .tickFormat((d) => {
        const val = +d;
        if (Math.abs(val) >= 100) return val.toFixed(0);
        if (Math.abs(val) >= 10) return val.toFixed(1);
        return val.toFixed(2);
      });

    g.append("g")
      .attr("transform", `translate(0,${height})`)
      .attr("class", "x-axis")
      .call(xAxis)
      .selectAll("text")
      .style("fill", "hsl(var(--muted-foreground))")
      .style("font-size", "9px");

    g.append("g")
      .attr("class", "y-axis")
      .call(yAxis)
      .selectAll("text")
      .style("fill", "hsl(var(--muted-foreground))")
      .style("font-size", "9px");

    // Style axis lines
    g.selectAll(".domain, .tick line").style("stroke", "hsl(var(--border))");

    // Add grid lines
    g.append("g")
      .attr("class", "grid")
      .selectAll("line")
      .data(yScale.ticks(4))
      .join("line")
      .attr("x1", 0)
      .attr("x2", width)
      .attr("y1", (d) => yScale(d))
      .attr("y2", (d) => yScale(d))
      .attr("stroke", "hsl(var(--border))")
      .attr("stroke-opacity", 0.3);

    // Y-axis label
    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -height / 2)
      .attr("y", -38)
      .attr("text-anchor", "middle")
      .attr("fill", "hsl(var(--muted-foreground))")
      .attr("font-size", "9px")
      .text(yLabel);

    // Build path data
    const pathData: [number, number][] = [];
    for (let i = startIdx; i <= endIdx; i++) {
      const x = xScale(frequencies[i]);
      const y = yScale(convertedData[i]);
      if (isFinite(x) && isFinite(y)) {
        pathData.push([x, y]);
      }
    }

    // Create line generator
    const line = d3.line().x((d) => d[0]).y((d) => d[1]);

    // Draw phase/group delay line
    g.append("path")
      .attr("fill", "none")
      .attr("stroke", phaseViewMode === "phase" ? "hsl(var(--accent))" : "hsl(var(--chart-2))")
      .attr("stroke-width", 1.5)
      .attr("d", line(pathData));

    // Add zero line for reference
    if (yMin < 0 && yMax > 0) {
      g.append("line")
        .attr("x1", 0)
        .attr("x2", width)
        .attr("y1", yScale(0))
        .attr("y2", yScale(0))
        .attr("stroke", "hsl(var(--muted-foreground))")
        .attr("stroke-width", 1)
        .attr("stroke-dasharray", "4,4")
        .attr("stroke-opacity", 0.5);
    }
  }, [dimensions, phaseData, showPhase, phaseViewMode, phaseUnwrap, sampleRate, zoomDomain]);

  // Format frequency for display
  const formatFreq = useCallback((freq: number) => {
    if (freq >= 1000) return `${(freq / 1000).toFixed(1)}k`;
    return `${Math.round(freq)}`;
  }, []);

  // Reset zoom callback
  const resetZoom = useCallback(() => {
    setZoomDomain(null);
  }, []);

  // Title based on mode
  const title = analysisMode === "coherence" ? "Coherence Analysis" : "Frequency Spectrum";

  // Whether coherence mode is ready (has required probes selected)
  const coherenceReady = analysisMode === "coherence" &&
    referenceProbe && measurementProbe &&
    referenceProbe !== measurementProbe;

  // Determine if we're in transfer mode
  const isTransferMode = mode === "transfer" && referenceSpectrum !== null;

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold text-muted-foreground">
            {analysisMode === "coherence" ? title : (isTransferMode ? "Transfer Function" : "Frequency Spectrum")}
          </h3>
          {isTransferMode && referenceName && (
            <Badge variant="secondary" className="text-xs">
              ref: {referenceName}
            </Badge>
          )}
          {zoomDomain && (
            <Badge
              variant="outline"
              className="cursor-pointer text-xs"
              onClick={resetZoom}
            >
              {formatFreq(zoomDomain[0])}–{formatFreq(zoomDomain[1])} Hz ✕
            </Badge>
          )}
        </div>
        {analysisMode === "spectrum" && (
          <div className="flex gap-1">
            <Button
              variant={logScale ? "default" : "outline"}
              size="sm"
              className="text-xs h-6 px-2"
              onClick={() => setLogScale(true)}
            >
              dB
            </Button>
            <Button
              variant={!logScale ? "default" : "outline"}
              size="sm"
              className="text-xs h-6 px-2"
              onClick={() => setLogScale(false)}
            >
              Linear
            </Button>
            <Button
              variant={showPeaks ? "default" : "outline"}
              size="sm"
              className="text-xs h-6 px-2"
              onClick={() => setShowPeaks(!showPeaks)}
            >
              Peaks
            </Button>
            {isTransferMode && (
              <Button
                variant={showPhase ? "default" : "outline"}
                size="sm"
                className="text-xs h-6 px-2"
                onClick={() => setShowPhase(!showPhase)}
              >
                Phase
              </Button>
            )}
          </div>
        )}
      </div>

      {/* Coherence mode: probe selection */}
      {analysisMode === "coherence" && probeNames.length >= 2 && (
        <div className="flex items-center gap-4 mb-2 text-xs">
          <div className="flex items-center gap-1">
            <span className="text-muted-foreground">Reference:</span>
            <div className="flex gap-1">
              {probeNames.map((name) => (
                <Badge
                  key={`ref-${name}`}
                  variant={referenceProbe === name ? "default" : "outline"}
                  className="cursor-pointer"
                  onClick={() => onReferenceProbeChange?.(name)}
                >
                  {name}
                </Badge>
              ))}
            </div>
          </div>
          <div className="flex items-center gap-1">
            <span className="text-muted-foreground">Measurement:</span>
            <div className="flex gap-1">
              {probeNames.map((name) => (
                <Badge
                  key={`meas-${name}`}
                  variant={measurementProbe === name ? "default" : "outline"}
                  className={`cursor-pointer ${name === referenceProbe ? "opacity-50" : ""}`}
                  onClick={() => name !== referenceProbe && onMeasurementProbeChange?.(name)}
                >
                  {name}
                </Badge>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Coherence mode: not enough probes warning */}
      {analysisMode === "coherence" && probeNames.length < 2 && (
        <div className="flex items-center justify-center py-4 text-sm text-muted-foreground">
          Coherence analysis requires at least 2 probes
        </div>
      )}

      {/* Coherence mode: same probe selected warning */}
      {analysisMode === "coherence" && referenceProbe && measurementProbe && referenceProbe === measurementProbe && (
        <div className="flex items-center justify-center py-2 text-xs text-warning">
          Select different probes for reference and measurement
        </div>
      )}

      <div ref={containerRef} className={`min-h-0 relative ${showPhase && isTransferMode ? "flex-[2]" : "flex-1"}`}>
        <svg ref={svgRef} width={dimensions.width} height={dimensions.height} />
        {isComputing && (
          <div className="absolute inset-0 flex items-center justify-center bg-background/50">
            <div className="flex items-center gap-2 text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span className="text-sm">
                {analysisMode === "coherence" ? "Computing coherence..." : "Computing spectrum..."}
              </span>
            </div>
          </div>
        )}
        {analysisMode === "coherence" && !coherenceReady && !isComputing && probeNames.length >= 2 && (
          <div className="absolute inset-0 flex items-center justify-center text-muted-foreground text-sm">
            Select reference and measurement probes above
          </div>
        )}
        {analysisMode !== "coherence" && !isComputing && !spectrum && data.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-sm text-muted-foreground">No probe data available</span>
          </div>
        )}
      </div>
      {/* Phase/Group Delay Plot */}
      {showPhase && isTransferMode && phaseData && (
        <div className="flex-1 min-h-[120px] max-h-[150px] mt-1 border-t border-border pt-1">
          <div className="flex items-center justify-between mb-1">
            <div className="flex gap-1">
              <Button
                variant={phaseViewMode === "phase" ? "default" : "outline"}
                size="sm"
                className="text-xs h-5 px-2"
                onClick={() => setPhaseViewMode("phase")}
              >
                Phase
              </Button>
              <Button
                variant={phaseViewMode === "groupDelay" ? "default" : "outline"}
                size="sm"
                className="text-xs h-5 px-2"
                onClick={() => setPhaseViewMode("groupDelay")}
              >
                Group Delay
              </Button>
            </div>
            {phaseViewMode === "phase" && (
              <Button
                variant={phaseUnwrap ? "default" : "outline"}
                size="sm"
                className="text-xs h-5 px-2"
                onClick={() => setPhaseUnwrap(!phaseUnwrap)}
              >
                Unwrap
              </Button>
            )}
          </div>
          <svg ref={phaseSvgRef} width={dimensions.width} height={120} />
        </div>
      )}
      {zoomDomain && (
        <div className="text-xs text-muted-foreground mt-1 text-center">
          Double-click to reset zoom
        </div>
      )}
    </div>
  );
}
