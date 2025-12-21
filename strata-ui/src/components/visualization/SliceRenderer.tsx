/**
 * SliceRenderer - 2D heatmap visualization of pressure field slices.
 *
 * Renders a 2D cross-section of the 3D pressure field using Canvas 2D API.
 * Supports real-time updates during playback with efficient pixel manipulation.
 */

import { useRef, useEffect, useCallback, forwardRef, useImperativeHandle } from "react";
import * as THREE from "three";
import { extractSlice, extractGeometrySlice, getSliceIndex, getAxisSize, getSlicePlaneLabel } from "../../lib/sliceUtils";
import { applyPressureColormap, getSymmetricRange } from "../../lib/colormap";
import type { SliceAxis } from "../../stores/simulationStore";

export interface SliceRendererProps {
  /** 3D pressure field data */
  pressure: Float32Array | null;
  /** Grid dimensions [nx, ny, nz] */
  shape: [number, number, number];
  /** Grid resolution in meters */
  resolution: number;
  /** Axis perpendicular to the slice plane */
  axis: SliceAxis;
  /** Position along the axis (0-1 normalized) */
  position: number;
  /** Whether to show axis labels */
  showLabels?: boolean;
  /** Whether to show the colorbar */
  showColorbar?: boolean;
  /** 3D geometry mask (1=air, 0=solid) */
  geometry?: Uint8Array | null;
  /** Whether to show geometry overlay */
  showGeometry?: boolean;
}

export interface SliceRendererHandle {
  /** Get the canvas element for export */
  getCanvas: () => HTMLCanvasElement | null;
  /** Force a render update */
  render: () => void;
}

// Pre-allocated color for performance
const tempColor = new THREE.Color();

// Geometry overlay color (semi-transparent dark gray)
const GEOMETRY_COLOR = { r: 74, g: 74, b: 74, a: 180 }; // #4a4a4a with alpha

export const SliceRenderer = forwardRef<SliceRendererHandle, SliceRendererProps>(
  function SliceRenderer(
    {
      pressure,
      shape,
      resolution,
      axis,
      position,
      showLabels = true,
      showColorbar = true,
      geometry = null,
      showGeometry = false,
    },
    ref
  ) {
    const containerRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const colorbarCanvasRef = useRef<HTMLCanvasElement>(null);

    // Expose methods via ref
    useImperativeHandle(ref, () => ({
      getCanvas: () => canvasRef.current,
      render: () => renderSlice(),
    }));

    // Calculate slice info
    const axisSize = getAxisSize(shape, axis);
    const sliceIndex = getSliceIndex(position, axisSize);
    const physicalPosition = sliceIndex * resolution;

    // Render the slice to canvas
    const renderSlice = useCallback(() => {
      if (!canvasRef.current || !pressure || pressure.length === 0) {
        return;
      }

      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      // Extract 2D slice from 3D data
      const slice = extractSlice(pressure, shape, axis, position, resolution);

      // Resize canvas to match slice dimensions
      // Use a reasonable display size while maintaining aspect ratio
      const maxDisplaySize = 800;
      const aspectRatio = slice.width / slice.height;
      let displayWidth: number;
      let displayHeight: number;

      if (aspectRatio >= 1) {
        displayWidth = Math.min(maxDisplaySize, slice.width * 4);
        displayHeight = displayWidth / aspectRatio;
      } else {
        displayHeight = Math.min(maxDisplaySize, slice.height * 4);
        displayWidth = displayHeight * aspectRatio;
      }

      canvas.width = slice.width;
      canvas.height = slice.height;
      canvas.style.width = `${displayWidth}px`;
      canvas.style.height = `${displayHeight}px`;

      // Get pressure range for color mapping (symmetric around zero)
      const [min, max] = getSymmetricRange(slice.data);

      // Create image data for pixel manipulation
      const imageData = ctx.createImageData(slice.width, slice.height);
      const pixels = imageData.data;

      // Apply colormap to each pixel
      for (let i = 0; i < slice.data.length; i++) {
        const value = slice.data[i];
        applyPressureColormap(value, min, max, tempColor);

        // Note: Canvas Y is inverted (0 at top), so we flip vertically
        const row = Math.floor(i / slice.width);
        const col = i % slice.width;
        const flippedRow = slice.height - 1 - row;
        const pixelIndex = (flippedRow * slice.width + col) * 4;

        pixels[pixelIndex] = Math.round(tempColor.r * 255);
        pixels[pixelIndex + 1] = Math.round(tempColor.g * 255);
        pixels[pixelIndex + 2] = Math.round(tempColor.b * 255);
        pixels[pixelIndex + 3] = 255;
      }

      // Draw to canvas
      ctx.putImageData(imageData, 0, 0);

      // Render geometry overlay if enabled
      if (showGeometry && geometry && geometry.length > 0) {
        renderGeometryOverlay(ctx, slice.width, slice.height);
      }

      // Render colorbar if enabled
      if (showColorbar && colorbarCanvasRef.current) {
        renderColorbar(min, max);
      }
    }, [pressure, shape, axis, position, resolution, showColorbar, geometry, showGeometry]);

    // Render the colorbar
    const renderColorbar = useCallback((min: number, max: number) => {
      const canvas = colorbarCanvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const width = 20;
      const height = 200;
      canvas.width = width;
      canvas.height = height;

      // Create gradient from blue (bottom) to white (middle) to red (top)
      const imageData = ctx.createImageData(width, height);
      const pixels = imageData.data;

      for (let y = 0; y < height; y++) {
        // Map y to value: bottom = min, top = max
        const t = 1 - y / (height - 1);
        const value = min + t * (max - min);
        applyPressureColormap(value, min, max, tempColor);

        for (let x = 0; x < width; x++) {
          const idx = (y * width + x) * 4;
          pixels[idx] = Math.round(tempColor.r * 255);
          pixels[idx + 1] = Math.round(tempColor.g * 255);
          pixels[idx + 2] = Math.round(tempColor.b * 255);
          pixels[idx + 3] = 255;
        }
      }

      ctx.putImageData(imageData, 0, 0);
    }, []);

    // Render geometry overlay on top of pressure slice
    const renderGeometryOverlay = useCallback(
      (ctx: CanvasRenderingContext2D, width: number, height: number) => {
        if (!geometry) return;

        // Extract geometry slice
        const geoSlice = extractGeometrySlice(geometry, shape, axis, position);

        // Create overlay image data
        const overlayData = ctx.createImageData(width, height);
        const pixels = overlayData.data;

        // Apply geometry overlay (solid regions = 0 in mask)
        for (let i = 0; i < geoSlice.data.length; i++) {
          const isSolid = geoSlice.data[i] === 0;

          // Note: Canvas Y is inverted (0 at top), so we flip vertically
          const row = Math.floor(i / width);
          const col = i % width;
          const flippedRow = height - 1 - row;
          const pixelIndex = (flippedRow * width + col) * 4;

          if (isSolid) {
            pixels[pixelIndex] = GEOMETRY_COLOR.r;
            pixels[pixelIndex + 1] = GEOMETRY_COLOR.g;
            pixels[pixelIndex + 2] = GEOMETRY_COLOR.b;
            pixels[pixelIndex + 3] = GEOMETRY_COLOR.a;
          } else {
            // Transparent for air
            pixels[pixelIndex] = 0;
            pixels[pixelIndex + 1] = 0;
            pixels[pixelIndex + 2] = 0;
            pixels[pixelIndex + 3] = 0;
          }
        }

        // Create temporary canvas for compositing
        const tempCanvas = document.createElement("canvas");
        tempCanvas.width = width;
        tempCanvas.height = height;
        const tempCtx = tempCanvas.getContext("2d");
        if (!tempCtx) return;

        tempCtx.putImageData(overlayData, 0, 0);

        // Draw overlay on top with alpha compositing
        ctx.drawImage(tempCanvas, 0, 0);
      },
      [geometry, shape, axis, position]
    );

    // Re-render when dependencies change
    useEffect(() => {
      renderSlice();
    }, [renderSlice]);

    // Format pressure value for display
    const formatPressure = (value: number): string => {
      if (Math.abs(value) < 0.001) {
        return value.toExponential(1);
      }
      return value.toFixed(3);
    };

    // Get pressure range for colorbar labels
    const pressureRange = pressure ? getSymmetricRange(pressure) : [0, 0];

    return (
      <div
        ref={containerRef}
        className="w-full h-full flex items-center justify-center bg-background p-4"
      >
        <div className="flex items-center gap-4">
          {/* Main slice visualization */}
          <div className="relative">
            <canvas
              ref={canvasRef}
              className="border border-border rounded shadow-md"
              style={{ imageRendering: "pixelated" }}
            />

            {/* Axis labels */}
            {showLabels && pressure && (
              <>
                {/* Slice info overlay */}
                <div className="absolute top-2 left-2 bg-background/80 px-2 py-1 rounded text-xs">
                  <div className="font-medium">{getSlicePlaneLabel(axis)}</div>
                  <div className="text-muted-foreground">
                    {axis.toUpperCase()} = {(physicalPosition * 100).toFixed(1)} cm
                    <span className="text-muted-foreground/60"> (index {sliceIndex}/{axisSize - 1})</span>
                  </div>
                </div>
              </>
            )}
          </div>

          {/* Colorbar */}
          {showColorbar && pressure && (
            <div className="flex flex-col items-center">
              <div className="text-xs text-muted-foreground mb-1">
                {formatPressure(pressureRange[1])} Pa
              </div>
              <canvas
                ref={colorbarCanvasRef}
                className="border border-border rounded"
                style={{ width: "20px", height: "200px" }}
              />
              <div className="text-xs text-muted-foreground mt-1">
                {formatPressure(pressureRange[0])} Pa
              </div>
            </div>
          )}
        </div>

        {/* Empty state */}
        {!pressure && (
          <div className="text-muted-foreground text-sm">
            No pressure data available
          </div>
        )}
      </div>
    );
  }
);
