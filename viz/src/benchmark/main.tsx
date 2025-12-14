/**
 * Entry point for the benchmark page.
 */

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BenchmarkRunner, type BenchmarkConfig } from "./BenchmarkRunner";
import "../index.css";

// Parse config from URL params
function getConfigFromUrl(): Partial<BenchmarkConfig> {
  const params = new URLSearchParams(window.location.search);
  const config: Partial<BenchmarkConfig> = {};

  const gridSizes = params.get("gridSizes");
  if (gridSizes) {
    config.gridSizes = gridSizes.split(",").map(Number);
  }

  const frames = params.get("frames");
  if (frames) {
    config.framesPerBenchmark = Number(frames);
  }

  const warmup = params.get("warmup");
  if (warmup) {
    config.warmupFrames = Number(warmup);
  }

  const downsample = params.get("downsample");
  if (downsample !== null) {
    config.testDownsampling = downsample === "true";
  }

  const targetVoxels = params.get("targetVoxels");
  if (targetVoxels) {
    config.targetVoxels = Number(targetVoxels);
  }

  return config;
}

const config = getConfigFromUrl();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BenchmarkRunner
      config={config}
      onComplete={(results) => {
        console.log("Benchmark complete:", results);
        // Signal to Puppeteer that benchmark is complete
        (window as unknown as { __benchmarkComplete: boolean }).__benchmarkComplete = true;
      }}
      onProgress={(current, total, message) => {
        console.log(`[${current}/${total}] ${message}`);
      }}
    />
  </StrictMode>
);
