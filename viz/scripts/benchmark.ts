#!/usr/bin/env npx tsx

/**
 * Performance benchmark runner script.
 *
 * Launches a headless browser to run the benchmark page and collect results.
 *
 * Usage:
 *   pnpm benchmark                    # Run with default settings
 *   pnpm benchmark --output results.json
 *   pnpm benchmark --grid-sizes 50,100,150
 *   pnpm benchmark --frames 60
 *   pnpm benchmark --no-downsample
 *   pnpm benchmark --headless false   # Show browser window
 */

import { spawn, type ChildProcess } from "child_process";
import puppeteer from "puppeteer";
import * as fs from "fs";
import * as path from "path";

interface BenchmarkOptions {
  output: string;
  gridSizes: number[];
  frames: number;
  warmup: number;
  downsample: boolean;
  targetVoxels: number;
  headless: boolean;
  port: number;
}

function parseArgs(): BenchmarkOptions {
  const args = process.argv.slice(2);
  const options: BenchmarkOptions = {
    output: "benchmark-results.json",
    gridSizes: [50, 100, 150, 200],
    frames: 120,
    warmup: 30,
    downsample: true,
    targetVoxels: 262144,
    headless: true,
    port: 5174,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    const next = args[i + 1];

    switch (arg) {
      case "--output":
      case "-o":
        options.output = next;
        i++;
        break;
      case "--grid-sizes":
      case "-g":
        options.gridSizes = next.split(",").map(Number);
        i++;
        break;
      case "--frames":
      case "-f":
        options.frames = Number(next);
        i++;
        break;
      case "--warmup":
      case "-w":
        options.warmup = Number(next);
        i++;
        break;
      case "--no-downsample":
        options.downsample = false;
        break;
      case "--target-voxels":
        options.targetVoxels = Number(next);
        i++;
        break;
      case "--headless":
        options.headless = next !== "false";
        i++;
        break;
      case "--port":
      case "-p":
        options.port = Number(next);
        i++;
        break;
      case "--help":
      case "-h":
        printHelp();
        process.exit(0);
    }
  }

  return options;
}

function printHelp(): void {
  console.log(`
FDTD Visualizer Performance Benchmark

Usage:
  pnpm benchmark [options]

Options:
  -o, --output <file>       Output JSON file (default: benchmark-results.json)
  -g, --grid-sizes <sizes>  Comma-separated grid sizes (default: 50,100,150,200)
  -f, --frames <count>      Frames to collect per test (default: 120)
  -w, --warmup <count>      Warmup frames before collecting (default: 30)
  --no-downsample           Disable downsampling tests
  --target-voxels <count>   Target voxels for downsampling (default: 262144)
  --headless <bool>         Run headless browser (default: true)
  -p, --port <number>       Dev server port (default: 5174)
  -h, --help                Show this help message

Examples:
  pnpm benchmark
  pnpm benchmark -o results.json -g 50,100 -f 60
  pnpm benchmark --headless false  # Watch the benchmark run
`);
}

async function startDevServer(port: number): Promise<ChildProcess> {
  console.log(`Starting dev server on port ${port}...`);

  const server = spawn("npx", ["vite", "--port", String(port)], {
    cwd: path.dirname(path.dirname(new URL(import.meta.url).pathname)),
    stdio: ["ignore", "pipe", "pipe"],
  });

  // Wait for server to be ready
  await new Promise<void>((resolve, reject) => {
    const timeout = setTimeout(() => {
      reject(new Error("Dev server failed to start within 30 seconds"));
    }, 30000);

    server.stdout?.on("data", (data: Buffer) => {
      const output = data.toString();
      if (output.includes("Local:") || output.includes("ready in")) {
        clearTimeout(timeout);
        resolve();
      }
    });

    server.stderr?.on("data", (data: Buffer) => {
      const output = data.toString();
      // Vite sometimes outputs to stderr for non-errors
      if (output.includes("Local:") || output.includes("ready in")) {
        clearTimeout(timeout);
        resolve();
      }
    });

    server.on("error", (err) => {
      clearTimeout(timeout);
      reject(err);
    });

    server.on("exit", (code) => {
      if (code !== 0) {
        clearTimeout(timeout);
        reject(new Error(`Dev server exited with code ${code}`));
      }
    });
  });

  console.log("Dev server ready");
  return server;
}

async function runBenchmark(options: BenchmarkOptions): Promise<void> {
  let server: ChildProcess | null = null;

  try {
    // Start dev server
    server = await startDevServer(options.port);

    // Build URL with config params
    const url = new URL(`http://localhost:${options.port}/benchmark.html`);
    url.searchParams.set("gridSizes", options.gridSizes.join(","));
    url.searchParams.set("frames", String(options.frames));
    url.searchParams.set("warmup", String(options.warmup));
    url.searchParams.set("downsample", String(options.downsample));
    url.searchParams.set("targetVoxels", String(options.targetVoxels));

    console.log(`Opening benchmark at ${url.toString()}`);

    // Launch browser
    const browser = await puppeteer.launch({
      headless: options.headless,
      args: [
        "--enable-webgl",
        "--use-gl=swiftshader",
        "--no-sandbox",
        "--disable-setuid-sandbox",
      ],
    });

    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 720 });

    // Listen for console output
    page.on("console", (msg) => {
      const text = msg.text();
      if (text.startsWith("[")) {
        console.log(text);
      }
    });

    // Navigate to benchmark page
    await page.goto(url.toString(), { waitUntil: "networkidle0" });

    console.log("Waiting for benchmark to complete...");

    // Wait for benchmark to complete (with timeout)
    const maxWaitTime = 10 * 60 * 1000; // 10 minutes max
    const startTime = Date.now();

    while (Date.now() - startTime < maxWaitTime) {
      const isComplete = await page.evaluate(() => {
        return (window as unknown as { __benchmarkComplete?: boolean }).__benchmarkComplete === true;
      });

      if (isComplete) {
        break;
      }

      await new Promise((resolve) => setTimeout(resolve, 1000));
    }

    // Get results
    const results = await page.evaluate(() => {
      return (window as unknown as { __benchmarkResults?: unknown }).__benchmarkResults;
    });

    if (!results) {
      throw new Error("Failed to get benchmark results");
    }

    // Write results to file
    const outputPath = path.resolve(options.output);
    fs.writeFileSync(outputPath, JSON.stringify(results, null, 2));
    console.log(`\nResults written to ${outputPath}`);

    // Print summary
    console.log("\n=== Benchmark Summary ===\n");
    const resultData = results as {
      results: Array<{
        gridSize: number;
        downsampled: boolean;
        renderedVoxels: number;
        fps: { mean: number; p95: number };
        frameTime: { p95: number };
        memoryMB: number;
        timeToFirstFrame: number;
      }>;
    };

    console.log(
      "Grid Size".padEnd(12),
      "Voxels".padEnd(12),
      "FPS (mean)".padEnd(12),
      "FPS (p95)".padEnd(12),
      "Frame p95".padEnd(12),
      "Memory".padEnd(10),
      "TTFF"
    );
    console.log("-".repeat(80));

    for (const r of resultData.results) {
      const label = `${r.gridSize}³${r.downsampled ? " DS" : ""}`;
      const fpsColor =
        r.fps.mean >= 30 ? "\x1b[32m" : r.fps.mean >= 15 ? "\x1b[33m" : "\x1b[31m";
      const reset = "\x1b[0m";

      console.log(
        label.padEnd(12),
        String(r.renderedVoxels).padEnd(12),
        `${fpsColor}${r.fps.mean}${reset}`.padEnd(21), // Extra padding for color codes
        `${r.fps.p95}`.padEnd(12),
        `${r.frameTime.p95}ms`.padEnd(12),
        `${r.memoryMB}MB`.padEnd(10),
        `${r.timeToFirstFrame}ms`
      );
    }

    // Performance targets check
    console.log("\n=== Performance Targets ===\n");
    const targets = [
      { size: 100, minFps: 30 },
      { size: 150, minFps: 15 },
    ];

    for (const target of targets) {
      const result = resultData.results.find(
        (r) => r.gridSize === target.size && !r.downsampled
      );
      if (result) {
        const passed = result.fps.mean >= target.minFps;
        const icon = passed ? "\x1b[32m✓\x1b[0m" : "\x1b[31m✗\x1b[0m";
        console.log(
          `${icon} ${target.size}³ at ${target.minFps}+ FPS: ${result.fps.mean} FPS`
        );
      }
    }

    await browser.close();
  } finally {
    if (server) {
      server.kill();
    }
  }
}

// Main
const options = parseArgs();
runBenchmark(options).catch((err) => {
  console.error("Benchmark failed:", err);
  process.exit(1);
});
