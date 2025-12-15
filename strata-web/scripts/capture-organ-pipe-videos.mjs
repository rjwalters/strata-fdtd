#!/usr/bin/env node
/**
 * Capture videos of organ pipe demo simulations using Playwright
 *
 * This script captures frame-by-frame screenshots of the organ pipe demo
 * and optionally encodes them to MP4 using ffmpeg.
 *
 * Usage:
 *   node scripts/capture-organ-pipe-videos.mjs
 *
 * Environment:
 *   VIZ_URL - Base URL of the viz app (default: http://localhost:5173)
 *
 * Requires:
 *   - Dev server running (pnpm dev or pnpm preview)
 *   - Playwright installed (npx playwright install chromium)
 *   - ffmpeg (optional, for video encoding)
 */

import { chromium } from 'playwright'
import path from 'path'
import fs from 'fs'
import { execSync } from 'child_process'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const OUTPUT_DIR = path.join(__dirname, '..', '..', 'videos')
const BASE_URL = process.env.VIZ_URL || 'http://localhost:5173'
const FRAME_RATE = 15
const VIEWPORT = { width: 1280, height: 720 }

const PIPES = [
  { id: 'closed', name: 'Closed Pipe', buttonText: 'Closed Pipe' },
  { id: 'open', name: 'Open Pipe', buttonText: 'Open Pipe' },
  { id: 'half-open', name: 'Half-Open Pipe', buttonText: 'Half-Open Pipe' },
]

/**
 * Wait for the simulation store to be available on window
 */
async function waitForStore(page) {
  await page.waitForFunction(() => {
    return typeof window.__SIMULATION_STORE__ !== 'undefined'
  }, { timeout: 30000 })
}

/**
 * Wait for simulation to load (has frames and first snapshot loaded)
 */
async function waitForSimulationLoad(page) {
  // Wait for store to be available
  await waitForStore(page)

  // Wait for frames to be available and first snapshot loaded
  await page.waitForFunction(() => {
    const store = window.__SIMULATION_STORE__
    if (!store) return false
    const state = store.getState()
    return state.totalFrames > 0 && state.snapshots.size > 0
  }, { timeout: 30000 })

  // Additional settle time for WebGL
  await page.waitForTimeout(500)
}

/**
 * Get frame count from store
 */
async function getFrameCount(page) {
  return await page.evaluate(() => {
    const store = window.__SIMULATION_STORE__
    return store ? store.getState().totalFrames : 0
  })
}

/**
 * Set frame and wait for snapshot to load using the exposed store API
 */
async function setFrameAndWait(page, frame) {
  await page.evaluate(async (f) => {
    await window.__SIMULATION_STORE__.setFrame(f)
  }, frame)

  // Brief wait for WebGL render
  await page.waitForTimeout(50)
}

/**
 * Get pressure range for current frame to verify data loaded
 */
async function getPressureRange(page) {
  return await page.evaluate(() => {
    return window.__SIMULATION_STORE__.getPressureRange()
  })
}

/**
 * Capture video for a single pipe configuration
 */
async function captureVideo(page, pipe) {
  console.log(`\nCapturing ${pipe.name}...`)

  // Click the pipe button to load it
  const pipeButton = page.locator(`button:has-text("${pipe.buttonText}")`).first()
  await pipeButton.click()

  // Wait for simulation to load
  await waitForSimulationLoad(page)

  const frameCount = await getFrameCount(page)
  console.log(`  Loaded ${frameCount} frames`)

  // Find the canvas
  const canvas = page.locator('canvas').first()
  await canvas.waitFor({ state: 'visible' })

  // Get canvas dimensions
  const canvasBounds = await canvas.boundingBox()
  if (!canvasBounds) throw new Error('Could not get canvas bounds')

  // Create frames directory
  const framesDir = path.join(OUTPUT_DIR, `${pipe.id}-frames`)
  if (fs.existsSync(framesDir)) {
    fs.rmSync(framesDir, { recursive: true })
  }
  fs.mkdirSync(framesDir, { recursive: true })

  // Capture each frame using direct store access
  console.log(`  Capturing ${frameCount} frames...`)
  for (let i = 0; i < frameCount; i++) {
    // Set frame and wait for snapshot to load
    await setFrameAndWait(page, i)

    // Verify pressure data loaded (for debugging)
    if (i === 0 || i === frameCount - 1) {
      const [min, max] = await getPressureRange(page)
      console.log(`    Frame ${i}: pressure range [${min.toFixed(6)}, ${max.toFixed(6)}]`)
    }

    // Screenshot the canvas area
    const framePath = path.join(framesDir, `frame_${String(i).padStart(5, '0')}.png`)
    let retries = 3
    while (retries > 0) {
      try {
        await page.screenshot({
          path: framePath,
          clip: {
            x: canvasBounds.x,
            y: canvasBounds.y,
            width: canvasBounds.width,
            height: canvasBounds.height,
          }
        })
        break
      } catch (err) {
        retries--
        if (retries === 0) throw err
        console.log(`    Retry screenshot for frame ${i}...`)
        await page.waitForTimeout(500)
      }
    }

    if ((i + 1) % 25 === 0 || i === frameCount - 1) {
      console.log(`    Frame ${i + 1}/${frameCount}`)
    }
  }

  // Try to convert to video using ffmpeg
  const outputPath = path.join(OUTPUT_DIR, `${pipe.id}-pipe.mp4`)

  try {
    // Check if ffmpeg is available
    execSync('which ffmpeg', { stdio: 'pipe' })

    console.log(`  Encoding to ${outputPath}...`)
    execSync(`ffmpeg -y -framerate ${FRAME_RATE} -i "${framesDir}/frame_%05d.png" -c:v libx264 -pix_fmt yuv420p -crf 18 "${outputPath}"`, {
      stdio: 'inherit'
    })

    // Clean up frames
    fs.rmSync(framesDir, { recursive: true })
    console.log(`  Done: ${outputPath}`)
    return outputPath
  } catch {
    console.log(`  ffmpeg not available - keeping frames at ${framesDir}`)
    console.log(`  To encode manually: ffmpeg -framerate ${FRAME_RATE} -i "${framesDir}/frame_%05d.png" -c:v libx264 -pix_fmt yuv420p "${outputPath}"`)
    return framesDir
  }
}

async function main() {
  // Ensure output directory exists
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true })
  }

  console.log('Launching browser...')
  const browser = await chromium.launch({
    headless: false, // Use headed mode to see what's happening
  })

  const context = await browser.newContext({
    viewport: VIEWPORT,
  })

  const page = await context.newPage()

  try {
    // Navigate to app
    console.log(`Navigating to ${BASE_URL}...`)

    // Listen for console messages (helpful for debugging)
    page.on('console', msg => {
      if (msg.type() === 'error') {
        console.log('Browser Error:', msg.text())
      }
    })
    page.on('pageerror', err => console.log('Page Error:', err.message))

    await page.goto(BASE_URL, { waitUntil: 'domcontentloaded' })

    // Wait for React to render
    console.log('Waiting for app to render...')
    await page.waitForSelector('text=FDTD Visualizer', { timeout: 30000 })
    await page.waitForTimeout(1000)

    // Click "Organ Pipe Modes" button to enter demo
    console.log('Clicking Organ Pipe Modes button...')
    const demoButton = page.locator('button').filter({ hasText: 'Organ Pipe Modes' }).first()
    await demoButton.click({ timeout: 10000 })

    // Wait for initial load
    await waitForSimulationLoad(page)

    // Verify store is accessible
    const storeAvailable = await page.evaluate(() => !!window.__SIMULATION_STORE__)
    if (!storeAvailable) {
      throw new Error('Simulation store not exposed on window. Check simulationStore.ts')
    }
    console.log('Store API available')

    // Capture each pipe type
    const videos = []
    for (const pipe of PIPES) {
      const videoPath = await captureVideo(page, pipe)
      videos.push(videoPath)
    }

    console.log('\n✓ All videos captured:')
    videos.forEach(v => console.log(`  ${v}`))

  } finally {
    await browser.close()
  }
}

main().catch(err => {
  console.error('Error:', err)
  process.exit(1)
})
