/**
 * Demo configuration for auto-loading simulation data from R2 storage.
 *
 * Maps demo IDs (used in URL routes) to their R2 storage paths.
 */

/**
 * R2 bucket base URL from environment.
 * Falls back to empty string if not configured.
 */
const R2_BUCKET_URL = import.meta.env.VITE_R2_BUCKET_URL || '';

/**
 * Demo metadata for display and loading.
 */
export interface DemoConfig {
  /** Display title for the demo */
  title: string;
  /** Short description */
  description: string;
  /** Category for filtering */
  category: string;
  /** Path within R2 bucket (relative to bucket root) */
  path: string;
}

/**
 * Available demos mapped by their URL-safe ID.
 */
export const DEMOS: Record<string, DemoConfig> = {
  'open-pipe': {
    title: 'Open Pipe',
    description: 'Acoustic simulation of an open-ended pipe showing standing wave patterns and harmonics.',
    category: 'Acoustics',
    path: 'demos/organ-pipes/open-pipe',
  },
  'closed-pipe': {
    title: 'Closed Pipe',
    description: 'Acoustic simulation of a closed pipe demonstrating odd harmonic patterns.',
    category: 'Acoustics',
    path: 'demos/organ-pipes/closed-pipe',
  },
  'half-open-pipe': {
    title: 'Half-Open Pipe',
    description: 'Acoustic simulation showing the behavior of a pipe with one open and one closed end.',
    category: 'Acoustics',
    path: 'demos/organ-pipes/half-open-pipe',
  },
};

/**
 * Get the full R2 URL for a demo's base path.
 *
 * @param demoId - The demo ID from the URL route
 * @returns Full R2 URL to the demo directory, or null if demo not found
 */
export function getDemoBasePath(demoId: string): string | null {
  const demo = DEMOS[demoId];
  if (!demo) {
    return null;
  }

  if (!R2_BUCKET_URL) {
    console.warn('VITE_R2_BUCKET_URL not configured. Demo loading will fail.');
    return null;
  }

  return `${R2_BUCKET_URL}/${demo.path}`;
}

/**
 * Check if R2 storage is configured.
 */
export function isR2Configured(): boolean {
  return Boolean(R2_BUCKET_URL);
}

/**
 * Get demo config by ID.
 */
export function getDemoConfig(demoId: string): DemoConfig | null {
  return DEMOS[demoId] || null;
}
