/**
 * Python script validation using Skulpt parser
 *
 * Validates Python syntax and provides helpful error messages
 * for use in the Monaco editor.
 */

/// <reference path="./skulpt.d.ts" />

// Import Skulpt from npm package (not CDN)
import Sk from 'skulpt'

// Make Skulpt available globally for compatibility
if (typeof window !== 'undefined') {
  window.Sk = Sk
}

// TypeScript declaration for global Sk
declare global {
  interface Window {
    Sk: typeof Sk
  }
}

export interface ValidationError {
  line: number
  column: number
  message: string
  severity: 'error' | 'warning'
}

/**
 * Initialize Skulpt for Python parsing
 */
function initSkulpt() {
  if (typeof window.Sk === 'undefined') {
    throw new Error('Skulpt not loaded. Make sure skulpt.min.js is included.')
  }

  // Configure Skulpt for parsing only (no execution)
  window.Sk.configure({
    output: () => {}, // No output needed for validation
    read: () => '', // No file reading needed
    __future__: window.Sk.python3, // Use Python 3 syntax
  })
}

/**
 * Validate Python script syntax
 *
 * @param script - The Python script to validate
 * @returns Array of validation errors (empty if valid)
 */
export async function validatePythonScript(script: string): Promise<ValidationError[]> {
  const errors: ValidationError[] = []

  try {
    // Initialize Skulpt
    initSkulpt()

    // Try to parse the Python code
    // Skulpt will throw an error if there's a syntax error
    window.Sk.importMainWithBody('<stdin>', false, script, true)

    // Additional semantic checks for metamaterial API
    const semanticErrors = checkMetamaterialAPI(script)
    errors.push(...semanticErrors)
  } catch (error: any) {
    // Parse Skulpt error to extract line/column information
    const validationError = parseSkulptError(error)
    if (validationError) {
      errors.push(validationError)
    } else {
      // Fallback for unparseable errors
      errors.push({
        line: 1,
        column: 1,
        message: `Syntax error: ${error.message || String(error)}`,
        severity: 'error',
      })
    }
  }

  return errors
}

/**
 * Parse Skulpt error message to extract line and column information
 */
function parseSkulptError(error: any): ValidationError | null {
  const message = error.message || error.toString()

  // Skulpt error format: "SyntaxError: ... on line X"
  // or "ParseError: ... at line X column Y"
  const lineMatch = message.match(/line (\d+)/i)
  const columnMatch = message.match(/column (\d+)/i) || message.match(/col (\d+)/i)

  const line = lineMatch ? parseInt(lineMatch[1], 10) : 1
  const column = columnMatch ? parseInt(columnMatch[1], 10) : 1

  // Extract the error message without line/column info
  let cleanMessage = message
    .replace(/^(SyntaxError|ParseError):\s*/i, '')
    .replace(/\s+on line \d+/i, '')
    .replace(/\s+at line \d+ column \d+/i, '')
    .trim()

  // Make error messages more helpful
  cleanMessage = improveErrorMessage(cleanMessage)

  return {
    line,
    column,
    message: cleanMessage,
    severity: 'error',
  }
}

/**
 * Improve error messages to be more helpful
 */
function improveErrorMessage(message: string): string {
  const improvements: Record<string, string> = {
    'unexpected EOF': 'Unexpected end of file. Did you forget to close a bracket or indent a block?',
    'invalid syntax': 'Invalid Python syntax. Check for missing colons, brackets, or incorrect indentation.',
    'unindent does not match': 'Indentation error. Make sure your indentation is consistent (use spaces or tabs, not both).',
    'expected an indented block': 'Expected an indented block after colon (:). Add at least one indented line.',
    'unexpected indent': 'Unexpected indentation. Make sure indentation matches the previous block level.',
  }

  for (const [pattern, improvement] of Object.entries(improvements)) {
    if (message.toLowerCase().includes(pattern.toLowerCase())) {
      return improvement
    }
  }

  return message
}

/**
 * Check for common metamaterial API mistakes
 *
 * NOTE: Uses regex and string matching for simplicity. This approach has known edge cases:
 * - May match identifiers in comments (e.g., "# UniformGrid is great")
 * - May match in string literals (e.g., print("UniformGrid"))
 * - Does not perform true AST-based semantic analysis
 *
 * For more accurate validation, consider using Skulpt's AST walking after parsing.
 * The current approach provides helpful warnings for common mistakes while accepting
 * some false positives as a tradeoff for simplicity.
 */
function checkMetamaterialAPI(script: string): ValidationError[] {
  const warnings: ValidationError[] = []

  // Check for missing imports
  if (!script.includes('from metamaterial import') && !script.includes('import metamaterial')) {
    if (script.includes('UniformGrid') || script.includes('Scene') || script.includes('FDTDSolver')) {
      warnings.push({
        line: 1,
        column: 1,
        message: 'Missing metamaterial import. Add: from metamaterial import ...',
        severity: 'warning',
      })
    }
  }

  // Check for undefined grid before Scene
  if (script.includes('Scene(grid)')) {
    const gridDefPattern = /grid\s*=/
    const sceneDefPattern = /Scene\(grid\)/
    const gridMatch = gridDefPattern.exec(script)
    const sceneMatch = sceneDefPattern.exec(script)

    if (sceneMatch && (!gridMatch || sceneMatch.index < gridMatch.index)) {
      const line = script.substring(0, sceneMatch.index).split('\n').length
      warnings.push({
        line,
        column: 1,
        message: 'Scene() references grid before it is defined. Define grid first.',
        severity: 'warning',
      })
    }
  }

  // Check for negative resolution
  const resMatch = script.match(/resolution\s*=\s*(-[\d.e-]+)/i)
  if (resMatch) {
    const line = script.substring(0, resMatch.index).split('\n').length
    warnings.push({
      line,
      column: 1,
      message: 'Grid resolution should be positive.',
      severity: 'error',
    })
  }

  // Check for invalid grid shape (must be positive integers)
  const shapeMatch = script.match(/shape\s*=\s*\(([^)]+)\)/i)
  if (shapeMatch) {
    const shapeValues = shapeMatch[1].split(',').map(v => v.trim())
    if (shapeValues.length !== 3) {
      const line = script.substring(0, shapeMatch.index).split('\n').length
      warnings.push({
        line,
        column: 1,
        message: 'Grid shape must have exactly 3 dimensions (x, y, z).',
        severity: 'error',
      })
    } else {
      for (const val of shapeValues) {
        if (!/^\d+$/.test(val) || parseInt(val) <= 0) {
          const line = script.substring(0, shapeMatch.index).split('\n').length
          warnings.push({
            line,
            column: 1,
            message: 'Grid shape dimensions must be positive integers.',
            severity: 'error',
          })
          break
        }
      }
    }
  }

  return warnings
}

/**
 * Initialize Skulpt (now imported from npm, no dynamic loading needed)
 */
export async function loadSkulpt(): Promise<void> {
  // Skulpt is now imported at module level from npm package
  // This function kept for API compatibility but is now a no-op
  if (typeof window.Sk === 'undefined') {
    throw new Error('Skulpt failed to initialize from npm package')
  }
  return Promise.resolve()
}
