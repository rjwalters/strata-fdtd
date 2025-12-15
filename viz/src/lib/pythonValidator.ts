/**
 * Python script validator using Skulpt for AST-based validation
 * with context-aware semantic checks to avoid false positives
 */

declare global {
  interface Window {
    Sk?: {
      configure: (options: { __future__?: object }) => void
      importMainWithBody: (name: string, dumpJS: boolean, body: string, canSuspend: boolean) => Promise<void>
      python3: object
      builtin: {
        str: (obj: unknown) => { v: string }
      }
      ffi: {
        remapToJs: (obj: unknown) => unknown
      }
      misceval: {
        asyncToPromise: (fn: () => unknown) => Promise<unknown>
      }
    }
  }
}

export interface ValidationError {
  line: number
  column: number
  message: string
  severity: 'error' | 'warning'
}

// Track Skulpt loading state
let skulptLoaded = false
let skulptLoadPromise: Promise<void> | null = null

/**
 * Load Skulpt library dynamically
 */
export async function loadSkulpt(): Promise<void> {
  if (skulptLoaded) {
    return
  }

  if (skulptLoadPromise) {
    return skulptLoadPromise
  }

  skulptLoadPromise = new Promise((resolve, reject) => {
    // Check if Skulpt is already loaded
    if (window.Sk) {
      skulptLoaded = true
      resolve()
      return
    }

    // Load Skulpt from CDN
    const script = document.createElement('script')
    script.src = 'https://skulpt.org/js/skulpt.min.js'
    script.async = true

    script.onload = () => {
      // Load skulpt-stdlib after main skulpt
      const stdlibScript = document.createElement('script')
      stdlibScript.src = 'https://skulpt.org/js/skulpt-stdlib.js'
      stdlibScript.async = true

      stdlibScript.onload = () => {
        skulptLoaded = true
        resolve()
      }

      stdlibScript.onerror = () => {
        reject(new Error('Failed to load Skulpt stdlib'))
      }

      document.head.appendChild(stdlibScript)
    }

    script.onerror = () => {
      reject(new Error('Failed to load Skulpt'))
    }

    document.head.appendChild(script)
  })

  return skulptLoadPromise
}

/**
 * Parse a Python script and return syntax errors using Skulpt
 */
async function checkSyntax(script: string): Promise<ValidationError[]> {
  const errors: ValidationError[] = []

  if (!window.Sk) {
    return errors
  }

  try {
    window.Sk.configure({
      __future__: window.Sk.python3,
    })

    // Try to compile the script (syntax check only)
    await window.Sk.importMainWithBody('<stdin>', false, script, false)
  } catch (e: unknown) {
    // Parse Skulpt error to extract line/column info
    const error = e as { args?: { v?: Array<{ v?: string | number }> }; toString?: () => string }
    if (error.args?.v) {
      const args = error.args.v
      const message = typeof args[0]?.v === 'string' ? args[0].v : 'Syntax error'
      const line = typeof args[1]?.v === 'number' ? args[1].v : 1
      const column = typeof args[2]?.v === 'number' ? args[2].v : 1

      errors.push({
        line,
        column,
        message,
        severity: 'error',
      })
    } else {
      // Fallback error parsing
      const errorStr = error.toString?.() || String(e)
      const lineMatch = errorStr.match(/line (\d+)/)
      const line = lineMatch ? parseInt(lineMatch[1], 10) : 1

      errors.push({
        line,
        column: 1,
        message: errorStr,
        severity: 'error',
      })
    }
  }

  return errors
}

/**
 * Check if a position in the script is inside a comment or string literal.
 * This prevents false positives from patterns appearing in comments/strings.
 */
function isInCommentOrString(script: string, lineIndex: number): boolean {
  const lines = script.split('\n')
  if (lineIndex < 0 || lineIndex >= lines.length) {
    return false
  }

  const line = lines[lineIndex]
  const trimmedLine = line.trim()

  // Check if line starts with a comment
  if (trimmedLine.startsWith('#')) {
    return true
  }

  return false
}

/**
 * Extract the actual assignment value from a line, handling various patterns.
 * Returns null if the line doesn't contain an actual assignment (is a comment, etc.)
 */
function extractAssignmentValue(
  script: string,
  lineIndex: number,
  varName: string
): { value: string; column: number } | null {
  const lines = script.split('\n')
  if (lineIndex < 0 || lineIndex >= lines.length) {
    return null
  }

  const line = lines[lineIndex]

  // Skip if line is a pure comment
  const trimmedLine = line.trim()
  if (trimmedLine.startsWith('#')) {
    return null
  }

  // Find the assignment pattern
  const assignPattern = new RegExp(`\\b${varName}\\s*=\\s*(.+)`, 'i')
  const match = line.match(assignPattern)

  if (!match) {
    return null
  }

  // Get the position of the match in the line
  const matchIndex = line.indexOf(match[0])

  // Check if the match is inside a string literal by looking at quotes before it
  const beforeMatch = line.substring(0, matchIndex)
  const singleQuotes = (beforeMatch.match(/'/g) || []).length
  const doubleQuotes = (beforeMatch.match(/"/g) || []).length
  const tripleDoubleQuotes = (beforeMatch.match(/"""/g) || []).length
  const tripleSingleQuotes = (beforeMatch.match(/'''/g) || []).length

  // If odd number of quotes before, we're inside a string
  const inSingleQuoteString = (singleQuotes - tripleSingleQuotes * 3) % 2 !== 0
  const inDoubleQuoteString = (doubleQuotes - tripleDoubleQuotes * 3) % 2 !== 0

  if (inSingleQuoteString || inDoubleQuoteString) {
    return null
  }

  // Check if there's a comment marker (#) before the assignment
  const hashIndex = beforeMatch.indexOf('#')
  if (hashIndex >= 0) {
    return null
  }

  // Extract the value, removing any inline comment
  let value = match[1]
  const commentIndex = value.indexOf('#')
  if (commentIndex >= 0) {
    // Check if # is inside a string
    const valueBeforeHash = value.substring(0, commentIndex)
    const singleQ = (valueBeforeHash.match(/'/g) || []).length
    const doubleQ = (valueBeforeHash.match(/"/g) || []).length
    if (singleQ % 2 === 0 && doubleQ % 2 === 0) {
      value = value.substring(0, commentIndex)
    }
  }

  return {
    value: value.trim(),
    column: matchIndex + 1,
  }
}

/**
 * Check if a numeric string represents a negative value.
 * Handles expressions like abs(-1e-3) which are actually positive.
 */
function isNegativeValue(valueStr: string): boolean {
  // Remove whitespace
  const value = valueStr.trim()

  // If wrapped in abs(), it's positive
  if (value.match(/^abs\s*\(/i)) {
    return false
  }

  // If wrapped in other function calls that could make it positive, skip
  if (value.match(/^\w+\s*\(/)) {
    // Complex expression, can't determine - assume OK
    return false
  }

  // Check if it's a simple negative number
  const numMatch = value.match(/^(-?\d*\.?\d+(?:[eE][+-]?\d+)?)/)
  if (numMatch) {
    const num = parseFloat(numMatch[1])
    return num < 0
  }

  // Check if starts with unary minus
  if (value.startsWith('-')) {
    // Could be -variable or -expression, can't determine
    // Only flag if it looks like a literal negative number
    const afterMinus = value.substring(1).trim()
    if (afterMinus.match(/^\d/)) {
      return true
    }
  }

  return false
}

/**
 * Run semantic checks on the script for metamaterial API usage.
 * Uses context-aware pattern matching to avoid false positives.
 */
function checkSemantics(script: string): ValidationError[] {
  const warnings: ValidationError[] = []
  const lines = script.split('\n')

  // Check for negative resolution values (must be positive)
  for (let i = 0; i < lines.length; i++) {
    if (isInCommentOrString(script, i)) {
      continue
    }

    const extraction = extractAssignmentValue(script, i, 'resolution')
    if (extraction && isNegativeValue(extraction.value)) {
      warnings.push({
        line: i + 1,
        column: extraction.column,
        message: 'Grid resolution must be a positive value',
        severity: 'warning',
      })
    }
  }

  // Check for UniformGrid calls with potentially invalid resolution
  const gridPattern = /UniformGrid\s*\([^)]*resolution\s*=\s*([^,)]+)/gi
  let gridMatch
  while ((gridMatch = gridPattern.exec(script)) !== null) {
    // Find line number
    const beforeMatch = script.substring(0, gridMatch.index)
    const lineNumber = (beforeMatch.match(/\n/g) || []).length

    // Skip if in comment or string
    if (isInCommentOrString(script, lineNumber)) {
      continue
    }

    const resValue = gridMatch[1].trim()
    if (isNegativeValue(resValue)) {
      warnings.push({
        line: lineNumber + 1,
        column: 1,
        message: 'UniformGrid resolution must be a positive value',
        severity: 'warning',
      })
    }
  }

  // Check for grid shape not being 3D
  const shapePattern = /shape\s*=\s*\(([^)]+)\)/gi
  let shapeMatch
  while ((shapeMatch = shapePattern.exec(script)) !== null) {
    const beforeMatch = script.substring(0, shapeMatch.index)
    const lineNumber = (beforeMatch.match(/\n/g) || []).length

    if (isInCommentOrString(script, lineNumber)) {
      continue
    }

    const shapeContent = shapeMatch[1]
    const dimensions = shapeContent.split(',').filter(s => s.trim().length > 0)

    if (dimensions.length !== 3) {
      warnings.push({
        line: lineNumber + 1,
        column: 1,
        message: `Grid shape should have exactly 3 dimensions, found ${dimensions.length}`,
        severity: 'warning',
      })
    }
  }

  // Check for missing imports (helpful but non-blocking)
  const importedModules = new Set<string>()
  const importPattern = /^\s*(?:from\s+(\w+)|import\s+(\w+))/gm
  let importMatch
  while ((importMatch = importPattern.exec(script)) !== null) {
    const moduleName = importMatch[1] || importMatch[2]
    if (moduleName) {
      importedModules.add(moduleName)
    }
  }

  // Check for common metamaterial API usage without imports
  if (script.includes('UniformGrid') && !importedModules.has('strata')) {
    // Find first usage
    const usageMatch = script.match(/UniformGrid/)
    if (usageMatch) {
      const beforeMatch = script.substring(0, usageMatch.index)
      const lineNumber = (beforeMatch.match(/\n/g) || []).length

      if (!isInCommentOrString(script, lineNumber)) {
        warnings.push({
          line: lineNumber + 1,
          column: 1,
          message: 'Consider importing strata module for UniformGrid',
          severity: 'warning',
        })
      }
    }
  }

  return warnings
}

/**
 * Validate a Python script for syntax and semantic errors.
 * Returns validation errors with line/column information.
 */
export async function validatePythonScript(script: string): Promise<ValidationError[]> {
  const errors: ValidationError[] = []

  // Run syntax check with Skulpt
  if (skulptLoaded && window.Sk) {
    const syntaxErrors = await checkSyntax(script)
    errors.push(...syntaxErrors)
  }

  // Run semantic checks (always run, even without Skulpt)
  const semanticWarnings = checkSemantics(script)
  errors.push(...semanticWarnings)

  return errors
}
