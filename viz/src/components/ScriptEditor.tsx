/**
 * Monaco-based Python script editor component with AST validation
 */

import { useRef, useEffect, useState, useCallback, forwardRef, useImperativeHandle } from 'react'
import Editor from '@monaco-editor/react'
import type { Monaco } from '@monaco-editor/react'
import type { editor } from 'monaco-editor'
import {
  validatePythonScript,
  loadSkulpt,
  type ValidationError,
} from '../lib/pythonValidator'

interface ScriptEditorProps {
  value: string
  onChange: (value: string) => void
  onSave?: () => void
  onValidationChange?: (errors: ValidationError[]) => void
}

export interface ScriptEditorHandle {
  scrollToLine: (line: number, column?: number) => void
}

export const ScriptEditor = forwardRef<ScriptEditorHandle, ScriptEditorProps>(function ScriptEditor({
  value,
  onChange,
  onSave,
  onValidationChange,
}, ref) {
  const editorRef = useRef<editor.IStandaloneCodeEditor | null>(null)
  const monacoRef = useRef<Monaco | null>(null)
  const validationTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const highlightTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const decorationsRef = useRef<string[]>([])
  const [skulptLoaded, setSkulptLoaded] = useState(false)

  // Expose scrollToLine method to parent components
  useImperativeHandle(ref, () => ({
    scrollToLine: (line: number, column: number = 1) => {
      if (!editorRef.current || !monacoRef.current) return

      const editor = editorRef.current
      const monaco = monacoRef.current

      // Scroll to line and center it in the viewport
      editor.revealLineInCenter(line)

      // Set cursor at the specified position
      editor.setPosition({ lineNumber: line, column })

      // Focus the editor
      editor.focus()

      // Add temporary highlight decoration
      const model = editor.getModel()
      if (model) {
        // Clear previous highlight
        if (highlightTimeoutRef.current) {
          clearTimeout(highlightTimeoutRef.current)
        }

        // Apply new highlight decoration
        decorationsRef.current = editor.deltaDecorations(decorationsRef.current, [{
          range: new monaco.Range(line, 1, line, model.getLineMaxColumn(line)),
          options: {
            className: 'error-line-highlight',
            isWholeLine: true,
          }
        }])

        // Remove highlight after 2 seconds
        highlightTimeoutRef.current = setTimeout(() => {
          decorationsRef.current = editor.deltaDecorations(decorationsRef.current, [])
        }, 2000)
      }
    }
  }), [])

  // Load Skulpt on component mount
  useEffect(() => {
    loadSkulpt()
      .then(() => setSkulptLoaded(true))
      .catch(error => console.error('Failed to load Skulpt:', error))
  }, [])

  function handleEditorDidMount(editor: editor.IStandaloneCodeEditor, monaco: Monaco) {
    editorRef.current = editor
    monacoRef.current = monaco

    // Add keyboard shortcut for save (Ctrl+S / Cmd+S)
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => {
      onSave?.()
    })

    // Configure Python language features
    monaco.languages.setLanguageConfiguration('python', {
      comments: {
        lineComment: '#',
        blockComment: ["'''", "'''"],
      },
      brackets: [
        ['{', '}'],
        ['[', ']'],
        ['(', ')'],
      ],
      autoClosingPairs: [
        { open: '{', close: '}' },
        { open: '[', close: ']' },
        { open: '(', close: ')' },
        { open: '"', close: '"', notIn: ['string'] },
        { open: "'", close: "'", notIn: ['string', 'comment'] },
      ],
      surroundingPairs: [
        { open: '{', close: '}' },
        { open: '[', close: ']' },
        { open: '(', close: ')' },
        { open: '"', close: '"' },
        { open: "'", close: "'" },
      ],
      indentationRules: {
        increaseIndentPattern: /^\s*(def|class|for|if|elif|else|while|try|except|finally|with)\b.*:$/,
        decreaseIndentPattern: /^\s*(return|pass|break|continue|raise)\b/,
      },
    })
  }

  const validateScript = useCallback(async (script: string) => {
    if (!skulptLoaded || !editorRef.current || !monacoRef.current) {
      return
    }

    try {
      const errors = await validatePythonScript(script)

      // Notify parent component of validation results
      onValidationChange?.(errors)

      // Convert validation errors to Monaco markers
      const model = editorRef.current.getModel()
      if (!model) return

      const markers = errors.map(error => ({
        startLineNumber: error.line,
        startColumn: error.column,
        endLineNumber: error.line,
        endColumn: error.column + 10, // Highlight ~10 characters
        message: error.message,
        severity:
          error.severity === 'error'
            ? monacoRef.current!.MarkerSeverity.Error
            : monacoRef.current!.MarkerSeverity.Warning,
      }))

      monacoRef.current.editor.setModelMarkers(model, 'python-validator', markers)
    } catch (error) {
      console.error('Validation error:', error)
    }
  }, [skulptLoaded, onValidationChange])

  function handleEditorChange(value: string | undefined) {
    if (value !== undefined) {
      onChange(value)

      // Debounce validation to avoid validating on every keystroke
      if (validationTimeoutRef.current) {
        clearTimeout(validationTimeoutRef.current)
      }

      validationTimeoutRef.current = setTimeout(() => {
        validateScript(value)
      }, 500) // 500ms debounce
    }
  }

  // Focus editor and run initial validation on mount
  useEffect(() => {
    if (editorRef.current) {
      editorRef.current.focus()
    }
  }, [])

  // Run validation when Skulpt loads or value changes
  useEffect(() => {
    if (skulptLoaded && value) {
      validateScript(value)
    }
  }, [skulptLoaded, value, validateScript])

  // Cleanup timeouts on unmount
  useEffect(() => {
    return () => {
      if (validationTimeoutRef.current) {
        clearTimeout(validationTimeoutRef.current)
      }
      if (highlightTimeoutRef.current) {
        clearTimeout(highlightTimeoutRef.current)
      }
    }
  }, [])

  return (
    <Editor
      height="100%"
      defaultLanguage="python"
      value={value}
      onChange={handleEditorChange}
      onMount={handleEditorDidMount}
      theme="vs-dark"
      options={{
        minimap: { enabled: true },
        fontSize: 14,
        lineNumbers: 'on',
        scrollBeyondLastLine: false,
        automaticLayout: true,
        tabSize: 4,
        insertSpaces: true,
        wordWrap: 'on',
        folding: true,
        renderLineHighlight: 'all',
        cursorBlinking: 'smooth',
        smoothScrolling: true,
        contextmenu: true,
        quickSuggestions: {
          other: true,
          comments: false,
          strings: false,
        },
      }}
    />
  )
})
