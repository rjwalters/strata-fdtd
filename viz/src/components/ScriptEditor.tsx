/**
 * Monaco-based Python script editor component with AST validation
 */

import { useRef, useEffect, useState, useCallback } from 'react'
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

export function ScriptEditor({
  value,
  onChange,
  onSave,
  onValidationChange,
}: ScriptEditorProps) {
  const editorRef = useRef<editor.IStandaloneCodeEditor | null>(null)
  const monacoRef = useRef<Monaco | null>(null)
  const validationTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const [skulptLoaded, setSkulptLoaded] = useState(false)

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

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (validationTimeoutRef.current) {
        clearTimeout(validationTimeoutRef.current)
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
}
