import { useState, useEffect } from 'react';
import { Play, Square, RotateCcw, Save, Loader2 } from 'lucide-react';
import Prism from 'prismjs';
import 'prismjs/components/prism-python';

export default function CodeCell({
  initialCode = '',
  onRun,
  onStop,
  onSave,
  onReset,
  isRunning = false,
  isSaving = false,
  readOnly = false,
}) {
  const [code, setCode] = useState(initialCode);
  const [highlighted, setHighlighted] = useState('');

  useEffect(() => {
    setCode(initialCode);
  }, [initialCode]);

  useEffect(() => {
    const html = Prism.highlight(code, Prism.languages.python, 'python');
    setHighlighted(html);
  }, [code]);

  const handleRun = () => {
    if (onRun) onRun(code);
  };

  const handleSave = () => {
    if (onSave) onSave(code);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      handleRun();
    }
    if (e.key === 's' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      handleSave();
    }
    // Handle Tab for indentation
    if (e.key === 'Tab') {
      e.preventDefault();
      const start = e.target.selectionStart;
      const end = e.target.selectionEnd;
      const newCode = code.substring(0, start) + '    ' + code.substring(end);
      setCode(newCode);
      // Set cursor position after indent
      setTimeout(() => {
        e.target.selectionStart = e.target.selectionEnd = start + 4;
      }, 0);
    }
  };

  return (
    <div className="bg-zinc-900 rounded-lg overflow-hidden">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-zinc-800">
        <div className="flex items-center gap-2">
          {!isRunning ? (
            <button
              onClick={handleRun}
              disabled={readOnly}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-green-600 hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed text-white text-xs font-medium rounded transition-colors"
            >
              <Play className="w-3.5 h-3.5" />
              Run
            </button>
          ) : (
            <button
              onClick={onStop}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white text-xs font-medium rounded transition-colors"
            >
              <Square className="w-3.5 h-3.5" />
              Stop
            </button>
          )}
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs text-zinc-500">
            Ctrl+Enter to run
          </span>
          {onSave && (
            <button
              onClick={handleSave}
              disabled={isSaving}
              className="flex items-center gap-1.5 px-2 py-1.5 text-zinc-400 hover:text-white text-xs transition-colors"
              title="Save (Ctrl+S)"
            >
              {isSaving ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Save className="w-3.5 h-3.5" />
              )}
            </button>
          )}
          {onReset && (
            <button
              onClick={onReset}
              className="flex items-center gap-1.5 px-2 py-1.5 text-zinc-400 hover:text-white text-xs transition-colors"
              title="Reset to original"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Editor */}
      <div className="relative">
        <pre
          className="absolute inset-0 p-4 font-mono text-sm pointer-events-none overflow-hidden"
          aria-hidden="true"
        >
          <code
            className="language-python"
            dangerouslySetInnerHTML={{ __html: highlighted }}
          />
        </pre>
        <textarea
          value={code}
          onChange={(e) => setCode(e.target.value)}
          onKeyDown={handleKeyDown}
          readOnly={readOnly}
          className="w-full min-h-[200px] p-4 bg-transparent text-transparent caret-white font-mono text-sm resize-y focus:outline-none"
          spellCheck={false}
          autoCapitalize="off"
          autoComplete="off"
          autoCorrect="off"
        />
      </div>
    </div>
  );
}
