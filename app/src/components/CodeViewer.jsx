import { useState, useEffect } from 'react';
import { Code, Copy, Check, FileCode } from 'lucide-react';
import Prism from 'prismjs';
import 'prismjs/components/prism-python';
import 'prismjs/themes/prism-tomorrow.css';

export function CodeViewer({ files }) {
  const [activeFile, setActiveFile] = useState(0);
  const [codeContent, setCodeContent] = useState({});
  const [copied, setCopied] = useState(false);
  const [loading, setLoading] = useState(true);

  // Fetch code files
  useEffect(() => {
    const loadFiles = async () => {
      setLoading(true);
      const content = {};

      for (const file of files) {
        try {
          // In dev mode, we'll load from the parent directory
          const response = await fetch(`/${file}`);
          if (response.ok) {
            content[file] = await response.text();
          } else {
            content[file] = `// File not found: ${file}\n// Implement your code here!`;
          }
        } catch {
          content[file] = `// Could not load: ${file}\n// Make sure the dev server is configured to serve code files.`;
        }
      }

      setCodeContent(content);
      setLoading(false);
    };

    if (files.length > 0) {
      loadFiles();
    }
  }, [files]);

  // Highlight code
  useEffect(() => {
    Prism.highlightAll();
  }, [codeContent, activeFile]);

  const handleCopy = async () => {
    const currentFile = files[activeFile];
    const content = codeContent[currentFile] || '';

    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  if (files.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 bg-gray-800 rounded-lg">
        <p className="text-gray-500">No code files for this lecture</p>
      </div>
    );
  }

  const currentFile = files[activeFile];
  const currentContent = codeContent[currentFile] || '';

  return (
    <div className="flex flex-col bg-gray-900 rounded-lg overflow-hidden border border-gray-700">
      {/* File tabs */}
      <div className="flex items-center bg-gray-800 border-b border-gray-700">
        <div className="flex-1 flex overflow-x-auto">
          {files.map((file, index) => {
            const fileName = file.split('/').pop();
            return (
              <button
                key={file}
                onClick={() => setActiveFile(index)}
                className={`flex items-center gap-2 px-4 py-2 text-sm whitespace-nowrap transition-colors
                  ${index === activeFile
                    ? 'bg-gray-900 text-emerald-400 border-b-2 border-emerald-400'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-gray-700'
                  }`}
              >
                <FileCode size={14} />
                {fileName}
              </button>
            );
          })}
        </div>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 px-3 py-2 text-sm text-gray-400 hover:text-white transition-colors"
          title="Copy code"
        >
          {copied ? (
            <>
              <Check size={14} className="text-emerald-400" />
              <span className="text-emerald-400">Copied!</span>
            </>
          ) : (
            <>
              <Copy size={14} />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>

      {/* Code content */}
      <div className="overflow-auto max-h-[500px]">
        {loading ? (
          <div className="flex items-center justify-center h-48">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-emerald-400" />
          </div>
        ) : (
          <pre className="p-4 text-sm leading-relaxed">
            <code className="language-python">{currentContent}</code>
          </pre>
        )}
      </div>
    </div>
  );
}
