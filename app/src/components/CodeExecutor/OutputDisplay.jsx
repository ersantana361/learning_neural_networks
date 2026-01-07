import { useRef, useEffect } from 'react';

export default function OutputDisplay({ output, executionTime, status }) {
  const outputRef = useRef(null);

  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [output]);

  if (output.length === 0 && status === 'idle') {
    return (
      <div className="bg-zinc-900 rounded-lg p-4 text-zinc-500 text-sm font-mono">
        Output will appear here after running code...
      </div>
    );
  }

  return (
    <div className="bg-zinc-900 rounded-lg overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 border-b border-zinc-800">
        <span className="text-xs text-zinc-500 font-medium">Output</span>
        <div className="flex items-center gap-3">
          {executionTime !== null && (
            <span className="text-xs text-zinc-500">
              {executionTime}ms
            </span>
          )}
          {status === 'running' && (
            <span className="text-xs text-yellow-500 flex items-center gap-1">
              <span className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse" />
              Running
            </span>
          )}
          {status === 'completed' && (
            <span className="text-xs text-green-500">Completed</span>
          )}
          {status === 'failed' && (
            <span className="text-xs text-red-500">Failed</span>
          )}
          {status === 'timeout' && (
            <span className="text-xs text-orange-500">Timed out</span>
          )}
        </div>
      </div>

      <div
        ref={outputRef}
        className="p-4 max-h-80 overflow-y-auto font-mono text-sm"
      >
        {output.map((item, index) => (
          <div key={index} className="mb-2">
            {item.type === 'stdout' && (
              <pre className="text-zinc-300 whitespace-pre-wrap">{item.content}</pre>
            )}
            {item.type === 'stderr' && (
              <pre className="text-red-400 whitespace-pre-wrap">{item.content}</pre>
            )}
            {item.type === 'error' && (
              <pre className="text-red-500 whitespace-pre-wrap">{item.content}</pre>
            )}
            {item.type === 'plot' && (
              <img
                src={`data:image/png;base64,${item.content}`}
                alt="Plot output"
                className="max-w-full rounded mt-2"
              />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
