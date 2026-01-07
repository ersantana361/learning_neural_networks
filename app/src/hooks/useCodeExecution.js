import { useState, useCallback, useRef } from 'react';
import { useSession } from './useSession';

export function useCodeExecution() {
  const sessionId = useSession();
  const [output, setOutput] = useState([]);
  const [status, setStatus] = useState('idle');
  const [executionTime, setExecutionTime] = useState(null);
  const wsRef = useRef(null);

  const execute = useCallback(async (code, lectureId = null) => {
    setStatus('running');
    setOutput([]);
    setExecutionTime(null);

    try {
      const response = await fetch('/api/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          code,
          session_id: sessionId,
          lecture_id: lectureId,
        }),
      });

      const result = await response.json();

      if (result.stdout) {
        // Check for embedded plots
        const lines = result.stdout.split('\n');
        const plotLine = lines.find(l => l.startsWith('__PLOTS__:'));

        if (plotLine) {
          const plots = JSON.parse(plotLine.replace('__PLOTS__:', ''));
          const textOutput = lines.filter(l => !l.startsWith('__PLOTS__:')).join('\n');

          if (textOutput.trim()) {
            setOutput(prev => [...prev, { type: 'stdout', content: textOutput }]);
          }
          plots.forEach(plot => {
            setOutput(prev => [...prev, { type: 'plot', content: plot }]);
          });
        } else {
          setOutput(prev => [...prev, { type: 'stdout', content: result.stdout }]);
        }
      }

      if (result.stderr) {
        setOutput(prev => [...prev, { type: 'stderr', content: result.stderr }]);
      }

      setExecutionTime(result.execution_time_ms);
      setStatus(result.status);
    } catch (error) {
      setOutput([{ type: 'error', content: error.message }]);
      setStatus('failed');
    }
  }, [sessionId]);

  const executeStreaming = useCallback((code, lectureId = null) => {
    setStatus('running');
    setOutput([]);
    setExecutionTime(null);

    // Close existing connection
    if (wsRef.current) {
      wsRef.current.close();
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/api/execute/ws`);

    ws.onopen = () => {
      ws.send(JSON.stringify({
        code,
        session_id: sessionId,
        lecture_id: lectureId,
      }));
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case 'stdout':
          setOutput(prev => [...prev, { type: 'stdout', content: data.content }]);
          break;
        case 'stderr':
        case 'error':
          setOutput(prev => [...prev, { type: 'stderr', content: data.content }]);
          break;
        case 'status':
          setStatus(data.content);
          break;
        case 'execution_time':
          setExecutionTime(data.content);
          break;
      }
    };

    ws.onerror = () => {
      setOutput(prev => [...prev, { type: 'error', content: 'WebSocket error' }]);
      setStatus('failed');
    };

    ws.onclose = () => {
      if (status === 'running') {
        setStatus('completed');
      }
    };

    wsRef.current = ws;
  }, [sessionId, status]);

  const stop = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
    }
    setStatus('stopped');
  }, []);

  const clear = useCallback(() => {
    setOutput([]);
    setStatus('idle');
    setExecutionTime(null);
  }, []);

  return {
    execute,
    executeStreaming,
    stop,
    clear,
    output,
    status,
    executionTime,
    isRunning: status === 'running',
  };
}
