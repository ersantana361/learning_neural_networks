import { useEffect } from 'react';
import CodeCell from './CodeCell';
import OutputDisplay from './OutputDisplay';
import { useCodeExecution } from '../../hooks/useCodeExecution';
import { useUserCode } from '../../hooks/useUserCode';

export default function CodeExecutor({ lectureId, initialCode = '' }) {
  const {
    execute,
    stop,
    clear,
    output,
    status,
    executionTime,
    isRunning,
  } = useCodeExecution();

  const {
    code,
    setCode,
    saveCode,
    resetCode,
    loading,
    saving,
  } = useUserCode(lectureId);

  // Use initial code if no saved code exists
  useEffect(() => {
    if (!loading && !code && initialCode) {
      setCode(initialCode);
    }
  }, [loading, code, initialCode, setCode]);

  const handleRun = (codeToRun) => {
    execute(codeToRun, lectureId);
  };

  const handleReset = async () => {
    if (confirm('Reset to original code? Your changes will be lost.')) {
      await resetCode();
      setCode(initialCode);
      clear();
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-zinc-500">Loading...</div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <CodeCell
        initialCode={code || initialCode}
        onRun={handleRun}
        onStop={stop}
        onSave={saveCode}
        onReset={handleReset}
        isRunning={isRunning}
        isSaving={saving}
      />

      <OutputDisplay
        output={output}
        executionTime={executionTime}
        status={status}
      />
    </div>
  );
}
