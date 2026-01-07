import { useState, useCallback, useEffect } from 'react';
import { useSession } from './useSession';

export function useUserCode(lectureId) {
  const sessionId = useSession();
  const [code, setCode] = useState('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  // Load code on mount
  useEffect(() => {
    async function loadCode() {
      if (!lectureId) return;

      setLoading(true);
      try {
        const response = await fetch(`/api/code/${lectureId}?session_id=${sessionId}`);
        if (response.ok) {
          const data = await response.json();
          if (data && data.code) {
            setCode(data.code);
          }
        }
      } catch (error) {
        console.error('Failed to load code:', error);
      } finally {
        setLoading(false);
      }
    }

    loadCode();
  }, [lectureId, sessionId]);

  const saveCode = useCallback(async (newCode) => {
    if (!lectureId) return;

    setSaving(true);
    try {
      await fetch(`/api/code/${lectureId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          code: newCode,
          session_id: sessionId,
        }),
      });
      setCode(newCode);
    } catch (error) {
      console.error('Failed to save code:', error);
    } finally {
      setSaving(false);
    }
  }, [lectureId, sessionId]);

  const resetCode = useCallback(async () => {
    if (!lectureId) return;

    try {
      await fetch(`/api/code/${lectureId}?session_id=${sessionId}`, {
        method: 'DELETE',
      });
      setCode('');
    } catch (error) {
      console.error('Failed to reset code:', error);
    }
  }, [lectureId, sessionId]);

  return {
    code,
    setCode,
    saveCode,
    resetCode,
    loading,
    saving,
  };
}
