import { useState, useEffect } from 'react';

const SESSION_KEY = 'nn-session-id';

function generateSessionId() {
  return 'sess_' + Math.random().toString(36).substring(2, 15) +
         Math.random().toString(36).substring(2, 15);
}

export function useSession() {
  const [sessionId, setSessionId] = useState(() => {
    const stored = localStorage.getItem(SESSION_KEY);
    if (stored) return stored;

    const newId = generateSessionId();
    localStorage.setItem(SESSION_KEY, newId);
    return newId;
  });

  return sessionId;
}
