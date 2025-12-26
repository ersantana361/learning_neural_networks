import { useState, useEffect, useCallback } from 'react';

const STORAGE_KEY = 'nn-zero-to-hero-notes';

export function useNotes() {
  const [notes, setNotes] = useState(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      return stored ? JSON.parse(stored) : {};
    } catch {
      return {};
    }
  });

  // Persist to localStorage
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(notes));
  }, [notes]);

  // Get notes for a lecture
  const getNote = useCallback((lectureId) => {
    return notes[lectureId] || '';
  }, [notes]);

  // Update notes for a lecture
  const updateNote = useCallback((lectureId, content) => {
    setNotes(prev => ({
      ...prev,
      [lectureId]: content
    }));
  }, []);

  // Check if a lecture has notes
  const hasNotes = useCallback((lectureId) => {
    return !!notes[lectureId] && notes[lectureId].trim().length > 0;
  }, [notes]);

  // Delete notes for a lecture
  const deleteNote = useCallback((lectureId) => {
    setNotes(prev => {
      const next = { ...prev };
      delete next[lectureId];
      return next;
    });
  }, []);

  // Reset all notes
  const resetNotes = useCallback(() => {
    setNotes({});
  }, []);

  return {
    notes,
    getNote,
    updateNote,
    hasNotes,
    deleteNote,
    resetNotes
  };
}
