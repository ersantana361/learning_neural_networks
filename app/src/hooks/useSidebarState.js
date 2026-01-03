import { useState, useEffect, useCallback } from 'react';

const STORAGE_KEY = 'nn-zero-to-hero-sidebar';

export function useSidebarState() {
  const [isCollapsed, setIsCollapsed] = useState(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      return stored ? JSON.parse(stored) : false;
    } catch {
      return false;
    }
  });

  // Persist to localStorage
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(isCollapsed));
  }, [isCollapsed]);

  // Toggle sidebar state
  const toggleSidebar = useCallback(() => {
    setIsCollapsed(prev => !prev);
  }, []);

  return {
    isCollapsed,
    toggleSidebar
  };
}
