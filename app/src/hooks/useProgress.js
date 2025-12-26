import { useState, useEffect, useCallback } from 'react';
import { getAllTaskIds, getAllLectures } from '../data/course';

const STORAGE_KEY = 'nn-zero-to-hero-progress';

export function useProgress() {
  const [progress, setProgress] = useState(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      return stored ? JSON.parse(stored) : {};
    } catch {
      return {};
    }
  });

  // Persist to localStorage
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(progress));
  }, [progress]);

  // Toggle a task's completion status
  const toggleTask = useCallback((taskId) => {
    setProgress(prev => ({
      ...prev,
      [taskId]: !prev[taskId]
    }));
  }, []);

  // Check if a task is completed
  const isTaskComplete = useCallback((taskId) => {
    return !!progress[taskId];
  }, [progress]);

  // Get completion count for a lecture
  const getLectureProgress = useCallback((lectureId) => {
    const lecture = getAllLectures().find(l => l.id === lectureId);
    if (!lecture) return { completed: 0, total: 0 };

    const total = lecture.tasks.length;
    const completed = lecture.tasks.filter(t => progress[t.id]).length;

    return { completed, total };
  }, [progress]);

  // Check if a lecture is fully completed
  const isLectureComplete = useCallback((lectureId) => {
    const { completed, total } = getLectureProgress(lectureId);
    return total > 0 && completed === total;
  }, [getLectureProgress]);

  // Get overall course progress
  const getOverallProgress = useCallback(() => {
    const allTaskIds = getAllTaskIds();
    const total = allTaskIds.length;
    const completed = allTaskIds.filter(id => progress[id]).length;

    return {
      completed,
      total,
      percentage: total > 0 ? Math.round((completed / total) * 100) : 0
    };
  }, [progress]);

  // Reset all progress
  const resetProgress = useCallback(() => {
    setProgress({});
  }, []);

  return {
    progress,
    toggleTask,
    isTaskComplete,
    getLectureProgress,
    isLectureComplete,
    getOverallProgress,
    resetProgress
  };
}
