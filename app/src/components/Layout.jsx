import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { useSidebarState } from '../hooks/useSidebarState';

export function Layout({ getLectureProgress, isLectureComplete, getOverallProgress }) {
  const { isCollapsed, toggleSidebar } = useSidebarState();

  return (
    <div className="flex min-h-screen bg-gray-950 text-gray-100">
      <Sidebar
        getLectureProgress={getLectureProgress}
        isLectureComplete={isLectureComplete}
        getOverallProgress={getOverallProgress}
        isCollapsed={isCollapsed}
        toggleSidebar={toggleSidebar}
      />
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  );
}
