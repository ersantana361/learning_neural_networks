import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';

export function Layout({ getLectureProgress, isLectureComplete, getOverallProgress }) {
  return (
    <div className="flex min-h-screen bg-gray-950 text-gray-100">
      <Sidebar
        getLectureProgress={getLectureProgress}
        isLectureComplete={isLectureComplete}
        getOverallProgress={getOverallProgress}
      />
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  );
}
