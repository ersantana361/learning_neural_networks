import { Link, useLocation } from 'react-router-dom';
import { Home, CheckCircle2, Circle, ChevronRight, ExternalLink } from 'lucide-react';
import { course } from '../data/course';
import { ProgressBar } from './ProgressBar';

export function Sidebar({ getLectureProgress, isLectureComplete, getOverallProgress }) {
  const location = useLocation();
  const overall = getOverallProgress();

  return (
    <aside className="w-72 bg-gray-900 border-r border-gray-800 flex flex-col h-screen sticky top-0">
      {/* Header */}
      <div className="p-4 border-b border-gray-800">
        <Link to="/" className="flex items-center gap-2 text-white hover:text-emerald-400 transition-colors">
          <Home size={20} />
          <span className="font-semibold">NN: Zero to Hero</span>
        </Link>

        {/* Overall progress */}
        <div className="mt-4">
          <ProgressBar
            completed={overall.completed}
            total={overall.total}
            size="sm"
          />
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto p-2">
        {course.parts.map((part) => (
          <div key={part.id} className="mb-4">
            <h3 className="px-3 py-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">
              {part.title}
            </h3>
            <ul className="space-y-1">
              {part.lectures.map((lecture) => {
                const isActive = location.pathname === `/lecture/${lecture.id}`;
                const completed = isLectureComplete(lecture.id);
                const { completed: tasksDone, total } = getLectureProgress(lecture.id);

                return (
                  <li key={lecture.id}>
                    <Link
                      to={`/lecture/${lecture.id}`}
                      className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors
                        ${isActive
                          ? 'bg-emerald-900/50 text-emerald-300'
                          : 'text-gray-400 hover:bg-gray-800 hover:text-white'
                        }`}
                    >
                      {completed ? (
                        <CheckCircle2 size={16} className="text-emerald-400 flex-shrink-0" />
                      ) : (
                        <Circle size={16} className="flex-shrink-0" />
                      )}
                      <span className="flex-1 truncate">
                        {lecture.id}. {lecture.title}
                      </span>
                      {!completed && tasksDone > 0 && (
                        <span className="text-xs text-gray-500">
                          {tasksDone}/{total}
                        </span>
                      )}
                      <ChevronRight size={14} className="text-gray-600" />
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-gray-800">
        <a
          href={course.discord}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 text-sm text-gray-400 hover:text-white transition-colors"
        >
          <span>Join Discord</span>
          <ExternalLink size={14} />
        </a>
        <p className="mt-2 text-xs text-gray-600">
          By {course.instructor}
        </p>
      </div>
    </aside>
  );
}
