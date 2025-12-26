import { Link } from 'react-router-dom';
import { Clock, CheckCircle2, PlayCircle, ChevronRight } from 'lucide-react';
import { ProgressBar } from './ProgressBar';

export function LectureCard({ lecture, progress, isComplete }) {
  return (
    <Link
      to={`/lecture/${lecture.id}`}
      className="block bg-gray-800 rounded-xl p-5 hover:bg-gray-750 border border-gray-700 hover:border-emerald-500/50 transition-all group"
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 rounded-lg flex items-center justify-center text-lg font-bold
            ${isComplete
              ? 'bg-emerald-500/20 text-emerald-400'
              : 'bg-gray-700 text-gray-300'
            }`}
          >
            {lecture.id}
          </div>
          <div>
            <h3 className="font-semibold text-white group-hover:text-emerald-300 transition-colors">
              {lecture.title}
            </h3>
            <div className="flex items-center gap-2 text-sm text-gray-400 mt-0.5">
              <Clock size={14} />
              <span>{lecture.duration}</span>
            </div>
          </div>
        </div>
        {isComplete ? (
          <CheckCircle2 size={24} className="text-emerald-400" />
        ) : (
          <PlayCircle size={24} className="text-gray-500 group-hover:text-emerald-400 transition-colors" />
        )}
      </div>

      {/* Description */}
      <p className="text-sm text-gray-400 mb-4 line-clamp-2">
        {lecture.description}
      </p>

      {/* Topics */}
      <div className="flex flex-wrap gap-1.5 mb-4">
        {lecture.topics.slice(0, 3).map((topic, i) => (
          <span
            key={i}
            className="px-2 py-0.5 text-xs bg-gray-700 text-gray-300 rounded"
          >
            {topic}
          </span>
        ))}
        {lecture.topics.length > 3 && (
          <span className="px-2 py-0.5 text-xs bg-gray-700 text-gray-500 rounded">
            +{lecture.topics.length - 3} more
          </span>
        )}
      </div>

      {/* Progress */}
      <div className="flex items-center gap-3">
        <div className="flex-1">
          <ProgressBar
            completed={progress.completed}
            total={progress.total}
            size="sm"
            showLabel={false}
          />
        </div>
        <span className="text-xs text-gray-500">
          {progress.completed}/{progress.total}
        </span>
        <ChevronRight size={16} className="text-gray-600 group-hover:text-emerald-400 transition-colors" />
      </div>
    </Link>
  );
}
