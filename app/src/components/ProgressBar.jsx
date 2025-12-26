export function ProgressBar({ completed, total, showLabel = true, size = 'md' }) {
  const percentage = total > 0 ? Math.round((completed / total) * 100) : 0;

  const heights = {
    sm: 'h-1.5',
    md: 'h-2.5',
    lg: 'h-4'
  };

  return (
    <div className="w-full">
      <div className={`w-full bg-gray-700 rounded-full ${heights[size]}`}>
        <div
          className={`bg-gradient-to-r from-emerald-500 to-emerald-400 ${heights[size]} rounded-full transition-all duration-500 ease-out`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      {showLabel && (
        <p className="text-xs text-gray-400 mt-1">
          {completed}/{total} tasks ({percentage}%)
        </p>
      )}
    </div>
  );
}
