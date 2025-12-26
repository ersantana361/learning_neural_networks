import { Check, Circle } from 'lucide-react';

export function TaskList({ tasks, isTaskComplete, onToggle }) {
  return (
    <div className="space-y-2">
      {tasks.map((task) => {
        const completed = isTaskComplete(task.id);
        return (
          <button
            key={task.id}
            onClick={() => onToggle(task.id)}
            className={`w-full flex items-center gap-3 p-3 rounded-lg transition-all text-left
              ${completed
                ? 'bg-emerald-900/30 text-emerald-300 hover:bg-emerald-900/40'
                : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
              }`}
          >
            <div className={`flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center
              ${completed
                ? 'bg-emerald-500 text-white'
                : 'border-2 border-gray-500'
              }`}
            >
              {completed && <Check size={12} strokeWidth={3} />}
            </div>
            <span className={completed ? 'line-through opacity-75' : ''}>
              {task.label}
            </span>
          </button>
        );
      })}
    </div>
  );
}
