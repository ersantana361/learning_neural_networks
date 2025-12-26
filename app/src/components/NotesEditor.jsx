import { useState, useEffect } from 'react';
import { Save, FileText } from 'lucide-react';

export function NotesEditor({ lectureId, value, onChange }) {
  const [localValue, setLocalValue] = useState(value);
  const [saved, setSaved] = useState(true);

  useEffect(() => {
    setLocalValue(value);
  }, [value]);

  useEffect(() => {
    setSaved(localValue === value);
  }, [localValue, value]);

  // Auto-save with debounce
  useEffect(() => {
    const timer = setTimeout(() => {
      if (localValue !== value) {
        onChange(lectureId, localValue);
        setSaved(true);
      }
    }, 1000);

    return () => clearTimeout(timer);
  }, [localValue, lectureId, onChange, value]);

  const handleSave = () => {
    onChange(lectureId, localValue);
    setSaved(true);
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2 text-gray-400">
          <FileText size={16} />
          <span className="text-sm font-medium">Notes</span>
        </div>
        <div className="flex items-center gap-2">
          {!saved && (
            <span className="text-xs text-amber-400">Unsaved</span>
          )}
          <button
            onClick={handleSave}
            disabled={saved}
            className={`p-1.5 rounded transition-colors
              ${saved
                ? 'text-gray-600 cursor-not-allowed'
                : 'text-emerald-400 hover:bg-gray-700'
              }`}
            title="Save notes"
          >
            <Save size={16} />
          </button>
        </div>
      </div>
      <textarea
        value={localValue}
        onChange={(e) => setLocalValue(e.target.value)}
        placeholder="Write your notes here... (auto-saves after 1 second)"
        className="notes-editor flex-1 w-full p-4 bg-gray-800 border border-gray-700 rounded-lg
          text-gray-200 placeholder-gray-500 resize-none focus:outline-none focus:border-emerald-500
          transition-colors min-h-[200px]"
      />
    </div>
  );
}
