import { useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, ArrowRight, Clock, ExternalLink, Play, FileText, Code, BookOpen, CheckSquare, Pencil } from 'lucide-react';
import { getLectureById, getAllLectures, getPartForLecture } from '../data/course';
import { VideoPlayer } from '../components/VideoPlayer';
import { TaskList } from '../components/TaskList';
import { NotesEditor } from '../components/NotesEditor';
import { CodeViewer } from '../components/CodeViewer';
import { SummaryViewer } from '../components/SummaryViewer';
import { ProgressBar } from '../components/ProgressBar';

const tabs = [
  { id: 'video', label: 'Video', icon: Play },
  { id: 'summary', label: 'AI Summary', icon: FileText },
  { id: 'notes', label: 'My Notes', icon: Pencil },
  { id: 'code', label: 'Code', icon: Code },
];

export function Lecture({
  isTaskComplete,
  toggleTask,
  getLectureProgress,
  getNote,
  updateNote
}) {
  const { id } = useParams();
  const lecture = getLectureById(id);
  const allLectures = getAllLectures();
  const [activeTab, setActiveTab] = useState('video');

  if (!lecture) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-white mb-2">Lecture not found</h1>
          <Link to="/" className="text-emerald-400 hover:text-emerald-300">
            Back to Dashboard
          </Link>
        </div>
      </div>
    );
  }

  const part = getPartForLecture(lecture.id);
  const currentIndex = allLectures.findIndex(l => l.id === lecture.id);
  const prevLecture = currentIndex > 0 ? allLectures[currentIndex - 1] : null;
  const nextLecture = currentIndex < allLectures.length - 1 ? allLectures[currentIndex + 1] : null;
  const progress = getLectureProgress(lecture.id);
  const hasCode = lecture.codeFiles && lecture.codeFiles.length > 0;

  const availableTabs = tabs.filter(tab => {
    if (tab.id === 'code') return hasCode;
    return true;
  });

  return (
    <div className="p-8 max-w-6xl mx-auto">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-sm text-gray-400 mb-4">
        <Link to="/" className="hover:text-white transition-colors">Home</Link>
        <span>/</span>
        <span className="text-gray-500">{part?.title}</span>
        <span>/</span>
        <span className="text-white">Lecture {lecture.id}</span>
      </div>

      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-white mb-3">
          {lecture.id}. {lecture.title}
        </h1>
        <p className="text-gray-400 mb-4">{lecture.description}</p>
        <div className="flex items-center gap-4 text-gray-400">
          <div className="flex items-center gap-2">
            <Clock size={16} />
            <span>{lecture.duration}</span>
          </div>
          <ProgressBar
            completed={progress.completed}
            total={progress.total}
            size="sm"
          />
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Left Column - Main Content */}
        <div className="lg:col-span-2">
          {/* Tabs */}
          <div className="flex gap-1 mb-4 border-b border-gray-800 pb-px">
            {availableTabs.map(tab => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-t-lg transition-colors ${
                    activeTab === tab.id
                      ? 'bg-gray-800 text-emerald-400 border-b-2 border-emerald-400 -mb-px'
                      : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
                  }`}
                >
                  <Icon size={16} />
                  {tab.label}
                </button>
              );
            })}
          </div>

          {/* Tab Content */}
          <div className="bg-gray-800/30 rounded-xl p-4 min-h-[400px]">
            {activeTab === 'video' && (
              <VideoPlayer videoId={lecture.videoId} title={lecture.title} />
            )}
            {activeTab === 'summary' && (
              <SummaryViewer summaryFile={lecture.summaryFile} />
            )}
            {activeTab === 'notes' && (
              <NotesEditor
                lectureId={lecture.id}
                value={getNote(lecture.id)}
                onChange={updateNote}
              />
            )}
            {activeTab === 'code' && hasCode && (
              <CodeViewer files={lecture.codeFiles} />
            )}
          </div>
        </div>

        {/* Right Column - Sidebar */}
        <div className="space-y-6">
          {/* Tasks Card */}
          <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
            <div className="flex items-center gap-2 mb-3">
              <CheckSquare size={18} className="text-emerald-400" />
              <h2 className="text-lg font-semibold text-white">Tasks</h2>
            </div>
            <TaskList
              tasks={lecture.tasks}
              isTaskComplete={isTaskComplete}
              onToggle={toggleTask}
            />
          </div>

          {/* Key Concepts Card */}
          <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
            <div className="flex items-center gap-2 mb-3">
              <BookOpen size={18} className="text-emerald-400" />
              <h2 className="text-lg font-semibold text-white">Key Concepts</h2>
            </div>
            <ul className="space-y-2">
              {lecture.topics.map((topic, i) => (
                <li key={i} className="flex items-start gap-2 text-gray-300 text-sm">
                  <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full mt-1.5 flex-shrink-0" />
                  {topic}
                </li>
              ))}
            </ul>
          </div>

          {/* Resources Card */}
          {lecture.resources && lecture.resources.length > 0 && (
            <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
              <div className="flex items-center gap-2 mb-3">
                <ExternalLink size={18} className="text-emerald-400" />
                <h2 className="text-lg font-semibold text-white">Resources</h2>
              </div>
              <div className="space-y-2">
                {lecture.resources.map((resource, i) => (
                  <a
                    key={i}
                    href={resource.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 text-sm text-emerald-400 hover:text-emerald-300 transition-colors"
                  >
                    <ExternalLink size={12} />
                    {resource.label}
                  </a>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex justify-between mt-12 pt-8 border-t border-gray-800">
        {prevLecture ? (
          <Link
            to={`/lecture/${prevLecture.id}`}
            className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors group"
          >
            <ArrowLeft size={16} className="group-hover:-translate-x-1 transition-transform" />
            <div className="text-left">
              <p className="text-xs text-gray-500">Previous</p>
              <p>{prevLecture.title}</p>
            </div>
          </Link>
        ) : (
          <div />
        )}
        {nextLecture ? (
          <Link
            to={`/lecture/${nextLecture.id}`}
            className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors group"
          >
            <div className="text-right">
              <p className="text-xs text-gray-500">Next</p>
              <p>{nextLecture.title}</p>
            </div>
            <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform" />
          </Link>
        ) : (
          <div />
        )}
      </nav>
    </div>
  );
}
