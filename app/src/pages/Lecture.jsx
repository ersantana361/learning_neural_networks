import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, ArrowRight, Clock, ExternalLink } from 'lucide-react';
import { getLectureById, getAllLectures, getPartForLecture } from '../data/course';
import { VideoPlayer } from '../components/VideoPlayer';
import { TaskList } from '../components/TaskList';
import { NotesEditor } from '../components/NotesEditor';
import { CodeViewer } from '../components/CodeViewer';
import { ProgressBar } from '../components/ProgressBar';

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

  return (
    <div className="p-8 max-w-5xl mx-auto">
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
        <h1 className="text-3xl font-bold text-white mb-2">
          {lecture.id}. {lecture.title}
        </h1>
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

      {/* Video */}
      <section className="mb-8">
        <VideoPlayer videoId={lecture.videoId} title={lecture.title} />
      </section>

      {/* Two Column Layout */}
      <div className="grid gap-8 lg:grid-cols-2">
        {/* Left Column */}
        <div className="space-y-8">
          {/* Tasks */}
          <section>
            <h2 className="text-lg font-semibold text-white mb-3">Tasks</h2>
            <TaskList
              tasks={lecture.tasks}
              isTaskComplete={isTaskComplete}
              onToggle={toggleTask}
            />
          </section>

          {/* Topics */}
          <section>
            <h2 className="text-lg font-semibold text-white mb-3">Key Concepts</h2>
            <ul className="space-y-2">
              {lecture.topics.map((topic, i) => (
                <li key={i} className="flex items-start gap-2 text-gray-300">
                  <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full mt-2 flex-shrink-0" />
                  {topic}
                </li>
              ))}
            </ul>
          </section>

          {/* Resources */}
          {lecture.resources && lecture.resources.length > 0 && (
            <section>
              <h2 className="text-lg font-semibold text-white mb-3">Resources</h2>
              <div className="space-y-2">
                {lecture.resources.map((resource, i) => (
                  <a
                    key={i}
                    href={resource.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 text-emerald-400 hover:text-emerald-300 transition-colors"
                  >
                    <ExternalLink size={14} />
                    {resource.label}
                  </a>
                ))}
              </div>
            </section>
          )}
        </div>

        {/* Right Column - Notes */}
        <div>
          <NotesEditor
            lectureId={lecture.id}
            value={getNote(lecture.id)}
            onChange={updateNote}
          />
        </div>
      </div>

      {/* Code Viewer */}
      {lecture.codeFiles && lecture.codeFiles.length > 0 && (
        <section className="mt-8">
          <h2 className="text-lg font-semibold text-white mb-3">Implementation</h2>
          <CodeViewer files={lecture.codeFiles} />
        </section>
      )}

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
