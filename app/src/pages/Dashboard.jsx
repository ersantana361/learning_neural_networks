import { course } from '../data/course';
import { LectureCard } from '../components/LectureCard';
import { ProgressBar } from '../components/ProgressBar';
import { BookOpen, Clock, ExternalLink } from 'lucide-react';

export function Dashboard({ getLectureProgress, isLectureComplete, getOverallProgress }) {
  const overall = getOverallProgress();

  return (
    <div className="p-8 max-w-6xl mx-auto">
      {/* Hero */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-white mb-2">
          {course.title}
        </h1>
        <p className="text-lg text-gray-400 mb-4">
          {course.description}
        </p>

        <div className="flex flex-wrap items-center gap-6 text-sm text-gray-400">
          <div className="flex items-center gap-2">
            <BookOpen size={16} />
            <span>8 Lectures</span>
          </div>
          <div className="flex items-center gap-2">
            <Clock size={16} />
            <span>{course.totalDuration}</span>
          </div>
          <a
            href={course.website}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-emerald-400 hover:text-emerald-300 transition-colors"
          >
            <span>Course Website</span>
            <ExternalLink size={14} />
          </a>
        </div>
      </div>

      {/* Overall Progress Card */}
      <div className="bg-gradient-to-r from-emerald-900/40 to-gray-800 rounded-xl p-6 mb-8 border border-emerald-500/30">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-white">Your Progress</h2>
          <span className="text-3xl font-bold text-emerald-400">
            {overall.percentage}%
          </span>
        </div>
        <ProgressBar
          completed={overall.completed}
          total={overall.total}
          size="lg"
        />
      </div>

      {/* Lecture Parts */}
      {course.parts.map((part) => (
        <section key={part.id} className="mb-10">
          <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
            <span className="w-1.5 h-6 bg-emerald-500 rounded-full" />
            {part.title}
          </h2>

          <div className="grid gap-4 md:grid-cols-2">
            {part.lectures.map((lecture) => (
              <LectureCard
                key={lecture.id}
                lecture={lecture}
                progress={getLectureProgress(lecture.id)}
                isComplete={isLectureComplete(lecture.id)}
              />
            ))}
          </div>
        </section>
      ))}

      {/* Resources */}
      <section className="mt-12 p-6 bg-gray-800 rounded-xl border border-gray-700">
        <h2 className="text-lg font-semibold text-white mb-4">Official Repositories</h2>
        <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
          {[
            { name: 'micrograd', desc: 'Tiny autograd engine', url: 'https://github.com/karpathy/micrograd' },
            { name: 'makemore', desc: 'Character-level LM', url: 'https://github.com/karpathy/makemore' },
            { name: 'nanoGPT', desc: 'Minimal GPT', url: 'https://github.com/karpathy/nanoGPT' },
            { name: 'minbpe', desc: 'Minimal BPE tokenizer', url: 'https://github.com/karpathy/minbpe' },
          ].map((repo) => (
            <a
              key={repo.name}
              href={repo.url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex flex-col p-3 bg-gray-700 rounded-lg hover:bg-gray-600 transition-colors group"
            >
              <span className="font-medium text-white group-hover:text-emerald-300 transition-colors">
                {repo.name}
              </span>
              <span className="text-sm text-gray-400">{repo.desc}</span>
            </a>
          ))}
        </div>
      </section>
    </div>
  );
}
