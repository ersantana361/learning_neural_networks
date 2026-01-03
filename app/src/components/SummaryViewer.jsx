import { useState, useEffect } from 'react';
import { FileText, Clock, Tag, ChevronDown, ChevronRight, Quote, Lightbulb, BookOpen, Target } from 'lucide-react';

function parseFrontmatter(content) {
  const match = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
  if (!match) return { tags: [], body: content };

  const frontmatter = match[1];
  const body = match[2];

  const tags = [];
  const tagMatches = frontmatter.matchAll(/^\s+-\s+(.+)$/gm);
  for (const m of tagMatches) {
    tags.push(m[1].trim());
  }

  return { tags, body };
}

function parseMarkdown(body) {
  const lines = body.split('\n');
  const sections = [];
  let currentSection = null;
  let currentSegment = null;
  let collectingExcerpts = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmedLine = line.trim();

    // Section headers: *Introduction*, *Detailed Analysis*, *Conclusion*
    if (trimmedLine === '*Introduction*') {
      currentSection = { type: 'introduction', title: 'Introduction', items: [] };
      sections.push(currentSection);
      currentSegment = null;
      continue;
    }
    if (trimmedLine === '*Detailed Analysis*') {
      currentSection = { type: 'detailed', title: 'Detailed Analysis', segments: [] };
      sections.push(currentSection);
      currentSegment = null;
      continue;
    }
    if (trimmedLine === '*Conclusion*') {
      currentSection = { type: 'conclusion', title: 'Conclusion', items: [] };
      sections.push(currentSection);
      currentSegment = null;
      continue;
    }

    if (!currentSection) continue;

    // Segment headers in Detailed Analysis: • **Segment: Title (timestamp)**
    const segmentMatch = trimmedLine.match(/^•\s*\*\*Segment:\s*(.+?)\s*\((\d+:\d+.*?)\)\*\*$/);
    if (segmentMatch && currentSection.type === 'detailed') {
      currentSegment = {
        title: segmentMatch[1].trim(),
        timestamp: segmentMatch[2].trim(),
        excerpts: [],
        analysis: ''
      };
      currentSection.segments.push(currentSegment);
      collectingExcerpts = false;
      continue;
    }

    // Inside a segment
    if (currentSegment) {
      // Check for numbered items
      const numberedMatch = trimmedLine.match(/^(\d+)\.\s*\*(.+?)\*:\s*(.*)$/);
      if (numberedMatch) {
        const label = numberedMatch[2].trim();
        const content = numberedMatch[3].trim();

        if (label === 'Key verbatim excerpts') {
          collectingExcerpts = true;
        } else if (label === 'Technical analysis and implications') {
          collectingExcerpts = false;
          currentSegment.analysis = content;
        }
        continue;
      }

      // Excerpt bullets: • "quote"
      const excerptMatch = trimmedLine.match(/^•\s*"(.+)"$/);
      if (excerptMatch && collectingExcerpts) {
        currentSegment.excerpts.push(excerptMatch[1]);
        continue;
      }
    }

    // Introduction/Conclusion bullet points: • *Label*: Content
    const bulletMatch = trimmedLine.match(/^•\s*\*(.+?)\*:\s*(.+)$/);
    if (bulletMatch && (currentSection.type === 'introduction' || currentSection.type === 'conclusion')) {
      currentSection.items.push({
        label: bulletMatch[1].trim(),
        content: bulletMatch[2].trim()
      });
      continue;
    }
  }

  return sections;
}

function formatText(text) {
  // Convert markdown-style formatting to JSX
  // Handle `code` backticks
  const parts = text.split(/(`[^`]+`)/g);
  return parts.map((part, i) => {
    if (part.startsWith('`') && part.endsWith('`')) {
      return (
        <code key={i} className="px-1.5 py-0.5 bg-gray-700 rounded text-emerald-300 text-xs font-mono">
          {part.slice(1, -1)}
        </code>
      );
    }
    // Handle *italic* asterisks
    const italicParts = part.split(/(\*[^*]+\*)/g);
    return italicParts.map((ip, j) => {
      if (ip.startsWith('*') && ip.endsWith('*')) {
        return <em key={`${i}-${j}`} className="text-emerald-200">{ip.slice(1, -1)}</em>;
      }
      return ip;
    });
  });
}

function IntroductionSection({ section, isExpanded, onToggle }) {
  return (
    <div className="border border-gray-700 rounded-xl overflow-hidden">
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between p-4 bg-gradient-to-r from-emerald-900/30 to-gray-800/50 hover:from-emerald-900/40 transition-colors"
      >
        <div className="flex items-center gap-3">
          <Target size={20} className="text-emerald-400" />
          <h3 className="text-lg font-semibold text-white">{section.title}</h3>
        </div>
        {isExpanded ? (
          <ChevronDown size={20} className="text-gray-400" />
        ) : (
          <ChevronRight size={20} className="text-gray-400" />
        )}
      </button>
      {isExpanded && (
        <div className="p-5 space-y-4 bg-gray-900/30">
          {section.items.map((item, i) => (
            <div key={i} className="flex gap-3">
              <div className="flex-shrink-0 w-1 bg-emerald-500/50 rounded-full" />
              <div>
                <span className="text-emerald-400 font-medium text-sm">{item.label}</span>
                <p className="text-gray-300 text-sm leading-relaxed mt-1">
                  {formatText(item.content)}
                </p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function SegmentCard({ segment, isExpanded, onToggle }) {
  return (
    <div className="border border-gray-700 rounded-lg overflow-hidden bg-gray-800/30">
      <button
        onClick={onToggle}
        className="w-full flex items-start gap-3 p-4 hover:bg-gray-800/50 transition-colors text-left"
      >
        <div className="flex-shrink-0 mt-0.5">
          {isExpanded ? (
            <ChevronDown size={16} className="text-gray-400" />
          ) : (
            <ChevronRight size={16} className="text-gray-400" />
          )}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <Clock size={14} className="text-emerald-400 flex-shrink-0" />
            <span className="text-emerald-400 font-mono text-sm">{segment.timestamp}</span>
          </div>
          <h4 className="text-white font-medium">{segment.title}</h4>
        </div>
      </button>

      {isExpanded && (
        <div className="px-4 pb-4 space-y-4">
          {/* Key Excerpts */}
          {segment.excerpts.length > 0 && (
            <div className="pl-7">
              <div className="flex items-center gap-2 mb-2">
                <Quote size={14} className="text-yellow-500" />
                <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide">Key Excerpts</span>
              </div>
              <div className="space-y-2">
                {segment.excerpts.map((excerpt, i) => (
                  <blockquote
                    key={i}
                    className="border-l-2 border-yellow-500/50 pl-3 py-1 text-gray-300 text-sm italic bg-yellow-500/5 rounded-r"
                  >
                    "{excerpt}"
                  </blockquote>
                ))}
              </div>
            </div>
          )}

          {/* Technical Analysis */}
          {segment.analysis && (
            <div className="pl-7">
              <div className="flex items-center gap-2 mb-2">
                <Lightbulb size={14} className="text-blue-400" />
                <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide">Technical Analysis</span>
              </div>
              <p className="text-gray-300 text-sm leading-relaxed">
                {formatText(segment.analysis)}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function DetailedSection({ section, expandedSegments, toggleSegment }) {
  return (
    <div className="border border-gray-700 rounded-xl overflow-hidden">
      <div className="p-4 bg-gradient-to-r from-blue-900/30 to-gray-800/50">
        <div className="flex items-center gap-3">
          <BookOpen size={20} className="text-blue-400" />
          <h3 className="text-lg font-semibold text-white">{section.title}</h3>
          <span className="text-xs text-gray-500 bg-gray-700 px-2 py-0.5 rounded-full">
            {section.segments.length} segments
          </span>
        </div>
      </div>
      <div className="p-4 space-y-3 bg-gray-900/20">
        {section.segments.map((segment, i) => (
          <SegmentCard
            key={i}
            segment={segment}
            isExpanded={expandedSegments.has(i)}
            onToggle={() => toggleSegment(i)}
          />
        ))}
      </div>
    </div>
  );
}

function ConclusionSection({ section, isExpanded, onToggle }) {
  return (
    <div className="border border-gray-700 rounded-xl overflow-hidden">
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between p-4 bg-gradient-to-r from-purple-900/30 to-gray-800/50 hover:from-purple-900/40 transition-colors"
      >
        <div className="flex items-center gap-3">
          <Lightbulb size={20} className="text-purple-400" />
          <h3 className="text-lg font-semibold text-white">{section.title}</h3>
        </div>
        {isExpanded ? (
          <ChevronDown size={20} className="text-gray-400" />
        ) : (
          <ChevronRight size={20} className="text-gray-400" />
        )}
      </button>
      {isExpanded && (
        <div className="p-5 space-y-4 bg-gray-900/30">
          {section.items.map((item, i) => (
            <div key={i} className="flex gap-3">
              <div className="flex-shrink-0 w-1 bg-purple-500/50 rounded-full" />
              <div>
                <span className="text-purple-400 font-medium text-sm">{item.label}</span>
                <p className="text-gray-300 text-sm leading-relaxed mt-1">
                  {formatText(item.content)}
                </p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function SummaryViewer({ summaryFile }) {
  const [content, setContent] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedSections, setExpandedSections] = useState(new Set(['introduction', 'detailed', 'conclusion']));
  const [expandedSegments, setExpandedSegments] = useState(new Set([0]));

  useEffect(() => {
    if (!summaryFile) {
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    fetch(`/summaries/${summaryFile}`)
      .then(res => {
        if (!res.ok) throw new Error('Summary not found');
        return res.text();
      })
      .then(text => {
        const { tags, body } = parseFrontmatter(text);
        const sections = parseMarkdown(body);
        setContent({ tags, sections });
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, [summaryFile]);

  const toggleSection = (type) => {
    setExpandedSections(prev => {
      const next = new Set(prev);
      if (next.has(type)) {
        next.delete(type);
      } else {
        next.add(type);
      }
      return next;
    });
  };

  const toggleSegment = (index) => {
    setExpandedSegments(prev => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  };

  if (!summaryFile) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-gray-500">
        <FileText size={48} className="mb-4 opacity-50" />
        <p>No summary available for this lecture</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-pulse flex items-center gap-2 text-gray-400">
          <div className="w-4 h-4 bg-emerald-500/50 rounded-full animate-bounce" />
          <span>Loading summary...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-red-400">
        <FileText size={48} className="mb-4 opacity-50" />
        <p>Error loading summary: {error}</p>
      </div>
    );
  }

  if (!content) return null;

  return (
    <div className="space-y-6">
      {/* Tags */}
      {content.tags.length > 0 && (
        <div className="p-4 bg-gray-800/50 rounded-xl border border-gray-700">
          <div className="flex items-center gap-2 mb-3">
            <Tag size={16} className="text-emerald-400" />
            <span className="text-sm font-medium text-gray-300">Topics Covered</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {content.tags.map((tag, i) => (
              <span
                key={i}
                className="inline-flex items-center px-2.5 py-1 bg-gray-700/50 border border-gray-600 rounded-full text-xs text-gray-300 hover:bg-gray-700 transition-colors"
              >
                {tag}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Sections */}
      <div className="space-y-4">
        {content.sections.map((section, i) => {
          if (section.type === 'introduction') {
            return (
              <IntroductionSection
                key={i}
                section={section}
                isExpanded={expandedSections.has('introduction')}
                onToggle={() => toggleSection('introduction')}
              />
            );
          }
          if (section.type === 'detailed') {
            return (
              <DetailedSection
                key={i}
                section={section}
                expandedSegments={expandedSegments}
                toggleSegment={toggleSegment}
              />
            );
          }
          if (section.type === 'conclusion') {
            return (
              <ConclusionSection
                key={i}
                section={section}
                isExpanded={expandedSections.has('conclusion')}
                onToggle={() => toggleSection('conclusion')}
              />
            );
          }
          return null;
        })}
      </div>
    </div>
  );
}
