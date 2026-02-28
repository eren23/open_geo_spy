import { useState } from 'react';
import type { EvidenceItem } from '../../types';
import ConfidenceBadge from './ConfidenceBadge';

// ---------------------------------------------------------------------------
// Source badge color mapping
// ---------------------------------------------------------------------------

const SOURCE_STYLES: Record<string, string> = {
  exif: 'bg-purple-100 text-purple-700',
  vlm_analysis: 'bg-blue-100 text-blue-700',
  vlm_geo: 'bg-blue-100 text-blue-700',
  ocr: 'bg-cyan-100 text-cyan-700',
  geoclip: 'bg-green-100 text-green-700',
  streetclip: 'bg-green-100 text-green-700',
  serper: 'bg-orange-100 text-orange-700',
  google_maps: 'bg-red-100 text-red-700',
  osm: 'bg-yellow-100 text-yellow-700',
  browser: 'bg-gray-100 text-gray-700',
  reasoning: 'bg-indigo-100 text-indigo-700',
  user_hint: 'bg-pink-100 text-pink-700',
};

function sourceStyle(source: string): string {
  return SOURCE_STYLES[source] ?? 'bg-gray-100 text-gray-600';
}

// ---------------------------------------------------------------------------
// EvidenceCard
// ---------------------------------------------------------------------------

interface EvidenceCardProps {
  evidence: EvidenceItem;
}

export default function EvidenceCard({ evidence }: EvidenceCardProps) {
  const [expanded, setExpanded] = useState(false);

  const locationParts = [evidence.city, evidence.region, evidence.country].filter(Boolean);
  const hasCoords = evidence.latitude != null && evidence.longitude != null;

  return (
    <div className="rounded-lg border border-gray-200 bg-white overflow-hidden transition-shadow hover:shadow-sm">
      {/* Header - always visible */}
      <button
        type="button"
        onClick={() => setExpanded((prev) => !prev)}
        className="w-full flex items-start gap-3 px-4 py-3 text-left focus:outline-none focus:ring-2 focus:ring-inset focus:ring-blue-500"
      >
        {/* Source badge */}
        <span
          className={`flex-shrink-0 mt-0.5 rounded px-2 py-0.5 text-[11px] font-mono font-semibold ${sourceStyle(evidence.source)}`}
        >
          {evidence.source}
        </span>

        {/* Content preview */}
        <span className={`flex-grow text-sm text-gray-700 ${expanded ? '' : 'line-clamp-2'}`}>
          {evidence.content}
        </span>

        {/* Right side: confidence + chevron */}
        <span className="flex items-center gap-2 flex-shrink-0">
          <ConfidenceBadge confidence={evidence.confidence} />
          <svg
            className={`h-4 w-4 text-gray-400 transition-transform ${expanded ? 'rotate-180' : ''}`}
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={2}
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
          </svg>
        </span>
      </button>

      {/* Expanded details */}
      {expanded && (
        <div className="border-t border-gray-100 px-4 py-3 bg-gray-50 space-y-2">
          {/* Confidence bar */}
          <div>
            <span className="text-[11px] font-semibold uppercase tracking-wider text-gray-400">
              Confidence
            </span>
            <div className="mt-1 h-2 w-full rounded-full bg-gray-200 overflow-hidden">
              <div
                className={`h-full rounded-full transition-all ${
                  evidence.confidence > 0.7
                    ? 'bg-green-500'
                    : evidence.confidence > 0.4
                      ? 'bg-yellow-500'
                      : 'bg-red-500'
                }`}
                style={{ width: `${(evidence.confidence * 100).toFixed(1)}%` }}
              />
            </div>
            <p className="text-xs text-gray-500 mt-0.5">
              {(evidence.confidence * 100).toFixed(1)}%
            </p>
          </div>

          {/* Location info */}
          {locationParts.length > 0 && (
            <div>
              <span className="text-[11px] font-semibold uppercase tracking-wider text-gray-400">
                Location
              </span>
              <p className="text-sm text-gray-700">{locationParts.join(', ')}</p>
            </div>
          )}

          {/* Coordinates */}
          {hasCoords && (
            <div>
              <span className="text-[11px] font-semibold uppercase tracking-wider text-gray-400">
                Coordinates
              </span>
              <p className="text-sm font-mono text-gray-600">
                {evidence.latitude!.toFixed(6)}, {evidence.longitude!.toFixed(6)}
              </p>
            </div>
          )}

          {/* Full content */}
          <div>
            <span className="text-[11px] font-semibold uppercase tracking-wider text-gray-400">
              Full Content
            </span>
            <p className="text-sm text-gray-700 whitespace-pre-wrap mt-0.5">
              {evidence.content}
            </p>
          </div>

          {/* URL link */}
          {evidence.url && (
            <div>
              <span className="text-[11px] font-semibold uppercase tracking-wider text-gray-400">
                Source URL
              </span>
              <a
                href={evidence.url}
                target="_blank"
                rel="noopener noreferrer"
                className="block text-sm text-blue-600 hover:text-blue-800 hover:underline truncate mt-0.5"
              >
                {evidence.url}
              </a>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
