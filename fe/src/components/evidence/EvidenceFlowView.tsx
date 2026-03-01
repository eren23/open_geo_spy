import { useEffect, useRef } from 'react';
import type { LiveEvidence } from '../../types';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface EvidenceFlowViewProps {
  evidences: LiveEvidence[];
  /** Maximum items to display (newest first). */
  maxItems?: number;
  loading?: boolean;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SOURCE_COLORS: Record<string, string> = {
  exif: 'bg-purple-100 text-purple-700',
  vlm_analysis: 'bg-purple-100 text-purple-700',
  vlm_geo: 'bg-purple-100 text-purple-700',
  ocr: 'bg-purple-100 text-purple-700',
  geoclip: 'bg-green-100 text-green-700',
  streetclip: 'bg-green-100 text-green-700',
  serper: 'bg-orange-100 text-orange-700',
  google_maps: 'bg-orange-100 text-orange-700',
  osm: 'bg-orange-100 text-orange-700',
  browser: 'bg-orange-100 text-orange-700',
  reasoning: 'bg-indigo-100 text-indigo-700',
  user_hint: 'bg-indigo-100 text-indigo-700',
};

const DEFAULT_SOURCE_COLOR = 'bg-gray-100 text-gray-600';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function confidenceColor(confidence: number): string {
  if (confidence >= 0.7) return 'text-green-600';
  if (confidence >= 0.4) return 'text-yellow-600';
  return 'text-red-500';
}

function truncate(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen - 1) + '\u2026';
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function EvidenceFlowView({ evidences, maxItems = 20, loading = false }: EvidenceFlowViewProps) {
  const listRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new evidences arrive
  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [evidences.length]);

  if (evidences.length === 0 && !loading) return null;

  const visible = evidences.slice(-maxItems);

  return (
    <div className="rounded-lg border border-gray-200 bg-white overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-100">
        <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
          Evidence Feed
        </h4>
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-gray-400">{evidences.length} items</span>
          {loading && (
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500" />
            </span>
          )}
        </div>
      </div>

      {/* Evidence list */}
      <div
        ref={listRef}
        className="max-h-60 overflow-y-auto divide-y divide-gray-50"
      >
        {visible.map((ev) => (
          <div
            key={ev.id}
            className="px-3 py-2 hover:bg-gray-50 transition-colors animate-[fadeIn_0.3s_ease-out]"
          >
            <div className="flex items-center gap-2">
              <span
                className={`inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-medium ${
                  SOURCE_COLORS[ev.source] ?? DEFAULT_SOURCE_COLOR
                }`}
              >
                {ev.source}
              </span>
              <span className={`text-[10px] font-medium ${confidenceColor(ev.confidence)}`}>
                {(ev.confidence * 100).toFixed(0)}%
              </span>
            </div>
            <p className="text-[11px] text-gray-600 mt-0.5 leading-tight">
              {truncate(ev.content, 120)}
            </p>
          </div>
        ))}

        {loading && evidences.length === 0 && (
          <div className="px-3 py-4 text-center text-[11px] text-gray-400">
            Waiting for evidence...
          </div>
        )}
      </div>
    </div>
  );
}
