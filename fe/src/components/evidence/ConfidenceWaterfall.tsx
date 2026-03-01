import { useMemo } from 'react';
import type { EvidenceItem } from '../../types';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ConfidenceWaterfallProps {
  evidences: EvidenceItem[];
}

interface SourceSegment {
  source: string;
  count: number;
  avgConfidence: number;
  /** Weighted contribution: avgConfidence * count */
  contribution: number;
  /** Percentage of total contribution */
  percentage: number;
  color: string;
}

// ---------------------------------------------------------------------------
// Source color mapping (matches EvidenceCard conventions)
// ---------------------------------------------------------------------------

const SOURCE_COLORS: Record<string, { bg: string; bar: string; text: string }> = {
  exif:          { bg: 'bg-purple-100', bar: 'bg-purple-500', text: 'text-purple-700' },
  vlm_analysis:  { bg: 'bg-blue-100',   bar: 'bg-blue-500',   text: 'text-blue-700' },
  vlm_geo:       { bg: 'bg-blue-100',   bar: 'bg-blue-600',   text: 'text-blue-700' },
  ocr:           { bg: 'bg-cyan-100',    bar: 'bg-cyan-500',   text: 'text-cyan-700' },
  geoclip:       { bg: 'bg-green-100',   bar: 'bg-green-500',  text: 'text-green-700' },
  streetclip:    { bg: 'bg-green-100',   bar: 'bg-green-600',  text: 'text-green-700' },
  osv5m:         { bg: 'bg-emerald-100', bar: 'bg-emerald-500', text: 'text-emerald-700' },
  serper:        { bg: 'bg-orange-100',  bar: 'bg-orange-500', text: 'text-orange-700' },
  google_maps:   { bg: 'bg-red-100',     bar: 'bg-red-500',    text: 'text-red-700' },
  osm:           { bg: 'bg-yellow-100',  bar: 'bg-yellow-500', text: 'text-yellow-700' },
  browser:       { bg: 'bg-gray-100',    bar: 'bg-gray-500',   text: 'text-gray-700' },
  reasoning:     { bg: 'bg-indigo-100',  bar: 'bg-indigo-500', text: 'text-indigo-700' },
  user_hint:     { bg: 'bg-pink-100',    bar: 'bg-pink-500',   text: 'text-pink-700' },
};

function getSourceColors(source: string) {
  return SOURCE_COLORS[source] ?? { bg: 'bg-gray-100', bar: 'bg-gray-400', text: 'text-gray-600' };
}

// ---------------------------------------------------------------------------
// ConfidenceWaterfall
// ---------------------------------------------------------------------------

export default function ConfidenceWaterfall({ evidences }: ConfidenceWaterfallProps) {
  const { segments, totalContribution } = useMemo(() => {
    // Group by source
    const groups = new Map<string, EvidenceItem[]>();
    for (const ev of evidences) {
      const list = groups.get(ev.source) ?? [];
      list.push(ev);
      groups.set(ev.source, list);
    }

    // Calculate contributions
    const raw: Omit<SourceSegment, 'percentage'>[] = [];
    let total = 0;

    for (const [source, items] of groups) {
      const avg = items.reduce((s, e) => s + e.confidence, 0) / items.length;
      const contribution = avg * items.length;
      total += contribution;
      raw.push({
        source,
        count: items.length,
        avgConfidence: avg,
        contribution,
        color: getSourceColors(source).bar,
      });
    }

    // Sort by contribution descending
    raw.sort((a, b) => b.contribution - a.contribution);

    const segs: SourceSegment[] = raw.map((r) => ({
      ...r,
      percentage: total > 0 ? (r.contribution / total) * 100 : 0,
    }));

    return { segments: segs, totalContribution: total };
  }, [evidences]);

  if (evidences.length === 0) {
    return (
      <div className="rounded-xl border border-gray-200 bg-white p-8 text-center">
        <p className="text-sm text-gray-500">No evidence data available.</p>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-gray-200 bg-white overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200">
        <h3 className="text-sm font-semibold text-gray-900">Confidence Breakdown</h3>
        <p className="text-xs text-gray-500 mt-0.5">
          Contribution of each source (avg confidence x count)
        </p>
      </div>

      <div className="p-4 space-y-3">
        {/* Stacked bar */}
        <div className="rounded-full overflow-hidden h-6 flex bg-gray-100">
          {segments.map((seg) => (
            <div
              key={seg.source}
              className={`${seg.color} transition-all duration-300 relative group`}
              style={{ width: `${seg.percentage}%`, minWidth: seg.percentage > 0 ? 4 : 0 }}
              title={`${seg.source}: ${seg.percentage.toFixed(1)}%`}
            >
              {/* Tooltip on hover */}
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block z-10">
                <div className="bg-gray-900 text-white text-[10px] rounded px-2 py-1 whitespace-nowrap">
                  {seg.source}: {seg.percentage.toFixed(1)}%
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Legend / detail rows */}
        <div className="space-y-2">
          {segments.map((seg) => {
            const colors = getSourceColors(seg.source);
            return (
              <div key={seg.source} className="flex items-center gap-3">
                {/* Color swatch */}
                <div className={`flex-shrink-0 h-3 w-3 rounded-sm ${seg.color}`} />

                {/* Source name */}
                <span className={`text-xs font-semibold font-mono w-24 truncate ${colors.text}`}>
                  {seg.source}
                </span>

                {/* Bar */}
                <div className="flex-1 h-5 bg-gray-100 rounded-full overflow-hidden relative">
                  <div
                    className={`h-full rounded-full ${seg.color} transition-all duration-500`}
                    style={{ width: `${seg.percentage}%` }}
                  />
                </div>

                {/* Values */}
                <div className="flex items-center gap-2 flex-shrink-0 text-[11px]">
                  <span className="text-gray-500">
                    {seg.count}x
                  </span>
                  <span className="text-gray-500">
                    avg {(seg.avgConfidence * 100).toFixed(0)}%
                  </span>
                  <span className="font-semibold text-gray-800 w-12 text-right">
                    {seg.percentage.toFixed(1)}%
                  </span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Total bar */}
        <div className="border-t border-gray-200 pt-3 mt-3">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-xs font-semibold text-gray-700">
              Total Weighted Score
            </span>
            <span className="text-xs font-bold text-gray-900">
              {totalContribution.toFixed(2)}
            </span>
          </div>
          <div className="h-3 bg-gray-100 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full bg-gradient-to-r from-blue-500 to-blue-600 transition-all duration-500"
              style={{
                width: `${Math.min((totalContribution / Math.max(evidences.length, 1)) * 100, 100)}%`,
              }}
            />
          </div>
          <div className="flex justify-between mt-1 text-[10px] text-gray-400">
            <span>0</span>
            <span>
              Average: {evidences.length > 0 ? ((totalContribution / evidences.length) * 100).toFixed(1) : '0'}%
            </span>
            <span>100%</span>
          </div>
        </div>
      </div>
    </div>
  );
}
