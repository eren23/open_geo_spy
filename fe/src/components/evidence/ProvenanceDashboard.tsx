import { useMemo } from 'react';
import type { EvidenceItem } from '../../types';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ProvenanceDashboardProps {
  evidences: EvidenceItem[];
}

// ---------------------------------------------------------------------------
// Pipeline stages definition
// ---------------------------------------------------------------------------

const PIPELINE_STAGES = [
  {
    id: 'feature_extraction',
    label: 'Feature Extraction',
    sources: ['exif', 'vlm_analysis', 'vlm_geo', 'ocr'],
    color: 'bg-purple-500',
    lightColor: 'bg-purple-100',
    textColor: 'text-purple-700',
    borderColor: 'border-purple-300',
  },
  {
    id: 'ml_ensemble',
    label: 'ML Ensemble',
    sources: ['geoclip', 'streetclip', 'osv5m'],
    color: 'bg-green-500',
    lightColor: 'bg-green-100',
    textColor: 'text-green-700',
    borderColor: 'border-green-300',
  },
  {
    id: 'web_intelligence',
    label: 'Web Intelligence',
    sources: ['serper', 'google_maps', 'osm', 'browser'],
    color: 'bg-orange-500',
    lightColor: 'bg-orange-100',
    textColor: 'text-orange-700',
    borderColor: 'border-orange-300',
  },
  {
    id: 'reasoning',
    label: 'Reasoning',
    sources: ['reasoning', 'user_hint'],
    color: 'bg-indigo-500',
    lightColor: 'bg-indigo-100',
    textColor: 'text-indigo-700',
    borderColor: 'border-indigo-300',
  },
] as const;

// ---------------------------------------------------------------------------
// Source -> stage mapping
// ---------------------------------------------------------------------------

function getStageForSource(source: string): string {
  for (const stage of PIPELINE_STAGES) {
    if ((stage.sources as readonly string[]).includes(source)) return stage.id;
  }
  return 'unknown';
}

// ---------------------------------------------------------------------------
// ProvenanceDashboard
// ---------------------------------------------------------------------------

export default function ProvenanceDashboard({ evidences }: ProvenanceDashboardProps) {
  // Group evidence by source
  const sourceGroups = useMemo(() => {
    const groups = new Map<string, EvidenceItem[]>();
    for (const ev of evidences) {
      const list = groups.get(ev.source) ?? [];
      list.push(ev);
      groups.set(ev.source, list);
    }
    return groups;
  }, [evidences]);

  // Group sources by stage
  const stageGroups = useMemo(() => {
    const groups = new Map<string, { sources: Map<string, number>; total: number; avgConfidence: number }>();
    for (const stage of PIPELINE_STAGES) {
      groups.set(stage.id, { sources: new Map(), total: 0, avgConfidence: 0 });
    }
    groups.set('unknown', { sources: new Map(), total: 0, avgConfidence: 0 });

    for (const [source, items] of sourceGroups) {
      const stageId = getStageForSource(source);
      const group = groups.get(stageId)!;
      group.sources.set(source, items.length);
      group.total += items.length;
      const totalConf = items.reduce((s, e) => s + e.confidence, 0);
      group.avgConfidence =
        group.total > 0
          ? (group.avgConfidence * (group.total - items.length) + totalConf) / group.total
          : 0;
    }

    return groups;
  }, [sourceGroups]);

  // Final aggregated confidence
  const finalConfidence = useMemo(() => {
    if (evidences.length === 0) return 0;
    const total = evidences.reduce((s, e) => s + e.confidence, 0);
    return total / evidences.length;
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
        <h3 className="text-sm font-semibold text-gray-900">Evidence Provenance</h3>
        <p className="text-xs text-gray-500 mt-0.5">
          Flow of evidence through the pipeline stages
        </p>
      </div>

      {/* Three-column flow layout */}
      <div className="grid grid-cols-[1fr_auto_1fr_auto_1fr] items-stretch min-h-[280px]">
        {/* LEFT COLUMN: Evidence Sources */}
        <div className="p-4 space-y-2">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-3">
            Sources ({sourceGroups.size})
          </h4>
          {Array.from(sourceGroups.entries())
            .sort((a, b) => b[1].length - a[1].length)
            .map(([source, items]) => {
              const stageId = getStageForSource(source);
              const stage = PIPELINE_STAGES.find((s) => s.id === stageId);
              return (
                <div
                  key={source}
                  className={`flex items-center justify-between rounded-lg border px-3 py-2 ${stage?.borderColor ?? 'border-gray-200'} ${stage?.lightColor ?? 'bg-gray-50'}`}
                >
                  <span className={`text-xs font-semibold font-mono ${stage?.textColor ?? 'text-gray-600'}`}>
                    {source}
                  </span>
                  <span className={`text-xs font-bold ${stage?.textColor ?? 'text-gray-600'}`}>
                    {items.length}
                  </span>
                </div>
              );
            })}
        </div>

        {/* Arrow connector */}
        <div className="flex items-center px-2">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" className="text-gray-300">
            <path d="M5 12h14m0 0l-4-4m4 4l-4 4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>

        {/* MIDDLE COLUMN: Pipeline Stages */}
        <div className="p-4 space-y-3">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-3">
            Pipeline Stages
          </h4>
          {PIPELINE_STAGES.map((stage, idx) => {
            const group = stageGroups.get(stage.id);
            const count = group?.total ?? 0;
            const hasData = count > 0;

            return (
              <div key={stage.id}>
                {/* Stage card */}
                <div
                  className={`rounded-lg border-2 px-4 py-3 transition-colors ${
                    hasData ? stage.borderColor : 'border-gray-200'
                  } ${hasData ? 'bg-white' : 'bg-gray-50'}`}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <div className={`h-2.5 w-2.5 rounded-full ${hasData ? stage.color : 'bg-gray-300'}`} />
                    <span className={`text-xs font-semibold ${hasData ? 'text-gray-900' : 'text-gray-400'}`}>
                      {stage.label}
                    </span>
                  </div>
                  <div className="flex items-center gap-3 text-[11px]">
                    <span className={hasData ? 'text-gray-600' : 'text-gray-400'}>
                      {count} evidence{count !== 1 ? 's' : ''}
                    </span>
                    {hasData && group!.avgConfidence > 0 && (
                      <span className="text-gray-400">
                        avg {(group!.avgConfidence * 100).toFixed(0)}%
                      </span>
                    )}
                  </div>
                  {/* Source breakdown bar */}
                  {hasData && (
                    <div className="mt-2 flex rounded-full overflow-hidden h-1.5">
                      {Array.from(group!.sources.entries()).map(([src, cnt]) => (
                        <div
                          key={src}
                          className={stage.color}
                          style={{
                            width: `${(cnt / count) * 100}%`,
                            opacity: 0.5 + 0.5 * (cnt / count),
                          }}
                          title={`${src}: ${cnt}`}
                        />
                      ))}
                    </div>
                  )}
                </div>

                {/* Connector arrow between stages */}
                {idx < PIPELINE_STAGES.length - 1 && (
                  <div className="flex justify-center py-1">
                    <svg width="12" height="16" viewBox="0 0 12 16" fill="none" className="text-gray-300">
                      <path d="M6 0v12m0 0l-4-4m4 4l4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Arrow connector */}
        <div className="flex items-center px-2">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" className="text-gray-300">
            <path d="M5 12h14m0 0l-4-4m4 4l-4 4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>

        {/* RIGHT COLUMN: Final Prediction */}
        <div className="p-4 flex flex-col justify-center">
          <h4 className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-3">
            Output
          </h4>
          <div className="rounded-xl border-2 border-blue-300 bg-blue-50 p-4 text-center">
            <div className="mb-3">
              <span className="text-xs text-blue-600 font-semibold uppercase tracking-wider">
                Final Prediction
              </span>
            </div>
            {/* Confidence ring */}
            <div className="relative mx-auto mb-3" style={{ width: 80, height: 80 }}>
              <svg viewBox="0 0 36 36" className="w-full h-full">
                <path
                  d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                  fill="none"
                  stroke="#E5E7EB"
                  strokeWidth="3"
                />
                <path
                  d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                  fill="none"
                  stroke={finalConfidence > 0.7 ? '#22C55E' : finalConfidence > 0.4 ? '#EAB308' : '#EF4444'}
                  strokeWidth="3"
                  strokeDasharray={`${finalConfidence * 100}, 100`}
                  strokeLinecap="round"
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-lg font-bold text-gray-900">
                  {(finalConfidence * 100).toFixed(0)}%
                </span>
              </div>
            </div>
            <div className="space-y-1 text-xs text-gray-600">
              <p>
                <span className="font-medium">{evidences.length}</span> total evidence items
              </p>
              <p>
                <span className="font-medium">{sourceGroups.size}</span> unique sources
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
