import type { GroundingInfo, GroundingVerdict } from '../../types';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface GroundingViewProps {
  groundings: GroundingInfo[];
  /** Compact mode for sidebar embedding. */
  compact?: boolean;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const LEVEL_ORDER = ['continent', 'country', 'region', 'city', 'coordinates'];

const VERDICT_STYLES: Record<GroundingVerdict, { bg: string; text: string; border: string; icon: string }> = {
  GROUNDED: { bg: 'bg-green-50', text: 'text-green-700', border: 'border-green-300', icon: '\u2713\u2713' },
  SUPPORTED: { bg: 'bg-blue-50', text: 'text-blue-700', border: 'border-blue-300', icon: '\u2713' },
  UNCERTAIN: { bg: 'bg-yellow-50', text: 'text-yellow-700', border: 'border-yellow-300', icon: '?' },
  WEAKENED: { bg: 'bg-orange-50', text: 'text-orange-700', border: 'border-orange-300', icon: '\u25BC' },
  CONTRADICTED: { bg: 'bg-red-50', text: 'text-red-700', border: 'border-red-300', icon: '\u2717' },
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function levelLabel(level: string): string {
  return level.charAt(0).toUpperCase() + level.slice(1);
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function GroundingView({ groundings, compact = false }: GroundingViewProps) {
  if (groundings.length === 0) return null;

  // Sort by geographic hierarchy
  const sorted = [...groundings].sort(
    (a, b) => LEVEL_ORDER.indexOf(a.level) - LEVEL_ORDER.indexOf(b.level),
  );

  if (compact) {
    return (
      <div className="flex flex-col gap-1">
        {sorted.map((g) => {
          const style = VERDICT_STYLES[g.verdict] ?? VERDICT_STYLES.UNCERTAIN;
          return (
            <div
              key={g.level}
              className={`flex items-center justify-between rounded-md border px-2 py-1 ${style.border} ${style.bg}`}
            >
              <div className="flex items-center gap-1.5">
                <span className={`text-[10px] font-bold ${style.text}`}>{style.icon}</span>
                <span className="text-[11px] font-medium text-gray-700">{levelLabel(g.level)}</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="text-[11px] text-gray-600 truncate max-w-[100px]" title={g.value ?? undefined}>
                  {g.value || '\u2014'}
                </span>
                <span className={`text-[10px] font-semibold ${style.text}`}>
                  {(g.confidence * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          );
        })}
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-gray-200 bg-white overflow-hidden">
      {/* Header */}
      <div className="px-3 py-2 border-b border-gray-100">
        <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
          Grounding Verdicts
        </h4>
      </div>

      {/* Grounding levels */}
      <div className="p-3 space-y-2">
        {sorted.map((g, idx) => {
          const style = VERDICT_STYLES[g.verdict] ?? VERDICT_STYLES.UNCERTAIN;

          return (
            <div key={g.level}>
              <div className={`rounded-lg border px-3 py-2 ${style.border} ${style.bg}`}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className={`text-sm font-bold ${style.text}`}>{style.icon}</span>
                    <div>
                      <span className="text-xs font-semibold text-gray-800">{levelLabel(g.level)}</span>
                      {g.value && (
                        <span className="text-xs text-gray-600 ml-1.5">{g.value}</span>
                      )}
                    </div>
                  </div>
                  <div className="text-right">
                    <span className={`text-xs font-bold ${style.text}`}>
                      {g.verdict}
                    </span>
                    <span className="text-[10px] text-gray-400 ml-1.5">
                      {(g.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>

                {/* Evidence counts */}
                <div className="flex items-center gap-3 mt-1 text-[10px]">
                  <span className="text-green-600">{g.supporting_count} supporting</span>
                  <span className="text-red-500">{g.contradicting_count} contradicting</span>
                </div>
              </div>

              {/* Connector between levels */}
              {idx < sorted.length - 1 && (
                <div className="flex justify-center py-0.5">
                  <svg width="8" height="10" viewBox="0 0 8 10" fill="none" className="text-gray-300">
                    <path d="M4 0v7m0 0L1 4m3 3l3-3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
