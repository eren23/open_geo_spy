import type { StepTiming } from '../../types';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface TracingTimelineProps {
  steps: StepTiming[];
  /** When true, use compact single-row layout. */
  compact?: boolean;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const STEP_COLORS: Record<string, { bg: string; border: string; text: string; pulse: string }> = {
  feature_extraction: { bg: 'bg-purple-500', border: 'border-purple-400', text: 'text-purple-700', pulse: 'bg-purple-400' },
  ml_ensemble: { bg: 'bg-green-500', border: 'border-green-400', text: 'text-green-700', pulse: 'bg-green-400' },
  web_intelligence: { bg: 'bg-orange-500', border: 'border-orange-400', text: 'text-orange-700', pulse: 'bg-orange-400' },
  candidate_verification: { bg: 'bg-cyan-500', border: 'border-cyan-400', text: 'text-cyan-700', pulse: 'bg-cyan-400' },
  reasoning: { bg: 'bg-indigo-500', border: 'border-indigo-400', text: 'text-indigo-700', pulse: 'bg-indigo-400' },
  refinement_check: { bg: 'bg-amber-500', border: 'border-amber-400', text: 'text-amber-700', pulse: 'bg-amber-400' },
};

const DEFAULT_COLOR = { bg: 'bg-gray-500', border: 'border-gray-400', text: 'text-gray-700', pulse: 'bg-gray-400' };

const STATUS_ICONS: Record<string, string> = {
  running: '\u25B6',    // play
  completed: '\u2713',  // check
  error: '\u2717',      // x
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatDuration(ms?: number): string {
  if (ms == null) return '';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function stepLabel(name: string): string {
  return name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function TracingTimeline({ steps, compact = false }: TracingTimelineProps) {
  if (steps.length === 0) return null;

  const totalDuration = steps.reduce((sum, s) => sum + (s.duration_ms ?? 0), 0);
  const maxDuration = Math.max(...steps.map((s) => s.duration_ms ?? 0), 1);

  if (compact) {
    return (
      <div className="flex items-center gap-1 overflow-x-auto py-1">
        {steps.map((step, idx) => {
          const color = STEP_COLORS[step.name] ?? DEFAULT_COLOR;
          const widthPct = totalDuration > 0
            ? Math.max(8, ((step.duration_ms ?? 0) / totalDuration) * 100)
            : 100 / steps.length;

          return (
            <div key={step.name} className="flex items-center gap-1">
              <div
                className={`h-2 rounded-full ${step.status === 'running' ? 'animate-pulse ' + color.pulse : step.status === 'error' ? 'bg-red-500' : color.bg}`}
                style={{ width: `${widthPct}%`, minWidth: 16 }}
                title={`${stepLabel(step.name)}: ${formatDuration(step.duration_ms)}`}
              />
              {idx < steps.length - 1 && (
                <div className="w-0.5 h-0.5 rounded-full bg-gray-300 flex-shrink-0" />
              )}
            </div>
          );
        })}
        {totalDuration > 0 && (
          <span className="text-[10px] text-gray-400 ml-1 whitespace-nowrap flex-shrink-0">
            {formatDuration(totalDuration)}
          </span>
        )}
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-3">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Pipeline Timeline</h4>
        {totalDuration > 0 && (
          <span className="text-[10px] text-gray-400">Total: {formatDuration(totalDuration)}</span>
        )}
      </div>

      <div className="space-y-1.5">
        {steps.map((step) => {
          const color = STEP_COLORS[step.name] ?? DEFAULT_COLOR;
          const barWidth = maxDuration > 0
            ? Math.max(4, ((step.duration_ms ?? 0) / maxDuration) * 100)
            : 50;

          return (
            <div key={step.name} className="flex items-center gap-2">
              {/* Status icon */}
              <span
                className={`w-4 text-center text-[10px] flex-shrink-0 ${
                  step.status === 'error' ? 'text-red-500' : step.status === 'completed' ? color.text : 'text-gray-400'
                }`}
              >
                {STATUS_ICONS[step.status] ?? ''}
              </span>

              {/* Label */}
              <span className="text-[11px] text-gray-700 w-28 truncate flex-shrink-0" title={stepLabel(step.name)}>
                {stepLabel(step.name)}
              </span>

              {/* Bar */}
              <div className="flex-1 h-3 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    step.status === 'running'
                      ? 'animate-pulse ' + color.pulse
                      : step.status === 'error'
                        ? 'bg-red-400'
                        : color.bg
                  }`}
                  style={{ width: step.status === 'running' ? '60%' : `${barWidth}%` }}
                />
              </div>

              {/* Duration / evidence */}
              <span className="text-[10px] text-gray-400 w-16 text-right flex-shrink-0">
                {step.status === 'running' ? '...' : formatDuration(step.duration_ms)}
              </span>
              {step.evidence_count != null && step.evidence_count > 0 && (
                <span className="text-[10px] text-gray-300 w-8 text-right flex-shrink-0">
                  {step.evidence_count}ev
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
