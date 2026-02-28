import { useState } from 'react';
import type { ChatMessage } from '../../types';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface AgentStepMessageProps {
  message: ChatMessage;
}

// ---------------------------------------------------------------------------
// Step metadata extraction
// ---------------------------------------------------------------------------

/** Step labels matching the pipeline names used in useSession. */
const STEP_LABELS: Record<string, string> = {
  pipeline: 'Pipeline Started',
  feature_extraction: 'Feature Extraction',
  ml_ensemble: 'ML Ensemble',
  web_intelligence: 'Web Intelligence',
  candidate_verification: 'Visual Verification',
  reasoning: 'Reasoning & Verification',
};

/** Icon per step name (simple emoji-free SVG path data). */
const STEP_ICONS: Record<string, string> = {
  pipeline: 'M13 10V3L4 14h7v7l9-11h-7z', // bolt
  feature_extraction: 'M15 12a3 3 0 11-6 0 3 3 0 016 0z M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z', // eye
  ml_ensemble: 'M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z', // cpu
  web_intelligence: 'M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9', // globe
  candidate_verification: 'M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z', // check-circle
  reasoning: 'M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z', // lightbulb
};

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function StatusIndicator({ status }: { status: string }) {
  switch (status) {
    case 'running':
      return (
        <div className="h-4 w-4 rounded-full border-2 border-blue-500 border-t-transparent animate-spin" />
      );
    case 'completed':
      return (
        <div className="flex h-4 w-4 items-center justify-center rounded-full bg-green-500">
          <svg className="h-2.5 w-2.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
          </svg>
        </div>
      );
    case 'error':
      return (
        <div className="flex h-4 w-4 items-center justify-center rounded-full bg-red-500">
          <svg className="h-2.5 w-2.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </div>
      );
    default:
      return <div className="h-4 w-4 rounded-full border-2 border-gray-300" />;
  }
}

function StepIcon({ step }: { step: string }) {
  const d = STEP_ICONS[step];
  if (!d) {
    return (
      <div className="flex h-6 w-6 items-center justify-center rounded bg-gray-200 text-gray-500">
        <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
      </div>
    );
  }

  return (
    <div className="flex h-6 w-6 items-center justify-center rounded bg-gray-200 text-gray-600">
      <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d={d} />
      </svg>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

function AgentStepMessage({ message }: AgentStepMessageProps) {
  const [expanded, setExpanded] = useState(false);

  const meta = message.metadata ?? {};
  const step = (meta.step as string) || 'unknown';
  const status = (meta.status as string) || 'pending';
  const detail = (meta.detail as string | undefined);

  // Try to parse evidence count and duration from the detail string (e.g. "4 evidence in 320ms")
  const evidenceMatch = typeof detail === 'string' ? detail.match(/^(\d+)\s+evidence/) : null;
  const evidenceCount = evidenceMatch ? parseInt(evidenceMatch[1], 10) : null;
  const durationMatch = typeof detail === 'string' ? detail.match(/in\s+(\d+)ms/) : null;
  const durationMs = durationMatch ? parseInt(durationMatch[1], 10) : null;

  const label = STEP_LABELS[step] || step.replace(/_/g, ' ');

  const statusColor =
    status === 'running'
      ? 'text-blue-700'
      : status === 'completed'
        ? 'text-gray-700'
        : status === 'error'
          ? 'text-red-600'
          : 'text-gray-400';

  return (
    <div className="flex justify-start">
      <div className="max-w-[85%] rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm shadow-sm">
        {/* Header row */}
        <div className="flex items-center gap-2">
          <StepIcon step={step} />

          <span className={`font-medium ${statusColor}`}>{label}</span>

          <StatusIndicator status={status} />

          {evidenceCount !== null && evidenceCount > 0 && (
            <span className="rounded bg-gray-100 px-1.5 py-0.5 text-xs text-gray-500">
              {evidenceCount} evidence
            </span>
          )}

          {durationMs !== null && status === 'completed' && (
            <span className="rounded bg-gray-100 px-1.5 py-0.5 text-xs text-gray-400">
              {durationMs}ms
            </span>
          )}
        </div>

        {/* Expandable detail */}
        {detail && (
          <>
            <button
              type="button"
              onClick={() => setExpanded((prev) => !prev)}
              className="mt-1.5 text-xs text-gray-400 hover:text-gray-600 transition-colors"
            >
              {expanded ? 'Hide details' : 'Show details'}
            </button>

            {expanded && (
              <div className="mt-1 rounded bg-gray-50 px-2 py-1.5 text-xs text-gray-600">
                {detail}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default AgentStepMessage;
