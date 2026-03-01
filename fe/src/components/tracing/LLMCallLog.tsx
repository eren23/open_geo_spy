import { useState } from 'react';
import type { LLMCallInfo } from '../../types';
import { formatCost } from '../../utils/format';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface LLMCallLogProps {
  calls: LLMCallInfo[];
  /** Maximum calls to show before collapsing. */
  maxVisible?: number;
}

function formatLatency(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function shortModel(model: string): string {
  // Strip provider prefix (e.g., "google/gemini-2.5-flash" -> "gemini-2.5-flash")
  const parts = model.split('/');
  return parts[parts.length - 1];
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function LLMCallLog({ calls, maxVisible = 5 }: LLMCallLogProps) {
  const [expanded, setExpanded] = useState(false);

  if (calls.length === 0) return null;

  const visible = expanded ? calls : calls.slice(-maxVisible);
  const hiddenCount = calls.length - visible.length;

  const totalCost = calls.reduce((s, c) => s + c.cost_usd, 0);
  const totalTokens = calls.reduce((s, c) => s + c.input_tokens + c.output_tokens, 0);

  return (
    <div className="rounded-lg border border-gray-200 bg-white overflow-hidden">
      {/* Header */}
      <button
        type="button"
        className="w-full flex items-center justify-between px-3 py-2 border-b border-gray-100 hover:bg-gray-50 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
          LLM Calls ({calls.length})
        </h4>
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-gray-400">
            {formatCost(totalCost)} &middot; {totalTokens.toLocaleString()} tok
          </span>
          <svg
            className={`w-3 h-3 text-gray-400 transition-transform ${expanded ? 'rotate-180' : ''}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {/* Call list */}
      <div className="divide-y divide-gray-50">
        {hiddenCount > 0 && (
          <button
            type="button"
            className="w-full px-3 py-1.5 text-[10px] text-blue-500 hover:text-blue-700 hover:bg-blue-50 transition-colors"
            onClick={() => setExpanded(true)}
          >
            Show {hiddenCount} earlier call{hiddenCount !== 1 ? 's' : ''}...
          </button>
        )}

        {visible.map((call) => (
          <div key={call.id} className="px-3 py-2 hover:bg-gray-50 transition-colors">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="inline-flex items-center rounded bg-gray-100 px-1.5 py-0.5 text-[10px] font-mono text-gray-600">
                  {shortModel(call.model)}
                </span>
                <span className="text-[11px] text-gray-700">{call.purpose}</span>
              </div>
              <span className="text-[10px] text-gray-400">{formatLatency(call.latency_ms)}</span>
            </div>
            <div className="flex items-center gap-3 mt-1 text-[10px] text-gray-400">
              <span>{call.input_tokens.toLocaleString()} in</span>
              <span>{call.output_tokens.toLocaleString()} out</span>
              <span>{formatCost(call.cost_usd)}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
