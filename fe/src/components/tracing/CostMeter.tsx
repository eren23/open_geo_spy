import type { CostState } from '../../types';
import { formatCost } from '../../utils/format';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface CostMeterProps {
  cost: CostState;
  loading?: boolean;
}

function formatTokens(tokens: number): string {
  if (tokens < 1000) return `${tokens}`;
  if (tokens < 100_000) return `${(tokens / 1000).toFixed(1)}k`;
  return `${(tokens / 1000).toFixed(0)}k`;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function CostMeter({ cost, loading = false }: CostMeterProps) {
  if (cost.call_count === 0 && !loading) return null;

  return (
    <div
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium transition-colors ${
        loading
          ? 'border-blue-200 bg-blue-50 text-blue-600'
          : 'border-gray-200 bg-gray-50 text-gray-600'
      }`}
    >
      {loading && (
        <span className="relative flex h-2 w-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75" />
          <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500" />
        </span>
      )}
      <span>{formatCost(cost.total_usd)}</span>
      <span className="text-gray-300">|</span>
      <span>{formatTokens(cost.total_tokens)} tok</span>
      {cost.call_count > 0 && (
        <>
          <span className="text-gray-300">|</span>
          <span>{cost.call_count} calls</span>
        </>
      )}
    </div>
  );
}
