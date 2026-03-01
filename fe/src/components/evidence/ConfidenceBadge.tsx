// ---------------------------------------------------------------------------
// ConfidenceBadge -- color-coded confidence pill (0-1 scale)
// ---------------------------------------------------------------------------

interface ConfidenceBadgeProps {
  confidence: number;
}

export default function ConfidenceBadge({ confidence }: ConfidenceBadgeProps) {
  const pct = (confidence * 100).toFixed(0);

  let colorClasses: string;
  if (confidence > 0.7) {
    colorClasses = 'bg-green-100 text-green-700';
  } else if (confidence > 0.4) {
    colorClasses = 'bg-yellow-100 text-yellow-700';
  } else {
    colorClasses = 'bg-red-100 text-red-700';
  }

  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-semibold ${colorClasses}`}
    >
      {pct}%
    </span>
  );
}
