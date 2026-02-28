import { useMemo } from 'react';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface BatchItem {
  id: string;
  filename: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  result?: {
    name: string;
    country?: string;
    confidence: number;
    latitude?: number;
    longitude?: number;
  };
}

interface BatchDashboardProps {
  items: BatchItem[];
}

// ---------------------------------------------------------------------------
// Status styling helpers
// ---------------------------------------------------------------------------

function statusIndicator(status: BatchItem['status']) {
  switch (status) {
    case 'pending':
      return <span className="inline-block h-2.5 w-2.5 rounded-full bg-gray-300" title="Pending" />;
    case 'processing':
      return (
        <span className="relative flex h-2.5 w-2.5" title="Processing">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75" />
          <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-blue-500" />
        </span>
      );
    case 'completed':
      return <span className="inline-block h-2.5 w-2.5 rounded-full bg-green-500" title="Completed" />;
    case 'failed':
      return <span className="inline-block h-2.5 w-2.5 rounded-full bg-red-500" title="Failed" />;
  }
}

function statusLabel(status: BatchItem['status']): { text: string; className: string } {
  switch (status) {
    case 'pending':
      return { text: 'Pending', className: 'text-gray-500' };
    case 'processing':
      return { text: 'Processing', className: 'text-blue-600' };
    case 'completed':
      return { text: 'Completed', className: 'text-green-600' };
    case 'failed':
      return { text: 'Failed', className: 'text-red-600' };
  }
}

function confidenceBadge(confidence: number) {
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
    <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-semibold ${colorClasses}`}>
      {pct}%
    </span>
  );
}

// ---------------------------------------------------------------------------
// BatchDashboard
// ---------------------------------------------------------------------------

export default function BatchDashboard({ items }: BatchDashboardProps) {
  const stats = useMemo(() => {
    const total = items.length;
    const completed = items.filter((i) => i.status === 'completed').length;
    const processing = items.filter((i) => i.status === 'processing').length;
    const failed = items.filter((i) => i.status === 'failed').length;
    const pending = items.filter((i) => i.status === 'pending').length;
    return { total, completed, processing, failed, pending };
  }, [items]);

  if (items.length === 0) {
    return (
      <div className="rounded-xl border border-gray-200 bg-white p-8 text-center">
        <p className="text-sm text-gray-500">No batch items to display.</p>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-gray-200 bg-white overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200">
        <h3 className="text-sm font-semibold text-gray-900">Batch Processing</h3>
      </div>

      {/* Summary stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 px-4 py-3 bg-gray-50 border-b border-gray-200">
        <div className="flex flex-col items-center rounded-lg bg-white border border-gray-200 px-3 py-2">
          <span className="text-lg font-bold text-gray-900">{stats.total}</span>
          <span className="text-[11px] text-gray-500">Total</span>
        </div>
        <div className="flex flex-col items-center rounded-lg bg-white border border-green-200 px-3 py-2">
          <span className="text-lg font-bold text-green-600">{stats.completed}</span>
          <span className="text-[11px] text-gray-500">Completed</span>
        </div>
        <div className="flex flex-col items-center rounded-lg bg-white border border-blue-200 px-3 py-2">
          <span className="text-lg font-bold text-blue-600">{stats.processing}</span>
          <span className="text-[11px] text-gray-500">In Progress</span>
        </div>
        <div className="flex flex-col items-center rounded-lg bg-white border border-red-200 px-3 py-2">
          <span className="text-lg font-bold text-red-600">{stats.failed}</span>
          <span className="text-[11px] text-gray-500">Failed</span>
        </div>
      </div>

      {/* Progress bar */}
      {stats.total > 0 && (
        <div className="px-4 py-2 border-b border-gray-200">
          <div className="flex rounded-full overflow-hidden h-2 bg-gray-100">
            {stats.completed > 0 && (
              <div
                className="bg-green-500 transition-all duration-300"
                style={{ width: `${(stats.completed / stats.total) * 100}%` }}
              />
            )}
            {stats.processing > 0 && (
              <div
                className="bg-blue-500 transition-all duration-300"
                style={{ width: `${(stats.processing / stats.total) * 100}%` }}
              />
            )}
            {stats.failed > 0 && (
              <div
                className="bg-red-500 transition-all duration-300"
                style={{ width: `${(stats.failed / stats.total) * 100}%` }}
              />
            )}
          </div>
        </div>
      )}

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200 bg-gray-50">
              <th className="px-4 py-2.5 text-left text-[11px] font-semibold uppercase tracking-wider text-gray-500">
                #
              </th>
              <th className="px-4 py-2.5 text-left text-[11px] font-semibold uppercase tracking-wider text-gray-500">
                Filename
              </th>
              <th className="px-4 py-2.5 text-left text-[11px] font-semibold uppercase tracking-wider text-gray-500">
                Status
              </th>
              <th className="px-4 py-2.5 text-left text-[11px] font-semibold uppercase tracking-wider text-gray-500">
                Result
              </th>
              <th className="px-4 py-2.5 text-right text-[11px] font-semibold uppercase tracking-wider text-gray-500">
                Confidence
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {items.map((item, idx) => {
              const label = statusLabel(item.status);
              return (
                <tr
                  key={item.id}
                  className="hover:bg-gray-50 transition-colors"
                >
                  <td className="px-4 py-2.5 text-xs text-gray-400 font-mono">
                    {idx + 1}
                  </td>
                  <td className="px-4 py-2.5">
                    <span className="text-sm text-gray-800 truncate block max-w-[200px]" title={item.filename}>
                      {item.filename}
                    </span>
                  </td>
                  <td className="px-4 py-2.5">
                    <div className="flex items-center gap-2">
                      {statusIndicator(item.status)}
                      <span className={`text-xs font-medium ${label.className}`}>
                        {label.text}
                      </span>
                    </div>
                  </td>
                  <td className="px-4 py-2.5">
                    {item.result ? (
                      <div>
                        <span className="text-sm text-gray-800">{item.result.name}</span>
                        {item.result.country && (
                          <span className="text-xs text-gray-400 ml-1">({item.result.country})</span>
                        )}
                      </div>
                    ) : (
                      <span className="text-xs text-gray-400">--</span>
                    )}
                  </td>
                  <td className="px-4 py-2.5 text-right">
                    {item.result ? (
                      confidenceBadge(item.result.confidence)
                    ) : (
                      <span className="text-xs text-gray-400">--</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
