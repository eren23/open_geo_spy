type StepState = {
  name: string;
  status: 'pending' | 'running' | 'completed' | 'error';
  duration_ms?: number;
  evidence_count?: number;
  error?: string;
};

interface PipelineStatusProps {
  steps: StepState[];
  labels: Record<string, string>;
}

function PipelineStatus({ steps, labels }: PipelineStatusProps) {
  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-6">
      <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-4">
        Pipeline Progress
      </h3>
      <div className="space-y-3">
        {steps.map((step) => (
          <div key={step.name} className="flex items-center gap-3">
            {/* Status indicator */}
            <div className="flex-shrink-0">
              {step.status === 'pending' && (
                <div className="w-5 h-5 rounded-full border-2 border-gray-300" />
              )}
              {step.status === 'running' && (
                <div className="w-5 h-5 rounded-full border-2 border-blue-500 border-t-transparent animate-spin" />
              )}
              {step.status === 'completed' && (
                <div className="w-5 h-5 rounded-full bg-green-500 flex items-center justify-center">
                  <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
              )}
              {step.status === 'error' && (
                <div className="w-5 h-5 rounded-full bg-red-500 flex items-center justify-center">
                  <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </div>
              )}
            </div>

            {/* Label */}
            <span
              className={`text-sm flex-grow ${
                step.status === 'running'
                  ? 'text-blue-700 font-medium'
                  : step.status === 'completed'
                    ? 'text-gray-700'
                    : step.status === 'error'
                      ? 'text-red-600'
                      : 'text-gray-400'
              }`}
            >
              {labels[step.name] || step.name}
            </span>

            {/* Meta */}
            <div className="flex items-center gap-2 text-xs text-gray-400">
              {step.duration_ms !== undefined && (
                <span>{(step.duration_ms / 1000).toFixed(1)}s</span>
              )}
              {step.evidence_count !== undefined && step.evidence_count > 0 && (
                <span className="bg-gray-100 px-1.5 py-0.5 rounded">
                  {step.evidence_count} evidence
                </span>
              )}
              {step.error && (
                <span className="text-red-400 truncate max-w-32" title={step.error}>
                  {step.error}
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default PipelineStatus;
