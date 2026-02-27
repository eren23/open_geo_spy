import { useCallback, useState, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import { LocateResult, SSEEvent, locateImageStream } from './api';
import ResultDisplay from './components/ResultDisplay';
import PipelineStatus from './components/PipelineStatus';

type StepState = {
  name: string;
  status: 'pending' | 'running' | 'completed' | 'error';
  duration_ms?: number;
  evidence_count?: number;
  error?: string;
};

const STEPS = ['feature_extraction', 'ml_ensemble', 'web_intelligence', 'candidate_verification', 'reasoning'];
const STEP_LABELS: Record<string, string> = {
  feature_extraction: 'Feature Extraction',
  ml_ensemble: 'ML Ensemble',
  web_intelligence: 'Web Intelligence',
  candidate_verification: 'Visual Verification',
  reasoning: 'Reasoning & Verification',
};

function App() {
  const [result, setResult] = useState<LocateResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [locationHint, setLocationHint] = useState('');
  const [preview, setPreview] = useState<string | null>(null);
  const [steps, setSteps] = useState<StepState[]>([]);
  const cancelRef = useRef<(() => void) | null>(null);

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (!file) return;

      // Preview
      const url = URL.createObjectURL(file);
      setPreview(url);
      setResult(null);
      setError(null);
      setLoading(true);
      setSteps(STEPS.map((name) => ({ name, status: 'pending' })));

      const cancel = locateImageStream(
        file,
        locationHint || undefined,
        (event: SSEEvent) => {
          if (event.event === 'step_start' && event.step) {
            setSteps((prev) =>
              prev.map((s) => (s.name === event.step ? { ...s, status: 'running' } : s)),
            );
          } else if (event.event === 'step_complete' && event.step) {
            setSteps((prev) =>
              prev.map((s) =>
                s.name === event.step
                  ? {
                      ...s,
                      status: 'completed',
                      duration_ms: event.duration_ms,
                      evidence_count: event.evidence_count,
                    }
                  : s,
              ),
            );
          } else if (event.event === 'step_error' && event.step) {
            setSteps((prev) =>
              prev.map((s) =>
                s.name === event.step ? { ...s, status: 'error', error: event.error } : s,
              ),
            );
          } else if (event.event === 'result' && event.data) {
            setResult(event.data);
          } else if (event.event === 'error') {
            setError(event.error || 'Unknown error');
          }
        },
        () => setLoading(false),
        (err) => {
          setError(err.message);
          setLoading(false);
        },
      );

      cancelRef.current = cancel;
    },
    [locationHint],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpeg', '.jpg', '.png', '.webp'] },
    maxFiles: 1,
    disabled: loading,
  });

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-5xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900">OpenGeoSpy</h1>
          <p className="text-gray-500 mt-1">Multi-agent geolocation with evidence tracking</p>
        </div>

        {/* Upload Section */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-6">
          <input
            type="text"
            placeholder="Location hint (optional, e.g. 'Berlin, Germany')"
            value={locationHint}
            onChange={(e) => setLocationHint(e.target.value)}
            className="w-full px-4 py-2 mb-4 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            disabled={loading}
          />

          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
              isDragActive
                ? 'border-blue-500 bg-blue-50'
                : loading
                  ? 'border-gray-200 bg-gray-50 cursor-not-allowed'
                  : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
            }`}
          >
            <input {...getInputProps()} />
            {preview ? (
              <img
                src={preview}
                alt="Preview"
                className="max-h-64 mx-auto rounded-lg mb-3"
              />
            ) : null}
            <p className="text-gray-600">
              {isDragActive
                ? 'Drop image here...'
                : loading
                  ? 'Processing...'
                  : 'Drop an image here, or click to select'}
            </p>
          </div>
        </div>

        {/* Pipeline Status */}
        {steps.length > 0 && (
          <PipelineStatus steps={steps} labels={STEP_LABELS} />
        )}

        {/* Error */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <p className="text-red-700 text-sm">{error}</p>
          </div>
        )}

        {/* Result */}
        {result && <ResultDisplay result={result} />}
      </div>
    </div>
  );
}

export default App;
