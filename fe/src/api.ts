const API_URL = import.meta.env.VITE_API_URL || '';

export interface LocateResult {
  name: string;
  country?: string;
  region?: string;
  city?: string;
  latitude?: number;
  longitude?: number;
  confidence: number;
  reasoning: string;
  verified: boolean;
  verification_warning?: string;
  evidence_trail: EvidenceItem[];
  evidence_summary: Record<string, unknown>;
  pipeline_progress: PipelineProgress;
  total_evidence_count: number;
  elapsed_ms: number;
}

export interface EvidenceItem {
  source: string;
  content: string;
  confidence: number;
  latitude?: number;
  longitude?: number;
  country?: string;
  region?: string;
  city?: string;
  url?: string;
}

export interface PipelineStep {
  name: string;
  status: string;
  duration_ms: number;
  evidence_count: number;
  error?: string;
}

export interface PipelineProgress {
  steps: PipelineStep[];
  current_step: string;
  total_evidence: number;
  elapsed_ms: number;
}

export interface SSEEvent {
  event: string;
  step?: string;
  duration_ms?: number;
  evidence_count?: number;
  error?: string;
  data?: LocateResult;
}

export async function locateImage(
  file: File,
  locationHint?: string,
): Promise<LocateResult> {
  const formData = new FormData();
  formData.append('file', file);
  if (locationHint) {
    formData.append('location_hint', locationHint);
  }

  const resp = await fetch(`${API_URL}/api/locate`, {
    method: 'POST',
    body: formData,
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || 'Request failed');
  }

  return resp.json();
}

export function locateImageStream(
  file: File,
  locationHint: string | undefined,
  onEvent: (event: SSEEvent) => void,
  onDone: () => void,
  onError: (error: Error) => void,
): () => void {
  const formData = new FormData();
  formData.append('file', file);
  if (locationHint) {
    formData.append('location_hint', locationHint);
  }

  return streamSSE(`${API_URL}/api/locate/stream`, formData, onEvent, onDone, onError);
}

export function locateImageStreamV2(
  file: File,
  locationHint: string | undefined,
  onEvent: (event: SSEEvent) => void,
  onDone: () => void,
  onError: (error: Error) => void,
): () => void {
  const formData = new FormData();
  formData.append('file', file);
  if (locationHint) {
    formData.append('location_hint', locationHint);
  }

  return streamSSE(`${API_URL}/api/v2/locate/stream`, formData, onEvent, onDone, onError);
}

export function chatStream(
  sessionId: string,
  message: string,
  onEvent: (event: SSEEvent) => void,
  onDone: () => void,
  onError: (error: Error) => void,
): () => void {
  return streamSSE(
    `${API_URL}/api/chat/${sessionId}`,
    JSON.stringify({ message }),
    onEvent,
    onDone,
    onError,
    { 'Content-Type': 'application/json' },
  );
}

export async function getSession(sessionId: string): Promise<Record<string, unknown>> {
  const resp = await fetch(`${API_URL}/api/session/${sessionId}`);
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.json();
}

// --- Internal SSE helper ---

function streamSSE(
  url: string,
  body: FormData | string,
  onEvent: (event: SSEEvent) => void,
  onDone: () => void,
  onError: (error: Error) => void,
  extraHeaders?: Record<string, string>,
): () => void {
  const abortController = new AbortController();

  const headers: Record<string, string> = { ...extraHeaders };

  fetch(url, {
    method: 'POST',
    body,
    signal: abortController.signal,
    headers: body instanceof FormData ? headers : { ...headers, 'Content-Type': 'application/json' },
  })
    .then(async (response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      if (!response.body) throw new Error('No response body');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              onEvent(data);
            } catch {
              // Skip malformed events
            }
          }
        }
      }

      onDone();
    })
    .catch((err) => {
      if (err.name !== 'AbortError') {
        onError(err);
      }
    });

  return () => abortController.abort();
}
