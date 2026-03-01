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

export async function getSession(sessionId: string): Promise<Record<string, unknown>> {
  const resp = await fetch(`${API_URL}/api/session/${sessionId}`);
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.json();
}
