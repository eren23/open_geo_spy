import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { SSEEvent } from '../api';
import type {
  CandidateResult,
  ChatMessage,
  EvidenceSummary,
  PipelineResultMeta,
} from '../types';
import { useChatMessages } from './useChatMessages';

const API_URL = import.meta.env.VITE_API_URL || '';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface UseSessionReturn {
  /** The current session ID (null until the first stream starts). */
  sessionId: string | null;
  /** Ranked candidate results from the V2 pipeline. */
  candidates: CandidateResult[];
  /** Full chat message history (user + assistant + system/step). */
  messages: ChatMessage[];
  /** True while any stream (locate or chat) is active. */
  loading: boolean;
  /** Latest error, if any. */
  error: Error | null;
  /** Evidence summary from the pipeline result. */
  evidenceSummary: EvidenceSummary | null;
  /** Top-level pipeline result metadata. */
  pipelineResult: PipelineResultMeta | null;
  /** Rank of the currently selected candidate (1-based). */
  selectedCandidateRank: number;
  /** The currently selected candidate (derived). */
  selectedCandidate: CandidateResult | null;
  /** Select a candidate by rank. */
  selectCandidate: (rank: number) => void;
  /**
   * Start a new locate session.
   * Streams from `/api/v2/locate/stream` and populates candidates + messages.
   */
  create: (file: File, locationHint?: string) => void;
  /**
   * Send a follow-up chat message in the current session.
   * Streams from `/api/chat/{session_id}`.
   */
  sendMessage: (text: string) => void;
  /** Abort any in-flight stream and reset all state. */
  cancel: () => void;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Opens a POST fetch to `url` with `body`, reads the SSE stream line-by-line,
 * and calls `onEvent` for each parsed JSON payload.
 * Returns a cleanup function that aborts the request.
 */
function streamSSE(
  url: string,
  body: FormData | string,
  onEvent: (event: Record<string, unknown>) => void,
  onDone: () => void,
  onError: (err: Error) => void,
): () => void {
  const controller = new AbortController();

  const headers: Record<string, string> = {};
  const isJSON = typeof body === 'string';
  if (isJSON) {
    headers['Content-Type'] = 'application/json';
  }

  fetch(`${API_URL}${url}`, {
    method: 'POST',
    headers,
    body,
    signal: controller.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        const errBody = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errBody.detail || `HTTP ${response.status}`);
      }
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

        for (const rawLine of lines) {
          const line = rawLine.replace(/\r$/, '');
          if (line.startsWith('data: ')) {
            try {
              const parsed = JSON.parse(line.slice(6));
              onEvent(parsed);
            } catch {
              // Skip malformed lines
            }
          }
        }
      }

      onDone();
    })
    .catch((err) => {
      if (err.name !== 'AbortError') {
        onError(err instanceof Error ? err : new Error(String(err)));
      }
    });

  return () => controller.abort();
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

/**
 * Session state management for the OpenGeoSpy chat UI.
 *
 * - `create(file, hint?)` streams the V2 locate pipeline and populates
 *   candidates and chat messages (including step progress).
 * - `sendMessage(text)` streams a follow-up chat request against the
 *   existing session.
 */
export function useSession(): UseSessionReturn {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [candidates, setCandidates] = useState<CandidateResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [evidenceSummary, setEvidenceSummary] = useState<EvidenceSummary | null>(null);
  const [pipelineResult, setPipelineResult] = useState<PipelineResultMeta | null>(null);
  const [selectedCandidateRank, setSelectedCandidateRank] = useState(1);

  // Restore from sessionStorage on mount
  const [restored, setRestored] = useState(false);
  useEffect(() => {
    if (restored) return;
    setRestored(true);
    try {
      const saved = sessionStorage.getItem('ogspy_session');
      if (saved) {
        const data = JSON.parse(saved);
        if (data.sessionId) setSessionId(data.sessionId);
        if (data.candidates?.length) setCandidates(data.candidates);
        if (data.evidenceSummary) setEvidenceSummary(data.evidenceSummary);
      }
    } catch { /* ignore */ }
  }, [restored]);

  // Persist session state for page refresh recovery (only when not loading)
  useEffect(() => {
    if (!loading && sessionId && candidates.length > 0) {
      try {
        sessionStorage.setItem('ogspy_session', JSON.stringify({
          sessionId,
          candidates,
          evidenceSummary,
        }));
      } catch { /* ignore */ }
    }
  }, [loading, sessionId, candidates, evidenceSummary]);

  const selectedCandidate = useMemo(
    () => candidates.find((c) => c.rank === selectedCandidateRank) ?? null,
    [candidates, selectedCandidateRank],
  );

  const selectCandidate = useCallback((rank: number) => {
    setSelectedCandidateRank(rank);
  }, []);

  const { messages, addMessage, addAgentStep, clear: clearMessages } = useChatMessages();
  const cancelRef = useRef<(() => void) | null>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ---- create ----------------------------------------------------------

  const create = useCallback(
    (file: File, locationHint?: string) => {
      // Abort any existing stream
      cancelRef.current?.();

      // Reset state
      setCandidates([]);
      setError(null);
      setSessionId(null);
      setEvidenceSummary(null);
      setPipelineResult(null);
      setSelectedCandidateRank(1);
      clearMessages();
      setLoading(true);

      // Add an initial message so the chat view appears immediately
      addAgentStep('pipeline', 'running', 'Analyzing image...');

      // Build form data
      const formData = new FormData();
      formData.append('file', file);
      if (locationHint) {
        formData.append('location_hint', locationHint);
      }

      // SSE timeout: if no events for 5 min, treat as error
      // (first run may download ML models which takes several minutes)
      const TIMEOUT_MS = 300_000;
      const resetTimeout = () => {
        if (timeoutRef.current) clearTimeout(timeoutRef.current);
        timeoutRef.current = setTimeout(() => {
          cancelRef.current?.();
          setError(new Error('Pipeline timed out (no response for 5 minutes)'));
          setLoading(false);
        }, TIMEOUT_MS);
      };
      resetTimeout();

      const cancel = streamSSE(
        '/api/v2/locate/stream',
        formData,
        (event) => {
          resetTimeout();

          const ev = event as unknown as SSEEvent & {
            data?: Record<string, unknown>;
            candidates?: CandidateResult[];
            session_id?: string;
          };

          switch (ev.event) {
            case 'step_start':
              if (ev.step) addAgentStep(ev.step, 'running');
              break;

            case 'step_complete':
              if (ev.step) {
                const detail = ev.evidence_count != null
                  ? `${ev.evidence_count} evidence in ${ev.duration_ms ?? 0}ms`
                  : undefined;
                addAgentStep(ev.step, 'completed', detail);
              }
              break;

            case 'step_error':
              if (ev.step) addAgentStep(ev.step, 'error', ev.error);
              break;

            case 'result': {
              const data = ev.data ?? (ev as unknown as Record<string, unknown>);
              const resultCandidates = (data.candidates ?? []) as CandidateResult[];
              const sid = (data.session_id ?? null) as string | null;

              setCandidates(resultCandidates);
              if (sid) setSessionId(sid);

              // Mark the pipeline step as completed
              addAgentStep('pipeline', 'completed', 'Analysis complete');

              // Store evidence summary
              const rawSummary = data.evidence_summary as Record<string, unknown> | undefined;
              if (rawSummary) {
                setEvidenceSummary({
                  sources: (rawSummary.sources ?? []) as string[],
                  countries: (rawSummary.countries_mentioned ?? []) as string[],
                  agreement_score: (rawSummary.agreement_score ?? 0) as number,
                  centroid: rawSummary.centroid
                    ? { latitude: (rawSummary.centroid as any).lat, longitude: (rawSummary.centroid as any).lon }
                    : undefined,
                  top_evidence: (rawSummary.top_evidence as any[] | undefined)?.map((e: any) => ({
                    ...e,
                    confidence: e.confidence ?? 0,
                  })) as EvidenceSummary['top_evidence'],
                });
              }

              // Store pipeline result metadata
              setPipelineResult({
                elapsed_ms: (data.elapsed_ms ?? 0) as number,
                total_evidence_count: (data.total_evidence_count ?? 0) as number,
                verified: (data.verified ?? false) as boolean,
                verification_warning: data.verification_warning as string | undefined,
                reasoning: data.reasoning as string | undefined,
              });

              // Produce a structured summary assistant message
              if (resultCandidates.length > 0) {
                const top = resultCandidates[0];
                const confPct = (top.confidence * 100).toFixed(0);
                const totalEvidence = (data.total_evidence_count ?? 0) as number;
                const sources = rawSummary?.sources as string[] | undefined;
                const countries = rawSummary?.countries_mentioned as string[] | undefined;
                const agreementScore = rawSummary?.agreement_score as number | undefined;

                let summary = `I identified **${top.name}** with **${confPct}%** confidence.\n\n`;

                if (totalEvidence > 0 || sources?.length) {
                  const parts: string[] = [];
                  if (totalEvidence > 0) parts.push(`${totalEvidence} pieces`);
                  if (sources?.length) parts.push(`from ${sources.length} sources (${sources.join(', ')})`);
                  summary += `**Evidence:** ${parts.join(' ')}\n`;
                }

                if (countries?.length) {
                  summary += `**Countries mentioned:** ${countries.join(', ')}\n`;
                }

                if (agreementScore != null) {
                  summary += `**Agreement score:** ${(agreementScore * 100).toFixed(0)}%\n`;
                }

                summary += `\n${top.reasoning}`;

                if (resultCandidates.length > 1) {
                  summary += `\n\nI also found ${resultCandidates.length - 1} alternative candidate${resultCandidates.length > 2 ? 's' : ''} — click them on the map or sidebar to compare.`;
                }

                addMessage({
                  role: 'assistant',
                  content: summary,
                  timestamp: new Date().toISOString(),
                  metadata: { candidates: resultCandidates.length },
                });
              }
              break;
            }

            case 'error':
              setError(new Error(ev.error || 'Unknown pipeline error'));
              break;
          }
        },
        () => {
          if (timeoutRef.current) clearTimeout(timeoutRef.current);
          timeoutRef.current = null;
          setLoading(false);
        },
        (err) => {
          if (timeoutRef.current) clearTimeout(timeoutRef.current);
          timeoutRef.current = null;
          setError(err);
          setLoading(false);
        },
      );

      cancelRef.current = cancel;
    },
    [addAgentStep, addMessage, clearMessages],
  );

  // ---- sendMessage -----------------------------------------------------

  const sendMessage = useCallback(
    (text: string) => {
      if (!sessionId) {
        setError(new Error('No active session'));
        return;
      }

      // Abort any in-flight stream
      cancelRef.current?.();

      // Append user message
      addMessage({
        role: 'user',
        content: text,
        timestamp: new Date().toISOString(),
      });

      setError(null);
      setLoading(true);

      let assistantBuffer = '';

      const cancel = streamSSE(
        `/api/chat/${sessionId}`,
        JSON.stringify({ message: text }),
        (event) => {
          const ev = event as Record<string, unknown>;

          switch (ev.event) {
            case 'chat_token':
              // Streaming token
              assistantBuffer += (ev.token as string) || '';
              break;

            case 'chat_message':
              // Complete assistant message
              addMessage({
                role: 'assistant',
                content: (ev.content as string) || assistantBuffer,
                timestamp: new Date().toISOString(),
                metadata: ev.metadata as Record<string, unknown> | undefined,
              });
              assistantBuffer = '';
              break;

            case 'candidates_update':
              // The chat handler may return updated candidates
              if (Array.isArray(ev.candidates)) {
                setCandidates(ev.candidates as CandidateResult[]);
              }
              break;

            case 'error':
              setError(new Error((ev.error as string) || 'Chat error'));
              break;
          }
        },
        () => {
          // If we accumulated tokens but never got a chat_message event,
          // flush the buffer as the assistant response.
          if (assistantBuffer) {
            addMessage({
              role: 'assistant',
              content: assistantBuffer,
              timestamp: new Date().toISOString(),
            });
            assistantBuffer = '';
          }
          setLoading(false);
        },
        (err) => {
          setError(err);
          setLoading(false);
        },
      );

      cancelRef.current = cancel;
    },
    [sessionId, addMessage],
  );

  // ---- cancel ----------------------------------------------------------

  const cancel = useCallback(() => {
    cancelRef.current?.();
    cancelRef.current = null;
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    timeoutRef.current = null;
    setLoading(false);
    setSessionId(null);
    setCandidates([]);
    setEvidenceSummary(null);
    setPipelineResult(null);
    setSelectedCandidateRank(1);
    clearMessages();
    try { sessionStorage.removeItem('ogspy_session'); } catch { /* ignore */ }
  }, [clearMessages]);

  return {
    sessionId,
    candidates,
    messages,
    loading,
    error,
    evidenceSummary,
    pipelineResult,
    selectedCandidateRank,
    selectedCandidate,
    selectCandidate,
    create,
    sendMessage,
    cancel,
  };
}

export default useSession;
