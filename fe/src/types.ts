/**
 * Shared TypeScript types for OpenGeoSpy frontend.
 *
 * Mirrors the backend Pydantic schemas in src/api/schemas.py
 * and extends them with frontend-specific types.
 */

// Re-export types already defined in api.ts
export type { EvidenceItem, PipelineStep, SSEEvent } from './api';

import type { EvidenceItem } from './api';

// ---------------------------------------------------------------------------
// Evidence summary & pipeline result (from backend SSE result event)
// ---------------------------------------------------------------------------

/** Top-level evidence summary returned alongside candidates. */
export interface EvidenceSummary {
  sources: string[];
  countries: string[];
  agreement_score: number;
  centroid?: { latitude: number; longitude: number };
  top_evidence?: EvidenceItem[];
}

/** Top-level pipeline result metadata. */
export interface PipelineResultMeta {
  elapsed_ms: number;
  total_evidence_count: number;
  verified: boolean;
  verification_warning?: string;
  reasoning?: string;
}

// ---------------------------------------------------------------------------
// Multi-candidate result
// ---------------------------------------------------------------------------

/** A ranked location candidate from the V2 pipeline. */
export interface CandidateResult {
  rank: number;
  name: string;
  country?: string;
  region?: string;
  city?: string;
  latitude?: number;
  longitude?: number;
  confidence: number;
  reasoning: string;
  evidence_trail: EvidenceItem[];
  visual_match_score?: number;
  source_diversity: number;
}

// ---------------------------------------------------------------------------
// Search graph (V2)
// ---------------------------------------------------------------------------

export interface SearchGraphNode {
  id: string;
  query: string;
  intent: string;
  status: string;
  provider: string;
  parent_id?: string;
  evidence_count: number;
  best_confidence: number;
  duration_ms: number;
}

export interface SearchGraphEdge {
  source_id: string;
  target_id: string;
  relationship: string;
}

export interface SearchGraphData {
  nodes: SearchGraphNode[];
  edges: SearchGraphEdge[];
  root_ids: string[];
  stats: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: string;
  metadata?: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Tracing & Cost
// ---------------------------------------------------------------------------

/** A single LLM call recorded during the pipeline. */
export interface LLMCallInfo {
  id: string;
  model: string;
  purpose: string;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  latency_ms: number;
  timestamp: string;
}

/** Running cost state accumulated during the pipeline. */
export interface CostState {
  total_usd: number;
  total_tokens: number;
  call_count: number;
}

// ---------------------------------------------------------------------------
// Grounding
// ---------------------------------------------------------------------------

/** Grounding verdict for a geographic level. */
export type GroundingVerdict =
  | 'GROUNDED'
  | 'SUPPORTED'
  | 'UNCERTAIN'
  | 'WEAKENED'
  | 'CONTRADICTED';

/** Grounding result for a single geographic level. */
export interface GroundingInfo {
  level: string;
  value: string | null;
  verdict: GroundingVerdict;
  confidence: number;
  supporting_count: number;
  contradicting_count: number;
}

// ---------------------------------------------------------------------------
// Live evidence feed
// ---------------------------------------------------------------------------

/** A single evidence item discovered during pipeline execution. */
export interface LiveEvidence {
  id: string;
  source: string;
  content: string;
  confidence: number;
  timestamp: string;
}

// ---------------------------------------------------------------------------
// Pipeline step timing
// ---------------------------------------------------------------------------

/** Timing info for a completed pipeline step. */
export interface StepTiming {
  name: string;
  status: 'running' | 'completed' | 'error';
  start_time: number;
  duration_ms?: number;
  evidence_count?: number;
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

export interface SessionState {
  session_id: string;
  candidates: CandidateResult[];
  evidence_count: number;
  search_graph?: SearchGraphData;
  messages: ChatMessage[];
}
