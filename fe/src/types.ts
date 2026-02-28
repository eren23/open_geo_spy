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
// Session
// ---------------------------------------------------------------------------

export interface SessionState {
  session_id: string;
  candidates: CandidateResult[];
  evidence_count: number;
  search_graph?: SearchGraphData;
  messages: ChatMessage[];
}
