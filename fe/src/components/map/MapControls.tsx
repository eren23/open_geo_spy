import { useState, type RefObject } from 'react';
import type { Map as LeafletMap } from 'leaflet';
import type { CandidateResult, EvidenceItem } from '../../types';
import EvidenceCard from '../evidence/EvidenceCard';
import ConfidenceWaterfall from '../evidence/ConfidenceWaterfall';
import ConfidenceBadge from '../evidence/ConfidenceBadge';
import ProvenanceDashboard from '../evidence/ProvenanceDashboard';

// ---------------------------------------------------------------------------
// Rank badge color helpers (mirrors CandidateMarkers)
// ---------------------------------------------------------------------------

const RANK_BG: Record<number, string> = {
  1: 'bg-yellow-500',
  2: 'bg-gray-400',
  3: 'bg-amber-700',
};

function rankBg(rank: number): string {
  return RANK_BG[rank] ?? 'bg-gray-500';
}

// ---------------------------------------------------------------------------
// MapControls
// ---------------------------------------------------------------------------

interface MapControlsProps {
  candidates: CandidateResult[];
  mapRef: RefObject<LeafletMap | null>;
  selectedCandidateRank: number;
  selectCandidate: (rank: number) => void;
  pipelineEvidences?: EvidenceItem[];
}

export default function MapControls({
  candidates,
  mapRef,
  selectedCandidateRank,
  selectCandidate,
  pipelineEvidences,
}: MapControlsProps) {
  const [detailExpanded, setDetailExpanded] = useState(true);
  const [provenanceExpanded, setProvenanceExpanded] = useState(false);

  const validCandidates = candidates.filter(
    (c) => c.latitude != null && c.longitude != null,
  );

  if (validCandidates.length === 0) return null;

  const selected = validCandidates.find((c) => c.rank === selectedCandidateRank) ?? validCandidates[0];

  // Use selected candidate's evidence trail, or fall back to full pipeline evidences
  const provenanceEvidences = selected?.evidence_trail.length
    ? selected.evidence_trail
    : (pipelineEvidences ?? []);

  function zoomAndSelect(c: CandidateResult) {
    selectCandidate(c.rank);
    mapRef.current?.flyTo([c.latitude!, c.longitude!], 14, { duration: 1.2 });
  }

  return (
    <div className="flex flex-col gap-2 w-72 max-h-[calc(50vh-16px)] md:max-h-[calc(100vh-100px)] overflow-y-auto">
      {/* Header */}
      <span className="text-[11px] font-semibold uppercase tracking-wider text-gray-400">
        Candidates
      </span>

      {/* Candidate list */}
      {validCandidates.map((c) => {
        const isSelected = c.rank === selected.rank;
        const locationParts = [c.city, c.region, c.country].filter(Boolean);

        return (
          <button
            key={c.rank}
            type="button"
            onClick={() => zoomAndSelect(c)}
            className={`flex flex-col gap-1 rounded-lg px-3 py-2 text-left text-sm
                       bg-white border transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500
                       ${isSelected ? 'border-blue-500 bg-blue-50 ring-1 ring-blue-200' : 'border-gray-200 hover:border-blue-400 hover:bg-blue-50'}`}
          >
            <div className="flex items-center gap-2 w-full">
              <span
                className={`inline-flex items-center justify-center w-5 h-5 rounded-full text-[10px] font-bold text-white flex-shrink-0 ${rankBg(c.rank)}`}
              >
                {c.rank}
              </span>
              <span className="truncate font-medium text-gray-800">{c.name}</span>
              <span className="ml-auto flex-shrink-0">
                <ConfidenceBadge confidence={c.confidence} />
              </span>
            </div>
            {locationParts.length > 0 && (
              <span className="text-[11px] text-gray-500 pl-7 truncate">
                {locationParts.join(', ')}
              </span>
            )}
            <div className="flex items-center gap-2 pl-7">
              {c.source_diversity > 0 && (
                <span className="rounded bg-blue-50 px-1.5 py-0.5 text-[10px] text-blue-600 font-medium">
                  {c.source_diversity} sources
                </span>
              )}
              {c.evidence_trail.length > 0 && (
                <span className="rounded bg-gray-100 px-1.5 py-0.5 text-[10px] text-gray-500">
                  {c.evidence_trail.length} evidence
                </span>
              )}
            </div>
          </button>
        );
      })}

      {/* Selected candidate detail panel */}
      {selected && (
        <div className="rounded-xl border border-gray-200 bg-white overflow-hidden mt-1">
          {/* Detail header */}
          <button
            type="button"
            onClick={() => setDetailExpanded((v) => !v)}
            className="w-full flex items-center justify-between px-3 py-2 text-left bg-gray-50 border-b border-gray-200 hover:bg-gray-100 transition-colors"
          >
            <span className="text-xs font-semibold text-gray-700">
              Details — #{selected.rank} {selected.name}
            </span>
            <svg
              className={`h-4 w-4 text-gray-400 transition-transform ${detailExpanded ? 'rotate-180' : ''}`}
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={2}
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
            </svg>
          </button>

          {detailExpanded && (
            <div className="p-3 space-y-3">
              {/* Reasoning */}
              <div>
                <span className="text-[11px] font-semibold uppercase tracking-wider text-gray-400">
                  Reasoning
                </span>
                <p className="text-xs text-gray-700 mt-1 leading-relaxed whitespace-pre-wrap">
                  {selected.reasoning}
                </p>
              </div>

              {/* Visual match score */}
              {selected.visual_match_score != null && (
                <div className="flex items-center gap-2">
                  <span className="text-[11px] font-semibold uppercase tracking-wider text-gray-400">
                    Visual Match
                  </span>
                  <span className="rounded bg-indigo-50 px-2 py-0.5 text-[11px] font-semibold text-indigo-700">
                    {(selected.visual_match_score * 100).toFixed(0)}%
                  </span>
                </div>
              )}

              {/* Coordinates */}
              {selected.latitude != null && selected.longitude != null && (
                <div>
                  <span className="text-[11px] font-semibold uppercase tracking-wider text-gray-400">
                    Coordinates
                  </span>
                  <p className="text-xs font-mono text-gray-600 mt-0.5">
                    {selected.latitude.toFixed(4)}, {selected.longitude.toFixed(4)}
                  </p>
                </div>
              )}

              {/* Evidence trail */}
              {selected.evidence_trail.length > 0 && (
                <div>
                  <span className="text-[11px] font-semibold uppercase tracking-wider text-gray-400">
                    Evidence ({selected.evidence_trail.length} items)
                  </span>
                  <div className="mt-1.5 space-y-1.5">
                    {selected.evidence_trail.map((ev, i) => (
                      <EvidenceCard key={`${ev.source}-${i}`} evidence={ev} />
                    ))}
                  </div>
                </div>
              )}

              {/* Confidence waterfall */}
              {selected.evidence_trail.length > 0 && (
                <ConfidenceWaterfall evidences={selected.evidence_trail} />
              )}
            </div>
          )}
        </div>
      )}

      {/* Provenance Dashboard — collapsible pipeline flow viz */}
      {(provenanceEvidences.length > 0) && (
        <div className="rounded-xl border border-gray-200 bg-white overflow-hidden mt-1">
          <button
            type="button"
            onClick={() => setProvenanceExpanded((v) => !v)}
            className="w-full flex items-center justify-between px-3 py-2 text-left bg-gray-50 border-b border-gray-200 hover:bg-gray-100 transition-colors"
          >
            <span className="text-xs font-semibold text-gray-700">
              Evidence Provenance
              <span className="text-[10px] text-gray-400 ml-1">
                {selected.evidence_trail.length ? `(#${selected.rank})` : '(pipeline)'}
              </span>
            </span>
            <svg
              className={`h-4 w-4 text-gray-400 transition-transform ${provenanceExpanded ? 'rotate-180' : ''}`}
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={2}
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
            </svg>
          </button>
          {provenanceExpanded && (
            <ProvenanceDashboard evidences={provenanceEvidences} headless />
          )}
        </div>
      )}
    </div>
  );
}
