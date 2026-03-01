import { memo, useEffect, useState, type RefObject } from 'react';
import type { Map as LeafletMap } from 'leaflet';
import type { CandidateResult, EvidenceItem } from '../../types';
import EvidenceCard from '../evidence/EvidenceCard';
import ConfidenceWaterfall from '../evidence/ConfidenceWaterfall';
import ConfidenceBadge from '../evidence/ConfidenceBadge';
import ProvenanceDashboard from '../evidence/ProvenanceDashboard';

// ---------------------------------------------------------------------------
// Rank badge color helpers
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
// Tab types
// ---------------------------------------------------------------------------

type TabId = 'candidates' | 'evidence' | 'provenance';

const TABS: { id: TabId; label: string }[] = [
  { id: 'candidates', label: 'Candidates' },
  { id: 'evidence', label: 'Evidence' },
  { id: 'provenance', label: 'Provenance' },
];

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

export default memo(function MapControls({
  candidates,
  mapRef,
  selectedCandidateRank,
  selectCandidate,
  pipelineEvidences,
}: MapControlsProps) {
  const [activeTab, setActiveTab] = useState<TabId>('candidates');
  const [showAllEvidence, setShowAllEvidence] = useState(false);

  // Reset "show all" toggle when candidate changes
  useEffect(() => {
    setShowAllEvidence(false);
  }, [selectedCandidateRank]);

  const validCandidates = candidates.filter(
    (c) => c.latitude != null && c.longitude != null,
  );

  if (validCandidates.length === 0) return null;

  const selected = validCandidates.find((c) => c.rank === selectedCandidateRank) ?? validCandidates[0];

  // Use pipeline evidences when candidate has few, otherwise candidate's own
  const candidateEvidence = selected?.evidence_trail ?? [];
  const effectiveEvidence = candidateEvidence.length >= 3
    ? candidateEvidence
    : (pipelineEvidences?.length ? pipelineEvidences : candidateEvidence);

  const provenanceEvidences = pipelineEvidences?.length ? pipelineEvidences : candidateEvidence;

  const totalPipelineCount = pipelineEvidences?.length ?? 0;

  function zoomAndSelect(c: CandidateResult) {
    selectCandidate(c.rank);
    mapRef.current?.flyTo([c.latitude!, c.longitude!], 14, { duration: 1.2 });
  }

  // Evidence items to display (with show all toggle)
  const displayEvidence = showAllEvidence ? effectiveEvidence : effectiveEvidence.slice(0, 5);

  return (
    <div className="flex flex-col gap-1.5 w-80 max-w-[calc(100%-16px)] max-h-[calc(100vh-100px)] overflow-hidden">
      {/* Tab bar */}
      <div className="flex rounded-lg bg-gray-100 p-0.5">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => setActiveTab(tab.id)}
            className={`flex-1 text-[11px] font-medium py-1.5 rounded-md transition-colors
              ${activeTab === tab.id
                ? 'bg-white text-gray-900 shadow-sm'
                : 'text-gray-500 hover:text-gray-700'}`}
          >
            {tab.label}
            {tab.id === 'evidence' && totalPipelineCount > 0 && (
              <span className="ml-1 text-[10px] text-gray-400">({totalPipelineCount})</span>
            )}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-y-auto">
        {/* --- Candidates tab --- */}
        {activeTab === 'candidates' && (
          <div className="flex flex-col gap-1.5">
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

            {/* Selected candidate detail */}
            {selected && (
              <div className="rounded-xl border border-gray-200 bg-white overflow-hidden">
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
                </div>
              </div>
            )}
          </div>
        )}

        {/* --- Evidence tab --- */}
        {activeTab === 'evidence' && (
          <div className="flex flex-col gap-2">
            {/* Evidence count header */}
            <div className="flex items-center justify-between px-1">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-gray-400">
                Evidence for #{selected.rank} {selected.name}
              </span>
              {totalPipelineCount > 0 && candidateEvidence.length < totalPipelineCount && (
                <span className="text-[10px] text-gray-400">
                  {candidateEvidence.length} direct / {totalPipelineCount} total
                </span>
              )}
            </div>

            {/* Evidence cards */}
            {displayEvidence.length > 0 ? (
              <div className="space-y-1.5">
                {displayEvidence.map((ev, i) => (
                  <EvidenceCard key={`${ev.source}-${ev.content?.slice(0, 20)}-${i}`} evidence={ev} />
                ))}
              </div>
            ) : (
              <p className="text-xs text-gray-400 py-4 text-center">No evidence available</p>
            )}

            {/* Show all button */}
            {effectiveEvidence.length > 5 && (
              <button
                type="button"
                onClick={() => setShowAllEvidence((v) => !v)}
                className="text-[11px] text-blue-600 hover:text-blue-800 font-medium py-1"
              >
                {showAllEvidence ? 'Show less' : `Show all ${effectiveEvidence.length} items`}
              </button>
            )}

            {/* Confidence waterfall */}
            {effectiveEvidence.length > 0 && (
              <ConfidenceWaterfall evidences={effectiveEvidence} />
            )}
          </div>
        )}

        {/* --- Provenance tab --- */}
        {activeTab === 'provenance' && (
          <div>
            {provenanceEvidences.length > 0 ? (
              <ProvenanceDashboard evidences={provenanceEvidences} headless />
            ) : (
              <p className="text-xs text-gray-400 py-4 text-center">No provenance data</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
});
