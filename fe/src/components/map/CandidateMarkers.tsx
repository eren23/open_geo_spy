import { Marker, Popup } from 'react-leaflet';
import L from 'leaflet';
import type { CandidateResult } from '../../types';

// ---------------------------------------------------------------------------
// Color-coded numbered markers using SVG data URIs
// ---------------------------------------------------------------------------

const RANK_COLORS: Record<number, string> = {
  1: '#EAB308', // gold
  2: '#9CA3AF', // silver
  3: '#CD7F32', // bronze
};

function getMarkerColor(rank: number): string {
  return RANK_COLORS[rank] ?? '#6B7280'; // gray for rank > 3
}

function createNumberedIcon(rank: number): L.DivIcon {
  const color = getMarkerColor(rank);
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="30" height="42" viewBox="0 0 30 42">
      <path d="M15 0C6.716 0 0 6.716 0 15c0 10.5 15 27 15 27s15-16.5 15-27C30 6.716 23.284 0 15 0z"
            fill="${color}" stroke="#fff" stroke-width="1.5"/>
      <circle cx="15" cy="14" r="10" fill="#fff" opacity="0.9"/>
      <text x="15" y="18" text-anchor="middle" font-size="12" font-weight="bold"
            fill="${color}" font-family="system-ui, sans-serif">${rank}</text>
    </svg>`;

  return L.divIcon({
    html: svg,
    className: '', // remove default leaflet-div-icon styling
    iconSize: [30, 42],
    iconAnchor: [15, 42],
    popupAnchor: [0, -42],
  });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function truncate(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen).trimEnd() + '...';
}

// ---------------------------------------------------------------------------
// CandidateMarkers
// ---------------------------------------------------------------------------

interface CandidateMarkersProps {
  candidates: CandidateResult[];
  onSelectCandidate?: (rank: number) => void;
}

export default function CandidateMarkers({ candidates, onSelectCandidate }: CandidateMarkersProps) {
  const validCandidates = candidates.filter(
    (c) => c.latitude != null && c.longitude != null,
  );

  return (
    <>
      {validCandidates.map((candidate) => (
        <Marker
          key={`candidate-${candidate.rank}`}
          position={[candidate.latitude!, candidate.longitude!]}
          icon={createNumberedIcon(candidate.rank)}
          eventHandlers={
            onSelectCandidate
              ? { click: () => onSelectCandidate(candidate.rank) }
              : undefined
          }
        >
          <Popup>
            <div className="min-w-[200px] max-w-[280px]">
              {/* Header */}
              <div className="flex items-center gap-2 mb-1">
                <span
                  className="inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold text-white flex-shrink-0"
                  style={{ backgroundColor: getMarkerColor(candidate.rank) }}
                >
                  {candidate.rank}
                </span>
                <span className="font-semibold text-sm text-gray-900">
                  {candidate.name}
                </span>
              </div>

              {/* Location */}
              {(candidate.country || candidate.region || candidate.city) && (
                <p className="text-xs text-gray-500 mb-1.5">
                  {[candidate.city, candidate.region, candidate.country].filter(Boolean).join(', ')}
                </p>
              )}

              {/* Confidence */}
              <div className="flex items-center gap-1.5 text-xs mb-1.5">
                <span className="text-gray-500">Confidence:</span>
                <span
                  className={`font-medium ${
                    candidate.confidence > 0.7
                      ? 'text-green-600'
                      : candidate.confidence > 0.4
                        ? 'text-yellow-600'
                        : 'text-red-500'
                  }`}
                >
                  {(candidate.confidence * 100).toFixed(1)}%
                </span>
              </div>

              {/* Reasoning excerpt */}
              {candidate.reasoning && (
                <p className="text-[11px] text-gray-600 mb-1.5 leading-relaxed">
                  {truncate(candidate.reasoning, 120)}
                </p>
              )}

              {/* Stats row */}
              <div className="flex items-center gap-2 flex-wrap mb-1.5">
                {candidate.evidence_trail.length > 0 && (
                  <span className="rounded bg-gray-100 px-1.5 py-0.5 text-[10px] text-gray-600">
                    {candidate.evidence_trail.length} evidence items
                  </span>
                )}
                {candidate.source_diversity > 0 && (
                  <span className="rounded bg-blue-50 px-1.5 py-0.5 text-[10px] text-blue-600">
                    {candidate.source_diversity} sources
                  </span>
                )}
                {candidate.visual_match_score != null && (
                  <span className="rounded bg-indigo-50 px-1.5 py-0.5 text-[10px] text-indigo-600">
                    Visual: {(candidate.visual_match_score * 100).toFixed(0)}%
                  </span>
                )}
              </div>

              {/* Coordinates */}
              {candidate.latitude != null && candidate.longitude != null && (
                <p className="text-[10px] font-mono text-gray-400 mb-1">
                  {candidate.latitude.toFixed(4)}, {candidate.longitude.toFixed(4)}
                </p>
              )}

              {/* Hint */}
              <p className="text-[10px] text-gray-400 italic">
                Click sidebar for full details
              </p>
            </div>
          </Popup>
        </Marker>
      ))}
    </>
  );
}
