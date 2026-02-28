import type { Map as LeafletMap } from 'leaflet';
import type { CandidateResult } from '../../types';

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
  mapRef: React.RefObject<LeafletMap | null>;
}

export default function MapControls({ candidates, mapRef }: MapControlsProps) {
  const validCandidates = candidates.filter(
    (c) => c.latitude != null && c.longitude != null,
  );

  if (validCandidates.length === 0) return null;

  function zoomTo(lat: number, lng: number) {
    mapRef.current?.flyTo([lat, lng], 14, { duration: 1.2 });
  }

  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-[11px] font-semibold uppercase tracking-wider text-gray-400 mb-0.5">
        Candidates
      </span>
      {validCandidates.map((c) => (
        <button
          key={c.rank}
          type="button"
          onClick={() => zoomTo(c.latitude!, c.longitude!)}
          className="flex items-center gap-2 rounded-lg px-3 py-2 text-left text-sm
                     bg-white border border-gray-200 hover:border-blue-400 hover:bg-blue-50
                     transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <span
            className={`inline-flex items-center justify-center w-5 h-5 rounded-full text-[10px] font-bold text-white ${rankBg(c.rank)}`}
          >
            {c.rank}
          </span>
          <span className="truncate font-medium text-gray-800">{c.name}</span>
          <span className="ml-auto text-xs text-gray-400">
            {(c.confidence * 100).toFixed(0)}%
          </span>
        </button>
      ))}
    </div>
  );
}
