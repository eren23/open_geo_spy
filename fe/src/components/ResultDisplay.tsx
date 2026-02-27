import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import L from 'leaflet';
import type { LocateResult, EvidenceItem } from '../api';

// Fix leaflet marker icons for bundlers
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png';
import markerIcon from 'leaflet/dist/images/marker-icon.png';
import markerShadow from 'leaflet/dist/images/marker-shadow.png';

delete (L.Icon.Default.prototype as unknown as Record<string, unknown>)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
});

interface ResultDisplayProps {
  result: LocateResult;
}

function ResultDisplay({ result }: ResultDisplayProps) {
  const lat = result.latitude ?? 0;
  const lon = result.longitude ?? 0;
  const hasCoords = result.latitude != null && result.longitude != null;

  return (
    <div className="space-y-6">
      {/* Map */}
      {hasCoords && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
          <MapContainer
            center={[lat, lon]}
            zoom={13}
            style={{ height: '400px', width: '100%' }}
          >
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            />
            <Marker position={[lat, lon]}>
              <Popup>
                <strong>{result.name}</strong>
                <br />
                Confidence: {(result.confidence * 100).toFixed(1)}%
              </Popup>
            </Marker>
          </MapContainer>
        </div>
      )}

      {/* Location Details */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">{result.name}</h2>
            <p className="text-sm text-gray-500 mt-1">
              {[result.city, result.region, result.country].filter(Boolean).join(', ')}
            </p>
          </div>
          <div className="text-right">
            <div
              className={`text-2xl font-bold ${
                result.confidence > 0.7
                  ? 'text-green-600'
                  : result.confidence > 0.4
                    ? 'text-yellow-600'
                    : 'text-red-500'
              }`}
            >
              {(result.confidence * 100).toFixed(0)}%
            </div>
            <div className="text-xs text-gray-400">confidence</div>
            {result.verified && (
              <span className="inline-block mt-1 text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full">
                Verified
              </span>
            )}
          </div>
        </div>

        {hasCoords && (
          <p className="text-sm text-gray-500 mb-3">
            {lat.toFixed(6)}, {lon.toFixed(6)}
          </p>
        )}

        {result.verification_warning && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 mb-3">
            <p className="text-yellow-700 text-sm">{result.verification_warning}</p>
          </div>
        )}

        {/* Reasoning */}
        {result.reasoning && (
          <div className="mt-4">
            <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-2">
              Reasoning
            </h3>
            <p className="text-sm text-gray-700 whitespace-pre-wrap">{result.reasoning}</p>
          </div>
        )}

        {/* Stats */}
        <div className="flex gap-4 mt-4 pt-4 border-t border-gray-100">
          <Stat label="Evidence" value={result.total_evidence_count.toString()} />
          <Stat label="Time" value={`${(result.elapsed_ms / 1000).toFixed(1)}s`} />
          <Stat
            label="Sources"
            value={
              (result.evidence_summary as Record<string, unknown>)?.sources
                ? String(
                    (
                      (result.evidence_summary as Record<string, unknown>).sources as string[]
                    ).length,
                  )
                : '0'
            }
          />
        </div>
      </div>

      {/* Evidence Trail */}
      {result.evidence_trail.length > 0 && (
        <details className="bg-white rounded-xl shadow-sm border border-gray-200">
          <summary className="px-6 py-4 cursor-pointer text-sm font-semibold text-gray-500 uppercase tracking-wide hover:bg-gray-50 rounded-xl">
            Evidence Trail ({result.evidence_trail.length} items)
          </summary>
          <div className="px-6 pb-4 space-y-2">
            {result.evidence_trail.map((e: EvidenceItem, i: number) => (
              <div
                key={i}
                className="flex items-start gap-3 py-2 border-t border-gray-50 first:border-0"
              >
                <span
                  className={`flex-shrink-0 text-xs font-mono px-2 py-0.5 rounded ${sourceColor(e.source)}`}
                >
                  {e.source}
                </span>
                <div className="flex-grow min-w-0">
                  <p className="text-sm text-gray-700 truncate">{e.content}</p>
                  {e.url && (
                    <a
                      href={e.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs text-blue-500 hover:underline"
                    >
                      {e.url}
                    </a>
                  )}
                </div>
                <span className="flex-shrink-0 text-xs text-gray-400">
                  {(e.confidence * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-lg font-semibold text-gray-900">{value}</div>
      <div className="text-xs text-gray-400">{label}</div>
    </div>
  );
}

function sourceColor(source: string): string {
  const colors: Record<string, string> = {
    exif: 'bg-purple-100 text-purple-700',
    vlm_analysis: 'bg-blue-100 text-blue-700',
    vlm_geo: 'bg-blue-100 text-blue-700',
    ocr: 'bg-cyan-100 text-cyan-700',
    geoclip: 'bg-green-100 text-green-700',
    streetclip: 'bg-green-100 text-green-700',
    serper: 'bg-orange-100 text-orange-700',
    google_maps: 'bg-red-100 text-red-700',
    osm: 'bg-yellow-100 text-yellow-700',
    browser: 'bg-gray-100 text-gray-700',
    reasoning: 'bg-indigo-100 text-indigo-700',
    user_hint: 'bg-pink-100 text-pink-700',
  };
  return colors[source] || 'bg-gray-100 text-gray-600';
}

export default ResultDisplay;
