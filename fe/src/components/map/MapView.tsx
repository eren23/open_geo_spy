import { useEffect, useRef } from 'react';
import { MapContainer, TileLayer, useMap } from 'react-leaflet';
import L from 'leaflet';
import type { CandidateResult } from '../../types';
import CandidateMarkers from './CandidateMarkers';

import 'leaflet/dist/leaflet.css';

// Fix leaflet default marker icons for bundlers
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png';
import markerIcon from 'leaflet/dist/images/marker-icon.png';
import markerShadow from 'leaflet/dist/images/marker-shadow.png';

delete (L.Icon.Default.prototype as unknown as Record<string, unknown>)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
});

// ---------------------------------------------------------------------------
// FitBounds - auto-zooms when candidates change
// ---------------------------------------------------------------------------

function FitBounds({ candidates }: { candidates: CandidateResult[] }) {
  const map = useMap();
  const prevLengthRef = useRef(candidates.length);

  useEffect(() => {
    const points = candidates
      .filter((c) => c.latitude != null && c.longitude != null)
      .map((c) => [c.latitude!, c.longitude!] as [number, number]);

    if (points.length === 0) return;

    // Only fit bounds when the candidates actually change
    if (points.length !== prevLengthRef.current || prevLengthRef.current === 0) {
      const bounds = L.latLngBounds(points.map(([lat, lng]) => L.latLng(lat, lng)));
      map.fitBounds(bounds, { padding: [40, 40], maxZoom: 14 });
    }

    prevLengthRef.current = points.length;
  }, [candidates, map]);

  return null;
}

// ---------------------------------------------------------------------------
// MapView
// ---------------------------------------------------------------------------

interface MapViewProps {
  candidates: CandidateResult[];
  /** Optional ref forwarded to the underlying Leaflet Map instance */
  mapRef?: React.MutableRefObject<L.Map | null>;
}

const DEFAULT_CENTER: [number, number] = [20, 0];
const DEFAULT_ZOOM = 2;

export default function MapView({ candidates, mapRef }: MapViewProps) {
  return (
    <MapContainer
      center={DEFAULT_CENTER}
      zoom={DEFAULT_ZOOM}
      className="h-full w-full"
      ref={(instance) => {
        if (mapRef && instance) {
          mapRef.current = instance;
        }
      }}
    >
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      />
      <CandidateMarkers candidates={candidates} />
      <FitBounds candidates={candidates} />
    </MapContainer>
  );
}
