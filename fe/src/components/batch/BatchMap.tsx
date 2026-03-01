import { useMemo, useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import type { BatchItem } from './BatchDashboard';

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
// Types
// ---------------------------------------------------------------------------

interface BatchMapProps {
  items: BatchItem[];
}

/** Completed item with valid coordinates */
interface MapItem {
  item: BatchItem;
  index: number; // 1-based display number
  lat: number;
  lng: number;
}

// ---------------------------------------------------------------------------
// Numbered marker icon (SVG data URI)
// ---------------------------------------------------------------------------

function createNumberedIcon(num: number): L.DivIcon {
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="30" height="42" viewBox="0 0 30 42">
      <path d="M15 0C6.716 0 0 6.716 0 15c0 10.5 15 27 15 27s15-16.5 15-27C30 6.716 23.284 0 15 0z"
            fill="#3B82F6" stroke="#fff" stroke-width="1.5"/>
      <circle cx="15" cy="14" r="10" fill="#fff" opacity="0.9"/>
      <text x="15" y="18" text-anchor="middle" font-size="${num > 99 ? 9 : 12}" font-weight="bold"
            fill="#3B82F6" font-family="system-ui, sans-serif">${num}</text>
    </svg>`;

  return L.divIcon({
    html: svg,
    className: '',
    iconSize: [30, 42],
    iconAnchor: [15, 42],
    popupAnchor: [0, -42],
  });
}

// ---------------------------------------------------------------------------
// FitBounds helper
// ---------------------------------------------------------------------------

function FitBounds({ mapItems }: { mapItems: MapItem[] }) {
  const map = useMap();
  const prevCountRef = useRef(mapItems.length);

  useEffect(() => {
    if (mapItems.length === 0) return;

    // Only refit when count changes
    if (mapItems.length !== prevCountRef.current || prevCountRef.current === 0) {
      const bounds = L.latLngBounds(
        mapItems.map((mi) => L.latLng(mi.lat, mi.lng)),
      );
      map.fitBounds(bounds, { padding: [40, 40], maxZoom: 14 });
    }

    prevCountRef.current = mapItems.length;
  }, [mapItems, map]);

  return null;
}

// ---------------------------------------------------------------------------
// BatchMap
// ---------------------------------------------------------------------------

const DEFAULT_CENTER: [number, number] = [20, 0];
const DEFAULT_ZOOM = 2;

export default function BatchMap({ items }: BatchMapProps) {
  const mapItems = useMemo<MapItem[]>(() => {
    const result: MapItem[] = [];
    let idx = 1;
    for (const item of items) {
      if (
        item.status === 'completed' &&
        item.result &&
        item.result.latitude != null &&
        item.result.longitude != null
      ) {
        result.push({
          item,
          index: idx++,
          lat: item.result.latitude,
          lng: item.result.longitude,
        });
      }
    }
    return result;
  }, [items]);

  if (items.length === 0) {
    return (
      <div className="rounded-xl border border-gray-200 bg-white p-8 text-center">
        <p className="text-sm text-gray-500">No batch items to display on map.</p>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-gray-200 bg-white overflow-hidden">
      {/* Header */}
      <div className="px-4 py-2.5 border-b border-gray-200 flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-gray-900">Batch Results Map</h3>
          <p className="text-xs text-gray-500">
            {mapItems.length} of {items.length} items with coordinates
          </p>
        </div>
      </div>

      {/* Map */}
      <div style={{ height: 400 }}>
        {mapItems.length > 0 ? (
          <MapContainer
            center={DEFAULT_CENTER}
            zoom={DEFAULT_ZOOM}
            className="h-full w-full"
          >
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            />
            {mapItems.map((mi) => (
              <Marker
                key={mi.item.id}
                position={[mi.lat, mi.lng]}
                icon={createNumberedIcon(mi.index)}
              >
                <Popup>
                  <div className="min-w-[180px]">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-blue-500 text-xs font-bold text-white">
                        {mi.index}
                      </span>
                      <span className="font-semibold text-sm text-gray-900 truncate">
                        {mi.item.filename}
                      </span>
                    </div>
                    {mi.item.result && (
                      <>
                        <p className="text-sm text-gray-700">{mi.item.result.name}</p>
                        {mi.item.result.country && (
                          <p className="text-xs text-gray-500">{mi.item.result.country}</p>
                        )}
                        <div className="flex items-center gap-1.5 text-xs mt-1">
                          <span className="text-gray-500">Confidence:</span>
                          <span
                            className={`font-medium ${
                              mi.item.result.confidence > 0.7
                                ? 'text-green-600'
                                : mi.item.result.confidence > 0.4
                                  ? 'text-yellow-600'
                                  : 'text-red-500'
                            }`}
                          >
                            {(mi.item.result.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                        <p className="text-[10px] text-gray-400 font-mono mt-1">
                          {mi.lat.toFixed(6)}, {mi.lng.toFixed(6)}
                        </p>
                      </>
                    )}
                  </div>
                </Popup>
              </Marker>
            ))}
            <FitBounds mapItems={mapItems} />
          </MapContainer>
        ) : (
          <div className="h-full flex items-center justify-center bg-gray-50">
            <div className="text-center">
              <svg
                className="h-10 w-10 text-gray-300 mx-auto mb-2"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth={1.5}
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M9 6.75V15m6-6v8.25m.503 3.498l4.875-2.437c.381-.19.622-.58.622-1.006V4.82c0-.836-.88-1.38-1.628-1.006l-3.869 1.934c-.317.159-.69.159-1.006 0L9.503 3.252a1.125 1.125 0 00-1.006 0L3.622 5.689C3.24 5.88 3 6.27 3 6.695V19.18c0 .836.88 1.38 1.628 1.006l3.869-1.934c.317-.159.69-.159 1.006 0l4.994 2.497c.317.158.69.158 1.006 0z"
                />
              </svg>
              <p className="text-sm text-gray-500">No completed items with coordinates yet.</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
