import { useMemo } from 'react';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface StreetViewEmbedProps {
  latitude: number;
  longitude: number;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

function isValidCoordinate(lat: number, lng: number): boolean {
  return (
    typeof lat === 'number' &&
    typeof lng === 'number' &&
    !Number.isNaN(lat) &&
    !Number.isNaN(lng) &&
    lat >= -90 &&
    lat <= 90 &&
    lng >= -180 &&
    lng <= 180
  );
}

// ---------------------------------------------------------------------------
// StreetViewEmbed -- Mapillary viewer at given coordinates
// ---------------------------------------------------------------------------

export default function StreetViewEmbed({ latitude, longitude }: StreetViewEmbedProps) {
  const valid = useMemo(
    () => isValidCoordinate(latitude, longitude),
    [latitude, longitude],
  );

  const mapillaryUrl = useMemo(() => {
    if (!valid) return '';
    // Mapillary embed URL centered on coordinates
    // Using the Mapillary map view centered at the given lat/lng with a reasonable zoom
    return `https://www.mapillary.com/embed?lat=${latitude}&lng=${longitude}&z=17&dateFrom=2020-01-01`;
  }, [latitude, longitude, valid]);

  if (!valid) {
    return (
      <div className="rounded-xl border border-gray-200 bg-white overflow-hidden">
        <div className="flex flex-col items-center justify-center h-64 bg-gray-50">
          {/* Map pin icon */}
          <svg
            className="h-12 w-12 text-gray-300 mb-3"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.5}
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M15 10.5a3 3 0 11-6 0 3 3 0 016 0z"
            />
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M19.5 10.5c0 7.142-7.5 11.25-7.5 11.25S4.5 17.642 4.5 10.5a7.5 7.5 0 1115 0z"
            />
          </svg>
          <p className="text-sm text-gray-500 font-medium">Invalid Coordinates</p>
          <p className="text-xs text-gray-400 mt-1">
            Provide valid latitude ({latitude}) and longitude ({longitude}) to view street-level imagery.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-gray-200 bg-white overflow-hidden">
      {/* Header */}
      <div className="px-4 py-2.5 border-b border-gray-200 flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-gray-900">Street View</h3>
          <p className="text-[11px] text-gray-500 font-mono">
            {latitude.toFixed(6)}, {longitude.toFixed(6)}
          </p>
        </div>
        <a
          href={`https://www.mapillary.com/app/?lat=${latitude}&lng=${longitude}&z=17`}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs text-blue-600 hover:text-blue-800 hover:underline flex items-center gap-1"
        >
          Open in Mapillary
          <svg
            className="h-3 w-3"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={2}
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M13.5 6H5.25A2.25 2.25 0 003 8.25v10.5A2.25 2.25 0 005.25 21h10.5A2.25 2.25 0 0018 18.75V10.5m-10.5 6L21 3m0 0h-5.25M21 3v5.25"
            />
          </svg>
        </a>
      </div>

      {/* Mapillary iframe */}
      <div className="relative" style={{ paddingBottom: '56.25%' /* 16:9 aspect */ }}>
        <iframe
          src={mapillaryUrl}
          title={`Mapillary street view at ${latitude}, ${longitude}`}
          className="absolute inset-0 w-full h-full"
          allow="fullscreen"
          loading="lazy"
          style={{ border: 'none' }}
        />
      </div>
    </div>
  );
}
