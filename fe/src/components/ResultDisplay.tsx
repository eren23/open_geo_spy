import React from 'react';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Fix for default marker icons in react-leaflet
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

interface Location {
  lat: number;
  lon: number;
  name: string;
  confidence: number;
  analysis?: {
    features?: any;
    description?: string;
    candidates?: any[];
    gemini_analysis?: string;
  };
  reasoning?: string;
}

interface ResultDisplayProps {
  location: Location;
}

function ResultDisplay({ location }: ResultDisplayProps) {
  if (!location) return null;

  return (
    <div style={{ marginTop: '20px' }}>
      <MapContainer 
        center={[location.lat, location.lon]} 
        zoom={13} 
        style={{ height: '400px', width: '100%', borderRadius: '8px', marginBottom: '20px' }}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />
        <Marker position={[location.lat, location.lon]}>
          <Popup>
            <strong>{location.name}</strong><br/>
            Confidence: {(location.confidence * 100).toFixed(1)}%
          </Popup>
        </Marker>
      </MapContainer>

      <h3>Location Details</h3>
      <p><strong>Name:</strong> {location.name}</p>
      <p><strong>Coordinates:</strong> {location.lat.toFixed(6)}, {location.lon.toFixed(6)}</p>
      <p><strong>Confidence:</strong> {(location.confidence * 100).toFixed(1)}%</p>
      {location.reasoning && (
        <p><strong>Reasoning:</strong> {location.reasoning}</p>
      )}

      {location.analysis && (
        <div style={{ marginTop: '20px' }}>
          <h3>Analysis Details</h3>
          
          <details style={{ marginBottom: '10px' }}>
            <summary>Features</summary>
            <pre style={{ whiteSpace: 'pre-wrap', fontSize: '14px', background: '#f5f5f5', padding: '10px' }}>
              {JSON.stringify(location.analysis.features, null, 2)}
            </pre>
          </details>

          {location.analysis.description && (
            <details style={{ marginBottom: '10px' }}>
              <summary>Description</summary>
              <div style={{ whiteSpace: 'pre-wrap', padding: '10px' }}>
                {location.analysis.description}
              </div>
            </details>
          )}

          {location.analysis.candidates && location.analysis.candidates.length > 0 && (
            <details style={{ marginBottom: '10px' }}>
              <summary>Location Candidates</summary>
              <div style={{ padding: '10px' }}>
                {location.analysis.candidates.map((candidate, index) => (
                  <div key={index} style={{ marginBottom: '10px' }}>
                    <p><strong>{candidate.name}</strong> ({candidate.lat}, {candidate.lon})</p>
                    <p style={{ fontSize: '14px', color: '#666' }}>
                      Confidence: {(candidate.confidence * 100).toFixed(1)}%
                      {candidate.source && ` • Source: ${candidate.source}`}
                    </p>
                  </div>
                ))}
              </div>
            </details>
          )}

          {location.analysis.gemini_analysis && (
            <details style={{ marginBottom: '10px' }}>
              <summary>Gemini Analysis</summary>
              <div style={{ whiteSpace: 'pre-wrap', padding: '10px' }}>
                {location.analysis.gemini_analysis}
              </div>
            </details>
          )}
        </div>
      )}
    </div>
  );
}

export default ResultDisplay; 