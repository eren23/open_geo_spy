import React, { useState } from 'react';
import FileUpload from './components/FileUpload';
import ResultDisplay from './components/ResultDisplay';
import './App.css';

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
}

function App() {
  const [location1, setLocation1] = useState<Location | null>(null);
  const [location2, setLocation2] = useState<Location | null>(null);
  const [loading1, setLoading1] = useState(false);
  const [loading2, setLoading2] = useState(false);

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '20px' }}>
      <h1 style={{ textAlign: 'center', marginBottom: '30px' }}>GeoLocator</h1>
      
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
        <div>
          <FileUpload
            onUploadSuccess={(result) => setLocation1(result.location)}
            setLoading={setLoading1}
            endpoint="/api/locate"
            title="Single Image Upload"
            description="Upload a single image for location analysis"
          />
          {loading1 && <p>Processing...</p>}
          {location1 && <ResultDisplay location={location1} />}
        </div>

        <div>
          <FileUpload
            onUploadSuccess={(result) => setLocation2(result.location)}
            setLoading={setLoading2}
            endpoint="/api/analyze-multimodal"
            title="Multi-Modal Analysis"
            description="Upload multiple images or videos for analysis"
          />
          {loading2 && <p>Processing...</p>}
          {location2 && <ResultDisplay location={location2} />}
        </div>
      </div>
    </div>
  );
}

export default App; 