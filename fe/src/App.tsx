import { useRef, useState } from 'react';
import type { Map as LeafletMap } from 'leaflet';
import { useSession } from './hooks/useSession';
import Layout from './components/Layout';
import UploadArea from './components/upload/UploadArea';
import ChatContainer from './components/chat/ChatContainer';
import MapView from './components/map/MapView';
import MapControls from './components/map/MapControls';

function App() {
  const {
    sessionId,
    candidates,
    messages,
    loading,
    error,
    selectedCandidateRank,
    selectCandidate,
    create,
    sendMessage,
  } = useSession();

  const [locationHint, setLocationHint] = useState('');
  const mapRef = useRef<LeafletMap | null>(null);

  const handleUpload = (file: File) => {
    create(file, locationHint || undefined);
  };

  const hasSession = !!sessionId;
  // Show chat view as soon as processing starts (loading after upload)
  // so streaming step progress is visible, not just a static spinner.
  const showChat = hasSession || (loading && messages.length > 0);

  // Left panel: upload or chat
  const leftPanel = (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200 bg-white flex-shrink-0">
        <h1 className="text-xl font-bold text-gray-900">OpenGeoSpy</h1>
        <p className="text-xs text-gray-500">Multi-agent geolocation with evidence tracking</p>
      </div>

      {!showChat ? (
        /* Upload view */
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          <input
            type="text"
            placeholder="Location hint (optional, e.g. 'Berlin, Germany')"
            value={locationHint}
            onChange={(e) => setLocationHint(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            disabled={loading}
          />
          <UploadArea onUpload={handleUpload} loading={loading} />
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-3">
              <p className="text-red-700 text-sm">{error.message}</p>
            </div>
          )}
        </div>
      ) : (
        /* Chat view — shown during processing AND after session is established */
        <div className="flex-1 overflow-hidden">
          <ChatContainer
            messages={messages}
            onSendMessage={sendMessage}
            loading={loading}
          />
        </div>
      )}
    </div>
  );

  // Right panel: map
  const rightPanel = (
    <div className="relative h-full">
      <MapView
        candidates={candidates}
        mapRef={mapRef}
        onSelectCandidate={selectCandidate}
      />
      {candidates.length > 0 && (
        <div className="absolute top-2 right-2 z-[1000]">
          <MapControls
            candidates={candidates}
            mapRef={mapRef}
            selectedCandidateRank={selectedCandidateRank}
            selectCandidate={selectCandidate}
          />
        </div>
      )}
    </div>
  );

  return <Layout left={leftPanel} right={rightPanel} />;
}

export default App;
