import { useCallback, useMemo, useRef, useState } from 'react';
import type { Map as LeafletMap } from 'leaflet';
import type { EvidenceItem } from './types';
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
    evidenceSummary,
    selectedCandidateRank,
    selectCandidate,
    create,
    sendMessage,
    cancel,
  } = useSession();

  const [locationHint, setLocationHint] = useState('');
  const mapRef = useRef<LeafletMap | null>(null);

  const handleUpload = (file: File) => {
    create(file, locationHint || undefined);
  };

  const handleNewAnalysis = useCallback(() => {
    cancel();
    setLocationHint('');
  }, [cancel]);

  // Aggregate all unique evidence items across candidates + pipeline summary
  const allEvidences = useMemo<EvidenceItem[]>(() => {
    const seen = new Set<string>();
    const merged: EvidenceItem[] = [];

    // First: add pipeline top_evidence (most comprehensive)
    if (evidenceSummary?.top_evidence?.length) {
      for (const ev of evidenceSummary.top_evidence) {
        const key = `${ev.source}:${ev.content}`;
        if (!seen.has(key)) {
          seen.add(key);
          merged.push(ev);
        }
      }
    }

    // Then: merge evidence from all candidates (to catch any missed)
    for (const c of candidates) {
      for (const ev of c.evidence_trail) {
        const key = `${ev.source}:${ev.content}`;
        if (!seen.has(key)) {
          seen.add(key);
          merged.push(ev);
        }
      }
    }

    return merged;
  }, [candidates, evidenceSummary]);

  const hasSession = !!sessionId;
  const showChat = hasSession || (loading && messages.length > 0);

  // Left panel: upload or chat
  const leftPanel = (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200 bg-white flex-shrink-0 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-gray-900">OpenGeoSpy</h1>
          <p className="text-xs text-gray-500">Multi-agent geolocation with evidence tracking</p>
        </div>
        {showChat && (
          <button
            type="button"
            onClick={handleNewAnalysis}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
            </svg>
            New Analysis
          </button>
        )}
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
        /* Chat view */
        <div className="flex-1 overflow-hidden flex flex-col">
          {error && (
            <div className="mx-4 mt-2 bg-red-50 border border-red-200 rounded-lg p-2 flex-shrink-0">
              <p className="text-red-700 text-xs">{error.message}</p>
            </div>
          )}
          <div className="flex-1 overflow-hidden">
            <ChatContainer
              messages={messages}
              onSendMessage={sendMessage}
              loading={loading}
            />
          </div>
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
            pipelineEvidences={allEvidences}
          />
        </div>
      )}
    </div>
  );

  return <Layout left={leftPanel} right={rightPanel} />;
}

export default App;
