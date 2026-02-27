# Changelog

All notable changes to OpenGeoSpy will be documented in this file.

---

## [0.1.0] — 2026-02-28

### Multi-Agent Pipeline Architecture

- LangGraph-ready 5-agent DAG: Feature Extraction → ML Ensemble + Web Intel (parallel) → Candidate Verification → Reasoning (`src/agents/`)
- Orchestrator with SSE streaming for real-time progress updates (`src/agents/orchestrator.py`)
- Evidence chain system with SHA-256 content-hash deduplication (`src/evidence/chain.py`)
- CoVe (Chain-of-Verification) hallucination detection on agent outputs (`src/evidence/verifier.py`)

### ML Models

- **GeoCLIP** (NeurIPS '23) — continuous GPS coordinate prediction from image embeddings (`src/models/geoclip_predictor.py`)
- **StreetCLIP** — zero-shot country/city classification across 70+ countries (`src/models/streetclip_predictor.py`)
- **VLM Geo-Reasoning** — Gemini 2.5 Pro structured location inference via OpenRouter (`src/models/vlm_geo.py`)
- **Visual Similarity Scorer** — StreetCLIP cosine similarity for candidate verification (`src/models/visual_similarity.py`)

### Feature Extraction

- VLM-powered OCR with category tagging: street signs, businesses, license plates, building info (`src/extraction/ocr.py`)
- VLM visual feature extraction: landmarks, architecture, vegetation, cultural indicators (`src/extraction/features.py`)
- EXIF GPS + camera metadata parsing via piexif (`src/extraction/metadata.py`)
- Environment classification (urban / rural / coastal / etc.) for evidence weighting

### Web Intelligence

- Tiered search strategy: Serper + OSM APIs (Tier 1, parallel) → stealth browser (Tier 2, fallback)
- 8-layer browser stealth stack: WebDriver, Chrome runtime, canvas, WebGL, plugins, permissions, hardware, headless detection (`src/browser/stealth.py`)
- Browser pool with semaphore-based rate limiting (`src/browser/browser_pool.py`)
- Google Maps scraping with coordinate extraction (`src/browser/search.py`)
- OSM Nominatim + Overpass query support (`src/geo/osm_client.py`)
- Serper web + image search client (`src/geo/serper_client.py`)
- Mapillary street-level imagery client (`src/geo/mapillary_client.py`)

### Candidate Visual Verification

- 4-strategy candidate generation: OCR business names, category search, landmarks, top evidence (`src/agents/candidate_verification_agent.py`)
- Reference image fetching via Serper image search + Mapillary
- StreetCLIP embedding comparison → `VISUAL_MATCH` evidence items

### API & Frontend

- FastAPI backend: `/api/locate` (sync), `/api/locate/stream` (SSE), `/api/health` (`src/api/app.py`)
- Pydantic request/response schemas with validation (`src/api/schemas.py`)
- React 18 + Vite + TypeScript + Tailwind CSS frontend (`fe/`)
- Interactive Leaflet map with location markers (`fe/src/components/ResultDisplay.tsx`)
- Real-time pipeline progress tracker (`fe/src/components/PipelineStatus.tsx`)
- Image upload with drag-and-drop via react-dropzone (`fe/src/components/FileUpload.tsx`)

### Infrastructure

- Pydantic BaseSettings with nested env var support (`LLM__API_KEY`, flat fallbacks) (`src/config/settings.py`)
- Docker Compose: API service (Python 3.11 + Chromium + Xvfb) + Frontend service (Node 20)
- Loguru structured logging (`src/utils/logging.py`)
- Tenacity retry patterns with exponential backoff (`src/utils/retry.py`)
- Utility belt: haversine distance, weighted centroid, coordinate validation, geographic spread (`src/utils/geo_math.py`)

### Bug Fixes (post-integration)

- Fixed GeoCLIP crash: pass file path string to `model.predict()`, not PIL Image
- Fixed browser search timeouts: truncate VLM `country_clues` to 5 words each (60-char cap) + 120-char query guard
- Fixed candidate verification: assemble `ocr_text` from actual OCR dict keys so Strategy B produces candidates

---

## Roadmap to 0.2.0

### Pipeline Intelligence

1. **Patching & Iterative Refinement** — After the first pass, let the reasoning agent identify weak evidence and re-run targeted searches. Patch gaps instead of one-shot.

2. **Search Graph Architecture** — Replace the flat query list with a directed search graph. Nodes are queries, edges are "refine" / "broaden" / "pivot". Each node tracks which evidence it produced, enabling backtracking and branch pruning.

3. **Smart Query Expansion** — Use LLM to generate search variations (synonyms, local-language translations, nearby landmark associations) instead of just truncating raw VLM output.

### Output & Interaction

4. **Multi-Candidate Final Output** — Return top-N ranked candidates with individual confidence scores, evidence trails, and visual match scores. Let the user pick or inspect the reasoning for each.

5. **Chat-Like Interaction Flow** — Conversational follow-up: user can ask "why not Turkey?", "zoom into the street signs", or "try searching for X". The pipeline re-runs targeted agents based on user hints.

6. **Streaming Chat UI** — Redesign the frontend as a chat interface with message bubbles for each agent step, inline map updates, and expandable evidence cards.

### Visualization & Explainability

7. **Search Graph Visualization** — Expose the search graph to the frontend as an interactive DAG (React Flow / dagre). Users see which search paths yielded results and which dead-ended.

8. **Evidence Provenance Dashboard** — Sankey or waterfall chart tracing how each piece of evidence flowed through agents and contributed to the final confidence score.

9. **GeoGuessr-Style Street View** — Embed Google Street View / Mapillary panoramas for top candidates. Let users visually verify without leaving the app.

### Reliability & Performance

10. **Confidence Calibration** — Benchmark confidence scores against a ground-truth dataset (e.g., GeoGuessr images with known locations). Calibrate so "80% confidence" means correct ~80% of the time.

11. **Caching & Dedup Layer** — Cache Serper / OSM / browser results by query hash. Avoid redundant API calls across pipeline runs and save costs.

12. **Alternative Search Providers** — Add Brave Search, SearXNG, Bing Visual Search as Tier 1 sources. Reduce single-provider dependency and cross-validate results.

### Extensibility

13. **Batch Mode** — Upload multiple images for batch geolocation with a summary dashboard (map with all pins, CSV export).

14. **Plugin System for New Models** — Hot-pluggable model registry so new geolocation models (PlonkV2, PIGEON, etc.) can be added without touching agent code.
