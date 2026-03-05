# Changelog

All notable changes to OpenGeoSpy will be documented in this file.

---

## [0.3.2] — 2025-01-14

### Bug Fixes

- **Fixed `SearchGraph.metadata` AttributeError**: Added missing `metadata` dict field to `SearchGraph` dataclass that was being accessed by `WebIntelAgent` for storing hints and country context.

### Chat Improvements

- **Hint-triggered searches**: When users provide location hints in chat (e.g., "narrow it down to Baden-Württemberg"), the system now runs **new targeted web searches** instead of just filtering existing evidence. This combines fresh search results with the hint context to find more specific locations.
- **Better hint parsing**: Improved prefix stripping to handle phrases like "narrow it down to" and "around" in addition to previous patterns.
- **Informative feedback**: Chat responses now indicate how many new evidence items were found from hint-triggered searches.

### Files Changed
- `src/search/graph.py` — Added `metadata: dict[str, Any]` field to `SearchGraph`
- `src/chat/handler.py` — `_handle_refine_hint()` now instantiates `WebIntelAgent` and runs fresh searches with user hints

---

## [0.3.1] — 2026-03-02

### Search Quality Improvements

This release implements all 15 items from the Search Quality Improvements roadmap.

#### P0 - Critical Fixes
- **Dynamic confidence scoring**: All search providers now use `calculate_search_confidence()` with query match boosts, coordinate presence, address detection, and position decay (was already implemented in `src/geo/confidence.py`)
- **Source attribution fix**: Brave search correctly uses `EvidenceSource.BRAVE` enum (was already fixed)

#### P1 - High Priority
- **Smarter query templates**: `QueryExpander` builds targeted queries using business names, streets, and landmarks from evidence context
- **Context compression**: Smart evidence compression prioritizes countries, cities, coordinates, and landmarks for LLM prompts
- **Country-to-language mapping**: Translation queries auto-detect target language from country (supports 25+ languages)
- **DBSCAN clustering**: Replaced O(n²) iterative clustering with sklearn's DBSCAN using haversine metric for proper geographic clustering
- **Smarter pruning**: Search nodes get retry counters; pruning only after N failed attempts (configurable `max_retries`)
- **Failed node tracking**: Search graph records failed query patterns to avoid similar queries in future expansions

#### P2 - Medium Priority
- **Cross-provider deduplication**: Hash-based deduplication across Serper, Brave, and SearXNG to prevent duplicate evidence
- **Coordinate validation**: `safe_coords()` helper validates lat/lon bounds before use
- **Country consensus fix**: Consensus threshold raised from 30% to 60% to prevent arbitrary dominant country on 50/50 splits
- **Negative evidence handling**: New `is_negative` field on Evidence; confidence penalty applied when contradictions found
- **Temporal weighting**: Newer evidence from refinement gets higher weight via recency-weighted confidence
- **Performance tracking**: `cost_effectiveness` field on SearchNode tracks evidence count per second
- **Geolocation reranker**: New `GeolocationReranker` class prioritizes results with coordinates, addresses, and location keywords

### Files Changed
- `src/scoring/config.py` — Country consensus threshold raised to 0.6
- `src/search/graph.py` — Added `retry_count`, `cost_effectiveness`, `failed_patterns` tracking
- `src/agents/web_intel_agent.py` — Cross-provider deduplication, failed pattern filtering
- `src/evidence/chain.py` — Negative evidence support, temporal weighting
- `src/agents/reasoning_agent.py` — DBSCAN clustering with haversine metric
- `src/geo/reranker.py` — **NEW** Geolocation reranker for search results
- `docs/SEARCH_QUALITY_IMPROVEMENTS.md` — All 15 items marked complete

---

## [0.3.0] — 2026-03-01

### Grounding-Based Scoring Architecture

- **ScoringConfig**: Single Pydantic model holding all 50+ scoring parameters (`src/scoring/config.py`), serializable to JSON for eval sweeps
- **GeoScorer**: Unified scoring API replacing scattered hardcoded weights across 12 files (`src/scoring/scorer.py`)
- **GroundingEngine**: Evidence-based verdicts (GROUNDED/SUPPORTED/UNCERTAIN/WEAKENED/CONTRADICTED) replace opaque confidence multipliers (`src/scoring/grounding.py`)
- **HierarchicalResolver**: Coarse-to-fine prediction with per-level grounding — continent, country, region, city, coordinates (`src/scoring/hierarchy.py`)
- **Continuous geo-agreement**: Linear interpolation replaces step-function discontinuities in agreement scoring

### Tracing & Observability

- **Full trace persistence**: Every pipeline run saved as JSONL — inputs, LLM calls, evidence, timing, costs (`src/tracing/recorder.py`)
- **LLM call instrumentation**: Token counts, costs, latency recorded for every API call (`src/tracing/instrumented_client.py`)
- **SQLite trace index**: Cross-run queries for accuracy stats, cost analysis, performance trends (`src/tracing/index.py`)
- **Live dashboard components**: TracingTimeline, CostMeter, LLMCallLog, EvidenceFlowView, GroundingView

### Evaluation Framework

- **Ground truth datasets**: Labeled image management with manifest.json format (`src/eval/dataset.py`)
- **Standard metrics**: Accuracy@{1,25,50,150,750}km, GCD error (median/mean/p90), ECE, country/city accuracy (`src/eval/metrics.py`)
- **LLM-as-judge**: Scores reasoning quality, evidence usage, confidence calibration, specificity (1-5 per dimension) (`src/eval/judge.py`)
- **Eval reports**: Console table, JSON, and markdown output formats with baseline comparison (`src/eval/report.py`)

### CLI & Headless Mode

- **`ogs locate`**: Single-image headless geolocation with JSON/table output
- **`ogs batch`**: Multi-image parallel processing with JSONL output
- **`ogs eval`**: Run evaluation on labeled datasets with automatic metrics
- **`ogs trace-stats`**: Query trace index for aggregate statistics
- **`ogs evolve`**: Analyze eval results and suggest/apply scoring weight adjustments

### Auto-Evolution

- **FailureAnalyzer**: Categorizes prediction failures — wrong_country, high_confidence_wrong, model_disagreement, etc. (`src/eval/evolution.py`)
- **WeightTuner**: Bayesian-style weight adjustment from eval results, outputs updated ScoringConfig (`src/eval/tuner.py`)
- **Versioned evolution**: Weight history with explanations at `data/evolution/history/`

---

## [0.2.0] — 2026-03-01

### Multi-Candidate Pipeline (v2 API)

- **V2 locate endpoint**: `GET /api/v2/locate/stream` returns ranked candidates with per-candidate evidence trails
- **Evidence clustering**: Haversine-based geographic proximity clustering (eps=50km) builds secondary candidates from evidence groups
- **Candidate ranking formula**: Composite score blending confidence, evidence count, source diversity, visual match, and country consensus
- **Candidate verification agent**: 4-strategy candidate generation (OCR businesses, category search, landmarks, top evidence) + StreetCLIP visual similarity scoring
- **Reverse geocoding enrichment**: Parallel Nominatim lookups fill city/region/country for cluster-based candidates

### Evidence & Scoring Improvements

- **Country consensus hierarchy**: Dominant country computed from full evidence chain; user hint gets 3x country votes; wrong-country candidates penalized at construction, hint-matching, and ranking stages
- **Hint boost/penalty**: Matching candidates 1.5x, non-matching 0.5x (was 1.3x match only)
- **Country penalty in ranking**: New `country_match` term (0.0 for wrong country with strong consensus, 1.0 otherwise)
- **Evidence redistribution**: Pipeline evidence redistributed to candidates by geographic proximity (200km radius), capped at 15 per candidate
- **Search graph architecture**: Directed graph tracking queries, results, and refinement edges

### Iterative Refinement

- **Refinement loop**: Reasoning agent identifies weak areas (low agreement, few sources, country disagreement, low confidence) and re-runs web intelligence (max 2 iterations)
- **Smart query expansion**: LLM generates search variations (synonyms, local-language translations, nearby landmarks) instead of raw VLM output truncation

### Chat Interaction

- **Chat handler**: POST `/api/chat/{session_id}` for conversational follow-up with LLM re-reasoning
- **Session state**: In-memory session tracking with evidence chain, candidates, search graph per session

### Frontend (React + Vite)

- **Chat UI**: Message bubbles for user/assistant, agent step messages with pipeline stage visualization
- **Evidence provenance dashboard**: Evidence grouped by pipeline stage (Feature Extraction, ML Ensemble, Web Intelligence, Reasoning) with per-stage metrics
- **Map controls sidebar**: Candidate list with rank/confidence, embedded provenance dashboard, selection controls
- **Batch upload UI**: Multi-file drag-and-drop with grid/map dashboard view
- **Search graph visualization**: Interactive DAG of search queries and results
- **Confidence waterfall**: Visual breakdown of confidence contributions

### Infrastructure

- **Docker improvements**: Model cache volume, Vite HMR with API proxy, health checks
- **Alternative search providers**: Brave Search, SearXNG support in web intelligence tier 1
- **Mapillary integration**: Street-level imagery fetching for candidate verification

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

## Delivered in 0.2.0 (from original roadmap)

The following items from the v0.1.0 roadmap were delivered in v0.2.0:

1. **Patching & Iterative Refinement** — Reasoning agent identifies weak areas and re-runs targeted web intelligence (max 2 iterations)
2. **Search Graph Architecture** — Directed graph tracking queries, results, and refinement edges
3. **Smart Query Expansion** — LLM generates search variations (synonyms, local-language translations, nearby landmarks)
4. **Multi-Candidate Final Output** — Top-N ranked candidates with individual confidence scores, evidence trails, and visual match scores
5. **Chat-Like Interaction Flow** — POST `/api/chat/{session_id}` for conversational follow-up with LLM re-reasoning
6. **Streaming Chat UI** — Chat interface with message bubbles for agent steps, inline map updates
7. **Search Graph Visualization** — Interactive DAG of search queries and results
8. **Evidence Provenance Dashboard** — Evidence grouped by pipeline stage with per-stage metrics
9. **Alternative Search Providers** — Brave Search, SearXNG support in web intelligence tier 1
10. **Batch Mode** — Multi-file drag-and-drop with grid/map dashboard view

### Deferred to v0.3.0

- **Confidence Calibration** — Eval framework with ground-truth benchmarking (→ Milestone C)
- **Caching & Dedup Layer** — Query hash caching for API calls
- **GeoGuessr-Style Street View** — Embedded panoramas for candidate verification
- **Plugin System for New Models** — Hot-pluggable model registry
