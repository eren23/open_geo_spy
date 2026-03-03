# Search Quality Improvements Plan

## Overview
This document tracks the implementation of search quality improvements identified in the codebase analysis.

## Status Legend
- ⬜ Not started
- 🔄 In progress
- ✅ Completed
- ❌ Blocked

---

## P0 - Critical (Must Fix)

### 1. Dynamic Confidence Scoring for Search Providers
**Files:** `src/geo/serper_client.py`, `src/geo/brave_client.py`, `src/geo/searxng_client.py`, `src/geo/osm_client.py`

**Problem:** All providers assign hardcoded confidence values regardless of result relevance.

**Solution:** Implement dynamic confidence based on:
- Query match in title/snippet
- Presence of coordinates
- Presence of address
- Result position decay

**Status:** ✅ Implemented in `src/geo/confidence.py` with `calculate_search_confidence()` and `calculate_osm_confidence()` functions. All providers use these functions.

---

### 2. Fix Source Attribution (Brave)
**Files:** `src/evidence/chain.py`, `src/geo/brave_client.py`

**Problem:** Brave results incorrectly use `EvidenceSource.SERPER`, corrupting source diversity calculations.

**Solution:** 
- Add `BRAVE` to EvidenceSource enum
- Update BraveClient to use correct source

**Status:** ✅ `BRAVE` enum exists in `src/evidence/chain.py` (line 27) and BraveClient uses `EvidenceSource.BRAVE` (line 111).

---

## P1 - High Priority

### 3. Smarter Query Templates in Expander
**Files:** `src/search/expander.py`

**Problem:** Generic templates like "exact location street address" produce low-quality queries.

**Solution:** Use specific evidence context (business names, streets, countries) to build targeted queries.

**Status:** ✅ Implemented in `src/search/expander.py` with `QueryExpander` class that uses `_build_refine_query()`, `_build_broaden_query()`, `_build_translate_query()` methods that extract business names, streets, and landmarks from evidence.

---

### 4. Better Context Compression in Smart Expander
**Files:** `src/search/smart_expander.py`

**Problem:** Evidence summary truncated to 500 chars loses critical geographic context.

**Solution:** Implement smart compression that prioritizes:
- Countries, cities, coordinates
- Business names, street names
- Evidence agreement scores

**Status:** ✅ Implemented in `src/search/smart_expander.py` with `_compress_evidence()` method (lines 58-75) that extracts country, city, landmarks, and type keywords with confidence scores.

---

### 5. Country-to-Language Mapping for Translation
**Files:** `src/search/smart_expander.py`

**Problem:** TRANSLATE intent lacks implementation for detecting target language.

**Solution:** Add country-to-language mapping and auto-detect translation needs.

**Status:** ✅ Implemented in `src/search/expander.py` with `COUNTRY_LANGUAGES` dict (lines 17-50) and `BUSINESS_TRANSLATIONS` dict (lines 52-64). The `_build_translate_query()` method uses these mappings.

---

### 6. Replace O(n²) Clustering with DBSCAN
**Files:** `src/agents/reasoning_agent.py`

**Problem:** Clustering has quadratic complexity, slow for large evidence sets.

**Solution:** Use sklearn's DBSCAN with haversine metric.

**Status:** ✅ Implemented in `src/agents/reasoning_agent.py` using `sklearn.cluster.DBSCAN` with haversine metric (import at line 20, usage in `_cluster_by_proximity()` method).

---

### 7. Smarter Pruning Logic in Search Graph
**Files:** `src/search/graph.py`

**Problem:** Nodes with zero evidence are immediately pruned, missing retry opportunities.

**Solution:** 
- Add retry counter to nodes
- Only prune after N failed attempts
- Keep initial queries longer

**Status:** ✅ Implemented with `retry_count` field on `SearchNode`, `max_retries` on `SearchGraph`, and `retryable_dead_ends()` method. The `dead_ends()` method now only returns nodes where `retry_count >= max_retries`.

---

### 8. Track Failed Node Patterns
**Files:** `src/search/graph.py`

**Problem:** Expansion logic ignores FAILED nodes, missing learning opportunities.

**Solution:** Analyze failure patterns to avoid similar queries.

**Status:** ✅ Implemented with `failed_patterns` list on `SearchGraph`, `record_failed_pattern()` and `matches_failed_pattern()` methods. The web intel agent filters queries matching failed patterns.

---

## P2 - Medium Priority

### 9. Cross-Provider Deduplication
**Files:** `src/agents/web_intel_agent.py`

**Problem:** Multiple providers searching same query create duplicate evidence.

**Solution:** Implement content-hash based deduplication across providers.

**Status:** ✅ Implemented in `_execute_pending_nodes()` with SHA256 hash-based deduplication using content + url + coordinates as key.

---

### 10. Coordinate Validation
**Files:** All provider clients

**Problem:** Coordinates used without bounds checking.

**Solution:** Add `_safe_coords()` helper with validation.

**Status:** ✅ Implemented in `src/geo/confidence.py` as `safe_coords()` function (lines 12-37) with lat/lon bounds validation. Also used in `Evidence.__post_init__()` for validation.

---

### 11. Country Consensus Edge Cases
**Files:** `src/agents/reasoning_agent.py`

**Problem:** 50/50 split produces arbitrary "dominant" country with boosted confidence.

**Solution:** Only apply consensus boost when dominance > 60%.

**Status:** ✅ Implemented in `src/scoring/config.py` with `CountryPenalty.consensus_threshold` set to 0.6 (60%). The scorer's `country_match_score()` and `country_penalty()` methods check this threshold.

---

### 12. Negative Evidence Handling
**Files:** `src/evidence/chain.py`, `src/agents/reasoning_agent.py`

**Problem:** System doesn't handle evidence that contradicts hypotheses.

**Solution:** Add negative evidence support and confidence reduction logic.

**Status:** ✅ Implemented with `is_negative` field on `Evidence` class, `negative_evidences()` method on `EvidenceChain`, and penalty applied in `agreement_score()` calculation.

---

### 13. Evidence Temporal Weighting
**Files:** `src/evidence/chain.py`

**Problem:** Newer evidence from refinement may be more accurate but weighted equally.

**Solution:** Add timestamp-based weighting favoring later evidence.

**Status:** ✅ Implemented with recency-weighted coordinates in `agreement_score()` and new `recency_weighted_confidence()` method. Timestamp field already existed on Evidence.

---

### 14. Query Performance Tracking
**Files:** `src/search/graph.py`

**Problem:** Duration tracked but not used for optimization.

**Solution:** Track cost-effectiveness ratio and avoid slow patterns.

**Status:** ✅ Implemented with `cost_effectiveness` field on `SearchNode` (evidence_count / duration_seconds), computed in web intel agent after node execution.

---

### 15. Geolocation Reranker
**Files:** `src/geo/` (new file)

**Problem:** Results not re-ranked for geolocation relevance.

**Solution:** Implement reranker that prioritizes coordinates, addresses, location mentions.

**Status:** ✅ Implemented as new file `src/geo/reranker.py` with `GeolocationReranker` class that scores results based on coordinate presence, address quality, location keywords, and source reliability.

---

## Implementation Order

1. **Phase 1 (P0):** Fix critical bugs
   - [x] Dynamic confidence scoring
   - [x] Source attribution fix

2. **Phase 2 (P1):** Improve search quality
   - [x] Smarter query templates
   - [x] Context compression
   - [x] Language mapping
   - [x] DBSCAN clustering
   - [x] Smarter pruning
   - [x] Failed node tracking

3. **Phase 3 (P2):** Polish and optimize
   - [x] Cross-provider deduplication
   - [x] Coordinate validation
   - [x] Country consensus fix
   - [x] Negative evidence
   - [x] Temporal weighting
   - [x] Performance tracking
   - [x] Geolocation reranker
