"""Chat follow-up handler.

Routes classified intents to appropriate actions: re-search,
counter-evidence lookup, candidate comparison, etc.
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator

from loguru import logger
from openai import AsyncOpenAI

from src.chat.intent import ChatIntent, classify_intent
from src.chat.session import Session, SessionManager
from src.config.settings import Settings
from src.config.llm import LLMCallType, get_llm_params
from src.evidence.chain import Evidence, EvidenceChain, EvidenceSource


class ChatHandler:
    """Handle chat follow-up messages on existing sessions."""

    def __init__(self, settings: Settings, session_manager: SessionManager):
        self.settings = settings
        self.sessions = session_manager
        self.client = AsyncOpenAI(
            base_url=settings.llm.base_url,
            api_key=settings.llm.api_key,
        )
        # Reasoning agent for re-running predictions after evidence updates
        from src.agents.reasoning_agent import ReasoningAgent
        self._reasoning_agent = ReasoningAgent(settings)

    def _build_chain_from_state(self, session: Session) -> EvidenceChain:
        """Build an EvidenceChain from session state, handling both Evidence objects and dicts."""
        chain = EvidenceChain()
        for e in session.pipeline_state.get("evidences", []):
            if isinstance(e, Evidence):
                chain.add(e)
            elif isinstance(e, dict):
                try:
                    chain.add(Evidence(
                        source=EvidenceSource(e.get("source", "reasoning")),
                        content=e.get("content", ""),
                        confidence=e.get("confidence", 0.5),
                        latitude=e.get("latitude"),
                        longitude=e.get("longitude"),
                        country=e.get("country"),
                        region=e.get("region"),
                        city=e.get("city"),
                        metadata=e.get("metadata", {}),
                    ))
                except (ValueError, KeyError):
                    logger.debug("Skipping malformed evidence dict: {}", e)
        return chain

    async def handle_message(
        self,
        session_id: str,
        user_message: str,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Process a user follow-up and yield SSE events."""
        session = self.sessions.get(session_id)
        if not session:
            yield {"event": "error", "error": "Session not found"}
            return

        session.add_message("user", user_message)

        # Classify intent
        intent = await classify_intent(
            user_message, self.client, self.settings
        )
        yield {"event": "intent", "intent": intent.value}

        # Route to handler
        handler_map = {
            ChatIntent.ASK_WHY_NOT: self._handle_why_not,
            ChatIntent.TRY_SEARCH: self._handle_try_search,
            ChatIntent.REFINE_HINT: self._handle_refine_hint,
            ChatIntent.COMPARE: self._handle_compare,
            ChatIntent.EXPLAIN: self._handle_explain,
            ChatIntent.ZOOM_FEATURE: self._handle_zoom_feature,
            ChatIntent.GENERAL: self._handle_general,
        }

        handler = handler_map.get(intent, self._handle_general)

        async for event in handler(session, user_message):
            yield event

        # Final message
        yield {"event": "chat_complete"}

    async def _handle_why_not(
        self, session: Session, message: str
    ) -> AsyncGenerator[dict, None]:
        """Handle 'why not X?' by searching for counter-evidence."""
        yield {"event": "chat_step", "step": "Analyzing counter-evidence..."}

        prompt = f"""The user asked: "{message}"

Current prediction: {json.dumps(session.prediction, default=str)[:500]}

Explain why the suggested location was NOT chosen over the current prediction.
Reference specific evidence that supports the current prediction and
any evidence that contradicts the user's suggested location.
Be specific and cite evidence sources."""

        try:
            params = get_llm_params(LLMCallType.CHAT_WHY_NOT, self.settings)
            resp = await self.client.chat.completions.create(
                **params,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = resp.choices[0].message.content
            session.add_message("assistant", answer, {"intent": "ask_why_not"})
            yield {"event": "chat_message", "role": "assistant", "content": answer}
        except Exception as e:
            yield {"event": "chat_error", "error": str(e)}

    async def _handle_try_search(
        self, session: Session, message: str
    ) -> AsyncGenerator[dict, None]:
        """Handle 'try searching for X' by running a new search."""
        yield {"event": "chat_step", "step": "Running new search..."}

        # Extract search query from message using configurable patterns
        from src.patterns import extract_search_query
        query = extract_search_query(message)

        try:
            from src.geo.serper_client import SerperClient

            if self.settings.geo.serper_api_key:
                serper = SerperClient(self.settings.geo.serper_api_key)
                results = await serper.search(query, num_results=5)
                new_evidences = serper.results_to_evidence(results, query)
                await serper.close()

                yield {
                    "event": "chat_step",
                    "step": f"Found {len(new_evidences)} new evidence items",
                }

                # Append new evidences to session pipeline state (cap at 200)
                existing_evidences = session.pipeline_state.get("evidences", [])
                existing_evidences.extend(new_evidences)
                if len(existing_evidences) > 200:
                    existing_evidences.sort(
                        key=lambda e: e.confidence if isinstance(e, Evidence) else e.get("confidence", 0),
                        reverse=True,
                    )
                    existing_evidences = existing_evidences[:200]
                session.pipeline_state["evidences"] = existing_evidences

                # Summarize findings
                if new_evidences:
                    evidence_text = "\n".join(
                        f"- {e.content} (conf={e.confidence:.2f})"
                        for e in new_evidences[:5]
                    )

                    # Re-run reasoning with augmented evidence
                    yield {"event": "chat_step", "step": "Re-analyzing with new evidence..."}
                    try:
                        chain = self._build_chain_from_state(session)
                        updated = await self._reasoning_agent.reason_multi_candidate(
                            chain, skip_verification=True,
                        )
                        if updated:
                            session.pipeline_state["ranked_candidates"] = updated
                            session.pipeline_state["prediction"] = updated[0]
                            yield {"event": "candidates_update", "candidates": updated}

                        answer = f"Search results for '{query}':\n\n{evidence_text}\n\nPredictions have been updated with this new evidence."
                    except Exception as re_err:
                        logger.warning("Re-reasoning after search failed: {}", re_err)
                        answer = f"Search results for '{query}':\n\n{evidence_text}"
                else:
                    answer = f"No results found for '{query}'."

                session.add_message("assistant", answer, {"intent": "try_search", "query": query})
                yield {"event": "chat_message", "role": "assistant", "content": answer}
            else:
                yield {"event": "chat_error", "error": "Search API not configured"}

        except Exception as e:
            yield {"event": "chat_error", "error": str(e)}

    async def _handle_refine_hint(
        self, session: Session, message: str
    ) -> AsyncGenerator[dict, None]:
        """Handle user providing a location hint.
        
        This:
        1. Adds the hint as evidence
        2. Runs FRESH discovery searches with hint + ALL existing context (ML predictions, OCR, landmarks, etc.)
        3. Only filters out ML model predictions that contradict the hint (keeps new search results)
        4. Re-runs reasoning with all evidence
        """
        yield {"event": "chat_step", "step": "Incorporating your hint..."}

        # Extract hint from message using configurable patterns
        from src.patterns import extract_location_hint
        hint_text = extract_location_hint(message)

        # Add as USER_HINT evidence to session state
        hint_evidence = Evidence(
            source=EvidenceSource.USER_HINT,
            content=f"User hint: {hint_text}",
            confidence=0.85,  # Higher confidence for explicit hints
            metadata={"hint": hint_text},
        )

        # Append hint evidence to session pipeline state
        evidences = session.pipeline_state.get("evidences", [])
        evidences.append(hint_evidence)
        session.pipeline_state["evidences"] = evidences

        # Build chain from current state (includes ALL existing evidence: ML, OCR, web, etc.)
        chain = self._build_chain_from_state(session)

        # --- Run FRESH discovery searches with FULL context + hint ---
        yield {"event": "chat_step", "step": f"Running fresh discovery searches for '{hint_text}'..."}
        
        new_evidence_count = 0
        try:
            from src.agents.web_intel_agent import WebIntelAgent
            from src.cache import CacheStore
            
            # IMPORTANT: Pass the FULL chain (with all ML predictions, OCR, features, etc.)
            # This allows the search to use ALL context for better query building
            search_chain = EvidenceChain()
            # First add the hint so it's prioritized
            search_chain.add(hint_evidence)
            # Then add ALL existing evidence for context-rich searching
            for e in chain.evidences:
                search_chain.add(e)
            
            # Get OCR/features from session if available
            ocr_result = session.pipeline_state.get("ocr_result")
            features = session.pipeline_state.get("features")
            
            # Run web intel agent with FULL context
            web_agent = WebIntelAgent(self.settings, cache=CacheStore.in_memory())
            new_chain, search_graph = await web_agent.search(
                evidence_chain=search_chain,  # Full context, not just hint!
                features=features,
                ocr_result=ocr_result,
                weak_areas=None,
            )
            await web_agent.close()
            
            # Merge new evidences into session
            new_evidence_count = len(new_chain.evidences)
            if new_evidence_count > 0:
                yield {"event": "chat_step", "step": f"Found {new_evidence_count} new evidence items!"}
                
                # Track new evidence hashes for smart filtering later
                new_evidence_hashes = {e.content_hash for e in new_chain.evidences}
                
                # Add new evidences to the main chain
                for e in new_chain.evidences:
                    chain.add(e)
                    
                logger.info("Hint search found {} new evidences for '{}'", new_evidence_count, hint_text)
            else:
                new_evidence_hashes = set()
        except Exception as e:
            logger.warning("Hint-based search failed: {}", e)
            yield {"event": "chat_step", "step": f"Search encountered an issue, proceeding with existing evidence..."}
            new_evidence_hashes = set()

        # Smart filtering: Only filter ML model predictions that contradict the hint
        # NEW web search results are kept unconditionally (they were found WITH hint context)
        # OLD web results are also filtered by country (they may be from wrong regions)
        from src.geo.country_matcher import extract_country_from_location
        hint_country = extract_country_from_location(hint_text)
        if hint_country:
            yield {"event": "chat_step", "step": f"Filtering predictions to {hint_text} region..."}
            
            # Filter ML model predictions (GeoCLIP, StreetCLIP, etc.) that don't match the hint
            # Also filter OLD web results that don't match (new ones are kept unconditionally)
            ml_sources = {
                EvidenceSource.GEOCLIP, 
                EvidenceSource.STREETCLIP, 
                EvidenceSource.PIGEON,
                EvidenceSource.VLM_GEO,
                EvidenceSource.VISUAL_MATCH,  # Also a visual model prediction
            }
            web_sources = {
                EvidenceSource.SERPER, 
                EvidenceSource.BRAVE, 
                EvidenceSource.SEARXNG, 
                EvidenceSource.BROWSER, 
                EvidenceSource.OSM, 
                EvidenceSource.GOOGLE_MAPS,
            }
            
            from src.geo.country_matcher import countries_match
            filtered_evidences = []
            for e in chain.evidences:
                # Always keep user hints
                if e.source == EvidenceSource.USER_HINT:
                    filtered_evidences.append(e)
                    continue
                    
                # Keep evidence without country info
                if not e.country:
                    filtered_evidences.append(e)
                    continue
                
                # Keep NEW web search results unconditionally (they were found with hint context)
                if e.source in web_sources and e.content_hash in new_evidence_hashes:
                    filtered_evidences.append(e)
                    continue
                    
                # For ML predictions and OLD web results: only keep if they match the hint
                if e.source in ml_sources or e.source in web_sources:
                    if countries_match(hint_country, e.country):
                        filtered_evidences.append(e)
                    else:
                        logger.debug("Filtered {} prediction from {} (hint={})", 
                                    e.source.value, e.country, hint_country)
                else:
                    # Keep other evidence types (reasoning, etc.)
                    filtered_evidences.append(e)
            
            # Rebuild chain with filtered evidences
            chain = EvidenceChain()
            for e in filtered_evidences:
                chain.add(e)
            
            logger.info(
                "Filtered evidence chain: {} -> {} evidences (hint={}, kept web results)",
                len(session.pipeline_state.get("evidences", [])),
                len(chain.evidences),
                hint_country,
            )

        # Update session evidences with the combined + filtered chain
        session.pipeline_state["evidences"] = chain.evidences

        # Re-run reasoning with augmented evidence
        yield {"event": "chat_step", "step": "Re-analyzing with your hint and new search results..."}
        try:
            # Pass the hint to reasoning for strong boosting
            updated = await self._reasoning_agent.reason_multi_candidate(
                chain, 
                skip_verification=True,
                hint=hint_text,  # Pass hint explicitly
            )
            if updated:
                session.pipeline_state["ranked_candidates"] = updated
                session.pipeline_state["prediction"] = updated[0]
                yield {"event": "candidates_update", "candidates": updated}

            if new_evidence_count > 0:
                answer = (
                    f"I ran fresh discovery searches for \"{hint_text}\" using all available context "
                    f"(OCR text, visual features, and ML predictions) and found {new_evidence_count} new evidence items. "
                    f"I've filtered out ML predictions from other regions while keeping relevant web search results, "
                    f"and re-analyzed everything. The predictions have been updated!"
                )
            else:
                answer = (
                    f"I've incorporated your hint: \"{hint_text}\" and re-analyzed the evidence. "
                    f"ML predictions from other regions have been filtered out, and candidates matching "
                    f"your hint have been strongly boosted."
                )
        except Exception as e:
            logger.warning("Re-reasoning after hint failed: {}", e)
            answer = (
                f"Thanks for the hint! I've noted: \"{hint_text}\" as additional evidence. "
                f"Re-analysis encountered an issue but the hint is saved."
            )

        session.add_message("assistant", answer, {"intent": "refine_hint", "hint": hint_text, "new_evidence_count": new_evidence_count})
        yield {"event": "chat_message", "role": "assistant", "content": answer}

    async def _handle_compare(
        self, session: Session, message: str
    ) -> AsyncGenerator[dict, None]:
        """Handle candidate comparison request."""
        candidates = session.candidates
        if len(candidates) < 2:
            answer = "Only one candidate available. Run the pipeline to generate multiple candidates."
            session.add_message("assistant", answer)
            yield {"event": "chat_message", "role": "assistant", "content": answer}
            return

        yield {"event": "chat_step", "step": "Comparing candidates..."}

        comparison = "## Candidate Comparison\n\n"
        for c in candidates[:3]:
            comparison += (
                f"**#{c.get('rank', '?')} {c.get('name', 'Unknown')}** "
                f"({c.get('country', '?')})\n"
                f"- Confidence: {c.get('confidence', 0):.0%}\n"
                f"- Coordinates: ({c.get('lat', '?')}, {c.get('lon', '?')})\n"
                f"- Evidence count: {len(c.get('evidence_trail', []))}\n"
                f"- Reasoning: {c.get('reasoning', 'N/A')[:150]}\n\n"
            )

        session.add_message("assistant", comparison, {"intent": "compare"})
        yield {"event": "chat_message", "role": "assistant", "content": comparison}

    async def _handle_explain(
        self, session: Session, message: str
    ) -> AsyncGenerator[dict, None]:
        """Explain evidence trail for a candidate."""
        yield {"event": "chat_step", "step": "Summarizing evidence..."}

        prompt = f"""Summarize the evidence that led to this prediction:

Prediction: {json.dumps(session.prediction, default=str)[:500]}

User question: {message}

Explain clearly which evidence sources contributed and how they support the conclusion."""

        try:
            params = get_llm_params(LLMCallType.CHAT_EXPLAIN, self.settings)
            resp = await self.client.chat.completions.create(
                **params,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = resp.choices[0].message.content
            session.add_message("assistant", answer, {"intent": "explain"})
            yield {"event": "chat_message", "role": "assistant", "content": answer}
        except Exception as e:
            yield {"event": "chat_error", "error": str(e)}

    async def _handle_zoom_feature(
        self, session: Session, message: str
    ) -> AsyncGenerator[dict, None]:
        """Handle zooming into a specific feature question."""
        yield {"event": "chat_step", "step": "Analyzing feature..."}

        prompt = f"""The user is asking about a specific visual feature from a geolocated image.

User question: {message}

The image was predicted to be at: {json.dumps(session.prediction, default=str)[:300]}

Provide insight about the visual feature they're asking about, based on the prediction context."""

        try:
            params = get_llm_params(LLMCallType.CHAT_ZOOM_FEATURE, self.settings)
            resp = await self.client.chat.completions.create(
                **params,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = resp.choices[0].message.content
            session.add_message("assistant", answer, {"intent": "zoom_feature"})
            yield {"event": "chat_message", "role": "assistant", "content": answer}
        except Exception as e:
            yield {"event": "chat_error", "error": str(e)}

    async def _handle_general(
        self, session: Session, message: str
    ) -> AsyncGenerator[dict, None]:
        """Handle general questions."""
        prompt = f"""You are a geolocation assistant. The user has submitted an image that was
analyzed and predicted to be at: {json.dumps(session.prediction, default=str)[:300]}

User message: {message}

Respond helpfully based on the geolocation context."""

        try:
            params = get_llm_params(LLMCallType.CHAT_GENERAL, self.settings)
            resp = await self.client.chat.completions.create(
                **params,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = resp.choices[0].message.content
            session.add_message("assistant", answer, {"intent": "general"})
            yield {"event": "chat_message", "role": "assistant", "content": answer}
        except Exception as e:
            yield {"event": "chat_error", "error": str(e)}
