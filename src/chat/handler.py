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
            user_message, self.client, self.settings.llm.fast_model
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
            resp = await self.client.chat.completions.create(
                model=self.settings.llm.reasoning_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1000,
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

        # Extract search query from message
        query = message.lower().replace("try searching for", "").replace("search for", "").strip()
        if not query:
            query = message

        try:
            from src.geo.serper_client import SerperClient

            if self.settings.geo.serper_api_key:
                serper = SerperClient(self.settings.geo.serper_api_key)
                results = await serper.search(query, num_results=5)
                evidences = serper.results_to_evidence(results, query)
                await serper.close()

                yield {
                    "event": "chat_step",
                    "step": f"Found {len(evidences)} new evidence items",
                }

                # Summarize findings
                if evidences:
                    evidence_text = "\n".join(
                        f"- {e.content} (conf={e.confidence:.2f})"
                        for e in evidences[:5]
                    )
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
        """Handle user providing a location hint."""
        yield {"event": "chat_step", "step": "Incorporating your hint..."}

        # Add as USER_HINT evidence
        hint_evidence = Evidence(
            source=EvidenceSource.USER_HINT,
            content=f"User hint: {message}",
            confidence=0.7,
        )

        answer = (
            f"Thanks for the hint! I've noted: \"{message}\". "
            f"In a full re-analysis, this would be incorporated as additional evidence "
            f"to refine the prediction."
        )
        session.add_message("assistant", answer, {"intent": "refine_hint"})
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
            resp = await self.client.chat.completions.create(
                model=self.settings.llm.fast_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800,
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
            resp = await self.client.chat.completions.create(
                model=self.settings.llm.fast_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=600,
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
            resp = await self.client.chat.completions.create(
                model=self.settings.llm.fast_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=600,
            )
            answer = resp.choices[0].message.content
            session.add_message("assistant", answer, {"intent": "general"})
            yield {"event": "chat_message", "role": "assistant", "content": answer}
        except Exception as e:
            yield {"event": "chat_error", "error": str(e)}
