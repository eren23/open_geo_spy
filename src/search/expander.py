"""Query expansion strategies.

Improved heuristic-based expansion using evidence context to generate
more targeted search queries instead of generic templates.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from src.evidence.chain import Evidence, EvidenceChain, EvidenceSource
from src.search.graph import QueryIntent, SearchGraph


# Country to language mapping for translation queries
COUNTRY_LANGUAGES: dict[str, str] = {
    "japan": "ja",
    "france": "fr",
    "germany": "de",
    "spain": "es",
    "italy": "it",
    "russia": "ru",
    "china": "zh",
    "korea": "ko",
    "south korea": "ko",
    "brazil": "pt",
    "portugal": "pt",
    "poland": "pl",
    "netherlands": "nl",
    "turkey": "tr",
    "thailand": "th",
    "vietnam": "vi",
    "indonesia": "id",
    "arabia": "ar",
    "egypt": "ar",
    "morocco": "ar",
    "israel": "he",
    "greece": "el",
    "czech": "cs",
    "czech republic": "cs",
    "hungary": "hu",
    "romania": "ro",
    "sweden": "sv",
    "norway": "no",
    "denmark": "da",
    "finland": "fi",
    "ukraine": "uk",
}

# Common business type translations for key languages
BUSINESS_TRANSLATIONS: dict[str, dict[str, str]] = {
    "ja": {"hotel": "ホテル", "station": "駅", "temple": "寺", "shrine": "神社", "restaurant": "レストラン"},
    "de": {"hotel": "Hotel", "station": "Bahnhof", "restaurant": "Restaurant", "street": "Straße"},
    "fr": {"hotel": "hôtel", "station": "gare", "restaurant": "restaurant", "street": "rue"},
    "es": {"hotel": "hotel", "station": "estación", "restaurant": "restaurante", "street": "calle"},
    "pt": {"hotel": "hotel", "station": "estação", "restaurant": "restaurante", "street": "rua"},
    "ru": {"hotel": "отель", "station": "станция", "street": "улица"},
    "zh": {"hotel": "酒店", "station": "站", "restaurant": "餐厅", "street": "街"},
    "ko": {"hotel": "호텔", "station": "역", "restaurant": "식당", "street": "거리"},
    "ar": {"hotel": "فندق", "street": "شارع"},
    "th": {"hotel": "โรงแรม", "temple": "วัด", "street": "ถนน"},
}


class QueryExpander:
    """Evidence-aware query expander.

    Generates targeted search queries using:
    - Business names from OCR
    - Street/landmark names
    - Country/region context
    - Local language translations
    """

    def suggest(
        self,
        graph: SearchGraph,
        evidence_chain: EvidenceChain,
        weak_areas: list[str] | None = None,
        ocr_result: dict[str, list[str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate expansion suggestions using evidence context.

        Args:
            graph: Search graph with node history
            evidence_chain: Evidence from prior agents
            weak_areas: Identified weakness areas from refinement
            ocr_result: Raw OCR results for business/street names

        Returns:
            List of dicts: {"query", "intent", "parent_id", "provider", "reason"}
        """
        suggestions: list[dict] = []

        # Extract context from evidence
        countries = list(set(evidence_chain.country_predictions))
        top_country = countries[0] if countries else None
        
        # Get cities from evidence
        cities = list(set(
            e.city for e in evidence_chain.evidences 
            if e.city
        ))
        top_city = cities[0] if cities else None
        
        # Extract business names and streets from OCR or evidence metadata
        business_names = self._extract_business_names(evidence_chain, ocr_result)
        street_names = self._extract_street_names(evidence_chain, ocr_result)
        landmark_names = self._extract_landmarks(evidence_chain)

        # Find productive completed nodes
        productive = [
            n for n in graph.nodes.values()
            if n.status.value == "completed" and n.evidence_count > 0
        ]

        if not productive and not weak_areas:
            return suggestions

        for node in productive[:3]:
            children = graph.get_children(node.id)
            existing_intents = {c.intent for c in children}

            # Smart refine: Use specific business/street names
            if QueryIntent.REFINE not in existing_intents:
                refine_query = self._build_refine_query(
                    node.query, business_names, street_names, top_city, top_country
                )
                if refine_query:
                    suggestions.append({
                        "query": refine_query,
                        "intent": QueryIntent.REFINE,
                        "parent_id": node.id,
                        "provider": node.provider,
                        "reason": "Refine with specific location details",
                    })

            # Smart broaden: Add geographic context
            if node.evidence_count < 3 and QueryIntent.BROADEN not in existing_intents:
                broaden_query = self._build_broaden_query(
                    node.query, top_city, top_country, landmark_names
                )
                if broaden_query:
                    suggestions.append({
                        "query": broaden_query,
                        "intent": QueryIntent.BROADEN,
                        "parent_id": node.id,
                        "provider": node.provider,
                        "reason": "Broaden with geographic context",
                    })

            # Translation: If country is known with non-English language
            if QueryIntent.TRANSLATE not in existing_intents and top_country:
                translate_query = self._build_translate_query(
                    node.query, top_country, business_names, street_names
                )
                if translate_query:
                    suggestions.append({
                        "query": translate_query,
                        "intent": QueryIntent.TRANSLATE,
                        "parent_id": node.id,
                        "provider": node.provider,
                        "reason": f"Translate for local language ({top_country})",
                        "language": COUNTRY_LANGUAGES.get(top_country.lower(), "en"),
                    })

        # Handle weak areas from refinement
        if weak_areas:
            suggestions.extend(
                self._handle_weak_areas(weak_areas, top_country, top_city, countries, landmark_names)
            )

        # Deduplicate by query text
        seen = set()
        unique = []
        for s in suggestions:
            q_lower = s["query"].lower().strip()
            if q_lower not in seen:
                seen.add(q_lower)
                unique.append(s)

        return unique[:8]

    def _extract_business_names(
        self, 
        evidence_chain: EvidenceChain, 
        ocr_result: dict[str, list[str]] | None
    ) -> list[str]:
        """Extract business names from OCR or evidence."""
        names = []
        
        # From OCR result
        if ocr_result:
            names.extend(ocr_result.get("business_names", [])[:3])
        
        # From evidence metadata
        for e in evidence_chain.evidences:
            if e.source == EvidenceSource.OCR:
                biz = e.metadata.get("business_name") or e.metadata.get("business_names", [])
                if isinstance(biz, list):
                    names.extend(biz[:2])
                elif biz:
                    names.append(biz)
        
        # Dedupe and return top 3
        return list(dict.fromkeys(names))[:3]

    def _extract_street_names(
        self, 
        evidence_chain: EvidenceChain, 
        ocr_result: dict[str, list[str]] | None
    ) -> list[str]:
        """Extract street names from OCR or evidence."""
        names = []
        
        if ocr_result:
            names.extend(ocr_result.get("street_signs", [])[:2])
        
        for e in evidence_chain.evidences:
            if e.source == EvidenceSource.OCR:
                streets = e.metadata.get("street_signs", [])
                if isinstance(streets, list):
                    names.extend(streets[:2])
        
        return list(dict.fromkeys(names))[:2]

    def _extract_landmarks(self, evidence_chain: EvidenceChain) -> list[str]:
        """Extract landmark names from evidence."""
        landmarks = []
        for e in evidence_chain.evidences:
            if e.metadata.get("landmarks"):
                lm = e.metadata["landmarks"]
                if isinstance(lm, list):
                    landmarks.extend(lm[:2])
            # Also check content for landmark keywords
            if any(kw in e.content.lower() for kw in ["temple", "church", "mosque", "castle", "palace", "tower", "bridge", "monument"]):
                landmarks.append(e.content[:50])
        
        return list(dict.fromkeys(landmarks))[:3]

    def _build_refine_query(
        self,
        original_query: str,
        business_names: list[str],
        street_names: list[str],
        city: str | None,
        country: str | None,
    ) -> str | None:
        """Build a refinement query using specific location details."""
        
        # Best: Business + Street
        if business_names and street_names:
            return f'"{business_names[0]}" "{street_names[0]}" location'
        
        # Good: Business + City/Country
        if business_names:
            biz = business_names[0]
            if city:
                return f'"{biz}" {city} address'
            elif country:
                return f'"{biz}" {country} location'
            else:
                return f'"{biz}" address location'
        
        # Okay: Street + City/Country
        if street_names:
            street = street_names[0]
            if city:
                return f'"{street}" {city}'
            elif country:
                return f'"{street}" {country}'
        
        # Fallback: Original + specific location keywords
        if city:
            return f"{original_query} {city} exact location"
        elif country:
            return f"{original_query} {country} coordinates"
        
        # Don't generate generic queries
        return None

    def _build_broaden_query(
        self,
        original_query: str,
        city: str | None,
        country: str | None,
        landmarks: list[str],
    ) -> str | None:
        """Build a broader query with geographic context."""
        
        if landmarks and country:
            return f"{landmarks[0]} {country} area"
        elif city and country:
            return f"{original_query} {city} {country} region"
        elif country:
            return f"{original_query} {country} area"
        else:
            return None  # Don't generate generic "area region" queries

    def _build_translate_query(
        self,
        original_query: str,
        country: str,
        business_names: list[str],
        street_names: list[str],
    ) -> str | None:
        """Build a query translated to local language."""
        lang = COUNTRY_LANGUAGES.get(country.lower())
        if not lang:
            return None
        
        translations = BUSINESS_TRANSLATIONS.get(lang, {})
        
        # Translate business type if present
        for eng, native in translations.items():
            if eng in original_query.lower():
                translated = original_query.lower().replace(eng, native)
                return f"{original_query} OR {translated}"
        
        # If we have business names, try with native script
        if business_names:
            biz = business_names[0]
            return f'"{biz}" {country}'
        
        return None

    def _handle_weak_areas(
        self,
        weak_areas: list[str],
        top_country: str | None,
        top_city: str | None,
        countries: list[str],
        landmarks: list[str],
    ) -> list[dict[str, Any]]:
        """Generate queries to address identified weak areas."""
        suggestions = []
        
        if "no_web_corroboration" in weak_areas:
            if top_city and top_country:
                suggestions.append({
                    "query": f"{top_city} {top_country} landmarks attractions",
                    "intent": QueryIntent.VERIFY,
                    "parent_id": None,
                    "provider": "serper",
                    "reason": "Find landmarks for web corroboration",
                })
            elif top_country:
                suggestions.append({
                    "query": f"famous landmarks {top_country}",
                    "intent": QueryIntent.VERIFY,
                    "parent_id": None,
                    "provider": "serper",
                    "reason": "Find landmarks for corroboration",
                })

        if "country_disagreement" in weak_areas and len(countries) >= 2:
            # Search for distinguishing features
            for country in countries[:2]:
                suggestions.append({
                    "query": f"distinctive architecture scenery {country}",
                    "intent": QueryIntent.VERIFY,
                    "parent_id": None,
                    "provider": "serper",
                    "reason": f"Verify country: {country}",
                })

        if "low_coordinate_confidence" in weak_areas and top_city:
            suggestions.append({
                "query": f"{top_city} coordinates center",
                "intent": QueryIntent.VERIFY,
                "parent_id": None,
                "provider": "serper",
                "reason": "Get precise coordinates for city",
            })

        return suggestions
