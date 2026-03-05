"""Tests for hint filtering and handling in the pipeline."""

import pytest

from src.evidence.chain import Evidence, EvidenceChain, EvidenceSource
from src.geo.country_matcher import countries_match, extract_country_from_location


class TestEvidenceChainFiltering:
    """Tests for evidence chain filtering by hint."""
    
    def create_evidence(
        self,
        source: EvidenceSource,
        country: str,
        confidence: float = 0.8,
        city: str = None,
    ) -> Evidence:
        """Helper to create evidence."""
        return Evidence(
            source=source,
            content=f"Location in {country}",
            confidence=confidence,
            country=country,
            city=city,
        )
    
    def test_filter_keeps_matching_country(self):
        """Evidence matching hint country is kept."""
        chain = EvidenceChain()
        chain.add(self.create_evidence(EvidenceSource.GEOCLIP, "Germany"))
        chain.add(self.create_evidence(EvidenceSource.VLM_GEO, "Germany"))
        
        filtered = chain.filter_by_hint("Germany", keep_non_geo=True)
        
        assert len(filtered.evidences) == 2
    
    def test_filter_removes_wrong_country(self):
        """Evidence from wrong country is removed."""
        chain = EvidenceChain()
        chain.add(self.create_evidence(EvidenceSource.GEOCLIP, "Germany"))
        chain.add(self.create_evidence(EvidenceSource.VLM_GEO, "France"))
        
        filtered = chain.filter_by_hint("Germany", keep_non_geo=True)
        
        assert len(filtered.evidences) == 1
        assert filtered.evidences[0].country == "Germany"
    
    def test_filter_keeps_user_hint(self):
        """User hint evidence is always kept."""
        chain = EvidenceChain()
        chain.add(Evidence(
            source=EvidenceSource.USER_HINT,
            content="User says Germany",
            confidence=0.8,
            metadata={"hint": "Germany"},
        ))
        chain.add(self.create_evidence(EvidenceSource.GEOCLIP, "France"))
        
        filtered = chain.filter_by_hint("Germany", keep_non_geo=True)
        
        assert len(filtered.evidences) == 1
        assert filtered.evidences[0].source == EvidenceSource.USER_HINT
    
    def test_filter_keeps_non_geo_if_requested(self):
        """Non-geo evidence is kept if keep_non_geo=True."""
        chain = EvidenceChain()
        chain.add(Evidence(
            source=EvidenceSource.VLM_ANALYSIS,
            content="Architecture style: European",
            confidence=0.6,
            # No country
        ))
        chain.add(self.create_evidence(EvidenceSource.GEOCLIP, "Germany"))
        
        filtered = chain.filter_by_hint("Germany", keep_non_geo=True)
        
        assert len(filtered.evidences) == 2
    
    def test_filter_removes_non_geo_if_requested(self):
        """Non-geo evidence is removed if keep_non_geo=False."""
        chain = EvidenceChain()
        chain.add(Evidence(
            source=EvidenceSource.VLM_ANALYSIS,
            content="Architecture style: European",
            confidence=0.6,
        ))
        chain.add(self.create_evidence(EvidenceSource.GEOCLIP, "Germany"))
        
        filtered = chain.filter_by_hint("Germany", keep_non_geo=False)
        
        assert len(filtered.evidences) == 1
        assert filtered.evidences[0].country == "Germany"
    
    def test_filter_handles_native_names(self):
        """Filtering works with native country names."""
        chain = EvidenceChain()
        chain.add(self.create_evidence(EvidenceSource.GEOCLIP, "Deutschland"))
        
        # Hint is "Germany", evidence is "Deutschland"
        filtered = chain.filter_by_hint("Germany", keep_non_geo=True)
        
        assert len(filtered.evidences) == 1
    
    def test_filter_handles_iso_codes(self):
        """Filtering works with ISO codes."""
        chain = EvidenceChain()
        chain.add(self.create_evidence(EvidenceSource.GEOCLIP, "DE"))
        
        # Hint is "Germany", evidence country is "DE"
        filtered = chain.filter_by_hint("Germany", keep_non_geo=True)
        
        assert len(filtered.evidences) == 1
    
    def test_filter_multiple_countries(self):
        """Filtering handles evidence from multiple countries."""
        chain = EvidenceChain()
        chain.add(self.create_evidence(EvidenceSource.GEOCLIP, "Germany", 0.9))
        chain.add(self.create_evidence(EvidenceSource.VLM_GEO, "France", 0.85))
        chain.add(self.create_evidence(EvidenceSource.STREETCLIP, "Germany", 0.8))
        chain.add(self.create_evidence(EvidenceSource.SERPER, "Germany", 0.7))
        
        filtered = chain.filter_by_hint("Germany", keep_non_geo=True)
        
        assert len(filtered.evidences) == 3
        for e in filtered.evidences:
            assert countries_match("Germany", e.country)


class TestCountryMatcherIntegration:
    """Tests for country matcher integration with evidence."""
    
    def test_extract_from_major_cities(self):
        """Major cities resolve to correct countries."""
        city_to_country = {
            "Berlin": "DE",
            "Munich": "DE",
            "Paris": "FR",
            "London": "GB",
            "Tokyo": "JP",
            "New York": "US",
            "Istanbul": "TR",
            "Barcelona": "ES",
            "Amsterdam": "NL",
            "Vienna": "AT",
            "Rome": "IT",
            "Prague": "CZ",
        }
        
        for city, expected_iso in city_to_country.items():
            iso = extract_country_from_location(city)
            assert iso == expected_iso, f"{city} should resolve to {expected_iso}, got {iso}"
    
    def test_countries_match_various_inputs(self):
        """Country matching handles various input formats."""
        # English names
        assert countries_match("Germany", "Germany")
        
        # Native names
        assert countries_match("Germany", "Deutschland")
        assert countries_match("Spain", "España")
        
        # Abbreviations
        assert countries_match("US", "United States")
        assert countries_match("USA", "United States")
        assert countries_match("UK", "United Kingdom")
        
        # Common aliases
        assert countries_match("Holland", "Netherlands")
        assert countries_match("America", "United States")


class TestHintExtraction:
    """Tests for extracting hints from evidence chains."""
    
    def test_get_hint_from_evidence(self):
        """Hint can be extracted from evidence chain."""
        chain = EvidenceChain()
        chain.add(Evidence(
            source=EvidenceSource.USER_HINT,
            content="User location hint: Germany",
            confidence=0.8,
            metadata={"hint": "Germany"},
        ))
        chain.add(Evidence(
            source=EvidenceSource.GEOCLIP,
            content="GeoCLIP prediction",
            confidence=0.7,
            country="France",
        ))
        
        hint = chain.get_hint_from_evidence()
        assert hint == "Germany"
    
    def test_get_hint_none_if_not_present(self):
        """Returns None if no hint in chain."""
        chain = EvidenceChain()
        chain.add(Evidence(
            source=EvidenceSource.GEOCLIP,
            content="GeoCLIP prediction",
            confidence=0.7,
            country="France",
        ))
        
        hint = chain.get_hint_from_evidence()
        assert hint is None


class TestScoringConfigHintValues:
    """Tests for scoring config values related to hints."""
    
    def test_hint_vote_multiplier(self):
        """Hint vote multiplier should be high."""
        from src.scoring.config import ScoringConfig
        config = ScoringConfig()
        
        # Should be >= 5 for strong hint authority
        assert config.country_penalty.hint_vote_multiplier >= 5
    
    def test_hint_boost_values(self):
        """Hint boost values should be significant."""
        from src.scoring.config import ScoringConfig
        config = ScoringConfig()
        
        # Match boost should be > 1.0
        assert config.hint.match_boost > 1.0
        # Strong match boost should be even higher
        assert config.hint.strong_match_boost > config.hint.match_boost
        # No-match penalty should be significant
        assert config.hint.no_match_penalty < 0.3
    
    def test_strong_hint_penalty(self):
        """Strong hint penalty should be heavy."""
        from src.scoring.config import ScoringConfig
        config = ScoringConfig()
        
        # Should be low to heavily penalize wrong-country candidates
        assert config.country_penalty.strong_hint_penalty < 0.2
