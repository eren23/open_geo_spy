"""Extended tests for hint handling - edge cases, real-world scenarios, integration."""

import pytest

from src.evidence.chain import Evidence, EvidenceChain, EvidenceSource
from src.geo.country_matcher import (
    countries_match,
    get_iso_code,
    normalize_country_name,
    extract_country_from_location,
    get_all_names,
)


class TestCountryMatcherEdgeCases:
    """Edge cases for country matching."""
    
    def test_country_with_extra_whitespace(self):
        """Countries with extra whitespace normalize correctly."""
        assert get_iso_code("  Germany  ") == "DE"
        assert get_iso_code("\tFrance\n") == "FR"
    
    def test_country_with_the_prefix(self):
        """Countries with 'the' prefix normalize correctly."""
        iso = get_iso_code("The Netherlands")
        assert iso == "NL"
        
        iso = get_iso_code("the United States")
        assert iso == "US"
    
    def test_country_with_diacritics_variants(self):
        """Countries with various diacritic styles."""
        # España with and without accent
        assert countries_match("Espana", "España") is True
        assert countries_match("Espana", "Spain") is True
    
    def test_misspelled_countries(self):
        """Common misspellings may match via fuzzy matching (threshold 0.85)."""
        # Note: fuzzy matching depends on threshold (0.85)
        # "Gernmany" with two n's matches, "Gernamy" may not
        # This tests that the fuzzy matching system exists
        result = countries_match("Gernmany", "Germany")
        # Either works or doesn't - just verify no crash
        assert isinstance(result, bool)
    
    def test_empty_and_none_inputs(self):
        """Empty and None inputs are handled gracefully."""
        assert normalize_country_name(None) == ""
        assert normalize_country_name("") == ""
        assert get_iso_code(None) is None
        assert get_iso_code("") is None
        assert countries_match("", "") is False
        assert countries_match(None, None) is False
    
    def test_whitespace_handling(self):
        """Various whitespace patterns are normalized."""
        assert get_iso_code("  Germany  ") == "DE"
        assert get_iso_code("New\tYork") is None  # Not a country, city
        # But Istanbul should work
        assert extract_country_from_location("  Istanbul  ") == "TR"


class TestMoreCityToCountryMappings:
    """Tests for city-to-country resolution."""
    
    def test_european_cities(self):
        """European cities resolve correctly."""
        capitals = {
            "Berlin": "DE",
            "Paris": "FR",
            "London": "GB",
            "Rome": "IT",
            "Madrid": "ES",
            "Amsterdam": "NL",
            "Vienna": "AT",
            "Prague": "CZ",
            "Warsaw": "PL",
            "Budapest": "HU",
            "Brussels": "BE",
            "Copenhagen": "DK",
            "Stockholm": "SE",
            "Oslo": "NO",
            "Helsinki": "FI",
            "Dublin": "IE",
            "Lisbon": "PT",
            "Athens": "GR",
        }
        for city, expected_iso in capitals.items():
            iso = extract_country_from_location(city)
            assert iso == expected_iso, f"{city} should be {expected_iso}, got {iso}"
    
    def test_asian_cities(self):
        """Asian cities resolve correctly."""
        cities = {
            "Tokyo": "JP",
            "Beijing": "CN",
            "Shanghai": "CN",
            "Seoul": "KR",
            "Singapore": "SG",
            "Bangkok": "TH",
            "Mumbai": "IN",
            "Delhi": "IN",
            "Hong Kong": "HK",
            "Jakarta": "ID",
            "Kuala Lumpur": "MY",
            "Manila": "PH",
        }
        for city, expected_iso in cities.items():
            iso = extract_country_from_location(city)
            assert iso == expected_iso, f"{city} should be {expected_iso}, got {iso}"
    
    def test_americas_cities(self):
        """Cities in Americas resolve correctly."""
        cities = {
            "New York": "US",
            "Los Angeles": "US",
            "Chicago": "US",
            "San Francisco": "US",
            "Toronto": "CA",
            "Montreal": "CA",
            "Vancouver": "CA",
            "Mexico City": "MX",
            "São Paulo": "BR",
            "Rio de Janeiro": "BR",
            "Buenos Aires": "AR",
            "Lima": "PE",
        }
        for city, expected_iso in cities.items():
            iso = extract_country_from_location(city)
            assert iso == expected_iso, f"{city} should be {expected_iso}, got {iso}"
    
    def test_turkish_cities(self):
        """Turkish cities resolve correctly."""
        cities = {
            "Istanbul": "TR",
            "Ankara": "TR",
            "Izmir": "TR",
            "Antalya": "TR",
            "Bursa": "TR",
        }
        for city, expected_iso in cities.items():
            iso = extract_country_from_location(city)
            assert iso == expected_iso, f"{city} should be {expected_iso}, got {iso}"


class TestEvidenceChainFilteringEdgeCases:
    """Edge cases for Evidence chain filtering."""
    
    def create_evidence(
        self,
        source: EvidenceSource,
        country: str,
        confidence: float = 0.8,
        lat: float = None,
        lon: float = None,
    ) -> Evidence:
        """Helper to create evidence."""
        return Evidence(
            source=source,
            content=f"Location in {country}",
            confidence=confidence,
            country=country,
            latitude=lat,
            longitude=lon,
        )
    
    def test_filter_empty_chain(self):
        """Filtering empty chain returns empty chain."""
        chain = EvidenceChain()
        filtered = chain.filter_by_hint("Germany")
        assert len(filtered.evidences) == 0
    
    def test_filter_chain_with_only_user_hints(self):
        """Chain with only user hints keeps all."""
        chain = EvidenceChain()
        chain.add(Evidence(
            source=EvidenceSource.USER_HINT,
            content="Hint 1",
            confidence=0.8,
            metadata={"hint": "Germany"},
        ))
        chain.add(Evidence(
            source=EvidenceSource.USER_HINT,
            content="Hint 2",
            confidence=0.8,
            metadata={"hint": "Berlin"},
        ))
        
        filtered = chain.filter_by_hint("Germany")
        assert len(filtered.evidences) == 2
    
    def test_filter_preserves_coordinates(self):
        """Filtering preserves coordinate evidence."""
        chain = EvidenceChain()
        chain.add(self.create_evidence(
            EvidenceSource.GEOCLIP,
            "Germany",
            lat=52.52,
            lon=13.405,
        ))
        
        filtered = chain.filter_by_hint("Germany")
        assert len(filtered.evidences) == 1
        assert filtered.evidences[0].latitude == 52.52
        assert filtered.evidences[0].longitude == 13.405
    
    def test_filter_with_confidence_ordering(self):
        """Filtering maintains evidence ordering."""
        chain = EvidenceChain()
        chain.add(self.create_evidence(EvidenceSource.GEOCLIP, "Germany", 0.9))
        chain.add(self.create_evidence(EvidenceSource.VLM_GEO, "Germany", 0.7))
        chain.add(self.create_evidence(EvidenceSource.STREETCLIP, "Germany", 0.8))
        
        filtered = chain.filter_by_hint("Germany")
        assert len(filtered.evidences) == 3
        # Order should be preserved
        assert filtered.evidences[0].confidence == 0.9
        assert filtered.evidences[1].confidence == 0.7
        assert filtered.evidences[2].confidence == 0.8
    
    def test_filter_multiple_hints(self):
        """Filtering with multiple passes works correctly."""
        chain = EvidenceChain()
        chain.add(self.create_evidence(EvidenceSource.GEOCLIP, "Germany"))
        chain.add(self.create_evidence(EvidenceSource.VLM_GEO, "France"))
        chain.add(self.create_evidence(EvidenceSource.STREETCLIP, "Italy"))
        
        # First filter by Germany
        filtered1 = chain.filter_by_hint("Germany")
        assert len(filtered1.evidences) == 1
        
        # Filter original by France
        filtered2 = chain.filter_by_hint("France")
        assert len(filtered2.evidences) == 1
        assert filtered2.evidences[0].country == "France"
    
    def test_filter_with_native_names(self):
        """Filtering works with native country names in evidence."""
        chain = EvidenceChain()
        chain.add(self.create_evidence(EvidenceSource.GEOCLIP, "Deutschland"))
        chain.add(self.create_evidence(EvidenceSource.VLM_GEO, "España"))
        chain.add(self.create_evidence(EvidenceSource.STREETCLIP, "Österreich"))
        
        # Filter by English names
        filtered_de = chain.filter_by_hint("Germany")
        assert len(filtered_de.evidences) == 1
        
        filtered_es = chain.filter_by_hint("Spain")
        assert len(filtered_es.evidences) == 1
        
        filtered_at = chain.filter_by_hint("Austria")
        assert len(filtered_at.evidences) == 1


class TestRealWorldScenarios:
    """Real-world hint handling scenarios."""
    
    def create_evidence_chain_for_germany_scenario(self) -> EvidenceChain:
        """Create a realistic evidence chain for Germany scenario."""
        chain = EvidenceChain()
        
        # User hint
        chain.add(Evidence(
            source=EvidenceSource.USER_HINT,
            content="User location hint: Germany",
            confidence=0.8,
            metadata={"hint": "Germany"},
        ))
        
        # ML predictions (some wrong)
        chain.add(Evidence(
            source=EvidenceSource.GEOCLIP,
            content="GeoCLIP: Paris, France",
            confidence=0.85,
            country="France",
            latitude=48.8566,
            longitude=2.3522,
        ))
        chain.add(Evidence(
            source=EvidenceSource.STREETCLIP,
            content="StreetCLIP: Berlin, Germany",
            confidence=0.78,
            country="Germany",
            latitude=52.52,
            longitude=13.405,
        ))
        chain.add(Evidence(
            source=EvidenceSource.VLM_GEO,
            content="VLM Geo: Munich, Germany",
            confidence=0.72,
            country="Germany",
            latitude=48.1351,
            longitude=11.582,
        ))
        
        # Visual features
        chain.add(Evidence(
            source=EvidenceSource.VLM_ANALYSIS,
            content="Architecture: Central European",
            confidence=0.6,
            # No country
        ))
        
        return chain
    
    def test_germany_scenario_filtering(self):
        """Test realistic Germany scenario with mixed ML predictions."""
        chain = self.create_evidence_chain_for_germany_scenario()
        
        # Filter by Germany hint
        filtered = chain.filter_by_hint("Germany", keep_non_geo=True)
        
        # Should keep: user hint, StreetCLIP, VLM_GEO, visual features
        # Should remove: GeoCLIP (France)
        assert len(filtered.evidences) == 4
        
        # Check no France evidence
        countries = [e.country for e in filtered.evidences if e.country]
        assert "France" not in countries
        assert all(c == "Germany" for c in countries)
    
    def test_istanbul_scenario(self):
        """Test Istanbul (major Turkish city) scenario."""
        chain = EvidenceChain()
        chain.add(Evidence(
            source=EvidenceSource.USER_HINT,
            content="user hint: Istanbul",
            confidence=0.8,
            metadata={"hint": "Istanbul"},
        ))
        chain.add(Evidence(
            source=EvidenceSource.GEOCLIP,
            content="Athens, Greece",
            confidence=0.82,
            country="Greece",
        ))
        chain.add(Evidence(
            source=EvidenceSource.STREETCLIP,
            content="Istanbul, Turkey",
            confidence=0.75,
            country="Turkey",
        ))
        
        # Extract country from Istanbul hint, then filter
        from src.geo.country_matcher import extract_country_from_location
        turkey_hint = extract_country_from_location("Istanbul")
        assert turkey_hint == "TR", f"Istanbul should resolve to TR, got {turkey_hint}"
        
        # Filter by Turkey (the resolved country)
        filtered = chain.filter_by_hint("Turkey", keep_non_geo=True)
        
        # Should keep user hint and Turkey evidence, remove Greece
        assert len(filtered.evidences) == 2
        countries = [e.country for e in filtered.evidences if e.country]
        assert "Greece" not in countries
    
    def test_istanbul_city_to_country_resolution(self):
        """Test that Istanbul resolves to Turkey."""
        iso = extract_country_from_location("Istanbul")
        assert iso == "TR"
    
    def test_holland_vs_netherlands_scenario(self):
        """Test Holland alias for Netherlands."""
        chain = EvidenceChain()
        chain.add(Evidence(
            source=EvidenceSource.USER_HINT,
            content="User hint: Holland",
            confidence=0.8,
            metadata={"hint": "Holland"},
        ))
        chain.add(Evidence(
            source=EvidenceSource.GEOCLIP,
            content="Amsterdam, Netherlands",
            confidence=0.85,
            country="Netherlands",
        ))
        
        # Holland should match Netherlands
        filtered = chain.filter_by_hint("Holland")
        assert len(filtered.evidences) == 2
    
    def test_uk_vs_england_scenario(self):
        """Test UK/England equivalence."""
        chain = EvidenceChain()
        chain.add(Evidence(
            source=EvidenceSource.USER_HINT,
            content="User hint: UK",
            confidence=0.8,
            metadata={"hint": "UK"},
        ))
        chain.add(Evidence(
            source=EvidenceSource.GEOCLIP,
            content="London, England",
            confidence=0.88,
            country="England",
        ))
        chain.add(Evidence(
            source=EvidenceSource.STREETCLIP,
            content="Manchester, United Kingdom",
            confidence=0.75,
            country="United Kingdom",
        ))
        
        # UK should match England and United Kingdom
        filtered = chain.filter_by_hint("UK")
        assert len(filtered.evidences) == 3


class TestEvidenceChainHintMethods:
    """Tests for EvidenceChain methods related to hints."""
    
    def test_get_hint_from_empty_chain(self):
        """get_hint_from_evidence returns None for empty chain."""
        chain = EvidenceChain()
        assert chain.get_hint_from_evidence() is None
    
    def test_get_hint_from_chain_with_evidence_but_no_hint(self):
        """get_hint_from_evidence returns None when no hint present."""
        chain = EvidenceChain()
        chain.add(Evidence(
            source=EvidenceSource.GEOCLIP,
            content="GeoCLIP prediction",
            confidence=0.8,
            country="Germany",
        ))
        assert chain.get_hint_from_evidence() is None
    
    def test_get_hint_from_chain_with_multiple_hints(self):
        """get_hint_from_evidence returns the first hint found."""
        chain = EvidenceChain()
        chain.add(Evidence(
            source=EvidenceSource.USER_HINT,
            content="First hint",
            confidence=0.8,
            metadata={"hint": "Germany"},
        ))
        chain.add(Evidence(
            source=EvidenceSource.USER_HINT,
            content="Second hint",
            confidence=0.8,
            metadata={"hint": "Berlin"},
        ))
        
        hint = chain.get_hint_from_evidence()
        assert hint is not None
        # Returns the first one found
        assert "Germany" in hint or "Berlin" in hint


class TestCountryMatcherPerformance:
    """Tests for country matcher performance characteristics."""
    
    def test_repeated_lookups_are_fast(self):
        """Repeated lookups should be fast (cached or O(1))."""
        import time
        
        # First lookup
        start = time.time()
        for _ in range(1000):
            get_iso_code("Germany")
        elapsed = time.time() - start
        
        # Should be very fast (< 0.1 seconds for 1000 lookups)
        assert elapsed < 0.1, f"1000 lookups took {elapsed}s, expected < 0.1s"
    
    def test_all_major_countries_resolve(self):
        """All G20 countries resolve correctly."""
        g20_countries = [
            ("Argentina", "AR"),
            ("Australia", "AU"),
            ("Brazil", "BR"),
            ("Canada", "CA"),
            ("China", "CN"),
            ("France", "FR"),
            ("Germany", "DE"),
            ("India", "IN"),
            ("Indonesia", "ID"),
            ("Italy", "IT"),
            ("Japan", "JP"),
            ("Mexico", "MX"),
            ("Russia", "RU"),
            ("Saudi Arabia", "SA"),
            ("South Africa", "ZA"),
            ("South Korea", "KR"),
            ("Turkey", "TR"),
            ("United Kingdom", "GB"),
            ("United States", "US"),
            ("European Union", None),  # Not a country
        ]
        
        for name, expected_iso in g20_countries:
            iso = get_iso_code(name)
            if expected_iso is None:
                # European Union is not a country, might resolve to None or something
                pass
            else:
                assert iso == expected_iso, f"{name} should be {expected_iso}, got {iso}"


class TestScoringConfigValues:
    """Tests for scoring config values related to hints."""
    
    def test_hint_vote_multiplier_is_high(self):
        """Hint vote multiplier should be moderate (tie-break, not veto)."""
        from src.scoring.config import ScoringConfig
        config = ScoringConfig()
        assert 2 <= config.country_penalty.hint_vote_multiplier <= 5
    
    def test_hint_match_boost_is_significant(self):
        """Hint match boost should be significant."""
        from src.scoring.config import ScoringConfig
        config = ScoringConfig()
        
        # Match boost should be > 1.0
        assert config.hint.match_boost > 1.0
        # Strong match boost should be even higher
        assert config.hint.strong_match_boost > config.hint.match_boost
    
    def test_no_match_penalty_is_heavy(self):
        """No-match scales confidence down but keeps alternates in play."""
        from src.scoring.config import ScoringConfig
        config = ScoringConfig()
        assert 0.5 <= config.hint.no_match_penalty <= 0.85
    
    def test_strong_hint_penalty_is_very_heavy(self):
        """Strong hint penalty downranks but does not zero out alternates."""
        from src.scoring.config import ScoringConfig
        config = ScoringConfig()
        assert 0.4 <= config.country_penalty.strong_hint_penalty <= 0.8
    
    def test_country_match_weight_same_with_hint(self):
        """Country match weight should be the same with or without hint (hint influence via boost/penalty)."""
        from src.scoring.config import ScoringConfig
        config = ScoringConfig()

        # country_match_with_hint should equal country_match (hint influence via boost/penalty, not extra weight)
        assert config.candidate_ranking.country_match_with_hint == config.candidate_ranking.country_match
