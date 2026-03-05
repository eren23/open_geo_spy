"""Tests for country matcher functionality."""

import pytest

from src.geo.country_matcher import (
    countries_match,
    get_iso_code,
    normalize_country_name,
    extract_country_from_location,
    get_all_names,
)


class TestNormalizeCountryName:
    """Tests for country name normalization."""
    
    def test_lowercase(self):
        assert normalize_country_name("GERMANY") == "germany"
        assert normalize_country_name("Germany") == "germany"
    
    def test_strip_whitespace(self):
        assert normalize_country_name("  Germany  ") == "germany"
    
    def test_remove_the_prefix(self):
        assert normalize_country_name("The United States") == "united states"
        assert normalize_country_name("the Netherlands") == "netherlands"
    
    def test_remove_punctuation(self):
        assert normalize_country_name("Germany!") == "germany"
        assert normalize_country_name("U.S.A.") == "usa"
    
    def test_empty_string(self):
        assert normalize_country_name("") == ""
        assert normalize_country_name(None) == ""


class TestGetIsoCode:
    """Tests for ISO code lookup."""
    
    def test_english_names(self):
        assert get_iso_code("Germany") == "DE"
        assert get_iso_code("France") == "FR"
        assert get_iso_code("United States") == "US"
    
    def test_native_names(self):
        assert get_iso_code("Deutschland") == "DE"
        assert get_iso_code("España") == "ES"
        # Note: Österreich with umlaut gets normalized to osterreich
        assert get_iso_code("Osterreich") == "AT"
    
    def test_native_names_variants(self):
        """Native names with/without accents work."""
        assert get_iso_code("Espana") == "ES"  # Without accent
        assert get_iso_code("España") == "ES"  # With accent (normalized)
    
    def test_abbreviations(self):
        assert get_iso_code("US") == "US"
        assert get_iso_code("USA") == "US"
        assert get_iso_code("UK") == "GB"
    
    def test_common_aliases(self):
        assert get_iso_code("Holland") == "NL"
        assert get_iso_code("America") == "US"
        assert get_iso_code("Great Britain") == "GB"
    
    def test_already_iso_code(self):
        assert get_iso_code("DE") == "DE"
        assert get_iso_code("FR") == "FR"
    
    def test_unknown_country(self):
        assert get_iso_code("Atlantis") is None


class TestCountriesMatch:
    """Tests for country matching."""
    
    def test_same_country_different_names(self):
        """Germany matches Deutschland"""
        assert countries_match("Germany", "Deutschland") is True
    
    def test_abbreviation_matches(self):
        """US matches United States"""
        assert countries_match("US", "United States") is True
        assert countries_match("USA", "United States") is True
    
    def test_uk_variants(self):
        """UK variants all match"""
        assert countries_match("UK", "United Kingdom") is True
        assert countries_match("Britain", "United Kingdom") is True
        assert countries_match("Great Britain", "United Kingdom") is True
        assert countries_match("England", "United Kingdom") is True
    
    def test_different_countries(self):
        """Different countries don't match"""
        assert countries_match("Germany", "France") is False
        assert countries_match("US", "Canada") is False
    
    def test_case_insensitive(self):
        """Matching is case insensitive"""
        assert countries_match("germany", "GERMANY") is True
        assert countries_match("Germany", "GERMANY") is True
    
    def test_holland_netherlands(self):
        """Holland matches Netherlands"""
        assert countries_match("Holland", "Netherlands") is True
    
    def test_fuzzy_typo_matching(self):
        """Fuzzy matching handles typos"""
        assert countries_match("Gernmany", "Germany") is True  # Typo
        assert countries_match("Frnce", "France") is True  # Typo
    
    def test_empty_inputs(self):
        """Empty inputs return False"""
        assert countries_match("", "Germany") is False
        assert countries_match("Germany", "") is False
        assert countries_match("", "") is False


class TestExtractCountryFromLocation:
    """Tests for extracting country from location strings."""
    
    def test_country_name(self):
        """Country names resolve to ISO codes"""
        assert extract_country_from_location("Germany") == "DE"
        assert extract_country_from_location("France") == "FR"
    
    def test_major_cities(self):
        """Major cities resolve to their countries"""
        assert extract_country_from_location("Berlin") == "DE"
        assert extract_country_from_location("Paris") == "FR"
        assert extract_country_from_location("London") == "GB"
        assert extract_country_from_location("Tokyo") == "JP"
    
    def test_city_state_combinations(self):
        """City with state info still works"""
        assert extract_country_from_location("Munich, Germany") == "DE"
        assert extract_country_from_location("Paris, France") == "FR"
    
    def test_unknown_location(self):
        """Unknown locations return None"""
        assert extract_country_from_location("Atlantis") is None
        assert extract_country_from_location("") is None


class TestGetAllNames:
    """Tests for getting all names for a country."""
    
    def test_germany(self):
        """Germany has multiple names"""
        names = get_all_names("DE")
        assert "germany" in names
        assert "deutschland" in names
    
    def test_us(self):
        """US has multiple aliases"""
        names = get_all_names("US")
        assert "united states" in names
        assert "usa" in names
        assert "us" in names
        assert "america" in names
    
    def test_unknown_iso(self):
        """Unknown ISO code returns empty list"""
        assert get_all_names("XX") == []


class TestHintCountryMatching:
    """Integration tests for hint-based country matching scenarios."""
    
    def test_user_hint_germany_matches_deutschland_prediction(self):
        """User says Germany, model predicts Deutschland - should match"""
        hint = "Germany"
        prediction_country = "Deutschland"
        assert countries_match(hint, prediction_country) is True
    
    def test_user_hint_uk_matches_england(self):
        """User says UK, model predicts England - should match"""
        hint = "UK"
        prediction_country = "England"
        assert countries_match(hint, prediction_country) is True
    
    def test_user_hint_holland_matches_netherlands(self):
        """User says Holland, model predicts Netherlands - should match"""
        hint = "Holland"
        prediction_country = "Netherlands"
        assert countries_match(hint, prediction_country) is True
    
    def test_user_hint_germany_doesnt_match_france(self):
        """User says Germany, model predicts France - should NOT match"""
        hint = "Germany"
        prediction_country = "France"
        assert countries_match(hint, prediction_country) is False
    
    def test_city_in_hint(self):
        """User provides city, we can extract country"""
        # Istanbul -> TR
        iso = extract_country_from_location("Istanbul")
        assert iso == "TR"
        
        # Barcelona -> ES
        iso = extract_country_from_location("Barcelona")
        assert iso == "ES"
