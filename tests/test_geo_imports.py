"""Import smoke tests for geo module dependencies.

Ensures country_codes exports what serper_client and other consumers need,
catching missing exports before deployment.
"""

import pytest


def test_country_codes_exports_required_symbols():
    """country_codes must export symbols required by serper_client."""
    import src.geo.country_codes as cc

    required = [
        "extract_country_from_location",
        "get_google_cr",
        "get_google_gl",
    ]
    for name in required:
        assert hasattr(cc, name), f"country_codes missing export: {name}"


def test_serper_client_imports_successfully():
    """SerperClient import pulls in country_codes with no ImportError."""
    from src.geo.serper_client import SerperClient

    assert SerperClient is not None


def test_get_google_gl_and_cr_values():
    """get_google_gl and get_google_cr return correct format for common codes."""
    from src.geo.country_codes import get_google_cr, get_google_gl

    assert get_google_gl("US") == "us"
    assert get_google_gl("TR") == "tr"
    assert get_google_gl("DE") == "de"
    assert get_google_gl("GB") == "uk"  # Special case

    assert get_google_cr("US") == "countryUS"
    assert get_google_cr("TR") == "countryTR"
    assert get_google_cr("GB") == "countryUK"  # Special case
    assert get_google_cr("") is None
    assert get_google_cr("XX") == "countryXX"
