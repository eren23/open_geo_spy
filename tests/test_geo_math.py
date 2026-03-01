"""Tests for geographic math utilities."""

import pytest

from src.utils.geo_math import (
    bounding_box,
    country_level_agreement,
    geographic_spread,
    haversine_distance,
    validate_coordinates,
    weighted_centroid,
)


class TestHaversine:
    def test_same_point_is_zero(self):
        assert haversine_distance(0, 0, 0, 0) == 0.0

    def test_known_distance(self):
        # Paris to London ≈ 344 km
        d = haversine_distance(48.8566, 2.3522, 51.5074, -0.1278)
        assert 340 < d < 350

    def test_antipodal(self):
        # Nearly half the earth's circumference
        d = haversine_distance(0, 0, 0, 180)
        assert d > 20000


class TestValidateCoordinates:
    def test_valid(self):
        assert validate_coordinates(0, 0)
        assert validate_coordinates(-90, -180)
        assert validate_coordinates(90, 180)

    def test_invalid(self):
        assert not validate_coordinates(91, 0)
        assert not validate_coordinates(0, 181)
        assert not validate_coordinates(-91, -181)


class TestBoundingBox:
    def test_small_radius(self):
        min_lat, min_lon, max_lat, max_lon = bounding_box(0, 0, 10)
        assert min_lat < 0 < max_lat
        assert min_lon < 0 < max_lon

    def test_near_pole(self):
        bb = bounding_box(89.5, 0, 100)
        assert bb[2] <= 90


class TestWeightedCentroid:
    def test_empty(self):
        assert weighted_centroid([]) is None

    def test_single_point(self):
        result = weighted_centroid([(10.0, 20.0, 1.0)])
        assert result == (10.0, 20.0)

    def test_equal_weights(self):
        result = weighted_centroid([(0.0, 0.0, 1.0), (10.0, 10.0, 1.0)])
        assert result is not None
        assert abs(result[0] - 5.0) < 0.01
        assert abs(result[1] - 5.0) < 0.01

    def test_zero_weights(self):
        assert weighted_centroid([(10, 20, 0), (30, 40, 0)]) is None


class TestGeographicSpread:
    def test_single_point(self):
        assert geographic_spread([(0, 0)]) == 0.0

    def test_close_points(self):
        spread = geographic_spread([(48.85, 2.35), (48.86, 2.36)])
        assert spread < 5  # km

    def test_far_points(self):
        spread = geographic_spread([(0, 0), (40, 40)])
        assert spread > 5000


class TestCountryAgreement:
    def test_empty(self):
        assert country_level_agreement([]) == 0.0

    def test_full_agreement(self):
        assert country_level_agreement(["France", "France", "France"]) == 1.0

    def test_split(self):
        score = country_level_agreement(["France", "Germany"])
        assert score == 0.5

    def test_case_insensitive(self):
        score = country_level_agreement(["france", "France", "FRANCE"])
        assert score == 1.0
