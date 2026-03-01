"""Tests for the Evidence and EvidenceChain classes."""

from src.evidence.chain import Evidence, EvidenceChain, EvidenceSource


class TestEvidence:
    def test_content_hash_deterministic(self):
        e1 = Evidence(source=EvidenceSource.GEOCLIP, content="test", confidence=0.5)
        e2 = Evidence(source=EvidenceSource.GEOCLIP, content="test", confidence=0.9)
        assert e1.content_hash == e2.content_hash

    def test_different_content_different_hash(self):
        e1 = Evidence(source=EvidenceSource.GEOCLIP, content="a", confidence=0.5)
        e2 = Evidence(source=EvidenceSource.GEOCLIP, content="b", confidence=0.5)
        assert e1.content_hash != e2.content_hash

    def test_confidence_clamped(self):
        e = Evidence(source=EvidenceSource.OCR, content="x", confidence=1.5)
        assert e.confidence == 1.0
        e2 = Evidence(source=EvidenceSource.OCR, content="y", confidence=-0.3)
        assert e2.confidence == 0.0

    def test_invalid_coordinates_cleared(self):
        e = Evidence(
            source=EvidenceSource.GEOCLIP,
            content="bad",
            confidence=0.5,
            latitude=200.0,
            longitude=500.0,
        )
        assert e.latitude is None
        assert e.longitude is None

    def test_valid_coordinates_kept(self):
        e = Evidence(
            source=EvidenceSource.GEOCLIP,
            content="good",
            confidence=0.5,
            latitude=48.85,
            longitude=2.35,
        )
        assert e.latitude == 48.85
        assert e.has_coordinates

    def test_to_dict_roundtrip(self):
        e = Evidence(
            source=EvidenceSource.SERPER,
            content="test",
            confidence=0.7,
            latitude=40.0,
            longitude=-74.0,
            country="US",
        )
        d = e.to_dict()
        e2 = Evidence.from_dict(d)
        assert e2.source == e.source
        assert e2.content == e.content
        assert e2.latitude == e.latitude


class TestEvidenceChain:
    def test_dedup_by_hash(self, sample_evidence):
        chain = EvidenceChain()
        assert chain.add(sample_evidence[0]) is True
        assert chain.add(sample_evidence[0]) is False  # duplicate
        assert len(chain.evidences) == 1

    def test_add_many_returns_count(self, sample_evidence):
        chain = EvidenceChain()
        added = chain.add_many(sample_evidence)
        assert added == len(sample_evidence)
        # Adding same again should add 0
        added2 = chain.add_many(sample_evidence)
        assert added2 == 0

    def test_centroid(self, sample_chain):
        centroid = sample_chain.location_cluster()
        assert centroid is not None
        lat, lon = centroid
        # Should be near Paris
        assert 48.0 < lat < 49.0
        assert 2.0 < lon < 3.0

    def test_agreement_score_high_for_agreement(self, sample_chain):
        score = sample_chain.agreement_score()
        # All evidence points to France/Paris, should be high
        assert score > 0.6

    def test_agreement_score_low_for_disagreement(self):
        chain = EvidenceChain()
        chain.add(Evidence(
            source=EvidenceSource.GEOCLIP, content="a",
            confidence=0.5, latitude=48.8, longitude=2.3, country="France",
        ))
        chain.add(Evidence(
            source=EvidenceSource.STREETCLIP, content="b",
            confidence=0.5, latitude=-33.8, longitude=151.2, country="Australia",
        ))
        assert chain.agreement_score() < 0.5

    def test_country_predictions(self, sample_chain):
        countries = sample_chain.country_predictions
        assert "France" in countries

    def test_by_source(self, sample_chain):
        ocr = sample_chain.by_source(EvidenceSource.OCR)
        assert len(ocr) == 1
        assert ocr[0].source == EvidenceSource.OCR

    def test_to_prompt_context(self, sample_chain):
        text = sample_chain.to_prompt_context()
        assert "[geoclip]" in text or "[vlm_geo]" in text
        assert "conf=" in text

    def test_clear(self, sample_chain):
        assert len(sample_chain.evidences) > 0
        sample_chain.clear()
        assert len(sample_chain.evidences) == 0
