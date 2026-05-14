"""
test_transform.py

Unit tests for src/pipeline/transform.py.

Tests cover the three core behaviours of the transform function:
  - URL deduplication
  - Segment routing
  - Sentiment scoring

Why test these specifically?
  Deduplication: was previously missing, which caused 952 duplicate rows
  to enter the warehouse undetected. A test here means that class of
  problem is caught in under a second rather than after a 38-minute load.

  Segment routing and sentiment: these are pure functions with no
  external dependencies. They are fast to test and the results feed
  directly into the warehouse. A bug here corrupts every downstream
  analysis without any obvious error message.

Usage:
  pytest tests/test_transform.py -v
"""

import pandas as pd
import pytest
from src.pipeline.transform import transform, _route_segment, _score_sentiment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_articles(*urls: str, themes: str = "") -> pd.DataFrame:
    """
    Build a minimal raw article DataFrame for testing.
    Only the columns transform() actually uses are populated.
    Other columns are left as empty strings to keep tests focused.
    """
    return pd.DataFrame({
        "url":           list(urls),
        "source_name":   ["test_source"] * len(urls),
        "seendate":      pd.to_datetime(["2025-05-07 00:00:00"] * len(urls)),
        "themes":        [themes] * len(urls),
        "persons":       [""] * len(urls),
        "organizations": [""] * len(urls),
        "language":      ["English"] * len(urls),
    })


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------

class TestDeduplication:

    def test_duplicate_urls_are_removed(self):
        """
        Two rows with the same URL should produce one article in the output.
        This is the exact scenario that caused 952 duplicate rows in the
        warehouse — GDELT republishes the same URL across multiple 15-minute
        windows and the pipeline must deduplicate before load.
        """
        df_raw = make_articles(
            "https://example.com/article-1",
            "https://example.com/article-1",  # duplicate
        )

        df_articles, _ = transform(df_raw)

        assert len(df_articles) == 1, (
            f"Expected 1 article after deduplication, got {len(df_articles)}"
        )

    def test_duplicate_url_entity_records_are_also_deduplicated(self):
        """
        Entity records must be extracted after deduplication, not before.
        If deduplication happened after entity extraction, the entity_records
        list would contain mentions from both the original and the duplicate row,
        leading to orphaned or doubled entity mentions in the warehouse.
        """
        df_raw = make_articles(
            "https://example.com/article-1",
            "https://example.com/article-1",  # duplicate
        )
        # Add a person to both rows so entity records would double if
        # deduplication happened after extraction
        df_raw["persons"] = "John Smith,1"

        df_articles, entity_records = transform(df_raw)

        article_urls_in_entities = {e["url"] for e in entity_records}
        mention_count = sum(
            1 for e in entity_records
            if e["url"] == "https://example.com/article-1"
        )

        assert len(df_articles) == 1
        assert mention_count == len(set(
            e["entity_name"] for e in entity_records
            if e["url"] == "https://example.com/article-1"
        )), "Entity mentions should not be doubled from duplicate rows"

    def test_unique_urls_are_all_kept(self):
        """
        Deduplication should not drop any rows when all URLs are unique.
        """
        df_raw = make_articles(
            "https://example.com/article-1",
            "https://example.com/article-2",
            "https://example.com/article-3",
        )

        df_articles, _ = transform(df_raw)

        assert len(df_articles) == 3, (
            f"Expected 3 articles, got {len(df_articles)} — "
            "deduplication must not remove unique URLs"
        )

    def test_empty_dataframe_returns_empty(self):
        """
        An empty input DataFrame should return an empty DataFrame and
        an empty entity list without raising an error.
        """
        df_raw = pd.DataFrame()
        df_articles, entity_records = transform(df_raw)

        assert df_articles.empty
        assert entity_records == []


# ---------------------------------------------------------------------------
# Segment routing tests
# ---------------------------------------------------------------------------

class TestSegmentRouting:

    def test_politics_theme_routes_correctly(self):
        assert _route_segment("POLITICS;TAX_FNCACT") == "Politics and Government"

    def test_business_theme_routes_correctly(self):
        assert _route_segment("ECON_POVERTY;TRADE_WTO") == "Business and Markets"

    def test_technology_theme_routes_correctly(self):
        assert _route_segment("TECH_INTERNET;CYBER_ATTACK") == "Technology"

    def test_sports_theme_routes_correctly(self):
        assert _route_segment("SPORTS;SPORT_FOOTBALL") == "Sports"

    def test_health_theme_routes_correctly(self):
        assert _route_segment("HEALTH_PANDEMIC;MEDICAL_VACCINE") == "Science and Health"

    def test_crime_theme_routes_correctly(self):
        assert _route_segment("CRIME_MURDER;LEGAL_PROCEEDINGS") == "Crime and Justice"

    def test_entertainment_theme_routes_correctly(self):
        assert _route_segment("ENTERTAIN_MUSIC;ARTS_FILM") == "Entertainment and Culture"

    def test_no_matching_theme_returns_general(self):
        """
        Articles with no theme tags or unrecognised tags should fall
        back to General, not raise an error or return None.
        """
        assert _route_segment("") == "General"
        assert _route_segment("UNKNOWN_TAG;ANOTHER_UNKNOWN") == "General"

    def test_null_theme_returns_general(self):
        """
        GDELT articles frequently have null theme fields.
        The router must handle this gracefully.
        """
        assert _route_segment(None) == "General"

    def test_first_match_wins(self):
        """
        When a themes string matches multiple segments, the first match
        in SEGMENT_MAP order should win. This test confirms the
        priority ordering is respected.
        """
        # POLITICS comes before ECON in SEGMENT_MAP, so this should
        # route to Politics and Government, not Business and Markets
        result = _route_segment("POLITICS;ECON_TRADE")
        assert result == "Politics and Government"


# ---------------------------------------------------------------------------
# Sentiment scoring tests
# ---------------------------------------------------------------------------

class TestSentimentScoring:

    def test_positive_text_returns_positive_label(self):
        score, label = _score_sentiment("excellent wonderful amazing")
        assert label == "POSITIVE"
        assert score >= 0.05

    def test_negative_text_returns_negative_label(self):
        score, label = _score_sentiment("terrible awful horrible disaster")
        assert label == "NEGATIVE"
        assert score <= -0.05

    def test_neutral_text_returns_neutral_label(self):
        # Domain names carry no sentiment — this mirrors real pipeline input
        score, label = _score_sentiment("iheart.com")
        assert label == "NEUTRAL"
        assert -0.05 < score < 0.05

    def test_empty_string_returns_neutral(self):
        score, label = _score_sentiment("")
        assert label == "NEUTRAL"
        assert score == 0.0

    def test_null_returns_neutral(self):
        score, label = _score_sentiment(None)
        assert label == "NEUTRAL"
        assert score == 0.0

    def test_score_is_rounded_to_4_decimal_places(self):
        score, _ = _score_sentiment("good news today")
        # Check that the score has at most 4 decimal places
        assert score == round(score, 4)


# ---------------------------------------------------------------------------
# Output schema tests
# ---------------------------------------------------------------------------

class TestOutputSchema:

    def test_output_contains_required_columns(self):
        """
        The warehouse loader expects specific columns from transform.
        Missing columns cause silent skips in load.py.
        """
        df_raw = make_articles("https://example.com/article-1")
        df_articles, _ = transform(df_raw)

        required_columns = [
            "url", "source_name", "seendate",
            "segment_name", "sentiment_score", "sentiment_label", "language"
        ]
        for col in required_columns:
            assert col in df_articles.columns, (
                f"Required column '{col}' missing from transform output"
            )

    def test_entity_records_have_required_keys(self):
        """
        Each entity record must have url, entity_name, and entity_type.
        Missing keys cause silent skips in load.py._insert_entities().
        """
        df_raw = make_articles("https://example.com/article-1")
        df_raw["persons"] = "Jane Doe,1"

        _, entity_records = transform(df_raw)

        assert len(entity_records) > 0
        for record in entity_records:
            assert "url" in record,         "Entity record missing 'url'"
            assert "entity_name" in record, "Entity record missing 'entity_name'"
            assert "entity_type" in record, "Entity record missing 'entity_type'"