"""Tests for sentiment analysis service."""

import pytest
from app.services.sentiment_service import SentimentService


class TestSentimentService:
    """Test cases for SentimentService."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.service = SentimentService()

    def test_analyze_quick_positive(self, sample_texts):
        """Test quick analysis identifies positive sentiment."""
        for text in sample_texts['positive']:
            result = self.service.analyze_quick(text)

            assert result['sentiment'] == 'positive'
            assert result['compound_score'] > 0.05
            assert 'scores' in result
            assert 'processing_time_ms' in result

    def test_analyze_quick_negative(self, sample_texts):
        """Test quick analysis identifies negative sentiment."""
        for text in sample_texts['negative']:
            result = self.service.analyze_quick(text)

            assert result['sentiment'] == 'negative'
            assert result['compound_score'] < -0.05
            assert 'scores' in result

    def test_analyze_quick_neutral(self, sample_texts):
        """Test quick analysis identifies neutral sentiment."""
        for text in sample_texts['neutral']:
            result = self.service.analyze_quick(text)

            assert result['sentiment'] == 'neutral'
            assert -0.05 <= result['compound_score'] <= 0.05

    def test_analyze_quick_empty_text(self):
        """Test quick analysis handles empty text."""
        result = self.service.analyze_quick('')

        assert result['sentiment'] == 'neutral'
        assert result['compound_score'] == 0.0
        assert result['confidence'] == 0.0

    def test_analyze_quick_whitespace_only(self):
        """Test quick analysis handles whitespace-only text."""
        result = self.service.analyze_quick('   \n\t   ')

        assert result['sentiment'] == 'neutral'
        assert result['compound_score'] == 0.0

    def test_analyze_emotions_returns_all_emotions(self):
        """Test emotion analysis returns all 7 emotions."""
        result = self.service.analyze_emotions("I am so happy today!")

        assert 'primary_emotion' in result
        assert 'confidence' in result
        assert 'emotion_scores' in result
        assert len(result['emotion_scores']) == 7

        expected_emotions = {'anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'}
        assert set(result['emotion_scores'].keys()) == expected_emotions

    def test_analyze_emotions_joy(self):
        """Test emotion analysis detects joy."""
        result = self.service.analyze_emotions("I am so incredibly happy and excited!")

        assert result['primary_emotion'] == 'joy'
        assert result['confidence'] > 0.5

    def test_analyze_emotions_sadness(self):
        """Test emotion analysis detects sadness."""
        result = self.service.analyze_emotions("I feel so sad and heartbroken.")

        assert result['primary_emotion'] == 'sadness'
        assert result['confidence'] > 0.3

    def test_analyze_emotions_anger(self):
        """Test emotion analysis detects anger."""
        result = self.service.analyze_emotions("I am furious and outraged about this!")

        assert result['primary_emotion'] == 'anger'
        assert result['confidence'] > 0.3

    def test_analyze_emotions_empty_text(self):
        """Test emotion analysis handles empty text."""
        result = self.service.analyze_emotions('')

        assert result['primary_emotion'] == 'neutral'
        assert result['confidence'] == 1.0

    def test_analyze_full_combines_both(self):
        """Test full analysis includes both sentiment and emotions."""
        result = self.service.analyze_full("I love this product so much!")

        # Sentiment fields
        assert 'sentiment' in result
        assert 'compound_score' in result
        assert 'scores' in result

        # Emotion fields
        assert 'primary_emotion' in result
        assert 'emotion_scores' in result
        assert 'confidence' in result

        # Metadata
        assert 'processing_time_ms' in result

    def test_batch_analyze_processes_all_texts(self, sample_texts):
        """Test batch analysis processes all texts."""
        texts = sample_texts['positive'] + sample_texts['negative'] + sample_texts['neutral']
        results = self.service.batch_analyze(texts, include_emotions=False)

        assert len(results) == len(texts)
        for i, result in enumerate(results):
            assert result['index'] == i
            assert 'sentiment' in result
            assert 'compound_score' in result

    def test_batch_analyze_with_emotions(self):
        """Test batch analysis with emotion detection."""
        texts = ["I am happy!", "I am sad.", "This is normal."]
        results = self.service.batch_analyze(texts, include_emotions=True)

        assert len(results) == 3
        for result in results:
            assert 'primary_emotion' in result
            assert 'confidence' in result

    def test_batch_analyze_empty_list(self):
        """Test batch analysis handles empty list."""
        results = self.service.batch_analyze([])
        assert results == []

    def test_batch_analyze_progress_callback(self):
        """Test batch analysis calls progress callback."""
        texts = ["Text 1", "Text 2", "Text 3"]
        progress_calls = []

        def callback(processed, total):
            progress_calls.append((processed, total))

        self.service.batch_analyze(texts, progress_callback=callback)

        assert len(progress_calls) > 0
        # Last call should have processed all texts
        assert progress_calls[-1][0] == len(texts)

    def test_preprocess_text_removes_urls(self):
        """Test text preprocessing removes URLs."""
        text = "Check this out https://example.com it's great!"
        cleaned = self.service._preprocess_text(text)

        assert 'https://' not in cleaned
        assert 'example.com' not in cleaned

    def test_preprocess_text_removes_html(self):
        """Test text preprocessing removes HTML tags."""
        text = "<p>Hello <strong>world</strong></p>"
        cleaned = self.service._preprocess_text(text)

        assert '<p>' not in cleaned
        assert '<strong>' not in cleaned
        assert 'Hello' in cleaned
        assert 'world' in cleaned

    def test_preprocess_text_normalizes_whitespace(self):
        """Test text preprocessing normalizes whitespace."""
        text = "Hello    world\n\n\ttest"
        cleaned = self.service._preprocess_text(text)

        assert '    ' not in cleaned
        assert '\n\n' not in cleaned

    def test_preprocess_text_truncates_long_text(self):
        """Test text preprocessing truncates very long text."""
        long_text = "word " * 2000  # ~10000 characters
        cleaned = self.service._preprocess_text(long_text)

        assert len(cleaned) <= 5000

    def test_compound_to_label_positive(self):
        """Test compound score to label conversion for positive."""
        assert self.service._compound_to_label(0.5) == 'positive'
        assert self.service._compound_to_label(0.06) == 'positive'

    def test_compound_to_label_negative(self):
        """Test compound score to label conversion for negative."""
        assert self.service._compound_to_label(-0.5) == 'negative'
        assert self.service._compound_to_label(-0.06) == 'negative'

    def test_compound_to_label_neutral(self):
        """Test compound score to label conversion for neutral."""
        assert self.service._compound_to_label(0.0) == 'neutral'
        assert self.service._compound_to_label(0.04) == 'neutral'
        assert self.service._compound_to_label(-0.04) == 'neutral'

    def test_get_model_info(self):
        """Test get_model_info returns expected structure."""
        info = self.service.get_model_info()

        assert 'vader' in info
        assert info['vader'] == 'loaded'
        assert 'emotion_model' in info
        assert 'device' in info
        assert info['device'] in ['cpu', 'cuda']

    def test_analyze_full_uses_cache(self, monkeypatch):
        """Ensure analyze_full returns cached results when available."""

        class DummyRedis:
            def __init__(self):
                self.store = {}

            def get(self, key):
                return self.store.get(key)

            def setex(self, key, ttl, value):
                self.store[key] = value

        service = SentimentService(cache_enabled=True)
        service.redis_client = DummyRedis()
        service.cache_enabled = True

        calls = {'quick': 0, 'emotion': 0}

        def fake_quick(text):
            calls['quick'] += 1
            return {
                'sentiment': 'positive',
                'compound_score': 0.9,
                'scores': {'pos': 0.9, 'neg': 0.0, 'neu': 0.1, 'compound': 0.9},
                'processing_time_ms': 1
            }

        def fake_emotions(text):
            calls['emotion'] += 1
            return {
                'primary_emotion': 'joy',
                'confidence': 0.8,
                'emotion_scores': {'joy': 0.8},
                'processing_time_ms': 2
            }

        monkeypatch.setattr(service, 'analyze_quick', fake_quick)
        monkeypatch.setattr(service, 'analyze_emotions', fake_emotions)

        text = "Caching works great!"

        first = service.analyze_full(text)
        second = service.analyze_full(text)

        assert first == second
        assert calls['quick'] == 1
        assert calls['emotion'] == 1
