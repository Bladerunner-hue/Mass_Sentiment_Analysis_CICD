"""Sentiment analysis service.

This module provides sentiment and emotion analysis using VADER for quick
sentiment scoring and Hugging Face transformers for emotion detection.
"""

import re
import time
import threading
from typing import Dict, List, Optional, Callable, Any

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentService:
    """Service for performing sentiment and emotion analysis.

    This service provides:
    - Quick VADER-based sentiment analysis (~0.3ms per text)
    - Transformer-based emotion detection using distilroberta (~40ms per text)
    - Combined full analysis with both sentiment and emotions
    - Efficient batch processing with GPU acceleration

    The transformer model is loaded lazily on first use to avoid startup overhead.

    Attributes:
        vader: VADER sentiment analyzer instance
        emotion_pipeline: Hugging Face pipeline for emotion detection (lazy loaded)
        model_name: Name of the Hugging Face model to use
    """

    # Emotion labels from j-hartmann/emotion-english-distilroberta-base
    EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

    # VADER threshold for sentiment classification
    POSITIVE_THRESHOLD = 0.05
    NEGATIVE_THRESHOLD = -0.05

    def __init__(self, model_name: str = None):
        """Initialize the sentiment service.

        Args:
            model_name: Hugging Face model name for emotion detection.
                       Defaults to j-hartmann/emotion-english-distilroberta-base
        """
        self.vader = SentimentIntensityAnalyzer()
        self.model_name = model_name or 'j-hartmann/emotion-english-distilroberta-base'

        # Lazy loaded components
        self._emotion_pipeline = None
        self._pipeline_lock = threading.Lock()
        self._device = None

    def _get_device(self) -> int:
        """Get the device for model inference.

        Returns:
            int: Device ID (0 for GPU, -1 for CPU)
        """
        if self._device is None:
            try:
                import torch
                self._device = 0 if torch.cuda.is_available() else -1
            except ImportError:
                self._device = -1
        return self._device

    def _load_emotion_model(self):
        """Load the emotion detection model (thread-safe, lazy loading)."""
        if self._emotion_pipeline is not None:
            return

        with self._pipeline_lock:
            # Double-check after acquiring lock
            if self._emotion_pipeline is not None:
                return

            from transformers import pipeline

            self._emotion_pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                device=self._get_device(),
                return_all_scores=True,
                truncation=True,
                max_length=512
            )

    @property
    def emotion_pipeline(self):
        """Get the emotion detection pipeline (lazy loaded).

        Returns:
            Pipeline: Hugging Face text classification pipeline
        """
        if self._emotion_pipeline is None:
            self._load_emotion_model()
        return self._emotion_pipeline

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis.

        Args:
            text: Raw input text

        Returns:
            str: Cleaned and normalized text
        """
        if not text:
            return ""

        # Convert to string if needed
        text = str(text)

        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Normalize whitespace
        text = ' '.join(text.split())

        # Limit length to prevent memory issues
        max_length = 5000
        if len(text) > max_length:
            text = text[:max_length]

        return text.strip()

    def _compound_to_label(self, compound: float) -> str:
        """Convert VADER compound score to sentiment label.

        Args:
            compound: VADER compound score (-1 to 1)

        Returns:
            str: Sentiment label (positive/negative/neutral)
        """
        if compound >= self.POSITIVE_THRESHOLD:
            return 'positive'
        elif compound <= self.NEGATIVE_THRESHOLD:
            return 'negative'
        else:
            return 'neutral'

    def analyze_quick(self, text: str) -> Dict[str, Any]:
        """Perform quick VADER sentiment analysis.

        This is extremely fast (~0.3ms per text) and suitable for
        real-time feedback or high-volume screening.

        Args:
            text: Text to analyze

        Returns:
            dict: Analysis result with keys:
                - sentiment: Label (positive/negative/neutral)
                - confidence: Absolute value of compound score
                - compound_score: Raw compound score (-1 to 1)
                - scores: Full VADER scores (pos, neg, neu, compound)
        """
        start_time = time.time()

        cleaned_text = self._preprocess_text(text)

        if not cleaned_text:
            return {
                'sentiment': 'neutral',
                'confidence': 0.0,
                'compound_score': 0.0,
                'scores': {'pos': 0.0, 'neg': 0.0, 'neu': 1.0, 'compound': 0.0},
                'processing_time_ms': 0
            }

        scores = self.vader.polarity_scores(cleaned_text)

        processing_time = int((time.time() - start_time) * 1000)

        return {
            'sentiment': self._compound_to_label(scores['compound']),
            'confidence': abs(scores['compound']),
            'compound_score': scores['compound'],
            'scores': scores,
            'processing_time_ms': processing_time
        }

    def analyze_emotions(self, text: str) -> Dict[str, Any]:
        """Perform transformer-based emotion detection.

        Uses j-hartmann/emotion-english-distilroberta-base to detect
        7 emotions: anger, disgust, fear, joy, neutral, sadness, surprise.

        Args:
            text: Text to analyze

        Returns:
            dict: Analysis result with keys:
                - primary_emotion: Detected primary emotion
                - confidence: Confidence score of primary emotion
                - emotion_scores: Dict of all emotion scores
                - processing_time_ms: Time taken in milliseconds
        """
        start_time = time.time()

        cleaned_text = self._preprocess_text(text)

        if not cleaned_text:
            default_scores = {e: 0.0 for e in self.EMOTIONS}
            default_scores['neutral'] = 1.0
            return {
                'primary_emotion': 'neutral',
                'confidence': 1.0,
                'emotion_scores': default_scores,
                'processing_time_ms': 0
            }

        # Get predictions from transformer model
        results = self.emotion_pipeline(cleaned_text)[0]

        # Convert to dict and find primary emotion
        emotion_scores = {r['label']: r['score'] for r in results}
        primary = max(results, key=lambda x: x['score'])

        processing_time = int((time.time() - start_time) * 1000)

        return {
            'primary_emotion': primary['label'],
            'confidence': primary['score'],
            'emotion_scores': emotion_scores,
            'processing_time_ms': processing_time
        }

    def analyze_full(self, text: str) -> Dict[str, Any]:
        """Perform full analysis with both sentiment and emotions.

        Combines VADER sentiment analysis with transformer-based
        emotion detection for comprehensive text analysis.

        Args:
            text: Text to analyze

        Returns:
            dict: Combined analysis result with all metrics
        """
        start_time = time.time()

        cleaned_text = self._preprocess_text(text)

        # Get VADER sentiment
        quick_result = self.analyze_quick(cleaned_text)

        # Get transformer emotions
        emotion_result = self.analyze_emotions(cleaned_text)

        total_time = int((time.time() - start_time) * 1000)

        return {
            'text': cleaned_text[:200] + '...' if len(cleaned_text) > 200 else cleaned_text,
            'sentiment': quick_result['sentiment'],
            'compound_score': quick_result['compound_score'],
            'scores': quick_result['scores'],
            'primary_emotion': emotion_result['primary_emotion'],
            'confidence': emotion_result['confidence'],
            'emotion_scores': emotion_result['emotion_scores'],
            'processing_time_ms': total_time
        }

    def batch_analyze(
        self,
        texts: List[str],
        batch_size: int = 32,
        include_emotions: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict[str, Any]]:
        """Efficiently analyze multiple texts in batches.

        Uses batch processing for transformer model to maximize throughput.
        GPU acceleration is used automatically when available.

        Args:
            texts: List of texts to analyze
            batch_size: Number of texts per batch for transformer
            include_emotions: Whether to include emotion detection
            progress_callback: Optional callback(processed, total) for progress updates

        Returns:
            list: List of analysis results for each text
        """
        if not texts:
            return []

        results = []
        total = len(texts)

        # Preprocess all texts
        cleaned_texts = [self._preprocess_text(t) for t in texts]

        # VADER analysis (always fast, do individually)
        vader_results = []
        for i, text in enumerate(cleaned_texts):
            if text:
                scores = self.vader.polarity_scores(text)
                vader_results.append({
                    'sentiment': self._compound_to_label(scores['compound']),
                    'compound_score': scores['compound'],
                    'scores': scores
                })
            else:
                vader_results.append({
                    'sentiment': 'neutral',
                    'compound_score': 0.0,
                    'scores': {'pos': 0.0, 'neg': 0.0, 'neu': 1.0, 'compound': 0.0}
                })

        # Emotion analysis in batches (if requested)
        emotion_results = []
        if include_emotions:
            # Filter out empty texts but track indices
            non_empty_indices = [i for i, t in enumerate(cleaned_texts) if t]
            non_empty_texts = [cleaned_texts[i] for i in non_empty_indices]

            # Process in batches
            batch_predictions = []
            for i in range(0, len(non_empty_texts), batch_size):
                batch = non_empty_texts[i:i + batch_size]
                batch_results = self.emotion_pipeline(batch)
                batch_predictions.extend(batch_results)

                if progress_callback:
                    processed = min(i + batch_size, len(non_empty_texts))
                    progress_callback(processed, total)

            # Map results back to original indices
            emotion_map = {}
            for idx, pred in zip(non_empty_indices, batch_predictions):
                emotion_scores = {r['label']: r['score'] for r in pred}
                primary = max(pred, key=lambda x: x['score'])
                emotion_map[idx] = {
                    'primary_emotion': primary['label'],
                    'confidence': primary['score'],
                    'emotion_scores': emotion_scores
                }

            # Fill in results for all texts
            default_emotion = {
                'primary_emotion': 'neutral',
                'confidence': 1.0,
                'emotion_scores': {e: 0.0 for e in self.EMOTIONS}
            }
            default_emotion['emotion_scores']['neutral'] = 1.0

            for i in range(total):
                emotion_results.append(emotion_map.get(i, default_emotion))
        else:
            # No emotion analysis requested
            emotion_results = [None] * total

        # Combine results
        for i in range(total):
            result = {
                'index': i,
                'sentiment': vader_results[i]['sentiment'],
                'compound_score': vader_results[i]['compound_score'],
                'scores': vader_results[i]['scores']
            }

            if emotion_results[i]:
                result['primary_emotion'] = emotion_results[i]['primary_emotion']
                result['confidence'] = emotion_results[i]['confidence']
                result['emotion_scores'] = emotion_results[i]['emotion_scores']

            results.append(result)

        if progress_callback:
            progress_callback(total, total)

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models.

        Returns:
            dict: Model information including device and status
        """
        return {
            'vader': 'loaded',
            'emotion_model': self.model_name,
            'emotion_model_loaded': self._emotion_pipeline is not None,
            'device': 'cuda' if self._get_device() == 0 else 'cpu'
        }
