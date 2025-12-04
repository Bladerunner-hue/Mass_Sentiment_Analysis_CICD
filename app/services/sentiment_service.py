"""Sentiment analysis service.

This module provides sentiment and emotion analysis using VADER for quick
sentiment scoring and Hugging Face transformers for emotion detection.
Custom BiLSTM + Attention model support is also available.
"""

import hashlib
import json
import os
import re
import time
import threading
import gc
from typing import Dict, List, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class SentimentService:
    """Service for performing sentiment and emotion analysis.

    This service provides:
    - Quick VADER-based sentiment analysis (~0.3ms per text)
    - Custom BiLSTM + Attention model for enhanced accuracy (primary/default)
    - Transformer-based emotion detection using distilroberta (~40ms per text) as fallback
    - Combined full analysis with both sentiment and emotions
    - Efficient batch processing with GPU acceleration

    The custom BiLSTM + Attention model is used as primary when CUSTOM_MODEL_PATH
    and CUSTOM_TOKENIZER_PATH are configured. Falls back to transformer model if
    custom model is unavailable or fails.

    Attributes:
        vader: VADER sentiment analyzer instance
        emotion_pipeline: Hugging Face pipeline for emotion detection (lazy loaded, fallback)
        custom_service: Custom BiLSTM + Attention service (lazy loaded, primary)
        model_name: Name of the fallback Hugging Face model to use
    """

    # Emotion labels from j-hartmann/emotion-english-distilroberta-base
    EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

    # VADER threshold for sentiment classification
    POSITIVE_THRESHOLD = 0.05
    NEGATIVE_THRESHOLD = -0.05

    # Precompiled regex patterns for faster preprocessing
    URL_PATTERN = re.compile(r"http\S+|www\.\S+")
    HTML_PATTERN = re.compile(r"<[^>]+>")
    WHITESPACE_PATTERN = re.compile(r"\s+")

    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        max_workers: int = 4,
        memory_threshold: float = 0.8,
        cache_enabled: bool = True,
        cache_ttl: int = 86400,
        redis_url: Optional[str] = None,
    ):
        """Initialize the sentiment service.

        Args:
            model_name: Hugging Face model name for emotion detection.
                       Defaults to j-hartmann/emotion-english-distilroberta-base
            cache_dir: Directory to cache downloaded models
            max_workers: Maximum worker threads for parallel processing
            memory_threshold: Memory usage threshold for cleanup (0-1)
        """
        self.vader = SentimentIntensityAnalyzer()
        self.model_name = model_name or "j-hartmann/emotion-english-distilroberta-base"
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), ".model_cache")

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Lazy loaded components
        self._emotion_pipeline: Any = None
        self._pipeline_lock = threading.Lock()
        self._device: Optional[int] = None
        self._custom_service: Any = None
        self._custom_lock = threading.Lock()

        # Check if custom model is available
        self._custom_model_available = self._check_custom_model_available()

        # Performance optimizations
        self.max_workers = max_workers
        self.memory_threshold = memory_threshold
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Caching configuration
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self.redis_client = None
        self._init_cache(redis_url)

    def _get_device(self) -> int:
        """Get the device for model inference.

        Returns:
            int: Device ID (0 for GPU, -1 for CPU)
        """
        if self._device is not None:
            return self._device
        try:
            import torch

            self._device = 0 if torch.cuda.is_available() else -1
        except ImportError:
            self._device = -1
        return self._device

    def _check_custom_model_available(self) -> bool:
        """Check if custom BiLSTM + Attention model should be used (environment variables set)."""
        model_path = os.getenv("CUSTOM_MODEL_PATH")
        tokenizer_path = os.getenv("CUSTOM_TOKENIZER_PATH")
        # Custom model is considered available if environment variables are set
        # (will attempt to load and fall back to transformer if loading fails)
        return bool(model_path and tokenizer_path)

    def _load_custom_service(self):
        """Load the custom sentiment service (thread-safe, lazy loading)."""
        if self._custom_service is not None:
            return

        with self._custom_lock:
            # Double-check after acquiring lock
            if self._custom_service is not None:
                return

            try:
                from app.services.custom_sentiment_service import CustomSentimentService

                self._custom_service = CustomSentimentService()
            except Exception:
                # If custom service fails to load, disable it
                self._custom_model_available = False
                self._custom_service = None

    @property
    def custom_service(self):
        """Get the custom sentiment service (lazy loaded).

        Returns:
            CustomSentimentService: Custom BiLSTM + Attention service or None
        """
        if self._custom_service is None and self._custom_model_available:
            self._load_custom_service()
        return self._custom_service

    def _load_emotion_model(self):
        """Load the emotion detection model (thread-safe, lazy loading with caching)."""
        if self._emotion_pipeline is not None:
            return

        with self._pipeline_lock:
            # Double-check after acquiring lock
            if self._emotion_pipeline is not None:
                return

            from transformers import pipeline

            torch_dtype = None
            model_kwargs = {}

            try:
                import torch

                # Prefer reduced precision on GPU for throughput
                if self._get_device() == 0:
                    torch_dtype = torch.float16

                    # Enable 8-bit loading when bitsandbytes is available
                    if os.getenv("ENABLE_INT8", "1") != "0":
                        try:
                            import bitsandbytes  # noqa: F401

                            model_kwargs["load_in_8bit"] = True
                        except Exception:
                            # Fall back silently if bitsandbytes is not present
                            pass
                else:
                    torch_dtype = torch.float32
            except ImportError:
                torch_dtype = None

            # Set cache directory for model downloads
            os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
            os.environ["HF_HOME"] = self.cache_dir

            self._emotion_pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                device=self._get_device(),
                return_all_scores=True,
                truncation=True,
                max_length=512,
                torch_dtype=torch_dtype,
                model_kwargs=model_kwargs,
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

    def _calculate_optimal_batch_size(self, num_texts: int, requested_batch_size: int) -> int:
        """Adjust batch size based on device availability and free memory."""
        requested = max(1, requested_batch_size)
        target = min(requested, max(1, num_texts))

        try:
            import torch

            if self._get_device() != 0 or not torch.cuda.is_available():
                return target

            properties = torch.cuda.get_device_properties(0)
            total_mem = getattr(properties, "total_memory", 0)
            used_mem = torch.cuda.memory_allocated(0)
            free_mem = max(0, total_mem - used_mem)

            if free_mem > 8e9:
                return min(64, target, num_texts)
            if free_mem > 4e9:
                return min(32, target, num_texts)
            return min(16, target, num_texts)
        except Exception:
            # If torch is unavailable or any check fails, use the requested value
            return target

    def _init_cache(self, redis_url: Optional[str]) -> None:
        """Initialize Redis client if caching is enabled."""
        if not self.cache_enabled:
            return

        try:
            from redis import Redis  # Local import to avoid hard dependency at import time

            url = redis_url or os.getenv("REDIS_URL")
            if url:
                self.redis_client = Redis.from_url(url, decode_responses=True)
            else:
                self.redis_client = Redis(
                    host=os.getenv("REDIS_HOST", "localhost"),
                    port=int(os.getenv("REDIS_PORT", 6379)),
                    db=int(os.getenv("REDIS_DB", 0)),
                    decode_responses=True,
                )
        except Exception:
            self.cache_enabled = False
            self.redis_client = None

    def _get_cache_key(self, text: str, analysis_type: str) -> str:
        """Generate a stable cache key for a text/analysis combination."""
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        return f"sentiment:{analysis_type}:{text_hash}"

    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Fetch a cached result if available."""
        if not self.cache_enabled or not self.redis_client:
            return None

        try:
            cached = self.redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception:
            self.cache_enabled = False
        return None

    def _set_cached_result(self, cache_key: str, value: Dict[str, Any]) -> None:
        """Store a result in cache."""
        if not self.cache_enabled or not self.redis_client:
            return

        try:
            self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(value))
        except Exception:
            self.cache_enabled = False

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
        text = self.URL_PATTERN.sub("", text)

        # Remove HTML tags
        text = self.HTML_PATTERN.sub("", text)

        # Normalize whitespace
        text = self.WHITESPACE_PATTERN.sub(" ", text)

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
            return "positive"
        elif compound <= self.NEGATIVE_THRESHOLD:
            return "negative"
        else:
            return "neutral"

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
                "sentiment": "neutral",
                "confidence": 0.0,
                "compound_score": 0.0,
                "scores": {"pos": 0.0, "neg": 0.0, "neu": 1.0, "compound": 0.0},
                "processing_time_ms": 0,
            }

        scores = self.vader.polarity_scores(cleaned_text)

        processing_time = int((time.time() - start_time) * 1000)

        return {
            "sentiment": self._compound_to_label(scores["compound"]),
            "confidence": abs(scores["compound"]),
            "compound_score": scores["compound"],
            "scores": scores,
            "processing_time_ms": processing_time,
        }

    def analyze_emotions(self, text: str) -> Dict[str, Any]:
        """Perform emotion detection using custom model (primary) or transformer fallback.

        Uses custom BiLSTM + Attention model as primary when configured, otherwise falls back to
        j-hartmann/emotion-english-distilroberta-base to detect
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
        cache_key = None

        if self.cache_enabled and cleaned_text:
            cache_key = self._get_cache_key(cleaned_text, "emotions")
            cached_result = self._get_cached_result(cache_key)
            if cached_result is not None:
                return cached_result

        if not cleaned_text:
            default_scores = {e: 0.0 for e in self.EMOTIONS}
            default_scores["neutral"] = 1.0
            return {
                "primary_emotion": "neutral",
                "confidence": 1.0,
                "emotion_scores": default_scores,
                "processing_time_ms": 0,
            }

        # Use custom model as primary (always try first)
        if self._custom_model_available:
            try:
                if self.custom_service:
                    result = self.custom_service.analyze(cleaned_text)
                    processing_time = int((time.time() - start_time) * 1000)

                    # Format result to match expected interface
                    custom_result = {
                        "primary_emotion": result.get("primary_emotion", "neutral"),
                        "confidence": result.get("confidence", 0.0),
                        "emotion_scores": result.get("emotion_scores", {}),
                        "processing_time_ms": processing_time,
                    }

                    if cache_key:
                        self._set_cached_result(cache_key, custom_result)

                    return custom_result
            except Exception:
                # Custom model failed, fall back to transformer
                pass

        # Fallback to transformer model
        results = self.emotion_pipeline(cleaned_text)[0]

        # Convert to dict and find primary emotion
        emotion_scores = {r["label"]: r["score"] for r in results}
        primary = max(results, key=lambda x: x["score"])

        processing_time = int((time.time() - start_time) * 1000)

        result = {
            "primary_emotion": primary["label"],
            "confidence": primary["score"],
            "emotion_scores": emotion_scores,
            "processing_time_ms": processing_time,
        }

        if cache_key:
            self._set_cached_result(cache_key, result)

        return result

    def analyze_full(self, text: str) -> Dict[str, Any]:
        """Perform full analysis with both sentiment and emotions.

        Uses custom BiLSTM + Attention model as primary for comprehensive analysis,
        otherwise falls back to combining VADER sentiment analysis with transformer-based emotion detection.

        Args:
            text: Text to analyze

        Returns:
            dict: Combined analysis result with all metrics
        """
        start_time = time.time()

        cleaned_text = self._preprocess_text(text)
        cache_key = None

        if self.cache_enabled and cleaned_text:
            cache_key = self._get_cache_key(cleaned_text, "full")
            cached_result = self._get_cached_result(cache_key)
            if cached_result is not None:
                return cached_result

        # Use custom model as primary (provides both sentiment and emotions)
        if self._custom_model_available:
            try:
                if self.custom_service:
                    custom_result = self.custom_service.analyze(cleaned_text)
                    total_time = int((time.time() - start_time) * 1000)

                    result = {
                        "text": (
                            cleaned_text[:200] + "..." if len(cleaned_text) > 200 else cleaned_text
                        ),
                        "sentiment": custom_result.get("sentiment", "neutral"),
                        "compound_score": custom_result.get("compound_score", 0.0),
                        "scores": custom_result.get("scores", {}),
                        "primary_emotion": custom_result.get("primary_emotion", "neutral"),
                        "confidence": custom_result.get("confidence", 0.0),
                        "emotion_scores": custom_result.get("emotion_scores", {}),
                        "processing_time_ms": total_time,
                        "model_used": "custom_bilstm_attention",
                    }

                    if cache_key:
                        self._set_cached_result(cache_key, result)

                    return result
            except Exception:
                # Custom model failed, fall back to VADER + transformer
                pass

        # Fallback to VADER + transformer
        quick_result = self.analyze_quick(cleaned_text)

        # Get transformer emotions
        emotion_result = self.analyze_emotions(cleaned_text)

        total_time = int((time.time() - start_time) * 1000)

        result = {
            "text": cleaned_text[:200] + "..." if len(cleaned_text) > 200 else cleaned_text,
            "sentiment": quick_result["sentiment"],
            "compound_score": quick_result["compound_score"],
            "scores": quick_result["scores"],
            "primary_emotion": emotion_result["primary_emotion"],
            "confidence": emotion_result["confidence"],
            "emotion_scores": emotion_result["emotion_scores"],
            "processing_time_ms": total_time,
            "model_used": "vader_transformer",
        }

        if cache_key:
            self._set_cached_result(cache_key, result)

        return result

    def batch_analyze(
        self,
        texts: List[str],
        batch_size: int = 32,
        include_emotions: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
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
                vader_results.append(
                    {
                        "sentiment": self._compound_to_label(scores["compound"]),
                        "compound_score": scores["compound"],
                        "scores": scores,
                    }
                )
            else:
                vader_results.append(
                    {
                        "sentiment": "neutral",
                        "compound_score": 0.0,
                        "scores": {"pos": 0.0, "neg": 0.0, "neu": 1.0, "compound": 0.0},
                    }
                )

        # Emotion analysis in batches (if requested)
        emotion_results = []
        if include_emotions:
            # Filter out empty texts but track indices
            non_empty_indices = [i for i, t in enumerate(cleaned_texts) if t]
            non_empty_texts = [cleaned_texts[i] for i in non_empty_indices]
            effective_batch_size = self._calculate_optimal_batch_size(
                len(non_empty_texts), batch_size
            )

            # Process in batches
            batch_predictions = []
            for i in range(0, len(non_empty_texts), effective_batch_size):
                batch = non_empty_texts[i : i + effective_batch_size]
                batch_results = self.emotion_pipeline(batch)
                batch_predictions.extend(batch_results)

                if progress_callback:
                    processed = min(i + effective_batch_size, len(non_empty_texts))
                    progress_callback(processed, total)

            # Map results back to original indices
            emotion_map = {}
            for idx, pred in zip(non_empty_indices, batch_predictions):
                emotion_scores = {r["label"]: r["score"] for r in pred}
                primary = max(pred, key=lambda x: x["score"])
                emotion_map[idx] = {
                    "primary_emotion": primary["label"],
                    "confidence": primary["score"],
                    "emotion_scores": emotion_scores,
                }

            # Fill in results for all texts
            default_emotion = {
                "primary_emotion": "neutral",
                "confidence": 1.0,
                "emotion_scores": {e: 0.0 for e in self.EMOTIONS},
            }
            default_emotion["emotion_scores"]["neutral"] = 1.0

            for i in range(total):
                emotion_results.append(emotion_map.get(i, default_emotion))
        else:
            # No emotion analysis requested
            emotion_results = [None] * total

        # Combine results
        for i in range(total):
            result = {
                "index": i,
                "sentiment": vader_results[i]["sentiment"],
                "compound_score": vader_results[i]["compound_score"],
                "scores": vader_results[i]["scores"],
            }

            if emotion_results[i]:
                result["primary_emotion"] = emotion_results[i]["primary_emotion"]
                result["confidence"] = emotion_results[i]["confidence"]
                result["emotion_scores"] = emotion_results[i]["emotion_scores"]

            results.append(result)

        if progress_callback:
            progress_callback(total, total)

        return results

    def _check_memory_usage(self) -> bool:
        """Check if memory usage is above threshold.

        Returns:
            bool: True if memory usage is high
        """
        if not HAS_PSUTIL:
            return False
        memory = psutil.virtual_memory()
        return memory.percent / 100 > self.memory_threshold

    def _cleanup_memory(self):
        """Force garbage collection and memory cleanup."""
        if hasattr(self, "_emotion_pipeline") and self._emotion_pipeline:
            # Clear any cached computations
            if hasattr(self._emotion_pipeline.model, "cache"):
                self._emotion_pipeline.model.cache.clear()

        gc.collect()

    def analyze_batch_parallel(
        self, texts: List[str], include_emotions: bool = True, batch_size: int = 16
    ) -> List[Dict[str, Any]]:
        """Analyze texts using parallel processing for better performance.

        Args:
            texts: List of texts to analyze
            include_emotions: Whether to include emotion analysis
            batch_size: Size of batches for processing

        Returns:
            List of analysis results
        """
        if not texts:
            return []

        # Split texts into chunks for parallel processing
        chunk_size = max(1, len(texts) // self.max_workers)
        text_chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]

        results = [None] * len(texts)
        futures = []

        # Submit parallel tasks
        for i, chunk in enumerate(text_chunks):
            start_idx = i * chunk_size
            future = self.executor.submit(
                self._process_chunk, chunk, start_idx, include_emotions, batch_size
            )
            futures.append((future, start_idx))

        # Collect results
        for future, start_idx in futures:
            try:
                chunk_results = future.result()
                for i, result in enumerate(chunk_results):
                    results[start_idx + i] = result
            except Exception as e:
                # Handle errors gracefully
                chunk_len = len(text_chunks[futures.index((future, start_idx))])
                for i in range(chunk_len):
                    results[start_idx + i] = {
                        "error": str(e),
                        "sentiment": "neutral",
                        "compound_score": 0.0,
                    }

        # Memory cleanup
        if self._check_memory_usage():
            self._cleanup_memory()

        return results

    def _process_chunk(
        self, texts: List[str], start_idx: int, include_emotions: bool, batch_size: int
    ) -> List[Dict[str, Any]]:
        """Process a chunk of texts.

        Args:
            texts: Texts to process
            start_idx: Starting index in original list
            include_emotions: Whether to include emotions
            batch_size: Batch size for transformer

        Returns:
            List of results for this chunk
        """
        return self.batch_analyze(texts, batch_size=batch_size, include_emotions=include_emotions)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models.

        Returns:
            dict: Model information including device and status
        """
        return {
            "vader": "loaded",
            "emotion_model": self.model_name,
            "emotion_model_loaded": self._emotion_pipeline is not None,
            "device": "cuda" if self._get_device() == 0 else "cpu",
        }
