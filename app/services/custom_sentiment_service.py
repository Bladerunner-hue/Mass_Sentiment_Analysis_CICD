"""Custom sentiment service backed by a BiLSTM + Attention model."""

import os
import re
import time
from typing import Any, Dict, List, Optional

from app.ml.inference.predictor import CustomModelPredictor
from app.services.sentiment_service import SentimentService


class CustomSentimentService:
    """Serve predictions from the custom PyTorch model with optional HF fallback."""

    URL_PATTERN = re.compile(r"http\S+|www\.\S+")
    HTML_PATTERN = re.compile(r"<[^>]+>")
    WHITESPACE_PATTERN = re.compile(r"\s+")

    def __init__(
        self,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        device: Optional[str] = None,
        fallback_to_transformer: bool = True,
    ):
        self.model_path = model_path or os.getenv("CUSTOM_MODEL_PATH", "models/checkpoints/bilstm_attention.pt")
        self.tokenizer_path = tokenizer_path or os.getenv("CUSTOM_TOKENIZER_PATH", "models/tokenizer.pkl")
        self.device = device or os.getenv("CUSTOM_MODEL_DEVICE")
        self.fallback_to_transformer = fallback_to_transformer

        self.predictor: Optional[CustomModelPredictor] = None
        self._load_predictor()

    def _load_predictor(self) -> None:
        try:
            self.predictor = CustomModelPredictor(
                model_path=self.model_path,
                tokenizer_path=self.tokenizer_path,
                device=self.device,
            )
        except Exception as exc:
            self.predictor = None
            if not self.fallback_to_transformer:
                raise exc

    def _preprocess(self, text: str) -> str:
        text = str(text or "")
        text = self.URL_PATTERN.sub("", text)
        text = self.HTML_PATTERN.sub(" ", text)
        text = self.WHITESPACE_PATTERN.sub(" ", text)
        return text.strip()

    def analyze(self, text: str) -> Dict[str, Any]:
        start = time.time()
        cleaned = self._preprocess(text)

        if not cleaned:
            return {
                "sentiment": "neutral",
                "primary_emotion": "neutral",
                "confidence": 0.0,
                "emotion_scores": {},
                "processing_time_ms": 0,
            }

        if self.predictor:
            prediction = self.predictor.predict([cleaned])[0]
            return {
                "text": cleaned,
                "sentiment": prediction["sentiment"],
                "compound_score": prediction.get("sentiment_confidence", 0.0),
                "scores": prediction.get("sentiment_scores", {}),
                "primary_emotion": prediction["primary_emotion"],
                "confidence": prediction.get("emotion_confidence", 0.0),
                "emotion_scores": prediction.get("emotion_scores", {}),
                "processing_time_ms": int((time.time() - start) * 1000),
            }

        # Fallback to existing transformer-based service
        transformer_service = SentimentService()
        return transformer_service.analyze_full(cleaned)

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not texts:
            return []
        cleaned = [self._preprocess(t) for t in texts]
        start = time.time()

        if self.predictor:
            predictions = self.predictor.predict(cleaned)
            results = []
            for pred in predictions:
                results.append(
                    {
                        "text": pred["text"],
                        "sentiment": pred["sentiment"],
                        "compound_score": pred.get("sentiment_confidence", 0.0),
                        "scores": pred.get("sentiment_scores", {}),
                        "primary_emotion": pred["primary_emotion"],
                        "confidence": pred.get("emotion_confidence", 0.0),
                        "emotion_scores": pred.get("emotion_scores", {}),
                    }
                )
            return results

        # Fallback path
        transformer_service = SentimentService()
        return transformer_service.batch_analyze(cleaned)
