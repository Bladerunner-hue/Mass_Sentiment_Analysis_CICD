"""FP16-capable predictor for the custom BiLSTM + Attention model."""

from typing import List, Dict, Any
import os

import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F

from app.ml.models.bilstm_attention import BiLSTMAttentionClassifier
from app.ml.preprocessing.tokenizer import CustomTokenizer


class CustomModelPredictor:
    """Load a trained model and run batched inference with optional FP16."""

    SENTIMENT_LABELS = ["positive", "negative", "neutral"]
    EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: str = None,
        max_length: int = 160,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_length = max_length

        self.tokenizer = CustomTokenizer.load(tokenizer_path)
        self.model = BiLSTMAttentionClassifier(
            vocab_size=self.tokenizer.vocab_size,
            dropout=0.4,
            hidden_dim=256,
            num_layers=2,
        )

        state = torch.load(model_path, map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        if self.device.type == "cuda":
            self.model.half()
        self.model.eval()

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not texts:
            return []

        encoded = self.tokenizer.batch_encode(
            texts,
            max_length=self.max_length,
            add_special_tokens=True,
            padding=True,
            truncation=True,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            with autocast(enabled=self.device.type == "cuda"):
                sentiment_logits, emotion_logits, _ = self.model(
                    input_ids, attention_mask=attention_mask
                )

            sentiment_probs = F.softmax(sentiment_logits.float(), dim=-1)
            emotion_probs = F.softmax(emotion_logits.float(), dim=-1)

        results = []
        for i, text in enumerate(texts):
            sent_prob, sent_idx = torch.max(sentiment_probs[i], dim=0)
            emo_prob, emo_idx = torch.max(emotion_probs[i], dim=0)

            sentiment_scores = {
                label: float(sentiment_probs[i, idx].item())
                for idx, label in enumerate(self.SENTIMENT_LABELS)
            }
            emotion_scores = {
                label: float(emotion_probs[i, idx].item())
                for idx, label in enumerate(self.EMOTION_LABELS)
            }

            results.append(
                {
                    "text": text,
                    "sentiment": self.SENTIMENT_LABELS[sent_idx],
                    "sentiment_confidence": float(sent_prob.item()),
                    "sentiment_scores": sentiment_scores,
                    "primary_emotion": self.EMOTION_LABELS[emo_idx],
                    "emotion_confidence": float(emo_prob.item()),
                    "emotion_scores": emotion_scores,
                }
            )

        return results
