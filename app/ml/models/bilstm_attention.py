"""BiLSTM + Attention classifier for sentiment and emotion."""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Simple additive attention that returns context vector and weights."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(
        self, lstm_output: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # lstm_output: (batch, seq_len, hidden_dim)
        scores = self.attention(lstm_output).squeeze(-1)  # (batch, seq_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores, dim=-1)  # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)  # (batch, hidden_dim)
        return context, weights


class BiLSTMAttentionClassifier(nn.Module):
    """Bidirectional LSTM with attention for joint sentiment/emotion classification."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_sentiment_classes: int = 3,
        num_emotion_classes: int = 7,
        dropout: float = 0.5,
        bidirectional: bool = True,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        if pretrained_embeddings is not None:
            with torch.no_grad():
                self.embedding.weight[: pretrained_embeddings.size(0)].copy_(pretrained_embeddings)
        self.embedding.weight.requires_grad = not freeze_embeddings

        self.embedding_dropout = nn.Dropout(dropout * 0.5)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = AttentionLayer(lstm_output_dim)

        self.fc1 = nn.Linear(lstm_output_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.LayerNorm(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)

        self.sentiment_head = nn.Linear(hidden_dim // 2, num_sentiment_classes)
        self.emotion_head = nn.Linear(hidden_dim // 2, num_emotion_classes)

        self.activation = nn.ReLU()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        embedded = self.embedding(input_ids)
        embedded = self.embedding_dropout(embedded)

        lstm_out, _ = self.lstm(embedded)
        context, attn_weights = self.attention(lstm_out, attention_mask)

        x = self.fc1(context)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout2(x)

        sentiment_logits = self.sentiment_head(x)
        emotion_logits = self.emotion_head(x)

        if return_attention:
            return sentiment_logits, emotion_logits, attn_weights
        return sentiment_logits, emotion_logits, None
