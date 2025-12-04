"""Sentiment analysis model for storing analysis results."""

from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, Text, Float, Integer, DateTime, ForeignKey, JSON, Index

from app.extensions import db


class SentimentAnalysis(db.Model):
    """Model for storing sentiment analysis results.

    Attributes:
        id: Primary key
        user_id: Foreign key to user who performed analysis
        input_text: Original text that was analyzed
        source: Source of analysis ('api', 'web', 'batch')

        # VADER sentiment scores
        sentiment: Overall sentiment label (positive/negative/neutral)
        compound_score: VADER compound score (-1 to 1)
        positive_score: VADER positive component
        negative_score: VADER negative component
        neutral_score: VADER neutral component

        # Transformer emotion scores
        primary_emotion: Detected primary emotion
        emotion_scores: JSON dict of all emotion scores
        confidence: Confidence score of primary emotion

        # Metadata
        processing_time_ms: Time taken to process in milliseconds
        text_length: Length of input text
        word_count: Number of words in input
        created_at: Timestamp of analysis
    """

    __tablename__ = "sentiment_analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    input_text: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="web")

    # VADER sentiment scores
    sentiment: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    compound_score: Mapped[float] = mapped_column(Float, nullable=False)
    positive_score: Mapped[float] = mapped_column(Float, default=0.0)
    negative_score: Mapped[float] = mapped_column(Float, default=0.0)
    neutral_score: Mapped[float] = mapped_column(Float, default=0.0)

    # Transformer emotion scores
    primary_emotion: Mapped[Optional[str]] = mapped_column(String(20), nullable=True, index=True)
    emotion_scores: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Metadata
    processing_time_ms: Mapped[int] = mapped_column(Integer, default=0)
    text_length: Mapped[int] = mapped_column(Integer, default=0)
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    stream_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    user = relationship("User", back_populates="analyses")

    # Composite indexes for common queries
    __table_args__ = (
        Index("idx_user_sentiment_date", "user_id", "sentiment", "created_at"),
        Index("idx_user_emotion_date", "user_id", "primary_emotion", "created_at"),
        Index("idx_source_date", "source", "created_at"),
    )

    def __repr__(self) -> str:
        """String representation of SentimentAnalysis."""
        return f"<SentimentAnalysis {self.id}: {self.sentiment}>"

    def to_dict(self, include_text: bool = True) -> dict:
        """Convert analysis to dictionary representation.

        Args:
            include_text: Whether to include the full input text

        Returns:
            dict: Analysis data dictionary
        """
        data = {
            "id": self.id,
            "user_id": self.user_id,
            "source": self.source,
            "sentiment": self.sentiment,
            "compound_score": round(self.compound_score, 4),
            "scores": {
                "positive": round(self.positive_score, 4),
                "negative": round(self.negative_score, 4),
                "neutral": round(self.neutral_score, 4),
                "compound": round(self.compound_score, 4),
            },
            "primary_emotion": self.primary_emotion,
            "emotion_scores": self.emotion_scores,
            "confidence": round(self.confidence, 4) if self.confidence else None,
            "processing_time_ms": self.processing_time_ms,
            "text_length": self.text_length,
            "word_count": self.word_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if include_text:
            data["input_text"] = self.input_text
        return data

    @classmethod
    def create_from_result(
        cls, user_id: Optional[int], text: str, result: dict, source: str = "web"
    ) -> "SentimentAnalysis":
        """Create a SentimentAnalysis instance from service result.

        Args:
            user_id: ID of the user performing analysis
            text: Original input text
            result: Result dictionary from SentimentService
            source: Source of analysis

        Returns:
            SentimentAnalysis: New analysis instance (not committed)
        """
        analysis = cls(
            user_id=user_id,
            input_text=text,
            source=source,
            sentiment=result.get("sentiment", "neutral"),
            compound_score=result.get("compound_score", 0.0),
            positive_score=result.get("scores", {}).get("pos", 0.0),
            negative_score=result.get("scores", {}).get("neg", 0.0),
            neutral_score=result.get("scores", {}).get("neu", 0.0),
            primary_emotion=result.get("primary_emotion"),
            emotion_scores=result.get("emotion_scores"),
            confidence=result.get("confidence"),
            processing_time_ms=result.get("processing_time_ms", 0),
            text_length=len(text),
            word_count=len(text.split()),
        )
        return analysis

    @staticmethod
    def get_user_stats(user_id: int) -> dict:
        """Get aggregated statistics for a user's analyses.

        Args:
            user_id: User ID to get stats for

        Returns:
            dict: Statistics dictionary
        """
        from sqlalchemy import func

        base_query = SentimentAnalysis.query.filter_by(user_id=user_id)

        total = base_query.count()

        if total == 0:
            return {
                "total_analyses": 0,
                "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0},
                "emotion_distribution": {},
                "average_confidence": 0,
                "average_processing_time_ms": 0,
            }

        # Sentiment distribution
        sentiment_counts = (
            db.session.query(SentimentAnalysis.sentiment, func.count(SentimentAnalysis.id))
            .filter_by(user_id=user_id)
            .group_by(SentimentAnalysis.sentiment)
            .all()
        )

        sentiment_dist = {s: c for s, c in sentiment_counts}

        # Emotion distribution
        emotion_counts = (
            db.session.query(SentimentAnalysis.primary_emotion, func.count(SentimentAnalysis.id))
            .filter(
                SentimentAnalysis.user_id == user_id, SentimentAnalysis.primary_emotion.isnot(None)
            )
            .group_by(SentimentAnalysis.primary_emotion)
            .all()
        )

        emotion_dist = {e: c for e, c in emotion_counts if e}

        # Averages
        averages = (
            db.session.query(
                func.avg(SentimentAnalysis.confidence),
                func.avg(SentimentAnalysis.processing_time_ms),
            )
            .filter_by(user_id=user_id)
            .first()
        )

        return {
            "total_analyses": total,
            "sentiment_distribution": {
                "positive": sentiment_dist.get("positive", 0),
                "negative": sentiment_dist.get("negative", 0),
                "neutral": sentiment_dist.get("neutral", 0),
            },
            "emotion_distribution": emotion_dist,
            "average_confidence": round(averages[0] or 0, 4),
            "average_processing_time_ms": round(averages[1] or 0, 2),
        }
