"""Batch job model for tracking CSV batch processing."""

from datetime import datetime
from typing import Optional, Dict, Any, List

from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, Text, Integer, DateTime, ForeignKey, JSON, Index

from app.extensions import db


class BatchJob(db.Model):
    """Model for tracking batch processing jobs.

    Attributes:
        id: Primary key
        user_id: Foreign key to user who submitted job
        celery_task_id: Celery task ID for progress tracking
        filename: Original uploaded filename
        status: Job status (pending/processing/completed/failed)

        # Processing stats
        total_records: Total number of records to process
        processed_records: Number of records processed so far
        failed_records: Number of records that failed processing

        # Results aggregation
        positive_count: Count of positive sentiments
        negative_count: Count of negative sentiments
        neutral_count: Count of neutral sentiments
        emotion_counts: JSON dict of emotion counts

        # Metadata
        error_message: Error message if job failed
        result_file_path: Path to results CSV file
        created_at: Job submission timestamp
        started_at: Job start timestamp
        completed_at: Job completion timestamp
    """

    __tablename__ = "batch_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False, index=True
    )
    celery_task_id: Mapped[Optional[str]] = mapped_column(
        String(50), unique=True, nullable=True, index=True
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False, index=True)

    # Processing stats
    total_records: Mapped[int] = mapped_column(Integer, default=0)
    processed_records: Mapped[int] = mapped_column(Integer, default=0)
    failed_records: Mapped[int] = mapped_column(Integer, default=0)

    # Results aggregation
    positive_count: Mapped[int] = mapped_column(Integer, default=0)
    negative_count: Mapped[int] = mapped_column(Integer, default=0)
    neutral_count: Mapped[int] = mapped_column(Integer, default=0)
    emotion_counts: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # Metadata
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    result_file_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="batch_jobs")

    # Composite indexes
    __table_args__ = (Index("idx_user_status_date", "user_id", "status", "created_at"),)

    # Status constants
    STATUS_PENDING = "pending"
    STATUS_PROCESSING = "processing"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"

    def __repr__(self) -> str:
        """String representation of BatchJob."""
        return f"<BatchJob {self.id}: {self.status}>"

    def to_dict(self) -> dict:
        """Convert batch job to dictionary representation.

        Returns:
            dict: Batch job data dictionary
        """
        return {
            "id": self.id,
            "user_id": self.user_id,
            "celery_task_id": self.celery_task_id,
            "filename": self.filename,
            "status": self.status,
            "total_records": self.total_records,
            "processed_records": self.processed_records,
            "failed_records": self.failed_records,
            "progress_percent": self.progress_percent,
            "sentiment_distribution": {
                "positive": self.positive_count,
                "negative": self.negative_count,
                "neutral": self.neutral_count,
            },
            "emotion_counts": self.emotion_counts or {},
            "error_message": self.error_message,
            "result_file_path": self.result_file_path,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
        }

    @property
    def progress_percent(self) -> int:
        """Calculate progress percentage.

        Returns:
            int: Progress percentage (0-100)
        """
        if self.total_records == 0:
            return 0
        return min(100, int((self.processed_records / self.total_records) * 100))

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate job duration in seconds.

        Returns:
            float or None: Duration in seconds if completed
        """
        if not self.started_at:
            return None

        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds()

    @property
    def is_pending(self) -> bool:
        """Check if job is pending."""
        return self.status == self.STATUS_PENDING

    @property
    def is_processing(self) -> bool:
        """Check if job is currently processing."""
        return self.status == self.STATUS_PROCESSING

    @property
    def is_completed(self) -> bool:
        """Check if job is completed."""
        return self.status == self.STATUS_COMPLETED

    @property
    def is_failed(self) -> bool:
        """Check if job has failed."""
        return self.status == self.STATUS_FAILED

    def start_processing(self, celery_task_id: str) -> None:
        """Mark job as started processing.

        Args:
            celery_task_id: Celery task ID
        """
        self.status = self.STATUS_PROCESSING
        self.celery_task_id = celery_task_id
        self.started_at = datetime.utcnow()

    def update_progress(
        self,
        processed: int,
        positive: int = 0,
        negative: int = 0,
        neutral: int = 0,
        emotion_counts: Optional[Dict[str, int]] = None,
        failed: int = 0,
    ) -> None:
        """Update job progress.

        Args:
            processed: Total records processed so far
            positive: Count of positive sentiments
            negative: Count of negative sentiments
            neutral: Count of neutral sentiments
            emotion_counts: Dictionary of emotion counts
            failed: Count of failed records
        """
        self.processed_records = processed
        self.positive_count = positive
        self.negative_count = negative
        self.neutral_count = neutral
        self.failed_records = failed
        if emotion_counts:
            self.emotion_counts = emotion_counts

    def mark_completed(self, result_file_path: Optional[str] = None) -> None:
        """Mark job as completed.

        Args:
            result_file_path: Path to results file
        """
        self.status = self.STATUS_COMPLETED
        self.completed_at = datetime.utcnow()
        self.result_file_path = result_file_path

    def mark_failed(self, error_message: str) -> None:
        """Mark job as failed.

        Args:
            error_message: Error message describing failure
        """
        self.status = self.STATUS_FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error_message

    @staticmethod
    def get_user_jobs(user_id: int, status: Optional[str] = None, limit: int = 20) -> List[Any]:
        """Get batch jobs for a user.

        Args:
            user_id: User ID
            status: Optional status filter
            limit: Maximum number of jobs to return

        Returns:
            list: List of BatchJob instances
        """
        query = BatchJob.query.filter_by(user_id=user_id)
        if status:
            query = query.filter_by(status=status)
        return query.order_by(BatchJob.created_at.desc()).limit(limit).all()

    @staticmethod
    def get_active_jobs_count(user_id: int) -> int:
        """Get count of active (pending/processing) jobs for a user.

        Args:
            user_id: User ID

        Returns:
            int: Count of active jobs
        """
        return BatchJob.query.filter(
            BatchJob.user_id == user_id,
            BatchJob.status.in_([BatchJob.STATUS_PENDING, BatchJob.STATUS_PROCESSING]),
        ).count()

    @staticmethod
    def cleanup_old_jobs(days: int = 30) -> int:
        """Delete completed jobs older than specified days.

        Args:
            days: Number of days to keep jobs

        Returns:
            int: Number of deleted jobs
        """
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(days=days)

        old_jobs = BatchJob.query.filter(
            BatchJob.status == BatchJob.STATUS_COMPLETED, BatchJob.completed_at < cutoff
        ).all()

        count = len(old_jobs)
        for job in old_jobs:
            db.session.delete(job)

        db.session.commit()
        return count
