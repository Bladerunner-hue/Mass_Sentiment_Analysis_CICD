"""Celery tasks for batch processing."""

import os
from datetime import datetime
from celery import shared_task, current_task

import pandas as pd

from app.extensions import db
from app.models.batch_job import BatchJob
from app.services.sentiment_service import SentimentService
from app.services.batch_service import BatchService


@shared_task(bind=True, max_retries=3)
def process_batch_file(self, batch_job_id: int, file_path: str, include_emotions: bool = True):
    """Process a batch CSV file for sentiment analysis.

    This task runs in the background and updates progress via Celery
    task state, which can be streamed to clients via SSE.

    Args:
        batch_job_id: ID of the BatchJob model instance
        file_path: Path to the uploaded CSV file
        include_emotions: Whether to include emotion detection
    """
    # Get the batch job from database
    batch_job = BatchJob.query.get(batch_job_id)

    if not batch_job:
        return {"error": "Batch job not found"}

    try:
        # Initialize services
        sentiment_service = SentimentService()
        batch_service = BatchService(sentiment_service=sentiment_service)

        # Read the CSV file
        df = pd.read_csv(file_path)

        # Find text column
        text_column = None
        for col in batch_service.ALLOWED_TEXT_COLUMNS:
            if col in df.columns:
                text_column = col
                break

        if not text_column:
            raise ValueError("No valid text column found in CSV")

        total_records = len(df)
        batch_job.total_records = total_records
        batch_job.start_processing(self.request.id)
        db.session.commit()

        # Initialize statistics
        stats = {"positive": 0, "negative": 0, "neutral": 0, "emotions": {}}

        def progress_callback(processed, total, current_stats):
            """Update progress in Celery task state and database."""
            progress_percent = int((processed / total) * 100)

            # Update Celery task state for SSE streaming
            self.update_state(
                state="PROGRESS",
                meta={
                    "progress": progress_percent,
                    "processed": processed,
                    "total": total,
                    "stats": current_stats,
                },
            )

            # Update database periodically (every 5%)
            if processed % max(1, total // 20) == 0 or processed == total:
                batch_job.update_progress(
                    processed=processed,
                    positive=current_stats.get("positive", 0),
                    negative=current_stats.get("negative", 0),
                    neutral=current_stats.get("neutral", 0),
                    emotion_counts=current_stats.get("emotions", {}),
                )
                db.session.commit()

        # Process the dataframe
        result_df = batch_service.process_dataframe(
            df,
            text_column,
            include_emotions=include_emotions,
            batch_size=32,
            progress_callback=progress_callback,
        )

        # Generate output file path
        base, ext = os.path.splitext(file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{base}_results_{timestamp}{ext}"

        # Save results
        result_df.to_csv(output_path, index=False)

        # Calculate final statistics
        final_stats = batch_service.generate_summary(result_df)

        # Mark job as completed
        batch_job.update_progress(
            processed=total_records,
            positive=int((result_df["Sentiment"] == "positive").sum()),
            negative=int((result_df["Sentiment"] == "negative").sum()),
            neutral=int((result_df["Sentiment"] == "neutral").sum()),
            emotion_counts=final_stats.get("emotion_distribution", {}),
        )
        batch_job.mark_completed(output_path)
        db.session.commit()

        return {
            "success": True,
            "batch_job_id": batch_job_id,
            "output_path": output_path,
            "total_processed": total_records,
            "stats": final_stats,
        }

    except Exception as e:
        # Mark job as failed
        batch_job.mark_failed(str(e))
        db.session.commit()

        # Re-raise for Celery retry mechanism
        raise self.retry(exc=e, countdown=60)


@shared_task
def cleanup_old_batch_files(days: int = 7):
    """Cleanup old batch result files.

    Args:
        days: Delete files older than this many days
    """
    from datetime import timedelta

    cutoff = datetime.utcnow() - timedelta(days=days)

    # Find completed jobs older than cutoff
    old_jobs = BatchJob.query.filter(
        BatchJob.status == BatchJob.STATUS_COMPLETED,
        BatchJob.completed_at < cutoff,
        BatchJob.result_file_path.isnot(None),
    ).all()

    deleted_files = 0
    for job in old_jobs:
        if job.result_file_path and os.path.exists(job.result_file_path):
            try:
                os.remove(job.result_file_path)
                deleted_files += 1
            except OSError:
                pass

    return {"jobs_checked": len(old_jobs), "files_deleted": deleted_files}


@shared_task
def cleanup_old_batch_jobs(days: int = 30):
    """Cleanup old batch job records.

    Args:
        days: Delete records older than this many days
    """
    deleted = BatchJob.cleanup_old_jobs(days=days)
    return {"deleted_jobs": deleted}
