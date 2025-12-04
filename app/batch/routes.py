"""Batch processing routes with SSE progress streaming."""

import os
import time
import json
import uuid

from flask import (
    render_template,
    redirect,
    url_for,
    flash,
    request,
    Response,
    stream_with_context,
    send_file,
    current_app,
    jsonify,
)
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename

from app.batch import bp
from app.main.forms import BatchUploadForm
from app.extensions import db
from app.models.batch_job import BatchJob
from app.services.batch_service import BatchService


def allowed_file(filename):
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"csv"}


@bp.route("/")
@login_required
def index():
    """Batch processing landing page."""
    form = BatchUploadForm()

    # Get user's batch jobs
    jobs = BatchJob.get_user_jobs(current_user.id, limit=20)

    # Count active jobs
    active_count = BatchJob.get_active_jobs_count(current_user.id)

    return render_template("batch/index.html", form=form, jobs=jobs, active_count=active_count)


@bp.route("/upload", methods=["POST"])
@login_required
def upload():
    """Handle CSV file upload and start processing."""
    form = BatchUploadForm()

    if not form.validate_on_submit():
        flash("Please select a valid CSV file.", "error")
        return redirect(url_for("batch.index"))

    file = form.file.data
    include_emotions = form.include_emotions.data

    if not file or not allowed_file(file.filename):
        flash("Invalid file. Please upload a CSV file.", "error")
        return redirect(url_for("batch.index"))

    # Check for too many active jobs
    if BatchJob.get_active_jobs_count(current_user.id) >= 5:
        flash("You have too many active batch jobs. Please wait for some to complete.", "error")
        return redirect(url_for("batch.index"))

    # Secure the filename and save
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    upload_folder = current_app.config.get("UPLOAD_FOLDER", "uploads")
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, unique_filename)
    file.save(file_path)

    # Validate the file
    batch_service = BatchService()
    with open(file_path, "rb") as f:
        validation = batch_service.validate_file(f, filename)

    if not validation["valid"]:
        os.remove(file_path)
        flash(f"Invalid file: {validation['error']}", "error")
        return redirect(url_for("batch.index"))

    # Create batch job
    batch_job = BatchJob(
        user_id=current_user.id,
        filename=filename,
        total_records=validation["row_count"],
        status=BatchJob.STATUS_PENDING,
    )
    db.session.add(batch_job)
    db.session.commit()

    # Start Celery task
    from app.batch.tasks import process_batch_file

    task = process_batch_file.delay(batch_job.id, file_path, include_emotions)

    # Update job with task ID
    batch_job.celery_task_id = task.id
    db.session.commit()

    flash(f"Processing started for {filename}. You can track progress below.", "success")
    return redirect(url_for("batch.job_detail", job_id=batch_job.id))


@bp.route("/jobs/<int:job_id>")
@login_required
def job_detail(job_id):
    """View batch job details and progress."""
    job = BatchJob.query.filter_by(id=job_id, user_id=current_user.id).first_or_404()

    return render_template("batch/job_detail.html", job=job)


@bp.route("/jobs/<int:job_id>/progress")
@login_required
def job_progress(job_id):
    """SSE endpoint for real-time progress updates."""
    job = BatchJob.query.filter_by(id=job_id, user_id=current_user.id).first_or_404()

    def generate():
        """Generate SSE events for job progress."""
        from app.batch.tasks import process_batch_file

        last_progress = -1

        while True:
            # Refresh job from database
            db.session.refresh(job)

            # Build progress data
            progress_data = {
                "status": job.status,
                "progress": job.progress_percent,
                "processed": job.processed_records,
                "total": job.total_records,
                "sentiment": {
                    "positive": job.positive_count,
                    "negative": job.negative_count,
                    "neutral": job.neutral_count,
                },
                "emotions": job.emotion_counts or {},
            }

            # Only send if progress changed
            if job.progress_percent != last_progress or job.status in ["completed", "failed"]:
                yield f"data: {json.dumps(progress_data)}\n\n"
                last_progress = job.progress_percent

            # Check if job is complete
            if job.status in ["completed", "failed"]:
                break

            # If task is running, also check Celery state
            if job.celery_task_id and job.status == "processing":
                task = process_batch_file.AsyncResult(job.celery_task_id)

                if task.state == "PROGRESS" and task.info:
                    progress_data.update(
                        {
                            "progress": task.info.get("progress", 0),
                            "processed": task.info.get("processed", 0),
                            "celery_state": task.state,
                        }
                    )
                    yield f"data: {json.dumps(progress_data)}\n\n"

            time.sleep(1)

        # Final status update
        final_data = {
            "status": job.status,
            "progress": 100 if job.status == "completed" else job.progress_percent,
            "processed": job.processed_records,
            "total": job.total_records,
            "sentiment": {
                "positive": job.positive_count,
                "negative": job.negative_count,
                "neutral": job.neutral_count,
            },
            "emotions": job.emotion_counts or {},
            "completed": job.status == "completed",
            "error": job.error_message if job.status == "failed" else None,
        }
        yield f"data: {json.dumps(final_data)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@bp.route("/jobs/<int:job_id>/download")
@login_required
def download_results(job_id):
    """Download batch job results as CSV."""
    job = BatchJob.query.filter_by(id=job_id, user_id=current_user.id).first_or_404()

    if job.status != BatchJob.STATUS_COMPLETED:
        flash("Job is not completed yet.", "error")
        return redirect(url_for("batch.job_detail", job_id=job_id))

    if not job.result_file_path or not os.path.exists(job.result_file_path):
        flash("Results file not found.", "error")
        return redirect(url_for("batch.job_detail", job_id=job_id))

    return send_file(
        job.result_file_path,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"sentiment_results_{job.filename}",
    )


@bp.route("/jobs/<int:job_id>/delete", methods=["POST"])
@login_required
def delete_job(job_id):
    """Delete a batch job and its result file."""
    job = BatchJob.query.filter_by(id=job_id, user_id=current_user.id).first_or_404()

    # Delete result file if exists
    if job.result_file_path and os.path.exists(job.result_file_path):
        try:
            os.remove(job.result_file_path)
        except OSError:
            pass

    # Delete job from database
    db.session.delete(job)
    db.session.commit()

    flash("Batch job deleted successfully.", "success")
    return redirect(url_for("batch.index"))


@bp.route("/api/status/<int:job_id>")
@login_required
def api_status(job_id):
    """API endpoint for job status (JSON)."""
    job = BatchJob.query.filter_by(id=job_id, user_id=current_user.id).first()

    if not job:
        return jsonify({"error": "Job not found"}), 404

    return jsonify(job.to_dict())
