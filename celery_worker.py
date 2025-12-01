"""Celery worker entry point.

This module initializes the Celery application for background task processing.

Usage:
    celery -A celery_worker.celery worker --loglevel=info

    For development with auto-reload:
    celery -A celery_worker.celery worker --loglevel=debug

    With Flower monitoring:
    celery -A celery_worker.celery flower --port=5555
"""

import os
from app import create_app

# Create Flask application
flask_app = create_app(os.environ.get('FLASK_CONFIG', 'development'))

# Get Celery instance from Flask app
celery = flask_app.extensions['celery']

# Import tasks to register them with Celery
from app.batch import tasks  # noqa: F401, E402


if __name__ == '__main__':
    celery.start()
