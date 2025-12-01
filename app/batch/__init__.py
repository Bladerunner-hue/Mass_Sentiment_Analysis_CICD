"""Batch processing blueprint."""

from flask import Blueprint

bp = Blueprint(
    'batch',
    __name__,
    template_folder='templates'
)

from app.batch import routes  # noqa: F401, E402
