"""Main web UI blueprint."""

from flask import Blueprint

bp = Blueprint("main", __name__, template_folder="templates")

from app.main import routes  # noqa: F401, E402
