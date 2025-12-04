"""Dataset management blueprint."""

from flask import Blueprint

bp = Blueprint("datasets", __name__, template_folder="../templates/datasets")

from app.datasets import routes  # noqa: E402, F401
