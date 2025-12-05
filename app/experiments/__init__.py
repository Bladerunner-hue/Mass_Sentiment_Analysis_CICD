"""Experiment and training playground blueprint."""

from flask import Blueprint

bp = Blueprint("experiments", __name__, template_folder="templates")

from app.experiments import routes  # noqa: E402,F401
