"""REST API blueprint with Swagger documentation.

This module sets up the Flask-RESTX API with automatic Swagger UI
at /api/v1/docs.
"""

from flask import Blueprint
from flask_restx import Api

bp = Blueprint("api", __name__)

authorizations = {
    "Bearer": {
        "type": "apiKey",
        "in": "header",
        "name": "Authorization",
        "description": "JWT token. Format: Bearer <token>",
    },
    "ApiKey": {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
        "description": "API Key for programmatic access",
    },
}

api = Api(
    bp,
    version="1.0",
    title="Sentiment Analysis API",
    description="Advanced sentiment and emotion analysis REST API. "
    "Supports single text analysis, batch processing, and real-time streaming.",
    doc="/docs",
    authorizations=authorizations,
    security="Bearer",
)

# Import and register namespaces
from app.api.routes import analysis_ns, batch_ns, auth_ns, stats_ns

api.add_namespace(analysis_ns, path="/analysis")
api.add_namespace(batch_ns, path="/batch")
api.add_namespace(auth_ns, path="/auth")
api.add_namespace(stats_ns, path="/stats")
