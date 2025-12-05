"""REST API routes and namespaces."""

import time
from functools import wraps
import re

from flask import request, current_app
from flask_restx import Namespace, Resource, fields
from flask_jwt_extended import (
    jwt_required,
    get_jwt_identity,
    create_access_token,
    create_refresh_token,
)
from marshmallow import ValidationError

from app.extensions import db, limiter
from app.models.user import User
from app.models.analysis import SentimentAnalysis
from app.api.schemas import TextInputSchema, BatchTextInputSchema
from app.services.sentiment_service import SentimentService
from app.services.custom_sentiment_service import CustomSentimentService


def validate_text_input(text: str, max_length: int = 5000) -> dict:
    """Validate and sanitize text input.

    Args:
        text: Input text to validate
        max_length: Maximum allowed length

    Returns:
        dict: Validation result with 'valid', 'text', and 'error' keys
    """
    if not text or not isinstance(text, str):
        return {"valid": False, "error": "Text is required and must be a string"}

    text = text.strip()
    if not text:
        return {"valid": False, "error": "Text cannot be empty"}

    if len(text) > max_length:
        return {"valid": False, "error": f"Text exceeds maximum length of {max_length} characters"}

    # Basic HTML tag removal (simple approach without bleach)
    import re as regex

    sanitized = regex.sub(r"<[^>]+>", "", text).strip()

    # Check for suspicious patterns (basic protection)
    suspicious_patterns = [
        r"<script",
        r"javascript:",
        r"on\w+\s*=",
        r"eval\s*\(",
        r"document\.",
        r"window\.",
        r"location\.",
    ]

    for pattern in suspicious_patterns:
        if re.search(pattern, sanitized, re.IGNORECASE):
            return {"valid": False, "error": "Text contains potentially dangerous content"}

    return {"valid": True, "text": sanitized}


def validate_batch_input(texts: list, max_texts: int = 1000) -> dict:
    """Validate batch text input.

    Args:
        texts: List of texts to validate
        max_texts: Maximum number of texts allowed

    Returns:
        dict: Validation result
    """
    if not texts or not isinstance(texts, list):
        return {"valid": False, "error": "Texts must be a non-empty list"}

    if len(texts) > max_texts:
        return {"valid": False, "error": f"Batch size exceeds maximum of {max_texts} texts"}

    if len(texts) == 0:
        return {"valid": False, "error": "Texts list cannot be empty"}

    validated_texts = []
    for i, text in enumerate(texts):
        validation = validate_text_input(text)
        if not validation["valid"]:
            return {"valid": False, "error": f'Text {i}: {validation["error"]}'}
        validated_texts.append(validation["text"])

    return {"valid": True, "texts": validated_texts}


# Initialize sentiment service
_sentiment_service = None


def get_sentiment_service():
    """Get or create sentiment service singleton."""
    global _sentiment_service
    if _sentiment_service is None:
        _sentiment_service = SentimentService()
    return _sentiment_service


_custom_service = None


def get_custom_service():
    """Get or create custom PyTorch sentiment service."""
    global _custom_service
    if _custom_service is None:
        try:
            _custom_service = CustomSentimentService()
        except Exception as exc:
            current_app.logger.warning("CustomSentimentService unavailable: %s", exc)
            _custom_service = None
    return _custom_service


# API Key authentication decorator
def api_key_required(f):
    """Decorator to require API key authentication."""

    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            return {"message": "API key required", "error": "missing_api_key"}, 401

        user = User.get_by_api_key(api_key)
        if not user:
            return {"message": "Invalid API key", "error": "invalid_api_key"}, 401

        # Store user in request context
        request.current_user = user
        return f(*args, **kwargs)

    return decorated


# ==============================================================================
# Analysis Namespace
# ==============================================================================

analysis_ns = Namespace("analysis", description="Sentiment and emotion analysis operations")

# Request/Response models for Swagger
text_input_model = analysis_ns.model(
    "TextInput",
    {
        "text": fields.String(
            required=True,
            description="Text to analyze",
            min_length=1,
            max_length=5000,
            example="I love this product! It exceeded all my expectations.",
        )
    },
)

sentiment_scores_model = analysis_ns.model(
    "SentimentScores",
    {
        "pos": fields.Float(description="Positive score"),
        "neg": fields.Float(description="Negative score"),
        "neu": fields.Float(description="Neutral score"),
        "compound": fields.Float(description="Compound score"),
    },
)

analysis_result_model = analysis_ns.model(
    "AnalysisResult",
    {
        "text": fields.String(description="Analyzed text"),
        "sentiment": fields.String(
            description="Overall sentiment", enum=["positive", "negative", "neutral"]
        ),
        "compound_score": fields.Float(description="VADER compound score"),
        "scores": fields.Nested(sentiment_scores_model),
        "primary_emotion": fields.String(
            description="Primary emotion",
            enum=["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
        ),
        "emotion_scores": fields.Raw(description="All emotion scores"),
        "confidence": fields.Float(description="Emotion confidence"),
        "processing_time_ms": fields.Integer(description="Processing time in ms"),
    },
)

batch_input_model = analysis_ns.model(
    "BatchInput",
    {
        "texts": fields.List(
            fields.String, required=True, description="List of texts to analyze", max_items=100
        ),
        "include_emotions": fields.Boolean(default=True, description="Include emotion detection"),
    },
)

batch_result_model = analysis_ns.model(
    "BatchResult",
    {
        "results": fields.List(fields.Nested(analysis_result_model)),
        "total": fields.Integer(description="Total texts analyzed"),
        "processing_time_ms": fields.Integer(description="Total processing time"),
    },
)


@analysis_ns.route("/analyze")
class AnalyzeText(Resource):
    """Single text sentiment and emotion analysis."""

    @analysis_ns.expect(text_input_model)
    @analysis_ns.marshal_with(analysis_result_model)
    @analysis_ns.doc(
        security="Bearer",
        responses={
            200: "Analysis completed successfully",
            400: "Invalid input",
            401: "Authentication required",
        },
    )
    @limiter.limit("50 per hour;200 per day")
    @jwt_required()
    def post(self):
        """Analyze sentiment and emotions of provided text.

        Returns detailed sentiment scores from VADER and emotion
        detection from transformer model.
        """
        schema = TextInputSchema()
        try:
            data = schema.load(request.get_json())
        except ValidationError as err:
            analysis_ns.abort(400, f"Validation error: {err.messages}")

        text = data["text"]

        # Perform analysis
        service = get_sentiment_service()
        result = service.analyze_full(text)

        # Save to database
        try:
            user_id = get_jwt_identity()
            analysis = SentimentAnalysis.create_from_result(
                user_id=user_id, text=text, result=result, source="api"
            )
            db.session.add(analysis)
            db.session.commit()
        except Exception as e:
            current_app.logger.error(f"Failed to save analysis: {e}")
            db.session.rollback()

        return result


@analysis_ns.route("/custom/analyze")
class CustomAnalyzeText(Resource):
    """Single text analysis using the custom PyTorch model (BiLSTM + Attention)."""

    @analysis_ns.expect(text_input_model)
    @analysis_ns.marshal_with(analysis_result_model)
    @analysis_ns.doc(
        security="Bearer",
        responses={
            200: "Analysis completed successfully",
            400: "Invalid input",
            503: "Custom model unavailable",
        },
    )
    @limiter.limit("50 per hour;200 per day")
    @jwt_required()
    def post(self):
        schema = TextInputSchema()
        try:
            data = schema.load(request.get_json())
        except ValidationError as err:
            analysis_ns.abort(400, f"Validation error: {err.messages}")

        service = get_custom_service()
        if service is None:
            analysis_ns.abort(503, "Custom model not available; falling back to transformer.")

        result = service.analyze(data["text"])
        return result


@analysis_ns.route("/analyze/quick")
class QuickAnalyze(Resource):
    """Quick VADER-only sentiment analysis."""

    @analysis_ns.expect(text_input_model)
    @analysis_ns.doc(
        security="Bearer",
        responses={
            200: "Analysis completed successfully",
            400: "Invalid input",
            401: "Authentication required",
        },
    )
    @jwt_required()
    def post(self):
        """Quick sentiment analysis using VADER only.

        Much faster than full analysis (~0.3ms vs ~40ms) but
        does not include emotion detection.
        """
        data = request.get_json()

        if not data or "text" not in data:
            analysis_ns.abort(400, "Text field is required")

        text = data["text"]
        if not text or not text.strip():
            analysis_ns.abort(400, "Text cannot be empty")

        service = get_sentiment_service()
        result = service.analyze_quick(text)

        return result


@analysis_ns.route("/batch")
class BatchAnalyze(Resource):
    """Batch text analysis."""

    @analysis_ns.expect(batch_input_model)
    @analysis_ns.marshal_with(batch_result_model)
    @analysis_ns.doc(
        security="Bearer",
        responses={
            200: "Batch analysis completed",
            400: "Invalid input",
            401: "Authentication required",
        },
    )
    @limiter.limit("10 per hour;50 per day")  # Stricter limits for batch processing
    @jwt_required()
    def post(self):
        """Analyze multiple texts in a single request.

        Supports up to 100 texts per request. Uses batch processing
        for efficient transformer inference.
        """
        schema = BatchTextInputSchema()
        try:
            data = schema.load(request.get_json())
        except ValidationError as err:
            analysis_ns.abort(400, f"Validation error: {err.messages}")

        texts = data["texts"]
        include_emotions = data.get("include_emotions", True)

        start_time = time.time()

        service = get_sentiment_service()
        results = service.batch_analyze(texts, include_emotions=include_emotions)

        processing_time = int((time.time() - start_time) * 1000)

        return {"results": results, "total": len(results), "processing_time_ms": processing_time}


# ==============================================================================
# Batch Job Namespace
# ==============================================================================

batch_ns = Namespace("batch", description="Batch processing job management")

batch_job_model = batch_ns.model(
    "BatchJob",
    {
        "id": fields.Integer(description="Job ID"),
        "filename": fields.String(description="Uploaded filename"),
        "status": fields.String(
            description="Job status", enum=["pending", "processing", "completed", "failed"]
        ),
        "total_records": fields.Integer(description="Total records"),
        "processed_records": fields.Integer(description="Records processed"),
        "progress_percent": fields.Integer(description="Progress percentage"),
        "sentiment_distribution": fields.Raw(description="Sentiment counts"),
        "created_at": fields.DateTime(description="Creation time"),
        "completed_at": fields.DateTime(description="Completion time"),
    },
)


@batch_ns.route("/jobs")
class BatchJobList(Resource):
    """List batch processing jobs."""

    @batch_ns.marshal_list_with(batch_job_model)
    @batch_ns.doc(
        security="Bearer", responses={200: "List of batch jobs", 401: "Authentication required"}
    )
    @jwt_required()
    def get(self):
        """Get list of batch jobs for current user."""
        from app.models.batch_job import BatchJob

        user_id = get_jwt_identity()
        jobs = BatchJob.get_user_jobs(user_id, limit=50)

        return [job.to_dict() for job in jobs]


@batch_ns.route("/jobs/<int:job_id>")
class BatchJobDetail(Resource):
    """Get batch job details."""

    @batch_ns.marshal_with(batch_job_model)
    @batch_ns.doc(
        security="Bearer",
        responses={200: "Job details", 401: "Authentication required", 404: "Job not found"},
    )
    @jwt_required()
    def get(self, job_id):
        """Get details of a specific batch job."""
        from app.models.batch_job import BatchJob

        user_id = get_jwt_identity()
        job = BatchJob.query.filter_by(id=job_id, user_id=user_id).first()

        if not job:
            batch_ns.abort(404, "Job not found")

        return job.to_dict()


# ==============================================================================
# Authentication Namespace
# ==============================================================================

auth_ns = Namespace("auth", description="Authentication operations")

login_model = auth_ns.model(
    "Login",
    {
        "email": fields.String(required=True, description="Email address"),
        "password": fields.String(required=True, description="Password"),
    },
)

register_model = auth_ns.model(
    "Register",
    {
        "username": fields.String(
            required=True, description="Username", min_length=3, max_length=64
        ),
        "email": fields.String(required=True, description="Email address"),
        "password": fields.String(required=True, description="Password", min_length=8),
    },
)

token_model = auth_ns.model(
    "Token",
    {
        "access_token": fields.String(description="JWT access token"),
        "refresh_token": fields.String(description="JWT refresh token"),
        "token_type": fields.String(description="Token type"),
        "expires_in": fields.Integer(description="Expiry in seconds"),
    },
)

user_model = auth_ns.model(
    "User",
    {
        "id": fields.Integer(description="User ID"),
        "username": fields.String(description="Username"),
        "email": fields.String(description="Email"),
        "is_admin": fields.Boolean(description="Admin status"),
        "created_at": fields.DateTime(description="Created at"),
    },
)


@auth_ns.route("/register")
class Register(Resource):
    """User registration."""

    @auth_ns.expect(register_model)
    @auth_ns.marshal_with(user_model, code=201)
    @auth_ns.doc(
        responses={
            201: "User registered successfully",
            400: "Validation error",
            409: "User already exists",
        }
    )
    def post(self):
        """Register a new user account."""
        data = request.get_json()

        if not data:
            auth_ns.abort(400, "Request body is required")

        username = data.get("username", "").strip()
        email = data.get("email", "").strip().lower()
        password = data.get("password", "")

        # Validation
        if not username or len(username) < 3:
            auth_ns.abort(400, "Username must be at least 3 characters")

        if not email or "@" not in email:
            auth_ns.abort(400, "Valid email is required")

        if not password or len(password) < 8:
            auth_ns.abort(400, "Password must be at least 8 characters")

        # Check existing users
        if User.get_by_email(email):
            auth_ns.abort(409, "Email already registered")

        if User.get_by_username(username):
            auth_ns.abort(409, "Username already taken")

        # Create user
        user = User(username=username, email=email)
        user.set_password(password)

        db.session.add(user)
        db.session.commit()

        return user.to_dict(include_email=True), 201


@auth_ns.route("/login")
class Login(Resource):
    """User login."""

    @auth_ns.expect(login_model)
    @auth_ns.marshal_with(token_model)
    @auth_ns.doc(
        responses={200: "Login successful", 400: "Missing credentials", 401: "Invalid credentials"}
    )
    def post(self):
        """Authenticate and receive JWT tokens."""
        data = request.get_json()

        if not data:
            auth_ns.abort(400, "Request body is required")

        email = data.get("email", "").strip().lower()
        password = data.get("password", "")

        if not email or not password:
            auth_ns.abort(400, "Email and password are required")

        user = User.get_by_email(email)

        if not user or not user.check_password(password):
            auth_ns.abort(401, "Invalid email or password")

        if not user.is_active:
            auth_ns.abort(401, "Account is disabled")

        # Update last login
        user.update_last_login()
        db.session.commit()

        # Generate tokens
        access_token = create_access_token(identity=user.id)
        refresh_token = create_refresh_token(identity=user.id)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "Bearer",
            "expires_in": current_app.config.get("JWT_ACCESS_TOKEN_EXPIRES", 3600),
        }


@auth_ns.route("/refresh")
class RefreshToken(Resource):
    """Refresh access token."""

    @auth_ns.marshal_with(token_model)
    @auth_ns.doc(
        security="Bearer", responses={200: "Token refreshed", 401: "Invalid refresh token"}
    )
    @jwt_required(refresh=True)
    def post(self):
        """Get new access token using refresh token."""
        user_id = get_jwt_identity()
        access_token = create_access_token(identity=user_id)

        return {
            "access_token": access_token,
            "refresh_token": None,
            "token_type": "Bearer",
            "expires_in": current_app.config.get("JWT_ACCESS_TOKEN_EXPIRES", 3600),
        }


@auth_ns.route("/me")
class CurrentUser(Resource):
    """Current user profile."""

    @auth_ns.marshal_with(user_model)
    @auth_ns.doc(security="Bearer", responses={200: "User profile", 401: "Not authenticated"})
    @jwt_required()
    def get(self):
        """Get current user profile."""
        user_id = get_jwt_identity()
        user = User.query.get(user_id)

        if not user:
            auth_ns.abort(401, "User not found")

        return user.to_dict(include_email=True)


@auth_ns.route("/api-key")
class APIKey(Resource):
    """API key management."""

    @auth_ns.doc(security="Bearer", responses={200: "API key generated", 401: "Not authenticated"})
    @jwt_required()
    def post(self):
        """Generate new API key for current user."""
        user_id = get_jwt_identity()
        user = User.query.get(user_id)

        if not user:
            auth_ns.abort(401, "User not found")

        api_key = user.generate_api_key()
        db.session.commit()

        return {
            "api_key": api_key,
            "message": "Store this key securely. It cannot be retrieved later.",
        }

    @auth_ns.doc(security="Bearer", responses={200: "API key revoked", 401: "Not authenticated"})
    @jwt_required()
    def delete(self):
        """Revoke API key for current user."""
        user_id = get_jwt_identity()
        user = User.query.get(user_id)

        if not user:
            auth_ns.abort(401, "User not found")

        user.revoke_api_key()
        db.session.commit()

        return {"message": "API key revoked"}


# ==============================================================================
# Statistics Namespace
# ==============================================================================

stats_ns = Namespace("stats", description="User statistics and analytics")

stats_model = stats_ns.model(
    "Stats",
    {
        "total_analyses": fields.Integer(description="Total analyses"),
        "sentiment_distribution": fields.Raw(description="Sentiment counts"),
        "emotion_distribution": fields.Raw(description="Emotion counts"),
        "average_confidence": fields.Float(description="Average confidence"),
        "average_processing_time_ms": fields.Float(description="Average processing time"),
    },
)

history_item_model = stats_ns.model(
    "HistoryItem",
    {
        "id": fields.Integer(description="Analysis ID"),
        "input_text": fields.String(description="Analyzed text"),
        "sentiment": fields.String(description="Sentiment"),
        "primary_emotion": fields.String(description="Primary emotion"),
        "confidence": fields.Float(description="Confidence"),
        "created_at": fields.DateTime(description="Timestamp"),
        "source": fields.String(description="Source"),
    },
)


@stats_ns.route("/")
class UserStats(Resource):
    """User statistics."""

    @stats_ns.marshal_with(stats_model)
    @stats_ns.doc(security="Bearer", responses={200: "User statistics", 401: "Not authenticated"})
    @jwt_required()
    def get(self):
        """Get statistics for current user."""
        user_id = get_jwt_identity()
        stats = SentimentAnalysis.get_user_stats(user_id)
        return stats


@stats_ns.route("/history")
class AnalysisHistory(Resource):
    """Analysis history."""

    @stats_ns.marshal_list_with(history_item_model)
    @stats_ns.doc(
        security="Bearer",
        params={
            "page": "Page number (default: 1)",
            "per_page": "Items per page (default: 20, max: 100)",
            "sentiment": "Filter by sentiment",
            "source": "Filter by source",
        },
        responses={200: "Analysis history", 401: "Not authenticated"},
    )
    @jwt_required()
    def get(self):
        """Get analysis history for current user."""
        user_id = get_jwt_identity()

        page = request.args.get("page", 1, type=int)
        per_page = min(request.args.get("per_page", 20, type=int), 100)
        sentiment = request.args.get("sentiment")
        source = request.args.get("source")

        query = SentimentAnalysis.query.filter_by(user_id=user_id)

        if sentiment:
            query = query.filter_by(sentiment=sentiment)
        if source:
            query = query.filter_by(source=source)

        query = query.order_by(SentimentAnalysis.created_at.desc())

        pagination = query.paginate(page=page, per_page=per_page, error_out=False)

        return [a.to_dict() for a in pagination.items]


@stats_ns.route("/history/<int:analysis_id>")
class AnalysisDetail(Resource):
    """Analysis detail."""

    @stats_ns.marshal_with(history_item_model)
    @stats_ns.doc(
        security="Bearer",
        responses={200: "Analysis details", 401: "Not authenticated", 404: "Analysis not found"},
    )
    @jwt_required()
    def get(self, analysis_id):
        """Get details of a specific analysis."""
        user_id = get_jwt_identity()
        analysis = SentimentAnalysis.query.filter_by(id=analysis_id, user_id=user_id).first()

        if not analysis:
            stats_ns.abort(404, "Analysis not found")

        return analysis.to_dict()

    @stats_ns.doc(
        security="Bearer",
        responses={204: "Analysis deleted", 401: "Not authenticated", 404: "Analysis not found"},
    )
    @jwt_required()
    def delete(self, analysis_id):
        """Delete a specific analysis."""
        user_id = get_jwt_identity()
        analysis = SentimentAnalysis.query.filter_by(id=analysis_id, user_id=user_id).first()

        if not analysis:
            stats_ns.abort(404, "Analysis not found")

        db.session.delete(analysis)
        db.session.commit()

        return "", 204
