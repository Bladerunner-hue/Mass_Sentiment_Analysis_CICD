"""Marshmallow schemas for API request/response validation."""

from marshmallow import Schema, fields, validate, post_load


class TextInputSchema(Schema):
    """Schema for single text analysis input."""

    text = fields.String(
        required=True,
        validate=validate.Length(min=1, max=5000),
        metadata={'description': 'Text to analyze (1-5000 characters)'}
    )


class BatchTextInputSchema(Schema):
    """Schema for batch text analysis input."""

    texts = fields.List(
        fields.String(validate=validate.Length(min=1, max=5000)),
        required=True,
        validate=validate.Length(min=1, max=100),
        metadata={'description': 'List of texts to analyze (1-100 items)'}
    )
    include_emotions = fields.Boolean(
        load_default=True,
        metadata={'description': 'Include emotion detection (slower but more detailed)'}
    )


class SentimentScoresSchema(Schema):
    """Schema for VADER sentiment scores."""

    positive = fields.Float(metadata={'description': 'Positive score (0-1)'})
    negative = fields.Float(metadata={'description': 'Negative score (0-1)'})
    neutral = fields.Float(metadata={'description': 'Neutral score (0-1)'})
    compound = fields.Float(metadata={'description': 'Compound score (-1 to 1)'})


class SentimentResultSchema(Schema):
    """Schema for sentiment analysis result."""

    text = fields.String(metadata={'description': 'Analyzed text (truncated if long)'})
    sentiment = fields.String(
        metadata={'description': 'Overall sentiment: positive, negative, or neutral'}
    )
    compound_score = fields.Float(
        metadata={'description': 'VADER compound score (-1 to 1)'}
    )
    scores = fields.Nested(
        SentimentScoresSchema,
        metadata={'description': 'Detailed VADER scores'}
    )
    primary_emotion = fields.String(
        allow_none=True,
        metadata={'description': 'Primary detected emotion'}
    )
    emotion_scores = fields.Dict(
        keys=fields.String(),
        values=fields.Float(),
        allow_none=True,
        metadata={'description': 'All emotion scores'}
    )
    confidence = fields.Float(
        allow_none=True,
        metadata={'description': 'Confidence in primary emotion (0-1)'}
    )
    processing_time_ms = fields.Integer(
        metadata={'description': 'Processing time in milliseconds'}
    )


class BatchResultSchema(Schema):
    """Schema for batch analysis result."""

    index = fields.Integer(metadata={'description': 'Index in input array'})
    sentiment = fields.String(metadata={'description': 'Sentiment label'})
    compound_score = fields.Float(metadata={'description': 'Compound score'})
    scores = fields.Nested(SentimentScoresSchema)
    primary_emotion = fields.String(allow_none=True)
    confidence = fields.Float(allow_none=True)


class BatchResponseSchema(Schema):
    """Schema for batch analysis response."""

    results = fields.List(
        fields.Nested(BatchResultSchema),
        metadata={'description': 'Analysis results for each input text'}
    )
    total = fields.Integer(metadata={'description': 'Total texts analyzed'})
    processing_time_ms = fields.Integer(
        metadata={'description': 'Total processing time'}
    )


class UserLoginSchema(Schema):
    """Schema for user login."""

    email = fields.Email(
        required=True,
        metadata={'description': 'User email address'}
    )
    password = fields.String(
        required=True,
        validate=validate.Length(min=6),
        load_only=True,
        metadata={'description': 'User password'}
    )


class UserRegisterSchema(Schema):
    """Schema for user registration."""

    username = fields.String(
        required=True,
        validate=validate.Length(min=3, max=64),
        metadata={'description': 'Username (3-64 characters)'}
    )
    email = fields.Email(
        required=True,
        metadata={'description': 'Email address'}
    )
    password = fields.String(
        required=True,
        validate=validate.Length(min=8),
        load_only=True,
        metadata={'description': 'Password (minimum 8 characters)'}
    )


class TokenResponseSchema(Schema):
    """Schema for JWT token response."""

    access_token = fields.String(metadata={'description': 'JWT access token'})
    refresh_token = fields.String(metadata={'description': 'JWT refresh token'})
    token_type = fields.String(
        dump_default='Bearer',
        metadata={'description': 'Token type'}
    )
    expires_in = fields.Integer(metadata={'description': 'Token expiry in seconds'})


class UserResponseSchema(Schema):
    """Schema for user response."""

    id = fields.Integer(metadata={'description': 'User ID'})
    username = fields.String(metadata={'description': 'Username'})
    email = fields.Email(metadata={'description': 'Email address'})
    is_admin = fields.Boolean(metadata={'description': 'Admin status'})
    created_at = fields.DateTime(metadata={'description': 'Account creation date'})
    analysis_count = fields.Integer(metadata={'description': 'Total analyses performed'})


class StatsResponseSchema(Schema):
    """Schema for user statistics response."""

    total_analyses = fields.Integer(metadata={'description': 'Total analyses performed'})
    sentiment_distribution = fields.Dict(
        keys=fields.String(),
        values=fields.Integer(),
        metadata={'description': 'Count by sentiment type'}
    )
    emotion_distribution = fields.Dict(
        keys=fields.String(),
        values=fields.Integer(),
        metadata={'description': 'Count by emotion type'}
    )
    average_confidence = fields.Float(
        metadata={'description': 'Average confidence score'}
    )
    average_processing_time_ms = fields.Float(
        metadata={'description': 'Average processing time'}
    )


class ErrorSchema(Schema):
    """Schema for error responses."""

    message = fields.String(metadata={'description': 'Error message'})
    error = fields.String(metadata={'description': 'Error code'})
    details = fields.Dict(
        allow_none=True,
        metadata={'description': 'Additional error details'}
    )


class AnalysisHistorySchema(Schema):
    """Schema for analysis history item."""

    id = fields.Integer(metadata={'description': 'Analysis ID'})
    input_text = fields.String(metadata={'description': 'Analyzed text'})
    sentiment = fields.String(metadata={'description': 'Sentiment result'})
    primary_emotion = fields.String(allow_none=True)
    confidence = fields.Float(allow_none=True)
    created_at = fields.DateTime(metadata={'description': 'Analysis timestamp'})
    source = fields.String(metadata={'description': 'Source of analysis'})


class PaginatedHistorySchema(Schema):
    """Schema for paginated history response."""

    items = fields.List(
        fields.Nested(AnalysisHistorySchema),
        metadata={'description': 'List of analyses'}
    )
    total = fields.Integer(metadata={'description': 'Total items'})
    page = fields.Integer(metadata={'description': 'Current page'})
    per_page = fields.Integer(metadata={'description': 'Items per page'})
    pages = fields.Integer(metadata={'description': 'Total pages'})
