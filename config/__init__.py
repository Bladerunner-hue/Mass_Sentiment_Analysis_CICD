"""Configuration module for Flask application."""

import os


class BaseConfig:
    """Base configuration with common settings."""

    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

    # SQLAlchemy settings
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
    }

    # JWT settings
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', SECRET_KEY)
    JWT_ACCESS_TOKEN_EXPIRES = 3600  # 1 hour
    JWT_REFRESH_TOKEN_EXPIRES = 2592000  # 30 days
    JWT_TOKEN_LOCATION = ['headers']
    JWT_HEADER_NAME = 'Authorization'
    JWT_HEADER_TYPE = 'Bearer'

    # Celery settings
    CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
    CELERY_TASK_SERIALIZER = 'json'
    CELERY_RESULT_SERIALIZER = 'json'
    CELERY_ACCEPT_CONTENT = ['json']
    CELERY_TIMEZONE = 'UTC'
    CELERY_TASK_TRACK_STARTED = True
    CELERY_TASK_TIME_LIMIT = 600  # 10 minutes

    # Cache settings
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300

    # File upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
    ALLOWED_EXTENSIONS = {'csv', 'txt'}

    # Rate limiting
    RATELIMIT_ENABLED = True
    RATELIMIT_DEFAULT = "200 per day;50 per hour"
    RATELIMIT_STORAGE_URL = os.environ.get('REDIS_URL', 'memory://')

    # Sentiment analysis settings
    SENTIMENT_MODEL_NAME = 'j-hartmann/emotion-english-distilroberta-base'
    SENTIMENT_BATCH_SIZE = 32
    SENTIMENT_MAX_TEXT_LENGTH = 5000

    # Pagination
    ITEMS_PER_PAGE = 20


class DevelopmentConfig(BaseConfig):
    """Development configuration."""

    DEBUG = True
    TESTING = False

    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL',
        'sqlite:///' + os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dev.db')
    )

    # Development cache
    CACHE_TYPE = 'simple'

    # Disable rate limiting in development
    RATELIMIT_ENABLED = False

    # More verbose logging
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(BaseConfig):
    """Production configuration."""

    DEBUG = False
    TESTING = False

    # PostgreSQL for production
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')

    # Redis cache for production
    CACHE_TYPE = 'redis'
    CACHE_REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/1')

    # Production rate limiting
    RATELIMIT_ENABLED = True
    RATELIMIT_STORAGE_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/2')

    # Production logging
    LOG_LEVEL = 'INFO'

    # Security settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

    # Enforce HTTPS
    PREFERRED_URL_SCHEME = 'https'


class TestingConfig(BaseConfig):
    """Testing configuration."""

    DEBUG = True
    TESTING = True

    # In-memory SQLite for tests
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

    # Disable CSRF for testing
    WTF_CSRF_ENABLED = False

    # Disable rate limiting in tests
    RATELIMIT_ENABLED = False

    # Use simple cache for tests
    CACHE_TYPE = 'simple'

    # Celery eager mode for synchronous execution
    CELERY_TASK_ALWAYS_EAGER = True
    CELERY_TASK_EAGER_PROPAGATES = True

    # Test logging
    LOG_LEVEL = 'DEBUG'

    # Disable login required for some tests
    LOGIN_DISABLED = False


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(config_name=None):
    """Get configuration class by name."""
    if config_name is None:
        config_name = os.environ.get('FLASK_CONFIG', 'development')
    return config.get(config_name, DevelopmentConfig)
