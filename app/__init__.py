"""Flask application factory.

This module contains the create_app() factory function that initializes
and configures the Flask application with all extensions and blueprints.
"""

import os
import logging
from logging.handlers import RotatingFileHandler

from flask import Flask, request


def create_app(config_name=None):
    """Create and configure the Flask application.

    Args:
        config_name: Configuration name ('development', 'production', 'testing')
                    If None, uses FLASK_CONFIG environment variable or 'development'

    Returns:
        Flask: Configured Flask application instance
    """
    app = Flask(__name__,
                template_folder='templates',
                static_folder='static')

    # Load configuration
    if config_name is None:
        config_name = os.environ.get('FLASK_CONFIG', 'development')

    from config import get_config
    app.config.from_object(get_config(config_name))

    # Ensure upload folder exists
    os.makedirs(app.config.get('UPLOAD_FOLDER', 'uploads'), exist_ok=True)

    # Initialize extensions
    _init_extensions(app)

    # Register blueprints
    _register_blueprints(app)

    # Register error handlers
    _register_error_handlers(app)

    # Configure logging
    _configure_logging(app)

    # Register shell context
    _register_shell_context(app)

    # Register CLI commands
    _register_cli_commands(app)

    return app


def _init_extensions(app):
    """Initialize Flask extensions.

    Args:
        app: Flask application instance
    """
    from app.extensions import (
        db, migrate, login_manager, jwt, cache, limiter, csrf, celery_init_app
    )

    # Initialize CORS
    from flask_cors import CORS
    if app.config.get('ENV') == 'production':
        allowed_origins = os.environ.get('ALLOWED_ORIGINS', '').split(',') \
            if os.environ.get('ALLOWED_ORIGINS') \
            else ['http://localhost:3000', 'http://localhost:5000']
        CORS(app, resources={
            r"/api/*": {
                "origins": allowed_origins,
                "methods": ["GET", "POST", "PUT", "DELETE"],
                "allow_headers": ["Content-Type", "Authorization"],
                "max_age": 3600
            }
        })
    else:
        CORS(app)  # Allow all in development

    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    jwt.init_app(app)

    # Configure JWT callbacks
    from app.models.user import User

    @jwt.user_identity_loader
    def user_identity_lookup(user):
        """Define how to serialize user identity to JWT."""
        return str(user)

    @jwt.user_lookup_loader
    def user_lookup_callback(_jwt_header, jwt_data):
        """Define how to deserialize user identity from JWT."""
        identity = jwt_data["sub"]
        return User.query.get(int(identity))

    cache.init_app(app)
    csrf.init_app(app)

    if app.config.get('RATELIMIT_ENABLED', True):
        limiter.init_app(app)

    # Initialize Celery
    celery_init_app(app)


def _register_blueprints(app):
    """Register application blueprints.

    Args:
        app: Flask application instance
    """
    from app.extensions import csrf

    # Main web UI blueprint
    from app.main import bp as main_bp
    app.register_blueprint(main_bp)

    # API blueprint (exempt from CSRF)
    from app.api import bp as api_bp
    app.register_blueprint(api_bp, url_prefix='/api/v1')
    csrf.exempt(api_bp)

    # Batch processing blueprint
    from app.batch import bp as batch_bp
    app.register_blueprint(batch_bp, url_prefix='/batch')

    # Authentication blueprint
    from app.auth import bp as auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')


def _register_error_handlers(app):
    """Register error handlers.

    Args:
        app: Flask application instance
    """
    from app.errors.handlers import (
        handle_400, handle_401, handle_403, handle_404, handle_500,
        handle_validation_error
    )
    from marshmallow import ValidationError

    app.register_error_handler(400, handle_400)
    app.register_error_handler(401, handle_401)
    app.register_error_handler(403, handle_403)
    app.register_error_handler(404, handle_404)
    app.register_error_handler(500, handle_500)
    app.register_error_handler(ValidationError, handle_validation_error)


def _configure_logging(app):
    """Configure application logging.

    Args:
        app: Flask application instance
    """
    log_level = getattr(logging, app.config.get('LOG_LEVEL', 'INFO'))

    # Remove default handlers
    app.logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    app.logger.addHandler(console_handler)

    # File handler (production only)
    if not app.debug and not app.testing:
        log_dir = os.path.join(os.path.dirname(app.root_path), 'logs')
        os.makedirs(log_dir, exist_ok=True)

        file_handler = RotatingFileHandler(
            os.path.join(log_dir, 'sentiment_analyzer.log'),
            maxBytes=10240000,  # 10MB
            backupCount=10
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        app.logger.addHandler(file_handler)

    app.logger.setLevel(log_level)
    app.logger.info(f'Sentiment Analyzer startup - config: {app.config.get("ENV", "development")}')


def _register_shell_context(app):
    """Register shell context for flask shell command.

    Args:
        app: Flask application instance
    """
    @app.shell_context_processor
    def make_shell_context():
        from app.extensions import db
        from app.models.user import User
        from app.models.analysis import SentimentAnalysis
        from app.models.batch_job import BatchJob
        return {
            'db': db,
            'User': User,
            'SentimentAnalysis': SentimentAnalysis,
            'BatchJob': BatchJob
        }


def _register_cli_commands(app):
    """Register custom CLI commands.

    Args:
        app: Flask application instance
    """
    @app.cli.command()
    def init_db():
        """Initialize the database."""
        from app.extensions import db
        db.create_all()
        print('Database initialized.')

    @app.cli.command()
    def seed_db():
        """Seed the database with sample data."""
        from app.extensions import db
        from app.models.user import User

        # Create admin user if not exists
        admin = User.query.filter_by(email='admin@example.com').first()
        if not admin:
            admin = User(
                username='admin',
                email='admin@example.com',
                is_admin=True
            )
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
            print('Admin user created: admin@example.com / admin123')
        else:
            print('Admin user already exists.')

    @app.cli.command()
    def clear_cache():
        """Clear the application cache."""
        from app.extensions import cache
        cache.clear()
        print('Cache cleared.')

    @app.cli.command()
    def download_model():
        """Download the sentiment analysis model."""
        from app.services.sentiment_service import SentimentService
        service = SentimentService()
        service._load_emotion_model()
        print('Model downloaded successfully.')
