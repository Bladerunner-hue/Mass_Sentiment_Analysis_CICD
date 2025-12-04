"""Flask extension instances.

All extensions are instantiated here and initialized in create_app().
This pattern avoids circular imports and allows for easier testing.
"""

from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_caching import Cache
from flask_jwt_extended import JWTManager
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect
from celery import Celery


# SQLAlchemy database instance
db = SQLAlchemy()

# Flask-Migrate for database migrations
migrate = Migrate()

# Flask-Login for session-based authentication
login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.login_message = "Please log in to access this page."
login_manager.login_message_category = "info"

# Flask-JWT-Extended for API token authentication
jwt = JWTManager()

# Flask-Caching for response caching
cache = Cache()

# Flask-Limiter for rate limiting
limiter = Limiter(key_func=get_remote_address, default_limits=["200 per day", "50 per hour"])

# CSRF protection
csrf = CSRFProtect()


def celery_init_app(app):
    """Initialize Celery with Flask application context.

    Args:
        app: Flask application instance

    Returns:
        Celery: Configured Celery application
    """

    class FlaskTask(Celery.Task):
        """Custom Task class that runs within Flask app context."""

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery_app = Celery(app.name)

    # Configure Celery from Flask config
    celery_app.config_from_object(
        {
            "broker_url": app.config.get("CELERY_BROKER_URL", "redis://localhost:6379/0"),
            "result_backend": app.config.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0"),
            "task_serializer": app.config.get("CELERY_TASK_SERIALIZER", "json"),
            "result_serializer": app.config.get("CELERY_RESULT_SERIALIZER", "json"),
            "accept_content": app.config.get("CELERY_ACCEPT_CONTENT", ["json"]),
            "timezone": app.config.get("CELERY_TIMEZONE", "UTC"),
            "task_track_started": app.config.get("CELERY_TASK_TRACK_STARTED", True),
            "task_time_limit": app.config.get("CELERY_TASK_TIME_LIMIT", 600),
            "task_always_eager": app.config.get("CELERY_TASK_ALWAYS_EAGER", False),
            "task_eager_propagates": app.config.get("CELERY_TASK_EAGER_PROPAGATES", False),
        }
    )

    celery_app.Task = FlaskTask
    celery_app.set_default()

    # Store celery instance in app extensions
    app.extensions["celery"] = celery_app

    return celery_app


@login_manager.user_loader
def load_user(user_id):
    """Load user by ID for Flask-Login.

    Args:
        user_id: User ID string

    Returns:
        User instance or None
    """
    from app.models.user import User

    return User.query.get(int(user_id))


@jwt.user_identity_loader
def user_identity_lookup(user):
    """Return user ID for JWT token creation.

    Args:
        user: User instance or user ID

    Returns:
        int: User ID
    """
    if hasattr(user, "id"):
        return user.id
    return user


@jwt.user_lookup_loader
def user_lookup_callback(_jwt_header, jwt_data):
    """Load user from JWT token data.

    Args:
        _jwt_header: JWT header (unused)
        jwt_data: JWT payload data

    Returns:
        User instance or None
    """
    from app.models.user import User

    identity = jwt_data["sub"]
    return User.query.get(identity)


@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    """Handle expired JWT tokens.

    Args:
        jwt_header: JWT header
        jwt_payload: JWT payload

    Returns:
        tuple: JSON response and status code
    """
    return {"message": "Token has expired", "error": "token_expired"}, 401


@jwt.invalid_token_loader
def invalid_token_callback(error):
    """Handle invalid JWT tokens.

    Args:
        error: Error message

    Returns:
        tuple: JSON response and status code
    """
    return {"message": "Invalid token", "error": "invalid_token"}, 401


@jwt.unauthorized_loader
def missing_token_callback(error):
    """Handle missing JWT tokens.

    Args:
        error: Error message

    Returns:
        tuple: JSON response and status code
    """
    return {"message": "Authorization token required", "error": "authorization_required"}, 401
