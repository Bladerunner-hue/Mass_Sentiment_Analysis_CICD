"""User model for authentication and authorization."""

from datetime import datetime
from typing import Optional

from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, Boolean, DateTime, Integer

from app.extensions import db


class User(UserMixin, db.Model):
    """User model for authentication.

    Attributes:
        id: Primary key
        username: Unique username
        email: Unique email address
        password_hash: Hashed password
        is_admin: Admin flag
        is_active: Account active flag
        created_at: Account creation timestamp
        last_login: Last login timestamp
        api_key: API key for programmatic access
    """

    __tablename__ = 'users'

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(120), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    api_key: Mapped[Optional[str]] = mapped_column(String(64), unique=True, nullable=True, index=True)

    # Relationships
    analyses = relationship('SentimentAnalysis', back_populates='user', lazy='dynamic',
                           cascade='all, delete-orphan')
    batch_jobs = relationship('BatchJob', back_populates='user', lazy='dynamic',
                             cascade='all, delete-orphan')

    def __repr__(self) -> str:
        """String representation of User."""
        return f'<User {self.username}>'

    def set_password(self, password: str) -> None:
        """Hash and set the user's password.

        Args:
            password: Plain text password to hash
        """
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """Verify a password against the stored hash.

        Args:
            password: Plain text password to verify

        Returns:
            bool: True if password matches, False otherwise
        """
        return check_password_hash(self.password_hash, password)

    def generate_api_key(self) -> str:
        """Generate a new API key for the user.

        Returns:
            str: The generated API key
        """
        import secrets
        self.api_key = secrets.token_urlsafe(32)
        return self.api_key

    def revoke_api_key(self) -> None:
        """Revoke the user's API key."""
        self.api_key = None

    def update_last_login(self) -> None:
        """Update the last login timestamp."""
        self.last_login = datetime.utcnow()

    def to_dict(self, include_email: bool = False) -> dict:
        """Convert user to dictionary representation.

        Args:
            include_email: Whether to include email in output

        Returns:
            dict: User data dictionary
        """
        data = {
            'id': self.id,
            'username': self.username,
            'is_admin': self.is_admin,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'analysis_count': self.analyses.count(),
            'batch_job_count': self.batch_jobs.count()
        }
        if include_email:
            data['email'] = self.email
        return data

    @staticmethod
    def get_by_api_key(api_key: str) -> Optional['User']:
        """Find a user by their API key.

        Args:
            api_key: API key to search for

        Returns:
            User or None: The user if found
        """
        if not api_key:
            return None
        return User.query.filter_by(api_key=api_key, is_active=True).first()

    @staticmethod
    def get_by_email(email: str) -> Optional['User']:
        """Find a user by their email address.

        Args:
            email: Email address to search for

        Returns:
            User or None: The user if found
        """
        return User.query.filter_by(email=email.lower()).first()

    @staticmethod
    def get_by_username(username: str) -> Optional['User']:
        """Find a user by their username.

        Args:
            username: Username to search for

        Returns:
            User or None: The user if found
        """
        return User.query.filter_by(username=username).first()
