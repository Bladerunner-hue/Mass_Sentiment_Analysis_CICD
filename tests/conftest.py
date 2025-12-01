"""Pytest configuration and fixtures."""

import os
import pytest
import tempfile

from app import create_app
from app.extensions import db
from app.models.user import User


@pytest.fixture(scope='session')
def app():
    """Create application for testing."""
    # Create a temporary database file
    db_fd, db_path = tempfile.mkstemp()

    app = create_app('testing')
    app.config.update({
        'SQLALCHEMY_DATABASE_URI': f'sqlite:///{db_path}',
        'SQLALCHEMY_ENGINE_OPTIONS': {},  # Disable pooling for SQLite
        'WTF_CSRF_ENABLED': False,
        'TESTING': True,
    })

    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()

    os.close(db_fd)
    os.unlink(db_path)


@pytest.fixture(scope='function')
def client(app):
    """Create a test client."""
    return app.test_client()


@pytest.fixture(scope='function')
def runner(app):
    """Create a CLI test runner."""
    return app.test_cli_runner()


@pytest.fixture(scope='function')
def db_session(app):
    """Create a database session for testing."""
    with app.app_context():
        db.session.begin_nested()
        yield db.session
        db.session.rollback()


@pytest.fixture(scope='function')
def test_user(app):
    """Create a test user."""
    with app.app_context():
        user = User(
            username='testuser',
            email='test@example.com'
        )
        user.set_password('testpassword123')
        db.session.add(user)
        db.session.commit()

        yield user

        # Cleanup
        db.session.delete(user)
        db.session.commit()


@pytest.fixture(scope='function')
def authenticated_client(app, test_user):
    """Create an authenticated test client."""
    client = app.test_client()

    with client.session_transaction() as session:
        session['_user_id'] = test_user.id
        session['_fresh'] = True

    return client


@pytest.fixture(scope='function')
def api_headers(app, test_user):
    """Get headers for API authentication."""
    from flask_jwt_extended import create_access_token

    with app.app_context():
        access_token = create_access_token(identity=test_user.id)

    return {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }


@pytest.fixture(scope='session')
def sample_texts():
    """Sample texts for testing sentiment analysis."""
    return {
        'positive': [
            "I love this product! It's absolutely amazing!",
            "Great experience, highly recommend!",
            "This made my day so much better!",
        ],
        'negative': [
            "This is terrible, worst experience ever.",
            "I hate this, completely disappointed.",
            "Awful service, would not recommend.",
        ],
        'neutral': [
            "The package arrived today.",
            "I went to the store.",
            "The meeting is scheduled for Monday.",
        ]
    }


@pytest.fixture(scope='session')
def sample_csv(tmp_path_factory):
    """Create a sample CSV file for batch testing."""
    import csv

    csv_path = tmp_path_factory.mktemp("data") / "sample.csv"

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Text'])
        writer.writerow(['I love this product!'])
        writer.writerow(['This is terrible.'])
        writer.writerow(['The package arrived.'])

    return str(csv_path)
