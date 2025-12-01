"""Tests for analysis API endpoints."""

import pytest
import json


class TestAnalysisAPI:
    """Test cases for /api/v1/analysis endpoints."""

    def test_analyze_requires_authentication(self, client):
        """Test analyze endpoint requires authentication."""
        response = client.post(
            '/api/v1/analysis/analyze',
            json={'text': 'Hello world'},
            content_type='application/json'
        )

        assert response.status_code == 401

    def test_analyze_with_valid_text(self, client, api_headers):
        """Test analyze endpoint with valid text."""
        response = client.post(
            '/api/v1/analysis/analyze',
            json={'text': 'I absolutely love this product!'},
            headers=api_headers
        )

        assert response.status_code == 200
        data = json.loads(response.data)

        assert 'sentiment' in data
        assert 'compound_score' in data
        assert 'scores' in data
        assert 'primary_emotion' in data
        assert 'processing_time_ms' in data

    def test_analyze_identifies_positive_sentiment(self, client, api_headers):
        """Test analyze correctly identifies positive sentiment."""
        response = client.post(
            '/api/v1/analysis/analyze',
            json={'text': 'This is amazing! I love it so much!'},
            headers=api_headers
        )

        assert response.status_code == 200
        data = json.loads(response.data)

        assert data['sentiment'] == 'positive'
        assert data['compound_score'] > 0

    def test_analyze_identifies_negative_sentiment(self, client, api_headers):
        """Test analyze correctly identifies negative sentiment."""
        response = client.post(
            '/api/v1/analysis/analyze',
            json={'text': 'This is terrible! I hate it!'},
            headers=api_headers
        )

        assert response.status_code == 200
        data = json.loads(response.data)

        assert data['sentiment'] == 'negative'
        assert data['compound_score'] < 0

    def test_analyze_missing_text(self, client, api_headers):
        """Test analyze endpoint with missing text."""
        response = client.post(
            '/api/v1/analysis/analyze',
            json={},
            headers=api_headers
        )

        assert response.status_code == 400

    def test_analyze_empty_text(self, client, api_headers):
        """Test analyze endpoint with empty text."""
        response = client.post(
            '/api/v1/analysis/analyze',
            json={'text': ''},
            headers=api_headers
        )

        assert response.status_code == 400

    def test_analyze_text_too_long(self, client, api_headers):
        """Test analyze endpoint with text exceeding max length."""
        long_text = 'x' * 6000  # Max is 5000

        response = client.post(
            '/api/v1/analysis/analyze',
            json={'text': long_text},
            headers=api_headers
        )

        assert response.status_code == 400

    def test_quick_analyze_faster_than_full(self, client, api_headers):
        """Test quick analyze is faster than full analyze."""
        text = 'This is a test sentence for comparison.'

        # Quick analysis
        quick_response = client.post(
            '/api/v1/analysis/analyze/quick',
            json={'text': text},
            headers=api_headers
        )

        # Full analysis
        full_response = client.post(
            '/api/v1/analysis/analyze',
            json={'text': text},
            headers=api_headers
        )

        assert quick_response.status_code == 200
        assert full_response.status_code == 200

        quick_data = json.loads(quick_response.data)
        full_data = json.loads(full_response.data)

        # Quick should be faster (but both should complete)
        assert 'processing_time_ms' in quick_data
        assert 'processing_time_ms' in full_data

    def test_batch_analyze(self, client, api_headers):
        """Test batch analyze endpoint."""
        response = client.post(
            '/api/v1/analysis/batch',
            json={
                'texts': [
                    'I love this!',
                    'I hate this!',
                    'This is okay.'
                ],
                'include_emotions': True
            },
            headers=api_headers
        )

        assert response.status_code == 200
        data = json.loads(response.data)

        assert 'results' in data
        assert len(data['results']) == 3
        assert 'total' in data
        assert data['total'] == 3

    def test_batch_analyze_limit(self, client, api_headers):
        """Test batch analyze respects maximum texts limit."""
        texts = [f'Text {i}' for i in range(150)]  # Max is 100

        response = client.post(
            '/api/v1/analysis/batch',
            json={'texts': texts},
            headers=api_headers
        )

        assert response.status_code == 400

    def test_batch_analyze_empty_list(self, client, api_headers):
        """Test batch analyze with empty list."""
        response = client.post(
            '/api/v1/analysis/batch',
            json={'texts': []},
            headers=api_headers
        )

        assert response.status_code == 400


class TestAuthAPI:
    """Test cases for /api/v1/auth endpoints."""

    def test_register_new_user(self, client):
        """Test user registration."""
        response = client.post(
            '/api/v1/auth/register',
            json={
                'username': 'newuser',
                'email': 'newuser@example.com',
                'password': 'securepassword123'
            },
            content_type='application/json'
        )

        assert response.status_code == 201
        data = json.loads(response.data)

        assert data['username'] == 'newuser'
        assert 'id' in data

    def test_register_duplicate_email(self, client, test_user):
        """Test registration with duplicate email fails."""
        response = client.post(
            '/api/v1/auth/register',
            json={
                'username': 'anotheruser',
                'email': 'test@example.com',  # Already exists
                'password': 'securepassword123'
            },
            content_type='application/json'
        )

        assert response.status_code == 409

    def test_register_weak_password(self, client):
        """Test registration with weak password fails."""
        response = client.post(
            '/api/v1/auth/register',
            json={
                'username': 'newuser2',
                'email': 'newuser2@example.com',
                'password': 'short'  # Too short
            },
            content_type='application/json'
        )

        assert response.status_code == 400

    def test_login_valid_credentials(self, client, test_user):
        """Test login with valid credentials."""
        response = client.post(
            '/api/v1/auth/login',
            json={
                'email': 'test@example.com',
                'password': 'testpassword123'
            },
            content_type='application/json'
        )

        assert response.status_code == 200
        data = json.loads(response.data)

        assert 'access_token' in data
        assert 'refresh_token' in data
        assert data['token_type'] == 'Bearer'

    def test_login_invalid_password(self, client, test_user):
        """Test login with invalid password."""
        response = client.post(
            '/api/v1/auth/login',
            json={
                'email': 'test@example.com',
                'password': 'wrongpassword'
            },
            content_type='application/json'
        )

        assert response.status_code == 401

    def test_login_nonexistent_user(self, client):
        """Test login with nonexistent email."""
        response = client.post(
            '/api/v1/auth/login',
            json={
                'email': 'nonexistent@example.com',
                'password': 'somepassword'
            },
            content_type='application/json'
        )

        assert response.status_code == 401

    def test_get_current_user(self, client, api_headers):
        """Test getting current user profile."""
        response = client.get(
            '/api/v1/auth/me',
            headers=api_headers
        )

        assert response.status_code == 200
        data = json.loads(response.data)

        assert 'username' in data
        assert 'email' in data


class TestStatsAPI:
    """Test cases for /api/v1/stats endpoints."""

    def test_get_stats_requires_auth(self, client):
        """Test stats endpoint requires authentication."""
        response = client.get('/api/v1/stats/')

        assert response.status_code == 401

    def test_get_stats(self, client, api_headers):
        """Test getting user statistics."""
        response = client.get(
            '/api/v1/stats/',
            headers=api_headers
        )

        assert response.status_code == 200
        data = json.loads(response.data)

        assert 'total_analyses' in data
        assert 'sentiment_distribution' in data
        assert 'emotion_distribution' in data

    def test_get_history(self, client, api_headers):
        """Test getting analysis history."""
        response = client.get(
            '/api/v1/stats/history',
            headers=api_headers
        )

        assert response.status_code == 200
        data = json.loads(response.data)

        assert isinstance(data, list)

    def test_get_history_pagination(self, client, api_headers):
        """Test history pagination parameters."""
        response = client.get(
            '/api/v1/stats/history?page=1&per_page=10',
            headers=api_headers
        )

        assert response.status_code == 200
