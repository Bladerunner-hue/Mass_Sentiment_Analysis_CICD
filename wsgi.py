"""WSGI entry point for production deployment.

This module provides the WSGI application instance for production
deployment with Gunicorn or other WSGI servers.

Usage:
    gunicorn wsgi:app -w 4 -b 0.0.0.0:5000

    Or for development:
    python wsgi.py
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from app import create_app

# Create the application instance
app = create_app(os.environ.get('FLASK_CONFIG', 'development'))

if __name__ == '__main__':
    # Development server
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'

    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
