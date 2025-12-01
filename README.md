# Mass Feeling Analysis CI/CD

## Overview

This project implements a comprehensive Flask-based sentiment analysis application with real-time streaming capabilities, machine learning models for emotion detection, and distributed processing. It provides REST API endpoints for single-text and batch sentiment analysis, supports CSV file processing via background jobs, includes real-time data streaming from Twitter, Reddit, and news feeds, and features JWT authentication and API key management.

## Features

- **Sentiment Analysis**: Dual ML approach using VADER (fast, rule-based) and Hugging Face transformers (accurate, emotion detection)
- **Real-Time Streaming**: Live sentiment analysis from Twitter API, Reddit, and RSS news feeds
- **REST API**: Endpoints for single text analysis, batch processing, streaming management, and user management
- **Batch Processing**: Asynchronous CSV file processing using Celery and Redis
- **Distributed Processing**: Apache Spark integration for large-scale ETL operations
- **Authentication**: JWT-based authentication with API key support
- **Database**: PostgreSQL with SQLAlchemy ORM and streaming data support
- **Caching**: Redis-backed caching for repeated analyses and pub/sub messaging
- **Monitoring**: Prometheus metrics and Grafana dashboard integration
- **CI/CD**: Jenkins pipeline for automated testing and deployment

## Architecture

- **Backend**: Flask application with factory pattern and real-time WebSocket support
- **ML Models**: VADER (~0.3ms) + Hugging Face emotion model (~40ms)
- **Streaming**: Twitter API v2, Reddit API, RSS feeds with Celery background processing
- **Task Queue**: Celery with Redis for background processing and streaming tasks
- **Database**: PostgreSQL with connection pooling and JSON metadata storage
- **Caching**: Redis for result caching, session storage, and real-time pub/sub
- **Distributed Processing**: Apache Spark for ETL pipelines and big data analytics
- **Monitoring**: Prometheus exporters and Grafana for real-time dashboards

## Prerequisites

- Python 3.11+
- PostgreSQL 14+
- Redis
- Apache Spark (optional, for distributed processing)
- Docker (optional, for containerized deployment)

### API Credentials (for streaming features)

- **Twitter API v2**: Bearer token for real-time tweet streaming
- **Reddit API**: Client ID, Client Secret, and User Agent for Reddit data access
- **News Feeds**: RSS/Atom feed URLs for news aggregation

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Bladerunner-hue/Mass_Feeling_Analysis_CICD.git
    cd Mass_Feeling_Analysis_CICD
    ```

2. Create virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    ```

4. Set up environment variables:

    ```bash
    cp .env.example .env
    # Edit .env with your database credentials, secrets, and API keys
    ```

    **Required for streaming features:**

    ```bash
    # Twitter API (optional)
    TWITTER_BEARER_TOKEN=your_twitter_bearer_token

    # Reddit API (optional)
    REDDIT_CLIENT_ID=your_reddit_client_id
    REDDIT_CLIENT_SECRET=your_reddit_client_secret
    REDDIT_USER_AGENT=SentimentAnalysisBot/1.0

    # Apache Spark (optional)
    SPARK_HOME=/path/to/spark
    ```

5. Initialize database:

    ```bash
    export FLASK_APP=wsgi.py
    flask db init  # If using migrations
    flask seed-db  # Create admin user
    ```

## Usage

### Running Locally

```bash
# Start Redis (if not running)
redis-server

# Start Celery worker (includes streaming tasks)
celery -A celery_worker.celery worker --loglevel=info

# Start Flask app
export FLASK_APP=wsgi.py
flask run
```

### Starting Streaming Services

```bash
# Start Twitter stream (requires TWITTER_BEARER_TOKEN)
python -c "
from app.streams.twitter_stream import stream_manager
stream_id = stream_manager.start_twitter_stream(['python', 'AI', 'sentiment'])
print(f'Started Twitter stream: {stream_id}')
"

# Start Reddit stream (requires Reddit API credentials)
python -c "
from app.streams.twitter_stream import stream_manager
stream_id = stream_manager.start_reddit_stream(['technology', 'programming'])
print(f'Started Reddit stream: {stream_id}')
"

# Start news feed processing
python -c "
from app.streams.twitter_stream import stream_manager
stream_id = stream_manager.start_news_stream([
    'http://feeds.feedburner.com/techcrunch',
    'http://rss.cnn.com/rss/edition.rss'
])
print(f'Started news stream: {stream_id}')
"
```

### API Endpoints

#### Authentication

- `POST /api/auth/login` - User login
- `POST /api/auth/register` - User registration

#### Sentiment Analysis

- `POST /api/v1/analyze` - Single text analysis
- `POST /api/v1/batch` - Batch text analysis
- `POST /api/v1/upload` - Upload CSV for batch processing

#### Streaming Management

- `POST /api/v1/streams/twitter/start` - Start Twitter stream
- `POST /api/v1/streams/reddit/start` - Start Reddit stream
- `POST /api/v1/streams/news/start` - Start news feed processing
- `GET /api/v1/streams/active` - Get active streams
- `DELETE /api/v1/streams/{stream_id}` - Stop specific stream

#### User Management

- `GET /api/users/profile` - Get user profile
- `POST /api/users/api-key` - Generate API key

### Example Usage

```python
import requests

# Single text analysis
response = requests.post('http://localhost:5000/api/v1/analyze',
    json={'text': 'I love this product!'},
    headers={'Authorization': 'Bearer <token>'}
)
print(response.json())
# {
#   'sentiment': 'POSITIVE',
#   'confidence': 0.95,
#   'emotions': {'joy': 0.8, 'trust': 0.6},
#   'processing_time': 0.042
# }

# Start Twitter stream
response = requests.post('http://localhost:5000/api/v1/streams/twitter/start',
    json={'keywords': ['python', 'AI'], 'duration_minutes': 60},
    headers={'Authorization': 'Bearer <token>'}
)
print(response.json())
# {
#   'stream_id': 'twitter_1234567890',
#   'status': 'started',
#   'keywords': ['python', 'AI']
# }

# Get active streams
response = requests.get('http://localhost:5000/api/v1/streams/active',
    headers={'Authorization': 'Bearer <token>'}
)
print(response.json())
# {
#   'streams': [
#     {
#       'id': 'twitter_1234567890',
#       'type': 'twitter',
#       'keywords': ['python', 'AI'],
#       'started_at': '2025-12-01T12:00:00Z'
#     }
#   ]
# }
```

## Project Structure

```text
Mass_Feeling_Analysis_CICD/
├── app/
│   ├── api/           # REST API routes
│   ├── auth/          # Authentication blueprints
│   ├── batch/         # Batch processing tasks
│   ├── errors/        # Error handlers
│   ├── main/          # Main application routes
│   ├── models/        # SQLAlchemy models
│   ├── services/      # Business logic services
│   ├── streams/       # Real-time streaming modules
│   │   ├── __init__.py
│   │   └── twitter_stream.py  # Twitter/Reddit/News streaming
│   ├── static/        # Static assets
│   └── templates/     # Jinja2 templates
├── config/            # Configuration classes
├── tests/             # Unit and integration tests
├── migrations/        # Database migrations
├── requirements.txt   # Python dependencies
├── requirements-dev.txt # Development dependencies
├── wsgi.py           # WSGI entry point
├── celery_worker.py  # Celery worker startup
├── test_streaming.py # Streaming infrastructure tests
├── Dockerfile        # Docker configuration
├── docker-compose.yml # Local development setup
├── Jenkinsfile       # CI/CD pipeline
└── README.md
```

## Configuration

Key environment variables:

```bash
# Flask
FLASK_APP=wsgi.py
FLASK_ENV=development

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/dbname

# JWT
JWT_SECRET_KEY=your-jwt-secret

# Redis
REDIS_URL=redis://localhost:6379/0

# Celery
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Streaming APIs (optional)
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=SentimentAnalysisBot/1.0

# Apache Spark (optional)
SPARK_HOME=/path/to/spark
PYSPARK_PYTHON=python3

# Monitoring (optional)
PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus
GRAFANA_URL=http://localhost:3000
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_api.py
```

## Deployment

### Docker

```bash
# Build and run
docker-compose up --build
```

### Jenkins CI/CD

The project includes a Jenkins pipeline (`Jenkinsfile`) that:

- Runs automated tests
- Performs security scanning
- Builds Docker images
- Deploys to staging/production

## Performance

- **VADER Analysis**: ~0.3ms per text (rule-based, no GPU required)
- **Emotion Analysis**: ~40ms per text (transformer model, GPU recommended)
- **Batch Processing**: Asynchronous with progress tracking
- **Streaming Processing**: Real-time analysis with Redis pub/sub distribution
- **Caching**: Redis-backed for repeated text analysis
- **Distributed Processing**: Apache Spark for large-scale ETL operations

## Custom PyTorch Model & Spark Pipeline

- BiLSTM + Attention model implemented under `app/ml/` with FP16 support for GPUs (Quadro RTX 5000 friendly).
- Optional Spark preprocessing/training helpers in `app/ml/spark` for TorchDistributor + Petastorm workflows.
- New Flask endpoint `/analysis/custom/analyze` uses the custom model when `CUSTOM_MODEL_PATH`/`CUSTOM_TOKENIZER_PATH` are present; falls back to the transformer otherwise.
- Optional heavy dependencies live in `requirements-ml.txt` to keep the base install lightweight.
- Configure with `.env`: `CUSTOM_MODEL_PATH`, `CUSTOM_TOKENIZER_PATH`, `CUSTOM_MODEL_DEVICE`, `TRAINING_DB_URL`, `SPARK_MASTER`.

## Dataset Export Helpers (Hugging Face + Kaggle)

- Export a Hugging Face dataset to Parquet:  
  `python scripts/export_hf_dataset.py --dataset carblacac/twitter-sentiment-analysis --split train --text-field text --sentiment-field sentiment --default-emotion neutral`

- Export a Kaggle dataset to Parquet (requires `~/.kaggle/kaggle.json`):  
  `python scripts/export_kaggle_dataset.py --dataset kazanova/sentiment140 --text-field text --sentiment-field sentiment --encoding latin-1`

Outputs land in `data/raw/hf/` or `data/raw/kaggle/` and are ready for Spark preprocessing or direct PyTorch training.

### Streaming Performance

- **Twitter API**: Real-time tweet processing with configurable keyword filtering
- **Reddit API**: Continuous subreddit monitoring with sentiment aggregation
- **News Feeds**: RSS/Atom feed polling with article extraction and analysis
- **Real-time Updates**: WebSocket/SSE support for live dashboard updates

## Real-Time Streaming Features

The application includes comprehensive real-time streaming capabilities for continuous sentiment analysis:

### Twitter Streaming

- **API Integration**: Twitter API v2 with filtered streaming
- **Keyword Filtering**: Real-time monitoring of specified keywords/hashtags
- **Data Collection**: Tweet text, metadata, engagement metrics
- **Sentiment Analysis**: Automatic analysis of all collected tweets

### Reddit Streaming

- **Subreddit Monitoring**: Continuous monitoring of specified subreddits
- **Content Analysis**: Posts and comments sentiment analysis
- **Community Insights**: Real-time sentiment trends by subreddit

### News Feed Processing

- **RSS/Atom Support**: Automated feed parsing and article extraction
- **Content Analysis**: Full article text extraction and sentiment analysis
- **Source Aggregation**: Multi-feed processing with source attribution

### Real-Time Architecture

- **Celery Tasks**: Background processing for all streaming operations
- **Redis Pub/Sub**: Real-time data distribution for dashboards
- **WebSocket Support**: Live updates for connected clients
- **Database Storage**: Persistent storage of all streaming analyses

### Monitoring & Analytics

- **Prometheus Metrics**: Performance monitoring and alerting
- **Grafana Integration**: Real-time dashboards and visualizations
- **Stream Management**: API endpoints for starting/stopping streams
- **Health Checks**: Stream status monitoring and error reporting

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please open an issue on GitHub.
