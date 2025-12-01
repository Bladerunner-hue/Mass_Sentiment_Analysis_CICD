# 🏗️ Architecture Decision Record

## Mass Sentiment Analysis CI/CD Project

**Last Updated:** December 1, 2025  
**Repository:** `Bladerunner-hue/Mass_Feeling_Analysis_CICD`  
**Branch:** `features`

---

## 📋 Executive Summary

This document captures the key architectural decisions for the Mass Sentiment Analysis platform. The system uses **Spark 4.0 Structured Streaming** as the primary streaming framework, **PostgreSQL** for data persistence, and **custom BiLSTM + Attention models** for sentiment analysis.

### Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Streaming** | Spark 4.0 Structured Streaming | Native streaming, no external dependencies (NO KAFKA) |
| **Database** | PostgreSQL | ACID compliance, JSONB support, excellent for time-series |
| **ML Model** | Custom BiLSTM + Attention | Better accuracy than generic transformers for sentiment |
| **GPU** | NVIDIA Quadro RTX 5000 | FP16 training, fast inference |
| **Queue** | Redis | Celery broker, pub/sub for real-time dashboards |
| **Fallback Model** | HuggingFace Transformers | Graceful degradation when custom model unavailable |

---

## 🚫 Explicitly NOT Using

The following technologies were **intentionally excluded**:

| Technology | Reason |
|------------|--------|
| ❌ **Kafka** | Overkill for our volume; Spark handles streaming natively |
| ❌ **Docker** | Development simplicity; containerize only for production |
| ❌ **Kubernetes** | Single-node deployment sufficient |
| ❌ **Elasticsearch** | PostgreSQL full-text search adequate |

---

## 🔄 Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SPARK 4.0 STRUCTURED STREAMING                    │
│                         (No Kafka Required)                              │
└─────────────────────────────────────────────────────────────────────────┘

     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
     │ Twitter API  │     │  Reddit API  │     │  RSS Feeds   │
     │    v2        │     │    PRAW      │     │  feedparser  │
     └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
            │                    │                    │
            ▼                    ▼                    ▼
     ┌─────────────────────────────────────────────────────────┐
     │              Spark Structured Streaming                  │
     │     ┌─────────────────────────────────────────────┐     │
     │     │         Custom Sentiment UDF                 │     │
     │     │    ┌─────────────────────────────────┐      │     │
     │     │    │  BiLSTM + Attention (Primary)   │      │     │
     │     │    │  ──────────────────────────────│      │     │
     │     │    │  HuggingFace (Fallback)        │      │     │
     │     │    └─────────────────────────────────┘      │     │
     │     └─────────────────────────────────────────────┘     │
     └──────────────────────────┬──────────────────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
     ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
     │  PostgreSQL  │   │   Parquet    │   │    Redis     │
     │ (Real-time)  │   │ (Historical) │   │  (Pub/Sub)   │
     └──────────────┘   └──────────────┘   └──────────────┘
            │                   │                   │
            └───────────────────┼───────────────────┘
                                ▼
                    ┌──────────────────────┐
                    │    Flask Dashboard   │
                    │    /api endpoints    │
                    └──────────────────────┘
```

---

## 📁 Project Structure

```
Mass_Feeling_Analysis_CICD/
├── app/
│   ├── main/                  # Flask blueprints & routes
│   ├── models/                # SQLAlchemy models
│   ├── services/              # Business logic
│   │   ├── sentiment_service.py    # Main service (custom model primary)
│   │   ├── custom_sentiment_service.py  # BiLSTM + Attention wrapper
│   │   └── batch_service.py        # Batch processing
│   ├── streams/               # Streaming (Tweepy for Flask routes)
│   │   └── twitter_stream.py       # Twitter/Reddit/News streaming
│   └── ml/
│       ├── models/            # PyTorch model definitions
│       ├── preprocessing/     # Tokenizer, text cleaning
│       ├── training/          # Training loops
│       ├── inference/         # Model serving
│       └── spark/             # Spark 4.0 integration
│           ├── streaming.py        # ✅ NEW: Spark Structured Streaming
│           ├── jobs.py             # Batch ETL jobs
│           └── data_processor.py   # Data preprocessing
├── database/
│   └── schema.sql             # ✅ NEW: PostgreSQL schema
├── data/
│   ├── raw/                   # Raw Parquet datasets
│   ├── processed/             # Training-ready data
│   ├── checkpoints/           # Spark checkpoints
│   └── streaming/             # Streaming output
├── models/
│   └── checkpoints/           # Trained model weights
├── scripts/                   # Data collection scripts
├── tests/                     # pytest test suite
└── config/                    # Configuration
```

---

## 🔧 Key Components

### 1. Spark 4.0 Structured Streaming (`app/ml/spark/streaming.py`)

**Purpose:** Real-time sentiment analysis at scale

**Features:**
- Micro-batch processing (10-second intervals)
- Watermark support for late data
- Dual sink: PostgreSQL + Parquet
- Custom UDF with BiLSTM model

**Usage:**
```python
from app.ml.spark.streaming import streaming_service

# Start Twitter stream
stream_id = streaming_service.start_twitter_stream(
    bearer_token=os.getenv("TWITTER_BEARER_TOKEN"),
    keywords=["python", "AI", "machine learning"]
)

# Start Reddit stream
stream_id = streaming_service.start_reddit_stream(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent="SentimentBot/1.0",
    subreddits=["machinelearning", "datascience"]
)

# Monitor
status = streaming_service.get_stream_status(stream_id)
```

### 2. Custom Sentiment Service (`app/services/sentiment_service.py`)

**Priority Order:**
1. **Custom BiLSTM + Attention** (if model files exist)
2. **HuggingFace Transformer** (fallback)
3. **VADER** (quick analysis, always available)

**Configuration:**
```bash
# .env
CUSTOM_MODEL_PATH=models/checkpoints/bilstm_attention.pt
CUSTOM_TOKENIZER_PATH=models/tokenizer.pkl
CUSTOM_MODEL_DEVICE=cuda
```

### 3. PostgreSQL Schema (`database/schema.sql`)

**Tables:**
- `streaming_sentiment_results` - Real-time streaming output
- `training_datasets` - Dataset registry
- `training_samples` - Individual training samples
- `models` - Model versioning
- `training_experiments` - Experiment tracking
- `api_credentials` - Secure credential storage
- `hourly_sentiment_stats` - Pre-aggregated metrics

---

## ⚙️ Environment Configuration

### Required Environment Variables

```bash
# PostgreSQL
DATABASE_URL=postgresql://user:pass@localhost:5432/dbname
DB_NAME=EffuzionBridge
DB_USER=EffuzionBridge
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432

# Spark 4.0 Streaming
SPARK_MASTER=local[*]
SPARK_EXECUTOR_MEMORY=32g
SPARK_DRIVER_MEMORY=8g
SPARK_STREAMING_CHECKPOINT=data/checkpoints/streaming
SPARK_STREAMING_OUTPUT=data/streaming/output
SPARK_TRIGGER_INTERVAL=10 seconds
SPARK_WATERMARK_DELAY=1 minute

# Custom Model
CUSTOM_MODEL_PATH=models/checkpoints/bilstm_attention.pt
CUSTOM_TOKENIZER_PATH=models/tokenizer.pkl
CUSTOM_MODEL_DEVICE=cuda

# FP16 Training
USE_FP16=true
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# APIs
TWITTER_BEARER_TOKEN=your_token
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
REDDIT_USER_AGENT=SentimentBot/1.0

# Redis
REDIS_URL=redis://localhost:6379/1
```

---

## 🚀 Deployment

### Development
```bash
# 1. Set up PostgreSQL
psql -U postgres -c "CREATE DATABASE EffuzionBridge;"
psql -U postgres -d EffuzionBridge -f database/schema.sql

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Flask
flask run

# 4. Start Spark streaming (separate terminal)
python -c "
from app.ml.spark.streaming import start_all_streams
start_all_streams(
    twitter_keywords=['AI', 'python'],
    subreddits=['datascience'],
    rss_feeds=['https://feeds.bbci.co.uk/news/technology/rss.xml']
)
"
```

### Production
```bash
# Use Spark cluster
export SPARK_MASTER=spark://master:7077

# Use PostgreSQL connection pool
export DATABASE_URL=postgresql://user:pass@db-cluster:5432/sentiment?pool_size=20
```

---

## 📊 Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Tweets/hour | > 100,000 | TBD |
| Latency (p99) | < 100ms | TBD |
| Model accuracy | > 85% | TBD |
| FP16 speedup | 2x | TBD |

---

## 🔮 Future Enhancements

1. **Kubernetes deployment** for auto-scaling
2. **MLflow integration** for experiment tracking
3. **Grafana dashboards** for monitoring
4. **A/B testing** for model comparison
5. **Active learning** pipeline for continuous improvement

---

## 📝 Change Log

| Date | Change | Author |
|------|--------|--------|
| 2025-12-01 | Initial architecture with Spark 4.0 Streaming | AI Assistant |
| 2025-12-01 | Removed Kafka dependency | AI Assistant |
| 2025-12-01 | Added PostgreSQL schema | AI Assistant |
| 2025-12-01 | Custom BiLSTM + Attention integration | AI Assistant |
