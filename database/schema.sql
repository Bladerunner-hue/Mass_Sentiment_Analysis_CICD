-- PostgreSQL Schema for Mass Sentiment Analysis
-- Spark 4.0 Streaming + Custom BiLSTM Model Architecture
-- 
-- This schema supports:
-- 1. Real-time streaming results from Spark Structured Streaming
-- 2. Training data management for custom PyTorch models
-- 3. Model versioning and experiment tracking
-- 4. API credentials secure storage

-- ============================================================================
-- CORE STREAMING TABLES
-- ============================================================================

-- Streaming sentiment results (Spark writes here)
CREATE TABLE IF NOT EXISTS streaming_sentiment_results (
    id BIGSERIAL PRIMARY KEY,
    tweet_id VARCHAR(64) NOT NULL,
    text TEXT NOT NULL,
    author_id VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    retweet_count BIGINT DEFAULT 0,
    like_count BIGINT DEFAULT 0,
    
    -- Sentiment analysis results
    sentiment VARCHAR(20) NOT NULL,
    compound_score FLOAT NOT NULL,
    primary_emotion VARCHAR(30),
    confidence FLOAT,
    model_used VARCHAR(50),
    
    -- Metadata
    keywords TEXT,
    source VARCHAR(20) NOT NULL DEFAULT 'twitter',
    
    -- Indexing
    CONSTRAINT uq_streaming_tweet UNIQUE (tweet_id, source)
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_streaming_sentiment ON streaming_sentiment_results(sentiment);
CREATE INDEX IF NOT EXISTS idx_streaming_emotion ON streaming_sentiment_results(primary_emotion);
CREATE INDEX IF NOT EXISTS idx_streaming_source ON streaming_sentiment_results(source);
CREATE INDEX IF NOT EXISTS idx_streaming_created_at ON streaming_sentiment_results(created_at);
CREATE INDEX IF NOT EXISTS idx_streaming_source_date ON streaming_sentiment_results(source, created_at);

-- ============================================================================
-- TRAINING DATA MANAGEMENT
-- ============================================================================

-- Training datasets registry
CREATE TABLE IF NOT EXISTS training_datasets (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    source VARCHAR(50) NOT NULL,  -- 'huggingface', 'kaggle', 'twitter_stream', 'manual'
    
    -- Dataset statistics
    total_samples BIGINT DEFAULT 0,
    train_samples BIGINT DEFAULT 0,
    validation_samples BIGINT DEFAULT 0,
    test_samples BIGINT DEFAULT 0,
    
    -- Class distribution
    positive_samples BIGINT DEFAULT 0,
    negative_samples BIGINT DEFAULT 0,
    neutral_samples BIGINT DEFAULT 0,
    
    -- File locations (Parquet)
    parquet_path TEXT NOT NULL,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Individual training samples (for fine-tuning)
CREATE TABLE IF NOT EXISTS training_samples (
    id BIGSERIAL PRIMARY KEY,
    dataset_id INTEGER REFERENCES training_datasets(id) ON DELETE CASCADE,
    
    -- Text content
    text TEXT NOT NULL,
    text_hash VARCHAR(32) NOT NULL,  -- MD5 hash for deduplication
    
    -- Labels
    sentiment VARCHAR(20),
    primary_emotion VARCHAR(30),
    
    -- Multi-label support
    emotion_scores JSONB,
    
    -- Metadata
    split VARCHAR(20) DEFAULT 'train',  -- train/validation/test
    source_id VARCHAR(100),  -- Original ID from source
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT uq_sample_hash UNIQUE (text_hash)
);

CREATE INDEX IF NOT EXISTS idx_training_samples_dataset ON training_samples(dataset_id);
CREATE INDEX IF NOT EXISTS idx_training_samples_split ON training_samples(split);
CREATE INDEX IF NOT EXISTS idx_training_samples_sentiment ON training_samples(sentiment);

-- ============================================================================
-- MODEL VERSIONING & EXPERIMENT TRACKING
-- ============================================================================

-- Model registry
CREATE TABLE IF NOT EXISTS models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,
    architecture VARCHAR(50) NOT NULL,  -- 'bilstm_attention', 'transformer', etc.
    
    -- Model paths
    model_path TEXT NOT NULL,
    tokenizer_path TEXT,
    config_path TEXT,
    
    -- Performance metrics
    accuracy FLOAT,
    f1_macro FLOAT,
    f1_weighted FLOAT,
    loss FLOAT,
    
    -- Training configuration
    hyperparameters JSONB,
    training_dataset_id INTEGER REFERENCES training_datasets(id),
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_production BOOLEAN DEFAULT FALSE,
    notes TEXT,
    
    CONSTRAINT uq_model_version UNIQUE (name, version)
);

-- Training experiments
CREATE TABLE IF NOT EXISTS training_experiments (
    id SERIAL PRIMARY KEY,
    experiment_name VARCHAR(100) NOT NULL,
    model_id INTEGER REFERENCES models(id),
    dataset_id INTEGER REFERENCES training_datasets(id),
    
    -- Training configuration
    epochs INTEGER NOT NULL,
    batch_size INTEGER NOT NULL,
    learning_rate FLOAT NOT NULL,
    use_fp16 BOOLEAN DEFAULT FALSE,
    optimizer VARCHAR(50) DEFAULT 'AdamW',
    scheduler VARCHAR(50),
    
    -- Spark configuration
    spark_master VARCHAR(100),
    num_executors INTEGER,
    executor_memory VARCHAR(20),
    
    -- Status
    status VARCHAR(20) DEFAULT 'pending',  -- pending/running/completed/failed
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Results
    final_train_loss FLOAT,
    final_val_loss FLOAT,
    best_epoch INTEGER,
    
    -- Logs
    log_path TEXT,
    error_message TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Epoch-level metrics
CREATE TABLE IF NOT EXISTS training_metrics (
    id BIGSERIAL PRIMARY KEY,
    experiment_id INTEGER REFERENCES training_experiments(id) ON DELETE CASCADE,
    epoch INTEGER NOT NULL,
    
    -- Metrics
    train_loss FLOAT,
    val_loss FLOAT,
    train_accuracy FLOAT,
    val_accuracy FLOAT,
    train_f1 FLOAT,
    val_f1 FLOAT,
    
    -- Timing
    epoch_duration_seconds FLOAT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_training_metrics_experiment ON training_metrics(experiment_id);

-- ============================================================================
-- API CREDENTIALS (Secure Storage)
-- ============================================================================

-- API credentials storage
CREATE TABLE IF NOT EXISTS api_credentials (
    id SERIAL PRIMARY KEY,
    service_name VARCHAR(50) NOT NULL,  -- 'twitter', 'reddit', 'newsapi', etc.
    credential_key VARCHAR(100) NOT NULL,  -- 'bearer_token', 'client_id', etc.
    credential_value TEXT NOT NULL,  -- Encrypted value
    encrypted BOOLEAN DEFAULT TRUE,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE,
    
    CONSTRAINT uq_credential UNIQUE (service_name, credential_key)
);

-- ============================================================================
-- STREAMING AGGREGATIONS (Pre-computed for dashboards)
-- ============================================================================

-- Hourly sentiment aggregations
CREATE TABLE IF NOT EXISTS hourly_sentiment_stats (
    id BIGSERIAL PRIMARY KEY,
    source VARCHAR(20) NOT NULL,
    hour_bucket TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Counts
    total_count BIGINT DEFAULT 0,
    positive_count BIGINT DEFAULT 0,
    negative_count BIGINT DEFAULT 0,
    neutral_count BIGINT DEFAULT 0,
    
    -- Averages
    avg_compound_score FLOAT,
    avg_confidence FLOAT,
    
    -- Emotion distribution
    emotion_counts JSONB,
    
    -- Top keywords
    top_keywords JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT uq_hourly_stats UNIQUE (source, hour_bucket)
);

CREATE INDEX IF NOT EXISTS idx_hourly_stats_source ON hourly_sentiment_stats(source);
CREATE INDEX IF NOT EXISTS idx_hourly_stats_bucket ON hourly_sentiment_stats(hour_bucket);

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- Real-time dashboard view
CREATE OR REPLACE VIEW v_realtime_sentiment AS
SELECT 
    source,
    sentiment,
    primary_emotion,
    COUNT(*) as count,
    AVG(compound_score) as avg_compound,
    AVG(confidence) as avg_confidence,
    MAX(created_at) as last_update
FROM streaming_sentiment_results
WHERE created_at >= NOW() - INTERVAL '1 hour'
GROUP BY source, sentiment, primary_emotion
ORDER BY count DESC;

-- Model performance comparison view
CREATE OR REPLACE VIEW v_model_comparison AS
SELECT 
    m.name,
    m.version,
    m.architecture,
    m.accuracy,
    m.f1_macro,
    m.f1_weighted,
    m.is_production,
    d.name as dataset_name,
    d.total_samples
FROM models m
LEFT JOIN training_datasets d ON m.training_dataset_id = d.id
ORDER BY m.f1_weighted DESC NULLS LAST;

-- ============================================================================
-- FUNCTIONS FOR AGGREGATION
-- ============================================================================

-- Function to update hourly stats (call from Spark or cron)
CREATE OR REPLACE FUNCTION update_hourly_sentiment_stats(
    p_source VARCHAR(20),
    p_hour TIMESTAMP WITH TIME ZONE
) RETURNS VOID AS $$
BEGIN
    INSERT INTO hourly_sentiment_stats (
        source, 
        hour_bucket,
        total_count,
        positive_count,
        negative_count,
        neutral_count,
        avg_compound_score,
        avg_confidence,
        emotion_counts
    )
    SELECT 
        source,
        date_trunc('hour', created_at) as hour_bucket,
        COUNT(*) as total_count,
        SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) as positive_count,
        SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) as negative_count,
        SUM(CASE WHEN sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral_count,
        AVG(compound_score) as avg_compound_score,
        AVG(confidence) as avg_confidence,
        jsonb_object_agg(
            COALESCE(primary_emotion, 'unknown'),
            emotion_cnt
        ) as emotion_counts
    FROM (
        SELECT 
            source,
            created_at,
            sentiment,
            compound_score,
            confidence,
            primary_emotion,
            COUNT(*) OVER (PARTITION BY primary_emotion) as emotion_cnt
        FROM streaming_sentiment_results
        WHERE source = p_source
          AND date_trunc('hour', created_at) = p_hour
    ) subq
    GROUP BY source, date_trunc('hour', created_at)
    ON CONFLICT (source, hour_bucket) 
    DO UPDATE SET
        total_count = EXCLUDED.total_count,
        positive_count = EXCLUDED.positive_count,
        negative_count = EXCLUDED.negative_count,
        neutral_count = EXCLUDED.neutral_count,
        avg_compound_score = EXCLUDED.avg_compound_score,
        avg_confidence = EXCLUDED.avg_confidence,
        emotion_counts = EXCLUDED.emotion_counts,
        created_at = CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- INITIAL DATA SETUP
-- ============================================================================

-- Insert default training dataset entry
INSERT INTO training_datasets (name, description, source, parquet_path)
VALUES (
    'initial_dataset',
    'Initial training dataset from HuggingFace/Kaggle',
    'huggingface',
    'data/processed/train'
) ON CONFLICT (name) DO NOTHING;

-- Insert placeholder model
INSERT INTO models (name, version, architecture, model_path, tokenizer_path, is_production)
VALUES (
    'bilstm_attention',
    '1.0.0',
    'bilstm_attention',
    'models/checkpoints/bilstm_attention.pt',
    'models/tokenizer.pkl',
    TRUE
) ON CONFLICT (name, version) DO NOTHING;
