-- PostgreSQL Schema for Spark Streaming and Training Data
-- Optimized for high-volume inserts and real-time queries

-- Extension for partitioning
CREATE EXTENSION IF NOT EXISTS pg_partman;

-- ============================================================================
-- TWITTER STREAMING TABLES
-- ============================================================================

-- Main table for Twitter sentiment results
CREATE TABLE IF NOT EXISTS twitter_sentiments (
    id BIGSERIAL,
    tweet_id VARCHAR(50) NOT NULL,
    text TEXT NOT NULL,
    user_id VARCHAR(50),
    created_at TIMESTAMP NOT NULL,
    processed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Tweet metadata
    lang VARCHAR(10),
    retweet_count INTEGER DEFAULT 0,
    favorite_count INTEGER DEFAULT 0,
    hashtags TEXT,
    
    -- Sentiment analysis results
    sentiment VARCHAR(20) NOT NULL,
    sentiment_confidence FLOAT,
    primary_emotion VARCHAR(20),
    emotion_confidence FLOAT,
    sentiment_scores TEXT,  -- JSON string
    emotion_scores TEXT,    -- JSON string
    
    -- Partitioning key
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Create partitions for current and next month
CREATE TABLE IF NOT EXISTS twitter_sentiments_2025_12 PARTITION OF twitter_sentiments
    FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');

CREATE TABLE IF NOT EXISTS twitter_sentiments_2026_01 PARTITION OF twitter_sentiments
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_twitter_sentiments_tweet_id ON twitter_sentiments(tweet_id);
CREATE INDEX IF NOT EXISTS idx_twitter_sentiments_created_at ON twitter_sentiments(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_twitter_sentiments_sentiment ON twitter_sentiments(sentiment);
CREATE INDEX IF NOT EXISTS idx_twitter_sentiments_emotion ON twitter_sentiments(primary_emotion);
CREATE INDEX IF NOT EXISTS idx_twitter_sentiments_user ON twitter_sentiments(user_id);

-- Composite index for trending queries
CREATE INDEX IF NOT EXISTS idx_twitter_sentiments_time_sentiment 
    ON twitter_sentiments(created_at DESC, sentiment);


-- ============================================================================
-- AGGREGATION TABLES
-- ============================================================================

-- Real-time sentiment trends (5-minute windows)
CREATE TABLE IF NOT EXISTS sentiment_trends (
    id BIGSERIAL PRIMARY KEY,
    window_start TIMESTAMP NOT NULL,
    window_end TIMESTAMP NOT NULL,
    sentiment VARCHAR(20) NOT NULL,
    tweet_count BIGINT NOT NULL,
    avg_confidence FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(window_start, window_end, sentiment)
);

CREATE INDEX IF NOT EXISTS idx_sentiment_trends_window ON sentiment_trends(window_start DESC, window_end DESC);
CREATE INDEX IF NOT EXISTS idx_sentiment_trends_sentiment ON sentiment_trends(sentiment);


-- Hourly sentiment aggregations (materialized view)
CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_sentiment_stats AS
SELECT 
    DATE_TRUNC('hour', created_at) as hour,
    sentiment,
    COUNT(*) as tweet_count,
    AVG(sentiment_confidence) as avg_confidence,
    COUNT(DISTINCT user_id) as unique_users,
    SUM(retweet_count::INTEGER) as total_retweets,
    SUM(favorite_count::INTEGER) as total_favorites
FROM twitter_sentiments
GROUP BY DATE_TRUNC('hour', created_at), sentiment;

CREATE UNIQUE INDEX IF NOT EXISTS idx_hourly_sentiment_stats ON hourly_sentiment_stats(hour, sentiment);

-- Refresh hourly stats (run every hour)
-- REFRESH MATERIALIZED VIEW CONCURRENTLY hourly_sentiment_stats;


-- Daily emotion distribution
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_emotion_distribution AS
SELECT 
    DATE_TRUNC('day', created_at) as day,
    primary_emotion,
    COUNT(*) as emotion_count,
    AVG(emotion_confidence) as avg_confidence
FROM twitter_sentiments
GROUP BY DATE_TRUNC('day', created_at), primary_emotion;

CREATE UNIQUE INDEX IF NOT EXISTS idx_daily_emotion_distribution ON daily_emotion_distribution(day, primary_emotion);


-- ============================================================================
-- TRAINING DATA TABLES (Enhanced from previous schema)
-- ============================================================================

-- Training texts with validation
CREATE TABLE IF NOT EXISTS training_texts (
    id BIGSERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    sentiment VARCHAR(20) NOT NULL CHECK (sentiment IN ('positive', 'negative', 'neutral')),
    emotion VARCHAR(20) NOT NULL CHECK (emotion IN ('anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise')),
    source VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    validated BOOLEAN DEFAULT FALSE,
    split VARCHAR(10) CHECK (split IN ('train', 'val', 'test')),
    
    -- Metadata
    language VARCHAR(10) DEFAULT 'en',
    confidence_score FLOAT,
    annotator_id INTEGER,
    
    -- Prevent duplicate texts
    CONSTRAINT unique_text_hash UNIQUE (MD5(text))
);

CREATE INDEX IF NOT EXISTS idx_training_texts_sentiment ON training_texts(sentiment);
CREATE INDEX IF NOT EXISTS idx_training_texts_emotion ON training_texts(emotion);
CREATE INDEX IF NOT EXISTS idx_training_texts_split ON training_texts(split);
CREATE INDEX IF NOT EXISTS idx_training_texts_created ON training_texts(created_at);


-- Spark batch jobs tracking
CREATE TABLE IF NOT EXISTS spark_batch_jobs (
    id SERIAL PRIMARY KEY,
    job_name VARCHAR(100),
    job_type VARCHAR(50) CHECK (job_type IN ('preprocessing', 'training', 'inference', 'streaming')),
    
    -- Spark config
    num_executors INTEGER,
    executor_cores INTEGER,
    executor_memory VARCHAR(10),
    
    -- Data info
    input_path TEXT,
    output_path TEXT,
    num_records_processed BIGINT,
    
    -- Status
    status VARCHAR(20) CHECK (status IN ('submitted', 'running', 'completed', 'failed')),
    error_message TEXT,
    
    -- Timestamps
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    
    -- Performance metrics
    duration_seconds INTEGER,
    throughput_records_per_sec FLOAT
);

CREATE INDEX IF NOT EXISTS idx_spark_jobs_status ON spark_batch_jobs(status);
CREATE INDEX IF NOT EXISTS idx_spark_jobs_type ON spark_batch_jobs(job_type);
CREATE INDEX IF NOT EXISTS idx_spark_jobs_submitted ON spark_batch_jobs(submitted_at DESC);


-- Training runs (enhanced)
CREATE TABLE IF NOT EXISTS training_runs (
    id SERIAL PRIMARY KEY,
    run_name VARCHAR(100) NOT NULL,
    model_architecture VARCHAR(50),
    hyperparameters JSONB,
    dataset_size INTEGER,
    train_samples INTEGER,
    val_samples INTEGER,
    
    -- Training metrics
    best_val_loss FLOAT,
    best_sentiment_accuracy FLOAT,
    best_emotion_accuracy FLOAT,
    epochs_completed INTEGER,
    
    -- Hardware info
    device_type VARCHAR(20),
    num_gpus INTEGER,
    num_workers INTEGER,
    use_fp16 BOOLEAN DEFAULT FALSE,
    
    -- Spark info (if distributed)
    spark_job_id INTEGER REFERENCES spark_batch_jobs(id),
    
    -- Timestamps
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    
    -- Model path
    model_checkpoint_path TEXT,
    tokenizer_path TEXT,
    
    -- Status
    status VARCHAR(20) CHECK (status IN ('running', 'completed', 'failed', 'stopped'))
);

CREATE INDEX IF NOT EXISTS idx_training_runs_status ON training_runs(status);
CREATE INDEX IF NOT EXISTS idx_training_runs_started ON training_runs(started_at DESC);


-- Epoch metrics (time-series)
CREATE TABLE IF NOT EXISTS epoch_metrics (
    id SERIAL PRIMARY KEY,
    training_run_id INTEGER REFERENCES training_runs(id) ON DELETE CASCADE,
    epoch INTEGER,
    
    -- Loss metrics
    train_loss FLOAT,
    val_loss FLOAT,
    
    -- Accuracy metrics  
    train_sentiment_acc FLOAT,
    val_sentiment_acc FLOAT,
    train_emotion_acc FLOAT,
    val_emotion_acc FLOAT,
    
    -- Per-class metrics
    sentiment_precision JSONB,
    sentiment_recall JSONB,
    sentiment_f1 JSONB,
    emotion_precision JSONB,
    emotion_recall JSONB,
    emotion_f1 JSONB,
    
    -- Timing
    training_time_seconds FLOAT,
    validation_time_seconds FLOAT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_epoch_metrics_run ON epoch_metrics(training_run_id, epoch);


-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to get sentiment distribution for a time range
CREATE OR REPLACE FUNCTION get_sentiment_distribution(
    start_time TIMESTAMP,
    end_time TIMESTAMP
)
RETURNS TABLE(
    sentiment VARCHAR(20),
    count BIGINT,
    percentage FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        t.sentiment,
        COUNT(*) as count,
        (COUNT(*) * 100.0 / SUM(COUNT(*)) OVER ()) as percentage
    FROM twitter_sentiments t
    WHERE t.created_at BETWEEN start_time AND end_time
    GROUP BY t.sentiment
    ORDER BY count DESC;
END;
$$ LANGUAGE plpgsql;


-- Function to get trending hashtags with sentiment
CREATE OR REPLACE FUNCTION get_trending_hashtags(
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    limit_count INTEGER DEFAULT 10
)
RETURNS TABLE(
    hashtag TEXT,
    tweet_count BIGINT,
    positive_count BIGINT,
    negative_count BIGINT,
    neutral_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    WITH hashtag_sentiments AS (
        SELECT 
            jsonb_array_elements_text(hashtags::jsonb) as hashtag,
            sentiment
        FROM twitter_sentiments
        WHERE created_at BETWEEN start_time AND end_time
        AND hashtags IS NOT NULL
        AND hashtags != '[]'
    )
    SELECT 
        hs.hashtag,
        COUNT(*) as tweet_count,
        COUNT(*) FILTER (WHERE sentiment = 'positive') as positive_count,
        COUNT(*) FILTER (WHERE sentiment = 'negative') as negative_count,
        COUNT(*) FILTER (WHERE sentiment = 'neutral') as neutral_count
    FROM hashtag_sentiments hs
    GROUP BY hs.hashtag
    ORDER BY tweet_count DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- MAINTENANCE
-- ============================================================================

-- Auto-vacuum settings for high-volume tables
ALTER TABLE twitter_sentiments SET (autovacuum_vacuum_scale_factor = 0.05);
ALTER TABLE sentiment_trends SET (autovacuum_vacuum_scale_factor = 0.1);

-- Partition maintenance (add this to cron)
-- SELECT create_parent('public.twitter_sentiments', 'created_at', 'native', 'monthly');
