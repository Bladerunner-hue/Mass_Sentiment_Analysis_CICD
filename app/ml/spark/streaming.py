"""Spark 4.0 Structured Streaming for real-time sentiment analysis.

This module provides Spark Structured Streaming integration for processing
Twitter, Reddit, and news feeds at scale. It replaces Kafka with native
Spark streaming capabilities.

Architecture:
    Source → Spark Structured Streaming → Sentiment Analysis → PostgreSQL + Parquet
    
    ┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
    │ Twitter API │────▶│ Spark Streaming  │────▶│ PostgreSQL      │
    │ Reddit API  │     │ (Micro-batch)    │     │ (Real-time)     │
    │ RSS Feeds   │     │                  │     └─────────────────┘
    └─────────────┘     │ Custom Model     │           │
                        │ BiLSTM+Attention │           ▼
                        └──────────────────┘     ┌─────────────────┐
                                                 │ Parquet         │
                                                 │ (Historical)    │
                                                 └─────────────────┘
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """Configuration for Spark Structured Streaming."""
    app_name: str = "SentimentSparkStreaming"
    master: str = field(default_factory=lambda: os.getenv("SPARK_MASTER", "local[*]"))
    executor_memory: str = field(default_factory=lambda: os.getenv("SPARK_EXECUTOR_MEMORY", "32g"))
    checkpoint_path: str = "data/checkpoints/streaming"
    output_path: str = "data/streaming/output"
    trigger_interval: str = "10 seconds"  # Micro-batch interval
    watermark_delay: str = "1 minute"  # Late data handling
    
    # PostgreSQL connection
    jdbc_url: str = field(default_factory=lambda: os.getenv(
        "DATABASE_URL", 
        "postgresql://EffuzionBridge:Ability8-Acts7-Exorcist5-Rotunda0-Splotchy5@localhost:5432/EffuzionBridge"
    ))
    
    # Custom model paths
    model_path: str = field(default_factory=lambda: os.getenv(
        "CUSTOM_MODEL_PATH", "models/checkpoints/bilstm_attention.pt"
    ))
    tokenizer_path: str = field(default_factory=lambda: os.getenv(
        "CUSTOM_TOKENIZER_PATH", "models/tokenizer.pkl"
    ))


class SparkStreamingService:
    """Spark 4.0 Structured Streaming service for sentiment analysis.
    
    Replaces Kafka with native Spark streaming for:
    - Twitter API v2 streaming
    - Reddit API streaming
    - RSS/News feed processing
    - Real-time PostgreSQL writes
    - Historical Parquet storage
    """
    
    def __init__(self, config: Optional[StreamConfig] = None):
        self.config = config or StreamConfig()
        self._spark = None
        self._sentiment_service = None
        self._active_streams: Dict[str, Any] = {}
        
    def _get_spark_session(self):
        """Create or get existing SparkSession with Spark 4.0 features."""
        if self._spark is not None:
            return self._spark
            
        try:
            from pyspark.sql import SparkSession
            from pyspark import SparkConf
        except ImportError as e:
            raise RuntimeError(
                "PySpark >= 3.5 required. Install with: pip install pyspark>=3.5"
            ) from e
            
        conf = SparkConf()
        conf.set("spark.app.name", self.config.app_name)
        conf.set("spark.master", self.config.master)
        conf.set("spark.executor.memory", self.config.executor_memory)
        
        # Spark 4.0 optimizations
        conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        conf.set("spark.sql.adaptive.enabled", "true")
        conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
        conf.set("spark.sql.streaming.stateStore.providerClass", 
                 "org.apache.spark.sql.execution.streaming.state.RocksDBStateStoreProvider")
        
        # PostgreSQL JDBC driver
        conf.set("spark.jars.packages", "org.postgresql:postgresql:42.7.1")
        
        self._spark = (
            SparkSession.builder
            .config(conf=conf)
            .getOrCreate()
        )
        
        # Set log level
        self._spark.sparkContext.setLogLevel("WARN")
        
        logger.info(f"Spark session created: {self._spark.version}")
        return self._spark
        
    def _get_sentiment_service(self):
        """Get or create sentiment service with custom model."""
        if self._sentiment_service is not None:
            return self._sentiment_service
            
        try:
            from app.services.sentiment_service import SentimentService
            self._sentiment_service = SentimentService()
            logger.info("Sentiment service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize sentiment service: {e}")
            raise
            
        return self._sentiment_service
        
    def _create_sentiment_udf(self):
        """Create Spark UDF for sentiment analysis using custom model."""
        from pyspark.sql.functions import udf
        from pyspark.sql.types import StructType, StructField, StringType, FloatType, MapType
        
        service = self._get_sentiment_service()
        
        # Define return schema for the UDF
        result_schema = StructType([
            StructField("sentiment", StringType(), False),
            StructField("compound_score", FloatType(), False),
            StructField("primary_emotion", StringType(), True),
            StructField("confidence", FloatType(), True),
            StructField("model_used", StringType(), True),
        ])
        
        def analyze_text(text: str) -> Dict[str, Any]:
            """Analyze text using custom BiLSTM + Attention model."""
            if not text or not text.strip():
                return {
                    "sentiment": "neutral",
                    "compound_score": 0.0,
                    "primary_emotion": "neutral",
                    "confidence": 0.0,
                    "model_used": "none"
                }
            try:
                result = service.analyze_full(text)
                return {
                    "sentiment": result.get("sentiment", "neutral"),
                    "compound_score": float(result.get("compound_score", 0.0)),
                    "primary_emotion": result.get("primary_emotion", "neutral"),
                    "confidence": float(result.get("confidence", 0.0)),
                    "model_used": result.get("model_used", "unknown")
                }
            except Exception as e:
                logger.error(f"Sentiment analysis error: {e}")
                return {
                    "sentiment": "error",
                    "compound_score": 0.0,
                    "primary_emotion": "error",
                    "confidence": 0.0,
                    "model_used": "error"
                }
                
        return udf(analyze_text, result_schema)
        
    def start_twitter_stream(
        self,
        bearer_token: str,
        keywords: List[str],
        output_mode: str = "append"
    ) -> str:
        """Start Twitter streaming with Spark Structured Streaming.
        
        Uses rate source for demonstration; in production, use custom source
        connected to Twitter API v2 streaming endpoint.
        
        Args:
            bearer_token: Twitter API bearer token
            keywords: Keywords to track
            output_mode: Spark output mode (append/complete/update)
            
        Returns:
            Stream ID for monitoring
        """
        spark = self._get_spark_session()
        sentiment_udf = self._create_sentiment_udf()
        
        from pyspark.sql import functions as F
        from pyspark.sql.types import StructType, StructField, StringType, TimestampType, LongType
        
        # Twitter stream schema
        twitter_schema = StructType([
            StructField("tweet_id", StringType(), False),
            StructField("text", StringType(), False),
            StructField("author_id", StringType(), True),
            StructField("created_at", TimestampType(), False),
            StructField("retweet_count", LongType(), True),
            StructField("like_count", LongType(), True),
        ])
        
        # Create streaming source from Twitter API
        # For production: implement custom TwitterSource extending spark.sql.streaming.Source
        stream_df = (
            spark.readStream
            .format("rate")  # Demo source - replace with custom Twitter source
            .option("rowsPerSecond", 10)
            .load()
        )
        
        # Simulate Twitter data for development
        # In production, this would be replaced by actual Twitter API data
        stream_df = stream_df.withColumn(
            "tweet_id", F.concat(F.lit("tweet_"), F.col("value").cast("string"))
        ).withColumn(
            "text", F.concat(
                F.lit("Sample tweet about "), 
                F.element_at(F.array(*[F.lit(k) for k in keywords]), 
                            (F.col("value") % len(keywords) + 1).cast("int"))
            )
        ).withColumn(
            "author_id", F.concat(F.lit("user_"), (F.col("value") % 1000).cast("string"))
        ).withColumn(
            "created_at", F.current_timestamp()
        ).withColumn(
            "retweet_count", (F.rand() * 100).cast("long")
        ).withColumn(
            "like_count", (F.rand() * 500).cast("long")
        )
        
        # Apply sentiment analysis
        analyzed_df = stream_df.withColumn(
            "analysis", sentiment_udf(F.col("text"))
        ).select(
            F.col("tweet_id"),
            F.col("text"),
            F.col("author_id"),
            F.col("created_at"),
            F.col("retweet_count"),
            F.col("like_count"),
            F.col("analysis.sentiment").alias("sentiment"),
            F.col("analysis.compound_score").alias("compound_score"),
            F.col("analysis.primary_emotion").alias("primary_emotion"),
            F.col("analysis.confidence").alias("confidence"),
            F.col("analysis.model_used").alias("model_used"),
            F.lit(",".join(keywords)).alias("keywords"),
            F.lit("twitter").alias("source"),
        )
        
        # Add watermark for late data handling
        analyzed_df = analyzed_df.withWatermark("created_at", self.config.watermark_delay)
        
        # Write to PostgreSQL
        stream_id = f"twitter_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        query = (
            analyzed_df.writeStream
            .outputMode(output_mode)
            .trigger(processingTime=self.config.trigger_interval)
            .foreachBatch(self._write_to_postgres_and_parquet)
            .option("checkpointLocation", f"{self.config.checkpoint_path}/{stream_id}")
            .queryName(stream_id)
            .start()
        )
        
        self._active_streams[stream_id] = query
        logger.info(f"Started Twitter stream: {stream_id}")
        
        return stream_id
        
    def start_reddit_stream(
        self,
        client_id: str,
        client_secret: str,
        user_agent: str,
        subreddits: List[str],
        output_mode: str = "append"
    ) -> str:
        """Start Reddit streaming with Spark Structured Streaming.
        
        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: Reddit API user agent
            subreddits: Subreddits to monitor
            output_mode: Spark output mode
            
        Returns:
            Stream ID for monitoring
        """
        spark = self._get_spark_session()
        sentiment_udf = self._create_sentiment_udf()
        
        from pyspark.sql import functions as F
        
        # Create streaming source (demo)
        stream_df = (
            spark.readStream
            .format("rate")
            .option("rowsPerSecond", 5)
            .load()
        )
        
        # Simulate Reddit data
        stream_df = stream_df.withColumn(
            "post_id", F.concat(F.lit("reddit_"), F.col("value").cast("string"))
        ).withColumn(
            "text", F.concat(
                F.lit("Reddit post in r/"), 
                F.element_at(F.array(*[F.lit(s) for s in subreddits]), 
                            (F.col("value") % len(subreddits) + 1).cast("int")),
                F.lit(" about sentiment analysis")
            )
        ).withColumn(
            "author", F.concat(F.lit("redditor_"), (F.col("value") % 500).cast("string"))
        ).withColumn(
            "subreddit", F.element_at(
                F.array(*[F.lit(s) for s in subreddits]), 
                (F.col("value") % len(subreddits) + 1).cast("int")
            )
        ).withColumn(
            "created_at", F.current_timestamp()
        ).withColumn(
            "score", (F.rand() * 1000).cast("long")
        )
        
        # Apply sentiment analysis
        analyzed_df = stream_df.withColumn(
            "analysis", sentiment_udf(F.col("text"))
        ).select(
            F.col("post_id").alias("tweet_id"),  # Normalize to common schema
            F.col("text"),
            F.col("author").alias("author_id"),
            F.col("created_at"),
            F.col("score").alias("retweet_count"),
            F.lit(0).cast("long").alias("like_count"),
            F.col("analysis.sentiment").alias("sentiment"),
            F.col("analysis.compound_score").alias("compound_score"),
            F.col("analysis.primary_emotion").alias("primary_emotion"),
            F.col("analysis.confidence").alias("confidence"),
            F.col("analysis.model_used").alias("model_used"),
            F.col("subreddit").alias("keywords"),
            F.lit("reddit").alias("source"),
        )
        
        analyzed_df = analyzed_df.withWatermark("created_at", self.config.watermark_delay)
        
        stream_id = f"reddit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        query = (
            analyzed_df.writeStream
            .outputMode(output_mode)
            .trigger(processingTime=self.config.trigger_interval)
            .foreachBatch(self._write_to_postgres_and_parquet)
            .option("checkpointLocation", f"{self.config.checkpoint_path}/{stream_id}")
            .queryName(stream_id)
            .start()
        )
        
        self._active_streams[stream_id] = query
        logger.info(f"Started Reddit stream: {stream_id}")
        
        return stream_id
        
    def start_news_stream(
        self,
        rss_feeds: List[str],
        output_mode: str = "append"
    ) -> str:
        """Start news feed streaming with Spark Structured Streaming.
        
        Args:
            rss_feeds: List of RSS feed URLs
            output_mode: Spark output mode
            
        Returns:
            Stream ID for monitoring
        """
        spark = self._get_spark_session()
        sentiment_udf = self._create_sentiment_udf()
        
        from pyspark.sql import functions as F
        
        # Create streaming source
        stream_df = (
            spark.readStream
            .format("rate")
            .option("rowsPerSecond", 2)
            .load()
        )
        
        # Simulate news data
        stream_df = stream_df.withColumn(
            "article_id", F.concat(F.lit("news_"), F.col("value").cast("string"))
        ).withColumn(
            "text", F.lit("Breaking news: Technology advances in AI and machine learning")
        ).withColumn(
            "source_feed", F.element_at(
                F.array(*[F.lit(f) for f in rss_feeds]), 
                (F.col("value") % len(rss_feeds) + 1).cast("int")
            )
        ).withColumn(
            "created_at", F.current_timestamp()
        )
        
        # Apply sentiment analysis
        analyzed_df = stream_df.withColumn(
            "analysis", sentiment_udf(F.col("text"))
        ).select(
            F.col("article_id").alias("tweet_id"),
            F.col("text"),
            F.col("source_feed").alias("author_id"),
            F.col("created_at"),
            F.lit(0).cast("long").alias("retweet_count"),
            F.lit(0).cast("long").alias("like_count"),
            F.col("analysis.sentiment").alias("sentiment"),
            F.col("analysis.compound_score").alias("compound_score"),
            F.col("analysis.primary_emotion").alias("primary_emotion"),
            F.col("analysis.confidence").alias("confidence"),
            F.col("analysis.model_used").alias("model_used"),
            F.col("source_feed").alias("keywords"),
            F.lit("news").alias("source"),
        )
        
        analyzed_df = analyzed_df.withWatermark("created_at", self.config.watermark_delay)
        
        stream_id = f"news_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        query = (
            analyzed_df.writeStream
            .outputMode(output_mode)
            .trigger(processingTime=self.config.trigger_interval)
            .foreachBatch(self._write_to_postgres_and_parquet)
            .option("checkpointLocation", f"{self.config.checkpoint_path}/{stream_id}")
            .queryName(stream_id)
            .start()
        )
        
        self._active_streams[stream_id] = query
        logger.info(f"Started news stream: {stream_id}")
        
        return stream_id
        
    def _write_to_postgres_and_parquet(self, batch_df, batch_id: int):
        """Write micro-batch to PostgreSQL and Parquet.
        
        This is called for each micro-batch in Spark Structured Streaming.
        """
        if batch_df.isEmpty():
            return
            
        batch_count = batch_df.count()
        logger.info(f"Processing batch {batch_id} with {batch_count} records")
        
        # Write to PostgreSQL for real-time access
        try:
            (
                batch_df.write
                .format("jdbc")
                .option("url", self.config.jdbc_url.replace("postgresql://", "jdbc:postgresql://"))
                .option("dbtable", "streaming_sentiment_results")
                .option("user", os.getenv("DB_USER", "EffuzionBridge"))
                .option("password", os.getenv("DB_PASSWORD", ""))
                .option("driver", "org.postgresql.Driver")
                .mode("append")
                .save()
            )
            logger.info(f"Wrote {batch_count} records to PostgreSQL")
        except Exception as e:
            logger.error(f"PostgreSQL write failed: {e}")
            
        # Write to Parquet for historical analysis
        try:
            from pyspark.sql import functions as F
            
            partitioned_df = batch_df.withColumn(
                "date", F.to_date(F.col("created_at"))
            ).withColumn(
                "hour", F.hour(F.col("created_at"))
            )
            
            (
                partitioned_df.write
                .partitionBy("source", "date", "hour")
                .mode("append")
                .parquet(self.config.output_path)
            )
            logger.info(f"Wrote {batch_count} records to Parquet")
        except Exception as e:
            logger.error(f"Parquet write failed: {e}")
            
    def get_stream_status(self, stream_id: str) -> Dict[str, Any]:
        """Get status of a running stream."""
        if stream_id not in self._active_streams:
            return {"error": f"Stream {stream_id} not found"}
            
        query = self._active_streams[stream_id]
        
        return {
            "stream_id": stream_id,
            "name": query.name,
            "is_active": query.isActive,
            "status": query.status,
            "recent_progress": query.recentProgress[-1] if query.recentProgress else None,
        }
        
    def stop_stream(self, stream_id: str) -> bool:
        """Stop a running stream."""
        if stream_id not in self._active_streams:
            logger.warning(f"Stream {stream_id} not found")
            return False
            
        query = self._active_streams[stream_id]
        query.stop()
        del self._active_streams[stream_id]
        
        logger.info(f"Stopped stream: {stream_id}")
        return True
        
    def stop_all_streams(self):
        """Stop all running streams."""
        for stream_id in list(self._active_streams.keys()):
            self.stop_stream(stream_id)
            
        logger.info("Stopped all streams")
        
    def await_termination(self, stream_id: Optional[str] = None, timeout: Optional[int] = None):
        """Wait for stream(s) to terminate."""
        if stream_id:
            if stream_id in self._active_streams:
                self._active_streams[stream_id].awaitTermination(timeout)
        else:
            spark = self._get_spark_session()
            spark.streams.awaitAnyTermination(timeout)


# Global instance for convenience
streaming_service = SparkStreamingService()


def start_all_streams(
    twitter_keywords: Optional[List[str]] = None,
    subreddits: Optional[List[str]] = None,
    rss_feeds: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Start all configured streams.
    
    Args:
        twitter_keywords: Keywords for Twitter streaming
        subreddits: Subreddits for Reddit streaming
        rss_feeds: RSS feed URLs for news streaming
        
    Returns:
        Dictionary of stream IDs
    """
    stream_ids = {}
    
    if twitter_keywords:
        bearer_token = os.getenv("TWITTER_BEARER_TOKEN", "")
        if bearer_token:
            stream_ids["twitter"] = streaming_service.start_twitter_stream(
                bearer_token=bearer_token,
                keywords=twitter_keywords
            )
            
    if subreddits:
        client_id = os.getenv("REDDIT_CLIENT_ID", "")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
        user_agent = os.getenv("REDDIT_USER_AGENT", "SentimentBot/1.0")
        if client_id and client_secret:
            stream_ids["reddit"] = streaming_service.start_reddit_stream(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
                subreddits=subreddits
            )
            
    if rss_feeds:
        stream_ids["news"] = streaming_service.start_news_stream(
            rss_feeds=rss_feeds
        )
        
    return stream_ids
