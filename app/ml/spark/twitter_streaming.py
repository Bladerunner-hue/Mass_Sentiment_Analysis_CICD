"""Spark Structured Streaming for real-time Twitter sentiment analysis."""

import json
from typing import Iterator, Dict, Any
from datetime import datetime

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col,
    from_json,
    udf,
    window,
    count,
    avg,
    current_timestamp,
    to_timestamp,
    expr,
)
from pyspark.sql.types import StructType, StructField, StringType, FloatType, MapType, TimestampType
import pandas as pd

from config.spark_config import get_spark_session
from app.ml.inference.predictor import CustomModelPredictor


class TwitterSparkStreaming:
    """
    Real-time Twitter sentiment analysis using Spark Structured Streaming.

    Features:
    - Consumes tweets from Kafka
    - Performs sentiment analysis using custom PyTorch model
    - Writes results to PostgreSQL
    - Computes real-time aggregations and trends
    - FP16 inference for speed
    """

    def __init__(
        self,
        kafka_servers: str = "localhost:9092",
        kafka_topic: str = "twitter_stream",
        postgres_url: str = "jdbc:postgresql://localhost:5432/sentiment_db",
        postgres_user: str = "postgres",
        postgres_password: str = "",
        model_path: str = "models/checkpoints/bilstm_attention.pt",
        tokenizer_path: str = "models/tokenizer.pkl",
        checkpoint_dir: str = "/tmp/spark_streaming_checkpoint",
    ):
        self.kafka_servers = kafka_servers
        self.kafka_topic = kafka_topic
        self.postgres_url = postgres_url
        self.postgres_user = postgres_user
        self.postgres_password = postgres_password
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.checkpoint_dir = checkpoint_dir

        # Initialize Spark session
        self.spark = get_spark_session(mode="streaming")

        # Initialize model (broadcasted to all executors)
        self.predictor = None

    def _get_predictor(self):
        """Lazy load predictor (called on executors)."""
        if self.predictor is None:
            self.predictor = CustomModelPredictor(
                model_path=self.model_path,
                tokenizer_path=self.tokenizer_path,
                device="cuda",  # Use GPU on executors
            )
        return self.predictor

    def _define_schema(self) -> StructType:
        """Define schema for incoming Twitter data."""
        return StructType(
            [
                StructField("id", StringType(), False),
                StructField("text", StringType(), False),
                StructField("user", StringType(), True),
                StructField("created_at", StringType(), False),
                StructField("lang", StringType(), True),
                StructField("retweet_count", StringType(), True),
                StructField("favorite_count", StringType(), True),
                StructField("hashtags", StringType(), True),
            ]
        )

    def _analyze_sentiment_udf(self):
        """Create UDF for sentiment analysis."""

        def analyze(text: str) -> Dict[str, Any]:
            """Analyze sentiment for a single text."""
            predictor = self._get_predictor()
            result = predictor.predict([text])[0]

            return {
                "sentiment": result["sentiment"],
                "sentiment_confidence": float(result.get("sentiment_confidence", 0.0)),
                "primary_emotion": result["primary_emotion"],
                "emotion_confidence": float(result.get("emotion_confidence", 0.0)),
                "sentiment_scores": json.dumps(result.get("sentiment_scores", {})),
                "emotion_scores": json.dumps(result.get("emotion_scores", {})),
            }

        return_schema = StructType(
            [
                StructField("sentiment", StringType()),
                StructField("sentiment_confidence", FloatType()),
                StructField("primary_emotion", StringType()),
                StructField("emotion_confidence", FloatType()),
                StructField("sentiment_scores", StringType()),
                StructField("emotion_scores", StringType()),
            ]
        )

        return udf(analyze, return_schema)

    def read_from_kafka(self) -> DataFrame:
        """Read streaming data from Kafka."""
        print(f"Reading from Kafka: {self.kafka_servers}, topic: {self.kafka_topic}")

        df = (
            self.spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", self.kafka_servers)
            .option("subscribe", self.kafka_topic)
            .option("startingOffsets", "latest")
            .option("maxOffsetsPerTrigger", 1000)
            .load()
        )

        # Parse JSON from Kafka value
        schema = self._define_schema()

        parsed_df = df.select(
            from_json(col("value").cast("string"), schema).alias("data"),
            col("timestamp").alias("kafka_timestamp"),
        ).select("data.*", "kafka_timestamp")

        # Convert created_at to timestamp
        parsed_df = parsed_df.withColumn(
            "created_at", to_timestamp(col("created_at"), "EEE MMM dd HH:mm:ss Z yyyy")
        )

        return parsed_df

    def analyze_stream(self, df: DataFrame) -> DataFrame:
        """Apply sentiment analysis to streaming data."""
        print("Applying sentiment analysis...")

        # Create sentiment analysis UDF
        analyze_udf = self._analyze_sentiment_udf()

        # Apply analysis
        analyzed_df = df.withColumn("analysis", analyze_udf(col("text")))

        # Flatten analysis results
        result_df = analyzed_df.select(
            col("id").alias("tweet_id"),
            col("text"),
            col("user"),
            col("created_at"),
            col("lang"),
            col("retweet_count"),
            col("favorite_count"),
            col("hashtags"),
            col("analysis.sentiment"),
            col("analysis.sentiment_confidence"),
            col("analysis.primary_emotion"),
            col("analysis.emotion_confidence"),
            col("analysis.sentiment_scores"),
            col("analysis.emotion_scores"),
            current_timestamp().alias("processed_at"),
        )

        return result_df

    def write_to_postgres(self, df: DataFrame, table_name: str = "twitter_sentiments"):
        """Write streaming results to PostgreSQL."""
        print(f"Writing to PostgreSQL table: {table_name}")

        def write_batch(batch_df: DataFrame, batch_id: int):
            """Write each micro-batch to PostgreSQL."""
            batch_df.write.format("jdbc").option("url", self.postgres_url).option(
                "dbtable", table_name
            ).option("user", self.postgres_user).option("password", self.postgres_password).option(
                "driver", "org.postgresql.Driver"
            ).mode(
                "append"
            ).save()

            print(f"Batch {batch_id} written: {batch_df.count()} records")

        query = (
            df.writeStream.foreachBatch(write_batch)
            .option("checkpointLocation", f"{self.checkpoint_dir}/postgres")
            .start()
        )

        return query

    def compute_aggregations(self, df: DataFrame) -> DataFrame:
        """Compute real-time aggregations for trends."""
        print("Computing real-time aggregations...")

        # Windowed aggregations (5-minute windows)
        agg_df = (
            df.withWatermark("created_at", "10 minutes")
            .groupBy(window(col("created_at"), "5 minutes"), col("sentiment"))
            .agg(
                count("*").alias("tweet_count"), avg("sentiment_confidence").alias("avg_confidence")
            )
        )

        return agg_df

    def write_aggregations(self, agg_df: DataFrame, table_name: str = "sentiment_trends"):
        """Write aggregations to PostgreSQL."""
        print(f"Writing aggregations to: {table_name}")

        def write_agg_batch(batch_df: DataFrame, batch_id: int):
            """Write aggregation batch."""
            # Flatten window struct
            flattened = batch_df.select(
                col("window.start").alias("window_start"),
                col("window.end").alias("window_end"),
                col("sentiment"),
                col("tweet_count"),
                col("avg_confidence"),
            )

            flattened.write.format("jdbc").option("url", self.postgres_url).option(
                "dbtable", table_name
            ).option("user", self.postgres_user).option("password", self.postgres_password).option(
                "driver", "org.postgresql.Driver"
            ).mode(
                "append"
            ).save()

            print(f"Aggregation batch {batch_id} written")

        query = (
            agg_df.writeStream.foreachBatch(write_agg_batch)
            .option("checkpointLocation", f"{self.checkpoint_dir}/aggregations")
            .start()
        )

        return query

    def start_pipeline(self):
        """Start the complete streaming pipeline."""
        print("=" * 60)
        print("Starting Twitter Sentiment Streaming Pipeline")
        print("=" * 60)

        # Read from Kafka
        tweets_df = self.read_from_kafka()

        # Analyze sentiment
        analyzed_df = self.analyze_stream(tweets_df)

        # Write raw results to PostgreSQL
        results_query = self.write_to_postgres(analyzed_df)

        # Compute and write aggregations
        agg_df = self.compute_aggregations(analyzed_df)
        agg_query = self.write_aggregations(agg_df)

        # Console output for monitoring (optional)
        console_query = (
            analyzed_df.select("tweet_id", "text", "sentiment", "primary_emotion")
            .writeStream.outputMode("append")
            .format("console")
            .option("truncate", False)
            .option("checkpointLocation", f"{self.checkpoint_dir}/console")
            .start()
        )

        print("\nStreaming pipeline started!")
        print("Press Ctrl+C to stop...\n")

        # Wait for termination
        try:
            self.spark.streams.awaitAnyTermination()
        except KeyboardInterrupt:
            print("\nStopping streaming pipeline...")
            results_query.stop()
            agg_query.stop()
            console_query.stop()
            print("Pipeline stopped.")


def main():
    """Main entry point for streaming application.

    Supports two modes:
    1. Kafka mode (default): Uses TwitterSparkStreaming class for Kafka-based streaming
    2. Direct API mode: Uses argparse args for standalone Twitter API streaming
    """
    import argparse
    import os
    from pyspark.sql import functions as F
    from app.services.twitter_service import XTwitterService

    parser = argparse.ArgumentParser(description="Spark streaming Twitter collector.")
    parser.add_argument(
        "--mode",
        default="api",
        choices=["api", "kafka"],
        help="Streaming mode: 'api' for direct Twitter API or 'kafka' for Kafka consumer",
    )
    parser.add_argument(
        "--query", default=os.getenv("X_STREAM_QUERY", "sentiment"), help="Twitter search query"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("X_STREAM_BATCH_SIZE", "50")),
        help="Tweets per micro-batch",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("X_STREAM_OUTPUT", "data/raw/twitter_stream_spark"),
        help="Output root for Parquet",
    )
    parser.add_argument(
        "--rows-per-second",
        type=int,
        default=1,
        help="Rate source rows per second to drive batches",
    )
    args = parser.parse_args()

    if args.mode == "kafka":
        # Kafka-based streaming using the full class
        streamer = TwitterSparkStreaming(
            kafka_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            kafka_topic=os.getenv("KAFKA_TOPIC_TWITTER", "twitter_stream"),
            postgres_url=os.getenv("POSTGRES_URL", "jdbc:postgresql://localhost:5432/sentiment_db"),
            postgres_user=os.getenv("POSTGRES_USER", "postgres"),
            postgres_password=os.getenv("POSTGRES_PASSWORD", ""),
            model_path=os.getenv("CUSTOM_MODEL_PATH", "models/checkpoints/bilstm_attention.pt"),
            tokenizer_path=os.getenv("CUSTOM_TOKENIZER_PATH", "models/tokenizer.pkl"),
        )
        streamer.start_pipeline()
    else:
        # Direct API mode - standalone Twitter API streaming
        def fetch_tweets(service: XTwitterService, query: str, batch_size: int) -> list:
            resp = service.search_recent_tweets(query=query, max_results=batch_size)
            if not isinstance(resp, dict) or "tweets" not in resp:
                return []
            return resp.get("tweets", [])

        service = XTwitterService()
        if not service.is_configured:
            raise RuntimeError("X_BEARER_TOKEN is not set; cannot start Spark Twitter streaming.")

        spark = (
            SparkSession.builder.appName("XTwitterSparkStreaming")
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.sql.streaming.schemaInference", "true")
            .getOrCreate()
        )

        rate_df = (
            spark.readStream.format("rate").option("rowsPerSecond", args.rows_per_second).load()
        )

        output_root = args.output

        def process_batch(_df: DataFrame, batch_id: int) -> None:
            tweets = fetch_tweets(service, args.query, args.batch_size)
            if not tweets:
                return
            sdf = spark.createDataFrame(tweets)
            now = datetime.utcnow()
            (
                sdf.withColumn("ingest_date", F.lit(now.date().isoformat()))
                .withColumn("ingest_hour", F.lit(now.hour))
                .write.mode("append")
                .partitionBy("ingest_date", "ingest_hour")
                .parquet(output_root)
            )

        query = (
            rate_df.writeStream.trigger(processingTime="30 seconds")
            .foreachBatch(process_batch)
            .option("checkpointLocation", f"{output_root}/_checkpoints")
            .start()
        )

        query.awaitTermination()


if __name__ == "__main__":
    main()
