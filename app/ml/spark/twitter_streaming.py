"""Spark Structured Streaming driver that pulls X (Twitter) data each micro-batch.

This uses a rate source to trigger batches; each batch queries the Twitter API
and appends results as Parquet partitions under the configured output path.
"""

import argparse
import os
from datetime import datetime
from typing import List

from pyspark.sql import SparkSession, functions as F

from app.services.twitter_service import XTwitterService


def fetch_tweets(service: XTwitterService, query: str, batch_size: int) -> List[dict]:
    resp = service.search_recent_tweets(query=query, max_results=batch_size)
    if not isinstance(resp, dict) or "tweets" not in resp:
        return []
    return resp.get("tweets", [])


def main():
    parser = argparse.ArgumentParser(description="Spark streaming Twitter collector.")
    parser.add_argument(
        "--query", default=os.getenv("X_STREAM_QUERY", "sentiment"), help="Twitter search query"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("X_STREAM_BATCH_SIZE", 50)),
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

    service = XTwitterService()
    if not service.is_configured:
        raise RuntimeError("X_BEARER_TOKEN is not set; cannot start Spark Twitter streaming.")

    spark = (
        SparkSession.builder.appName("XTwitterSparkStreaming")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.streaming.schemaInference", "true")
        .getOrCreate()
    )

    rate_df = spark.readStream.format("rate").option("rowsPerSecond", args.rows_per_second).load()

    output_root = args.output

    def process_batch(_df, batch_id: int):
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
