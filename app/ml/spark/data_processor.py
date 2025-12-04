"""Spark helper to clean/deduplicate raw Parquet data and write processed splits.

Usage:
    spark-submit app/ml/spark/data_processor.py \
        --raw data/raw \
        --processed data/processed

If Spark is unavailable, the script exits gracefully.
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw", help="Root path containing raw Parquet files")
    parser.add_argument(
        "--processed", default="data/processed", help="Output root for processed data"
    )
    args = parser.parse_args()

    try:
        from pyspark.sql import SparkSession, functions as F, Window
    except Exception:
        print("pyspark not installed; skip Spark processing.")
        return

    spark = (
        SparkSession.builder.appName("SentimentDataProcessor")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )

    df = spark.read.parquet(f"{args.raw}/**/*.parquet")
    if df.rdd.isEmpty():
        print("No Parquet files found under raw path.")
        spark.stop()
        return

    df = df.withColumn("text", F.trim(F.col("text")))
    df = df.filter(F.length("text") > 3)
    df = df.withColumn("text_hash", F.md5(F.col("text")))

    window = Window.partitionBy("text_hash").orderBy(F.lit(1))
    df = df.withColumn("rn", F.row_number().over(window)).filter(F.col("rn") == 1).drop("rn")

    # Simple stratified split based on sentiment
    train = df.sampleBy("sentiment", fractions=None, seed=42)  # random shuffle
    train.write.mode("overwrite").parquet(f"{args.processed}/train")

    spark.stop()


if __name__ == "__main__":
    main()
