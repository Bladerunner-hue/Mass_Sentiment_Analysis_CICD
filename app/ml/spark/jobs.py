"""Spark utilities for preprocessing and distributed PyTorch training."""

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class SparkPreprocessor:
    """Convert raw data sources (CSV/JSON) into Parquet partitions for training."""

    def __init__(self, app_name: str = "SentimentSparkPreprocess", master: str = "local[*]"):
        self.app_name = app_name
        self.master = master

    def _get_session(self):
        try:
            from pyspark.sql import SparkSession
        except Exception as exc:  # pragma: no cover - only executed when pyspark missing
            raise RuntimeError("pyspark is required for SparkPreprocessor") from exc

        return (
            SparkSession.builder.appName(self.app_name)
            .master(self.master)
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .getOrCreate()
        )

    def csv_to_parquet(self, input_path: str, output_path: str, text_column: str = "text") -> None:
        """Read CSV, standardize columns, and write Parquet for downstream training."""
        spark = self._get_session()
        df = spark.read.option("header", True).csv(input_path)
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in input data")

        # Simple schema alignment: keep text/sentiment/emotion if present
        select_cols = ["text"]
        rename_map = {}
        if text_column != "text":
            rename_map[text_column] = "text"

        for col in ["sentiment", "emotion", "split"]:
            if col in df.columns:
                select_cols.append(col)

        df = df.select(*select_cols)
        for source, target in rename_map.items():
            df = df.withColumnRenamed(source, target)

        df.write.mode("overwrite").parquet(output_path)
        logger.info("Wrote Parquet dataset to %s", output_path)
        spark.stop()


class SparkTorchTrainer:
    """Run distributed PyTorch training using TorchDistributor + Petastorm."""

    def __init__(self, app_name: str = "SentimentSparkTrain", master: str = "local[*]"):
        self.app_name = app_name
        self.master = master

    def _get_session(self):
        try:
            from pyspark.sql import SparkSession
        except Exception as exc:  # pragma: no cover - only executed when pyspark missing
            raise RuntimeError("pyspark is required for SparkTorchTrainer") from exc

        return (
            SparkSession.builder.appName(self.app_name)
            .master(self.master)
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .getOrCreate()
        )

    def run(
        self,
        train_fn: Callable,
        num_processes: int = 1,
        use_gpu: bool = True,
        local_mode: bool = True,
    ):
        """Execute a training function across Spark executors."""
        try:
            from pyspark.ml.torch.distributor import TorchDistributor
        except Exception as exc:  # pragma: no cover - only executed when pyspark missing
            raise RuntimeError("pyspark>=3.4 with TorchDistributor is required") from exc

        # Spark session needed for distributor
        spark = self._get_session()
        distributor = TorchDistributor(
            num_processes=num_processes,
            local_mode=local_mode,
            use_gpu=use_gpu,
        )
        result = distributor.run(train_fn)
        spark.stop()
        return result
