"""Spark 4.0 helpers for data preprocessing, streaming, and distributed training.

This module provides:
- SparkStreamingService: Real-time Structured Streaming for sentiment analysis
- SparkPreprocessor: Batch data preprocessing (CSV/JSON to Parquet)
- SparkTorchTrainer: Distributed PyTorch training with TorchDistributor

No Kafka required - uses native Spark Structured Streaming.
"""

from .jobs import SparkPreprocessor, SparkTorchTrainer  # noqa: F401
from .streaming import (  # noqa: F401
    SparkStreamingService,
    StreamConfig,
    streaming_service,
    start_all_streams,
)
