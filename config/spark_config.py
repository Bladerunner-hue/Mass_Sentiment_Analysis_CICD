"""Spark 4.0 Configuration for Streaming and Distributed Training."""

import os
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf


class SparkConfig:
    """Centralized Spark configuration for the application."""
    
    # Base configuration
    APP_NAME = "Mass_Sentiment_Analysis"
    MASTER = os.getenv("SPARK_MASTER", "local[*]")
    
    # Memory configuration (optimized for Quadro RTX 5000 16GB)
    DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "4g")
    EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY", "8g")
    EXECUTOR_CORES = int(os.getenv("SPARK_EXECUTOR_CORES", "4"))
    
    # GPU configuration
    GPU_PER_EXECUTOR = int(os.getenv("SPARK_GPU_PER_EXECUTOR", "1"))
    
    # Streaming configuration
    STREAMING_BATCH_INTERVAL = int(os.getenv("SPARK_STREAMING_BATCH_INTERVAL", "10"))  # seconds
    CHECKPOINT_DIR = os.getenv("SPARK_CHECKPOINT_DIR", "/tmp/spark_checkpoints")
    
    # Kafka configuration (for Twitter streaming)
    KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    KAFKA_TOPIC_TWITTER = os.getenv("KAFKA_TOPIC_TWITTER", "twitter_stream")
    
    # PostgreSQL configuration
    POSTGRES_URL = os.getenv(
        "POSTGRES_URL",
        "jdbc:postgresql://localhost:5432/sentiment_db"
    )
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
    
    # Petastorm configuration
    PETASTORM_CACHE_DIR = os.getenv("PETASTORM_CACHE_DIR", "file:///tmp/petastorm_cache")
    
    @classmethod
    def get_base_conf(cls) -> SparkConf:
        """Get base Spark configuration."""
        conf = SparkConf()
        conf.setAppName(cls.APP_NAME)
        conf.setMaster(cls.MASTER)
        
        # Memory settings
        conf.set("spark.driver.memory", cls.DRIVER_MEMORY)
        conf.set("spark.executor.memory", cls.EXECUTOR_MEMORY)
        conf.set("spark.executor.cores", str(cls.EXECUTOR_CORES))
        
        # GPU settings
        conf.set("spark.executor.resource.gpu.amount", str(cls.GPU_PER_EXECUTOR))
        conf.set("spark.task.resource.gpu.amount", "1")
        
        # PyArrow for better pandas conversion
        conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "10000")
        
        # Shuffle and memory optimization
        conf.set("spark.sql.shuffle.partitions", "200")
        conf.set("spark.memory.fraction", "0.8")
        conf.set("spark.memory.storageFraction", "0.3")
        
        return conf
    
    @classmethod
    def get_streaming_conf(cls) -> SparkConf:
        """Get Spark configuration for streaming applications."""
        conf = cls.get_base_conf()
        
        # Structured Streaming settings
        conf.set("spark.sql.streaming.checkpointLocation", cls.CHECKPOINT_DIR)
        conf.set("spark.sql.streaming.stateStore.providerClass", 
                "org.apache.spark.sql.execution.streaming.state.RocksDBStateStoreProvider")
        
        # Kafka settings
        conf.set("spark.jars.packages", 
                "org.apache.spark:spark-sql-kafka-0-10_2.12:4.0.0")
        
        # Backpressure
        conf.set("spark.streaming.backpressure.enabled", "true")
        conf.set("spark.streaming.kafka.maxRatePerPartition", "1000")
        
        return conf
    
    @classmethod
    def get_training_conf(cls) -> SparkConf:
        """Get Spark configuration for distributed training."""
        conf = cls.get_base_conf()
        
        # TorchDistributor settings
        conf.set("spark.task.maxFailures", "1")  # Fail fast for training
        conf.set("spark.python.worker.reuse", "false")  # Fresh workers for each task
        
        # Barrier execution (required for distributed training)
        conf.set("spark.scheduler.barrier.maxConcurrentTasksCheck.maxFailures", "10")
        conf.set("spark.scheduler.barrier.maxConcurrentTasksCheck.interval", "5s")
        
        # PostgreSQL JDBC driver
        conf.set("spark.jars.packages", 
                "org.postgresql:postgresql:42.7.0")
        
        # Petastorm configuration
        conf.set("spark.sql.parquet.compression.codec", "snappy")
        
        return conf
    
    @classmethod
    def create_session(cls, mode: str = "base") -> SparkSession:
        """
        Create Spark session with appropriate configuration.
        
        Args:
            mode: Configuration mode ('base', 'streaming', 'training')
            
        Returns:
            Configured SparkSession
        """
        if mode == "streaming":
            conf = cls.get_streaming_conf()
        elif mode == "training":
            conf = cls.get_training_conf()
        else:
            conf = cls.get_base_conf()
        
        spark = SparkSession.builder.config(conf=conf).getOrCreate()
        
        # Set log level
        spark.sparkContext.setLogLevel("WARN")
        
        return spark


def get_spark_session(mode: str = "base") -> SparkSession:
    """Convenience function to get or create Spark session."""
    return SparkConfig.create_session(mode)
