#!/usr/bin/env python
"""Initialize PostgreSQL database with schema for Spark 4.0 Streaming.

Usage:
    python scripts/init_database.py

This script:
1. Creates all required tables for streaming sentiment analysis
2. Sets up training data management tables
3. Creates model versioning tables
4. Initializes API credential storage
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def get_db_connection():
    """Get PostgreSQL connection using environment variables."""
    try:
        import psycopg2
    except ImportError:
        print("❌ psycopg2 not installed. Install with: pip install psycopg2-binary")
        sys.exit(1)

    db_url = os.getenv("DATABASE_URL")

    if db_url:
        # Parse DATABASE_URL
        from urllib.parse import urlparse

        result = urlparse(db_url)

        return psycopg2.connect(
            host=result.hostname,
            port=result.port or 5432,
            database=result.path[1:],  # Remove leading /
            user=result.username,
            password=result.password,
        )
    else:
        # Use individual env vars
        return psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", 5432)),
            database=os.getenv("DB_NAME", "EffuzionBridge"),
            user=os.getenv("DB_USER", "EffuzionBridge"),
            password=os.getenv("DB_PASSWORD", ""),
        )


def init_database():
    """Initialize database with schema."""
    print("🗄️  Initializing PostgreSQL database for Spark 4.0 Streaming...")

    # Load .env if exists
    env_path = project_root / ".env"
    if env_path.exists():
        from dotenv import load_dotenv

        load_dotenv(env_path)
        print("✅ Loaded .env file")

    # Read schema file
    schema_path = project_root / "database" / "schema.sql"
    if not schema_path.exists():
        print(f"❌ Schema file not found: {schema_path}")
        sys.exit(1)

    with open(schema_path, "r") as f:
        schema_sql = f.read()

    print(f"✅ Loaded schema from {schema_path}")

    # Connect and execute
    try:
        conn = get_db_connection()
        print(f"✅ Connected to PostgreSQL")

        with conn.cursor() as cur:
            # Execute schema
            cur.execute(schema_sql)
            conn.commit()

        print("✅ Schema created successfully")

        # Verify tables
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """
            )
            tables = [row[0] for row in cur.fetchall()]

        print(f"\n📋 Created tables ({len(tables)}):")
        for table in tables:
            print(f"   • {table}")

        conn.close()
        print("\n✅ Database initialization complete!")

    except Exception as e:
        print(f"❌ Database error: {e}")
        sys.exit(1)


def verify_spark_connection():
    """Verify Spark can connect to PostgreSQL."""
    print("\n🔥 Verifying Spark PostgreSQL connection...")

    try:
        from pyspark.sql import SparkSession

        spark = (
            SparkSession.builder.appName("PostgreSQLTest")
            .master("local[1]")
            .config("spark.jars.packages", "org.postgresql:postgresql:42.7.1")
            .getOrCreate()
        )

        # Read from PostgreSQL
        db_url = os.getenv("DATABASE_URL", "")
        if db_url:
            jdbc_url = db_url.replace("postgresql://", "jdbc:postgresql://")

            # Try to read a table
            df = (
                spark.read.format("jdbc")
                .option("url", jdbc_url)
                .option("dbtable", "models")
                .option("driver", "org.postgresql.Driver")
                .load()
            )

            count = df.count()
            print(f"✅ Spark can read from PostgreSQL (models table: {count} rows)")

        spark.stop()

    except Exception as e:
        print(f"⚠️  Spark PostgreSQL test failed: {e}")
        print("   This is optional - Flask can still use PostgreSQL directly")


if __name__ == "__main__":
    init_database()

    # Optional Spark verification
    try:
        verify_spark_connection()
    except Exception:
        pass
