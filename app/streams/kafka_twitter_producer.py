"""Kafka producer to ingest Twitter data for Spark streaming."""

import json
import time
import os
from typing import List, Optional
from kafka import KafkaProducer
from kafka.errors import KafkaError
import tweepy
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TwitterKafkaProducer:
    """
    Produce Twitter stream data to Kafka for Spark consumption.
    
    Features:
    - Twitter API v2 filtered stream
    - JSON serialization
    - Error handling and reconnection
    - Rate limiting awareness
    """
    
    def __init__(
        self,
        kafka_servers: str = "localhost:9092",
        kafka_topic: str = "twitter_stream",
        twitter_bearer_token: Optional[str] = None
    ):
        self.kafka_servers = kafka_servers
        self.kafka_topic = kafka_topic
        self.twitter_bearer_token = twitter_bearer_token or os.getenv("TWITTER_BEARER_TOKEN")
        
        if not self.twitter_bearer_token:
            raise ValueError("Twitter bearer token is required")
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',
            retries=3,
            max_in_flight_requests_per_connection=1
        )
        
        # Initialize Twitter client
        self.client = tweepy.Client(
            bearer_token=self.twitter_bearer_token,
            wait_on_rate_limit=True
        )
        
        logger.info(f"Kafka producer initialized: {kafka_servers}, topic: {kafka_topic}")
    
    def _format_tweet(self, tweet: tweepy.Tweet) -> dict:
        """Format tweet data for Kafka."""
        return {
            "id": str(tweet.id),
            "text": tweet.text,
            "user": tweet.author_id if hasattr(tweet, 'author_id') else None,
            "created_at": tweet.created_at.strftime("%a %b %d %H:%M:%S %z %Y") if tweet.created_at else None,
            "lang": tweet.lang if hasattr(tweet, 'lang') else None,
            "retweet_count": str(tweet.public_metrics.get('retweet_count', 0)) if hasattr(tweet, 'public_metrics') else "0",
            "favorite_count": str(tweet.public_metrics.get('like_count', 0)) if hasattr(tweet, 'public_metrics') else "0",
            "hashtags": json.dumps([tag['tag'] for tag in tweet.entities.get('hashtags', [])] if hasattr(tweet, 'entities') and tweet.entities else [])
        }
    
    def send_to_kafka(self, tweet_data: dict) -> bool:
        """Send tweet data to Kafka."""
        try:
            future = self.producer.send(self.kafka_topic, value=tweet_data)
            record_metadata = future.get(timeout=10)
            
            logger.debug(
                f"Sent to Kafka - Topic: {record_metadata.topic}, "
                f"Partition: {record_metadata.partition}, "
                f"Offset: {record_metadata.offset}"
            )
            return True
        except KafkaError as e:
            logger.error(f"Failed to send to Kafka: {e}")
            return False
    
    def setup_filtered_stream(
        self,
        keywords: List[str],
        languages: List[str] = ['en']
    ):
        """
        Setup Twitter filtered stream rules.
        
        Args:
            keywords: List of keywords to track
            languages: List of language codes
        """
        # Delete existing rules
        existing_rules = self.client.get_stream_rules()
        if existing_rules.data:
            rule_ids = [rule.id for rule in existing_rules.data]
            self.client.delete_stream_rules(rule_ids)
            logger.info(f"Deleted {len(rule_ids)} existing rules")
        
        # Create new rules
        rules = []
        for keyword in keywords:
            rule_value = f"{keyword} lang:{' OR lang:'.join(languages)}"
            rules.append(tweepy.StreamRule(value=rule_value))
        
        self.client.add_stream_rules(rules)
        logger.info(f"Added {len(rules)} stream rules")
    
    def start_streaming(
        self,
        keywords: List[str],
        languages: List[str] = ['en'],
        max_tweets: Optional[int] = None
    ):
        """
        Start streaming tweets to Kafka.
        
        Args:
            keywords: Keywords to track
            languages: Languages to filter
            max_tweets: Maximum number of tweets (None for infinite)
        """
        logger.info("=" * 60)
        logger.info("Starting Twitter to Kafka streaming")
        logger.info(f"Keywords: {keywords}")
        logger.info(f"Languages: {languages}")
        logger.info("=" * 60)
        
        # Setup filtered stream
        self.setup_filtered_stream(keywords, languages)
        
        # Start streaming
        tweet_count = 0
        
        try:
            # Stream tweets
            for response in self.client.search_recent_tweets(
                query=' OR '.join(keywords),
                max_results=100,
                tweet_fields=['created_at', 'lang', 'public_metrics', 'entities', 'author_id']
            ):
                if response.data:
                    for tweet in response.data:
                        tweet_data = self._format_tweet(tweet)
                        
                        if self.send_to_kafka(tweet_data):
                            tweet_count += 1
                            
                            if tweet_count % 100 == 0:
                                logger.info(f"Processed {tweet_count} tweets")
                            
                            if max_tweets and tweet_count >= max_tweets:
                                logger.info(f"Reached max tweets: {max_tweets}")
                                return
                
                # Rate limiting
                time.sleep(1)
        
        except KeyboardInterrupt:
            logger.info("\nStreaming interrupted by user")
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise
        finally:
            self.close()
            logger.info(f"Total tweets processed: {tweet_count}")
    
    def close(self):
        """Close Kafka producer."""
        self.producer.flush()
        self.producer.close()
        logger.info("Kafka producer closed")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Stream Twitter data to Kafka")
    parser.add_argument(
        '--keywords',
        nargs='+',
        default=['bitcoin', 'ethereum', 'crypto'],
        help='Keywords to track'
    )
    parser.add_argument(
        '--languages',
        nargs='+',
        default=['en'],
        help='Languages to filter'
    )
    parser.add_argument(
        '--max-tweets',
        type=int,
        default=None,
        help='Maximum number of tweets'
    )
    parser.add_argument(
        '--kafka-servers',
        default=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
        help='Kafka bootstrap servers'
    )
    parser.add_argument(
        '--kafka-topic',
        default=os.getenv('KAFKA_TOPIC_TWITTER', 'twitter_stream'),
        help='Kafka topic'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    producer = TwitterKafkaProducer(
        kafka_servers=args.kafka_servers,
        kafka_topic=args.kafka_topic
    )
    
    producer.start_streaming(
        keywords=args.keywords,
        languages=args.languages,
        max_tweets=args.max_tweets
    )


if __name__ == "__main__":
    main()
