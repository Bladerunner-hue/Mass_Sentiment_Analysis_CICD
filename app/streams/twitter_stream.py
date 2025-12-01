"""Twitter streaming integration for real-time sentiment analysis."""

import os
import json
from datetime import datetime
from typing import List

import redis
import tweepy
from celery import shared_task
from sqlalchemy.exc import SQLAlchemyError

from app.extensions import db
from app.models.analysis import SentimentAnalysis
from app.services.sentiment_service import SentimentService


class TwitterStreamListener(tweepy.StreamingClient):
    """Real-time Twitter stream processor for sentiment analysis."""

    def __init__(self, bearer_token: str, keywords: List[str]):
        super().__init__(bearer_token)
        self.keywords = keywords
        self.service = SentimentService()
        self.active_streams = set()

    def on_tweet(self, tweet):
        """Process each tweet in real-time."""
        try:
            # Skip retweets and replies for cleaner data
            if tweet.referenced_tweets:
                return

            # Analyze sentiment
            result = self.service.analyze_full(tweet.text)

            # Store in database
            analysis = SentimentAnalysis.create_from_result(
                user_id=None,  # System analysis
                text=tweet.text,
                result=result,
                source='twitter_stream'
            )
            analysis.stream_metadata = {
                'tweet_id': str(tweet.id),
                'author_id': str(tweet.author_id),
                'author_username': getattr(tweet.author, 'username', None),
                'created_at': str(tweet.created_at),
                'lang': getattr(tweet, 'lang', None),
                'retweet_count': getattr(tweet, 'retweet_count', 0),
                'like_count': getattr(tweet, 'like_count', 0),
                'keywords': self.keywords
            }

            db.session.add(analysis)
            db.session.commit()

            # Publish to Redis for real-time dashboard
            redis_client = redis.Redis.from_url(
                os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            )
            redis_client.publish('sentiment_updates', json.dumps({
                'type': 'twitter_sentiment',
                'data': {
                    'tweet_id': str(tweet.id),
                    'text': tweet.text[:100] + '...' if len(tweet.text) > 100
                           else tweet.text,
                    'sentiment': result['sentiment'],
                    'emotion': result['primary_emotion'],
                    'confidence': result['confidence'],
                    'timestamp': str(tweet.created_at)
                }
            }))

        except SQLAlchemyError as e:
            db.session.rollback()
            print(f"Database error processing tweet {tweet.id}: {e}")
        except Exception as e:
            print(f"Error processing tweet {tweet.id}: {e}")

    def on_errors(self, errors):
        """Handle streaming errors."""
        print(f"Twitter stream errors: {errors}")

    def on_connection_error(self):
        """Handle connection errors."""
        print("Twitter stream connection error")

    def on_disconnect(self):
        """Handle disconnection."""
        print("Twitter stream disconnected")


class RedditStreamProcessor:
    """Reddit streaming processor using PRAW."""

    def __init__(self, client_id: str, client_secret: str,
                 user_agent: str, subreddits: List[str]):
        try:
            import praw
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
            self.subreddits = subreddits
            self.service = SentimentService()
        except ImportError:
            raise ImportError("PRAW not installed. Run: pip install praw")

    def process_submission(self, submission):
        """Process a Reddit submission."""
        try:
            result = self.service.analyze_full(submission.selftext or submission.title)

            # Store in database
            analysis = SentimentAnalysis.create_from_result(
                user_id=None,  # System analysis
                text=submission.selftext or submission.title,
                result=result,
                source='reddit_stream'
            )
            analysis.stream_metadata = {
                'submission_id': submission.id,
                'subreddit': submission.subreddit.display_name,
                'author': str(submission.author) if submission.author else None,
                'created_utc': submission.created_utc,
                'score': submission.score,
                'num_comments': submission.num_comments,
                'url': submission.url
            }

            db.session.add(analysis)
            db.session.commit()

            # Publish to Redis
            redis_client = redis.Redis.from_url(
                os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            )
            redis_client.publish('sentiment_updates', json.dumps({
                'type': 'reddit_sentiment',
                'data': {
                    'submission_id': submission.id,
                    'text': (submission.selftext or submission.title)[:100] + '...',
                    'sentiment': result['sentiment'],
                    'emotion': result['primary_emotion'],
                    'confidence': result['confidence'],
                    'subreddit': submission.subreddit.display_name,
                    'timestamp': submission.created_utc
                }
            }))

        except Exception as e:
            print(f"Error processing Reddit submission {submission.id}: {e}")
            db.session.rollback()


@shared_task
def process_news_feeds(feed_urls: List[str]):
    """Process RSS/Atom news feeds for sentiment analysis."""
    try:
        import feedparser
        from newspaper import Article
    except ImportError:
        print("Required packages not installed. Install with: "
              "pip install feedparser newspaper3k")
        return

    from app.services.sentiment_service import SentimentService
    from app.models.analysis import SentimentAnalysis
    from app.extensions import db
    import redis
    import json

    service = SentimentService()

    for feed_url in feed_urls:
        try:
            feed = feedparser.parse(feed_url)

            for entry in feed.entries[:10]:  # Limit to 10 recent articles
                # per feed
                try:
                    # Extract article content
                    article = Article(entry.link)
                    article.download()
                    article.parse()

                    # Analyze sentiment
                    result = service.analyze_full(article.text)

                    # Store in database
                    analysis = SentimentAnalysis.create_from_result(
                        user_id=None,  # System analysis
                        text=article.title + " " + article.text[:500],
                        result=result,
                        source='news_feed'
                    )
                    analysis.stream_metadata = {
                        'feed_url': feed_url,
                        'article_url': entry.link,
                        'title': article.title,
                        'publish_date': getattr(entry, 'published', None),
                        'source': feed.feed.title if hasattr(feed.feed, 'title') else None
                    }

                    db.session.add(analysis)
                    db.session.commit()

                    # Publish to Redis
                    redis_client = redis.Redis.from_url(
                        os.getenv('REDIS_URL', 'redis://localhost:6379/0')
                    )
                    redis_client.publish('sentiment_updates', json.dumps({
                        'type': 'news_sentiment',
                        'data': {
                            'title': article.title,
                            'text': article.text[:200] + '...' if len(article.text) > 200 else article.text,
                            'sentiment': result['sentiment'],
                            'emotion': result['primary_emotion'],
                            'confidence': result['confidence'],
                            'source': feed.feed.title if hasattr(feed.feed, 'title') else None,
                            'url': entry.link
                        }
                    }))

                except Exception as e:
                    print(f"Error processing article {entry.link}: {e}")
                    db.session.rollback()

        except Exception as e:
            print(f"Error processing feed {feed_url}: {e}")


@shared_task
def start_twitter_stream(keywords: List[str], duration_minutes: int = 60):
    """Start Twitter stream in background for specified duration."""
    bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
    if not bearer_token:
        raise ValueError("TWITTER_BEARER_TOKEN environment variable required")

    stream = TwitterStreamListener(bearer_token, keywords)

    # Add filtering rules
    rules = [tweepy.StreamRule(keyword) for keyword in keywords]
    stream.add_rules(rules)

    # Start streaming
    try:
        print(f"Starting Twitter stream for keywords: {keywords}")
        stream.filter(threaded=True)

        # Keep stream alive for specified duration
        import time
        time.sleep(duration_minutes * 60)

    except Exception as e:
        print(f"Twitter streaming error: {e}")
    finally:
        # Clean up rules
        try:
            stream.delete_rules(stream.get_rules().data)
        except Exception:
            pass


@shared_task
def start_reddit_stream(subreddits: List[str], duration_minutes: int = 60):
    """Start Reddit stream in background."""
    try:
        import praw  # noqa: F401
    except ImportError:
        print("PRAW not installed. Install with: pip install praw")
        return

    client_id = os.getenv('REDDIT_CLIENT_ID')
    client_secret = os.getenv('REDDIT_CLIENT_SECRET')
    user_agent = os.getenv('REDDIT_USER_AGENT', 'SentimentAnalysisBot/1.0')

    if not client_id or not client_secret:
        raise ValueError("REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET required")

    processor = RedditStreamProcessor(client_id, client_secret, user_agent, subreddits)

    try:
        print(f"Starting Reddit stream for subreddits: {subreddits}")

        # Stream submissions from specified subreddits
        subreddit_str = '+'.join(subreddits)
        subreddit = processor.reddit.subreddit(subreddit_str)

        import time
        start_time = time.time()

        for submission in subreddit.stream.submissions():
            if time.time() - start_time > duration_minutes * 60:
                break

            processor.process_submission(submission)

    except Exception as e:
        print(f"Reddit streaming error: {e}")


class StreamManager:
    """Manager for coordinating multiple data streams."""

    def __init__(self):
        self.active_streams = {}

    def start_twitter_stream(self, keywords: List[str], duration_minutes: int = 60):
        """Start Twitter streaming."""
        task = start_twitter_stream.delay(keywords, duration_minutes)
        stream_id = f"twitter_{task.id}"
        self.active_streams[stream_id] = {
            'type': 'twitter',
            'task_id': task.id,
            'keywords': keywords,
            'started_at': datetime.utcnow()
        }
        return stream_id

    def start_reddit_stream(self, subreddits: List[str], duration_minutes: int = 60):
        """Start Reddit streaming."""
        task = start_reddit_stream.delay(subreddits, duration_minutes)
        stream_id = f"reddit_{task.id}"
        self.active_streams[stream_id] = {
            'type': 'reddit',
            'task_id': task.id,
            'subreddits': subreddits,
            'started_at': datetime.utcnow()
        }
        return stream_id

    def start_news_stream(self, feed_urls: List[str]):
        """Start news feed processing."""
        task = process_news_feeds.delay(feed_urls)
        stream_id = f"news_{task.id}"
        self.active_streams[stream_id] = {
            'type': 'news',
            'task_id': task.id,
            'feed_urls': feed_urls,
            'started_at': datetime.utcnow()
        }
        return stream_id

    def stop_stream(self, stream_id: str):
        """Stop a specific stream."""
        if stream_id in self.active_streams:
            task_id = self.active_streams[stream_id]['task_id']
            from flask import current_app
            celery = current_app.extensions['celery']
            celery.control.revoke(task_id, terminate=True)
            del self.active_streams[stream_id]
            return True
        return False

    def get_active_streams(self):
        """Get information about active streams."""
        return self.active_streams.copy()


# Global stream manager instance
stream_manager = StreamManager()
