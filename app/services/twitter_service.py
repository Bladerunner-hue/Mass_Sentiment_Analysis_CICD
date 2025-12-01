"""X (Twitter) API service for fetching tweets for sentiment analysis."""

import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests


class XTwitterService:
    """Service for interacting with X (Twitter) API v2.
    
    Requires:
    - X_BEARER_TOKEN: Bearer token for API v2 access
    - X_API_KEY: API Key (optional, for OAuth 1.0a)
    - X_API_SECRET: API Secret (optional, for OAuth 1.0a)
    - X_ACCESS_TOKEN: Access Token (optional, for OAuth 1.0a)
    - X_ACCESS_SECRET: Access Token Secret (optional, for OAuth 1.0a)
    """

    BASE_URL = "https://api.twitter.com/2"

    def __init__(self):
        self.bearer_token = os.environ.get("X_BEARER_TOKEN")
        self.api_key = os.environ.get("X_API_KEY")
        self.api_secret = os.environ.get("X_API_SECRET")
        self.access_token = os.environ.get("X_ACCESS_TOKEN")
        self.access_secret = os.environ.get("X_ACCESS_SECRET")
        
        self._session = requests.Session()
        if self.bearer_token:
            self._session.headers.update({
                "Authorization": f"Bearer {self.bearer_token}",
                "Content-Type": "application/json"
            })

    @property
    def is_configured(self) -> bool:
        """Check if the service is properly configured."""
        return bool(self.bearer_token)

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        method: str = "GET"
    ) -> Dict[str, Any]:
        """Make an API request with rate limit handling.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            method: HTTP method
            
        Returns:
            API response data
        """
        if not self.is_configured:
            return {"error": "X API not configured. Set X_BEARER_TOKEN environment variable."}
        
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            if method == "GET":
                response = self._session.get(url, params=params, timeout=30)
            else:
                response = self._session.post(url, json=params, timeout=30)
            
            # Handle rate limiting
            if response.status_code == 429:
                reset_time = int(response.headers.get("x-rate-limit-reset", 0))
                wait_seconds = max(reset_time - int(time.time()), 60)
                return {
                    "error": "Rate limit exceeded",
                    "retry_after_seconds": wait_seconds,
                    "rate_limit": True
                }
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def search_recent_tweets(
        self,
        query: str,
        max_results: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        next_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Search for recent tweets (last 7 days).
        
        Args:
            query: Search query (supports Twitter search operators)
            max_results: Maximum number of results (10-100)
            start_time: Start of time range
            end_time: End of time range
            next_token: Pagination token
            
        Returns:
            Search results with tweets and metadata
        """
        params = {
            "query": query,
            "max_results": min(max(max_results, 10), 100),
            "tweet.fields": "created_at,author_id,public_metrics,lang,source,context_annotations",
            "user.fields": "name,username,profile_image_url,verified",
            "expansions": "author_id"
        }
        
        if start_time:
            params["start_time"] = start_time.isoformat() + "Z"
        if end_time:
            params["end_time"] = end_time.isoformat() + "Z"
        if next_token:
            params["next_token"] = next_token
        
        result = self._make_request("tweets/search/recent", params)
        
        if "error" in result:
            return result
        
        # Process and enrich results
        tweets = result.get("data", [])
        users = {u["id"]: u for u in result.get("includes", {}).get("users", [])}
        
        processed_tweets = []
        for tweet in tweets:
            author = users.get(tweet.get("author_id"), {})
            processed_tweets.append({
                "id": tweet["id"],
                "text": tweet["text"],
                "created_at": tweet.get("created_at"),
                "lang": tweet.get("lang"),
                "source": tweet.get("source"),
                "author": {
                    "id": author.get("id"),
                    "name": author.get("name"),
                    "username": author.get("username"),
                    "verified": author.get("verified", False)
                },
                "metrics": tweet.get("public_metrics", {})
            })
        
        return {
            "tweets": processed_tweets,
            "meta": result.get("meta", {}),
            "next_token": result.get("meta", {}).get("next_token")
        }

    def get_user_tweets(
        self,
        username: str,
        max_results: int = 100,
        exclude_replies: bool = True,
        exclude_retweets: bool = True
    ) -> Dict[str, Any]:
        """Get tweets from a specific user.
        
        Args:
            username: Twitter username (without @)
            max_results: Maximum number of tweets
            exclude_replies: Whether to exclude replies
            exclude_retweets: Whether to exclude retweets
            
        Returns:
            User's tweets with metadata
        """
        # First, get user ID
        user_result = self._make_request(f"users/by/username/{username}")
        
        if "error" in user_result:
            return user_result
        
        user_data = user_result.get("data", {})
        user_id = user_data.get("id")
        
        if not user_id:
            return {"error": f"User not found: {username}"}
        
        # Get user's tweets
        params = {
            "max_results": min(max(max_results, 5), 100),
            "tweet.fields": "created_at,public_metrics,lang,source",
        }
        
        excludes = []
        if exclude_replies:
            excludes.append("replies")
        if exclude_retweets:
            excludes.append("retweets")
        if excludes:
            params["exclude"] = ",".join(excludes)
        
        result = self._make_request(f"users/{user_id}/tweets", params)
        
        if "error" in result:
            return result
        
        return {
            "user": {
                "id": user_data.get("id"),
                "name": user_data.get("name"),
                "username": user_data.get("username")
            },
            "tweets": result.get("data", []),
            "meta": result.get("meta", {})
        }

    def get_trending_topics(
        self,
        woeid: int = 1  # 1 = Worldwide
    ) -> Dict[str, Any]:
        """Get trending topics for a location.
        
        Note: This endpoint requires Twitter API v1.1 and OAuth 1.0a authentication.
        
        Args:
            woeid: Yahoo! Where On Earth ID (1 = Worldwide, 23424977 = USA)
            
        Returns:
            Trending topics
        """
        # This would require OAuth 1.0a authentication
        # For now, return a message about the limitation
        return {
            "error": "Trending topics require Twitter API v1.1 with OAuth 1.0a authentication",
            "suggestion": "Use search_recent_tweets with popular hashtags instead"
        }

    def stream_sample(self, callback, duration_seconds: int = 60) -> Dict[str, Any]:
        """Stream a sample of real-time tweets.
        
        Note: Requires Elevated access level.
        
        Args:
            callback: Function to call with each tweet
            duration_seconds: How long to stream
            
        Returns:
            Stream statistics
        """
        if not self.is_configured:
            return {"error": "X API not configured"}
        
        url = f"{self.BASE_URL}/tweets/sample/stream"
        params = {
            "tweet.fields": "created_at,author_id,lang,source"
        }
        
        tweet_count = 0
        start_time = time.time()
        
        try:
            with self._session.get(url, params=params, stream=True, timeout=duration_seconds + 10) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if time.time() - start_time > duration_seconds:
                        break
                    
                    if line:
                        import json
                        try:
                            tweet = json.loads(line)
                            if "data" in tweet:
                                callback(tweet["data"])
                                tweet_count += 1
                        except json.JSONDecodeError:
                            continue
            
            return {
                "success": True,
                "tweets_collected": tweet_count,
                "duration_seconds": time.time() - start_time
            }
            
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "tweets_collected": tweet_count}

    def collect_for_sentiment_analysis(
        self,
        query: str,
        target_count: int = 1000,
        languages: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Collect tweets for sentiment analysis.
        
        Args:
            query: Search query
            target_count: Target number of tweets to collect
            languages: List of language codes to filter (e.g., ['en', 'es'])
            
        Returns:
            Collection of tweets ready for sentiment analysis
        """
        if languages:
            lang_filter = " (" + " OR ".join([f"lang:{lang}" for lang in languages]) + ")"
            query = query + lang_filter
        
        # Add filter to exclude retweets for cleaner data
        query = f"{query} -is:retweet"
        
        all_tweets = []
        next_token = None
        
        while len(all_tweets) < target_count:
            result = self.search_recent_tweets(
                query=query,
                max_results=100,
                next_token=next_token
            )
            
            if "error" in result:
                if result.get("rate_limit"):
                    # Rate limited, return what we have
                    break
                return result
            
            tweets = result.get("tweets", [])
            if not tweets:
                break
            
            all_tweets.extend(tweets)
            next_token = result.get("next_token")
            
            if not next_token:
                break
            
            # Small delay to be nice to the API
            time.sleep(1)
        
        # Format for sentiment analysis
        return {
            "success": True,
            "query": query,
            "total_collected": len(all_tweets),
            "tweets": [
                {
                    "id": t["id"],
                    "text": t["text"],
                    "created_at": t.get("created_at"),
                    "lang": t.get("lang"),
                    "username": t.get("author", {}).get("username"),
                    "retweets": t.get("metrics", {}).get("retweet_count", 0),
                    "likes": t.get("metrics", {}).get("like_count", 0)
                }
                for t in all_tweets[:target_count]
            ]
        }
