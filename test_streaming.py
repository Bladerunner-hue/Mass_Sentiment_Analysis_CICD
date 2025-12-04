#!/usr/bin/env python3
"""Test script for streaming infrastructure."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.streams.twitter_stream import stream_manager
from app.services.sentiment_service import SentimentService


def test_streaming_infrastructure():
    """Test the streaming infrastructure components."""
    print("Testing streaming infrastructure...")

    # Create Flask app
    app = create_app("testing")

    with app.app_context():
        # Test sentiment service
        service = SentimentService()
        test_text = "This is a great day for testing!"
        result = service.analyze_full(test_text)

        print(f"✓ Sentiment analysis works: {result['sentiment']}")

        # Test stream manager
        active_count = len(stream_manager.active_streams)
        print(f"✓ Stream manager initialized: {active_count} active streams")

        # Test analysis creation
        from app.models.analysis import SentimentAnalysis

        analysis = SentimentAnalysis.create_from_result(
            user_id=None, text=test_text, result=result, source="test"
        )

        print(f"✓ Analysis creation works: {analysis.sentiment}")

        print("All streaming infrastructure tests passed!")


if __name__ == "__main__":
    test_streaming_infrastructure()
